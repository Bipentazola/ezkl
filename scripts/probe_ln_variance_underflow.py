#!/usr/bin/env python3

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
EZKL = REPO / "target" / "release" / "ezkl"
D_MODEL = 64
SEED = 0
ERRLINE = re.compile(r"\[E\].*")
TOOLARGE = re.compile(r"integer (\d+) is too large")


def export_ln_onnx(path, d):
    import torch
    import torch.nn as nn

    class LN(nn.Module):
        def __init__(self, d):
            super().__init__()
            self.ln = nn.LayerNorm(d)

        def forward(self, x):
            return self.ln(x)

    torch.manual_seed(SEED)
    torch.onnx.export(
        LN(d).eval(),
        torch.randn(1, 1, d),
        str(path),
        input_names=["x"],
        output_names=["y"],
        opset_version=17,
        dynamo=False,
    )


def run(cmd, env=None):
    return subprocess.run(
        [str(c) for c in cmd],
        cwd=str(REPO),
        env=dict(os.environ, **(env or {})),
        capture_output=True,
        text=True,
    )


def trial(onnx_path, wd, sigma, scale, fusion, div_new, seed=SEED, debug=False):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(D_MODEL) * sigma
    dpath = wd / "in.json"
    json.dump({"input_data": [x.tolist()]}, open(dpath, "w"))

    env = {"EZKL_DIV_ROUND_NEAREST": str(div_new), "EZKL_RESCALE_FUSION": str(fusion)}
    if debug:
        env["RUST_LOG"] = "debug"

    s, c, w = wd / "s.json", wd / "ln.compiled", wd / "w.json"
    r = run(
        [
            EZKL,
            "gen-settings",
            "-M",
            onnx_path,
            "-O",
            s,
            "--input-scale",
            scale,
            "--param-scale",
            scale,
        ],
        env,
    )
    if r.returncode:
        return "SETTINGS_ERR", (ERRLINE.findall(r.stderr) or [""])[-1], None, x
    r = run(
        [EZKL, "compile-circuit", "-M", onnx_path, "-S", s, "--compiled-circuit", c],
        env,
    )
    if r.returncode:
        return "COMPILE_ERR", (ERRLINE.findall(r.stderr) or [""])[-1], None, x

    r = run([EZKL, "gen-witness", "-M", c, "-D", dpath, "-O", w], env)
    blob = (r.stderr or "") + (r.stdout or "")
    if r.returncode == 0:
        return "OK", "", None, x
    line = (ERRLINE.findall(blob) or [""])[0]
    m = TOOLARGE.search(blob)
    return "FAIL", line.strip(), (int(m.group(1)) if m else None), x


def stats(x, scale):
    v = float(np.var(x))
    return {
        "var": v,
        "var_x_2S": v * 2**scale,
        "frac_x_quantized_to_zero": float((np.round(x * 2**scale) == 0).mean()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--out", default="results/ln_variance_underflow/ln_variance_underflow.json"
    )
    ap.add_argument("--scales", default="7,9,11,13")
    args = ap.parse_args()
    scales = [int(s) for s in args.scales.split(",")]

    out = {
        "d_model": D_MODEL,
        "seed": SEED,
        "scales": scales,
        "criterion": "var(x) * 2^S < 0.5  =>  gen-witness FAIL",
        "sweeps": [],
        "threshold": [],
        "witness_integer": [],
    }

    with tempfile.TemporaryDirectory() as td:
        wd = Path(td)
        onnx_path = wd / "ln.onnx"
        export_ln_onnx(onnx_path, D_MODEL)

        coarse = [
            1.0,
            0.5,
            0.25,
            0.125,
            0.0884,
            0.0625,
            0.05,
            0.04,
            0.03,
            0.02,
            0.01,
            0.004,
        ]
        fine = [0.0700, 0.0690, 0.0680, 0.0670, 0.0660, 0.0650, 0.0640, 0.0630, 0.0620]
        for sigma in coarse + fine:
            rec = {"sigma": sigma, "scale": 7}
            for tag, (fus, dv) in {"both_on": (1, 1), "both_off": (0, 0)}.items():
                st, line, val, x = trial(onnx_path, wd, sigma, 7, fus, dv)
                rec[tag] = {"status": st, "error": line, "witness_integer": val}
                rec.update(stats(x, 7))
            rec["optimizations_agree"] = (
                rec["both_on"]["status"] == rec["both_off"]["status"]
                and rec["both_on"]["witness_integer"]
                == rec["both_off"]["witness_integer"]
            )
            out["sweeps"].append(rec)
            print(
                "  sigma=%-8.4f var*2^7=%-9.4f 归零%5.1f%%  两侧=%-5s %s"
                % (
                    sigma,
                    rec["var_x_2S"],
                    rec["frac_x_quantized_to_zero"] * 100,
                    rec["both_on"]["status"],
                    "一致" if rec["optimizations_agree"] else "**不一致**",
                )
            )

        print()
        for S in scales:
            pred = (0.5 / 2**S) ** 0.5
            grid = sorted(
                {
                    pred * f
                    for f in (2.0, 1.6, 1.3, 1.15, 1.05, 1.0, 0.95, 0.85, 0.7, 0.5)
                },
                reverse=True,
            )
            last_ok = first_fail = None
            for sigma in grid:
                st, _, _, x = trial(onnx_path, wd, sigma, S, 1, 1)
                st_stats = stats(x, S)
                if st == "OK":
                    last_ok = (sigma, st_stats["var_x_2S"])
                elif first_fail is None:
                    first_fail = (sigma, st_stats["var_x_2S"])
            out["threshold"].append(
                {
                    "scale": S,
                    "predicted_sigma_star": pred,
                    "last_ok_sigma": last_ok[0] if last_ok else None,
                    "last_ok_var_x_2S": last_ok[1] if last_ok else None,
                    "first_fail_sigma": first_fail[0] if first_fail else None,
                    "first_fail_var_x_2S": first_fail[1] if first_fail else None,
                }
            )
            print(
                "  scale %-3d 预测 sigma*=%.5f  最后OK var*2^S=%.3f  首个FAIL var*2^S=%.3f"
                % (
                    S,
                    pred,
                    last_ok[1] if last_ok else -1,
                    first_fail[1] if first_fail else -1,
                )
            )

        print()
        for S in scales:
            sigma = 0.3 * (0.5 / 2**S) ** 0.5
            st, line, val, _ = trial(onnx_path, wd, sigma, S, 1, 1, debug=True)
            exp = 2 ** (S + 52)
            out["witness_integer"].append(
                {
                    "scale": S,
                    "sigma": sigma,
                    "status": st,
                    "witness_integer": val,
                    "expected_2_pow_S_plus_52": exp,
                    "matches": val == exp,
                    "error": line,
                }
            )
            print(
                "  scale %-3d 见证整数 %-22s 2^(S+52)=%-22s %s"
                % (S, val, exp, "命中" if val == exp else "不符")
            )

    op = REPO / args.out
    op.parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(op, "w"), indent=2)
    print("\n写入 %s" % args.out)


if __name__ == "__main__":
    sys.exit(main())
