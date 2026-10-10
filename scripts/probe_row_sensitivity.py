#!/usr/bin/env python3

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
EZKL = REPO / "target" / "release" / "ezkl"
GEN = REPO / "scripts" / "gen_decode_step_onnx.py"

VIS = [
    "--input-visibility",
    "polycommit",
    "--output-visibility",
    "polycommit",
    "--param-visibility",
    "fixed",
]
OPT_ENV = {"EZKL_DIV_ROUND_NEAREST": "1", "EZKL_RESCALE_FUSION": "1"}


def run(*args, cwd, check=True):
    import os

    e = dict(os.environ)
    for v in ("EZKL_DIV_ROUND_NEAREST", "EZKL_RESCALE_FUSION"):
        e.pop(v, None)
    e.update(OPT_ENV)
    r = subprocess.run(
        [str(a) for a in args], cwd=cwd, capture_output=True, text=True, env=e
    )
    if check and r.returncode != 0:
        sys.exit(f"command failed: {' '.join(map(str, args))}\n{r.stdout}\n{r.stderr}")
    return r.stdout + r.stderr


def logrows_for(n):
    return max(1, (n - 1).bit_length())


def gen_settings(mdir):
    run(
        EZKL,
        "gen-settings",
        "-M",
        "model.onnx",
        "-O",
        "settings.json",
        "-K",
        24,
        *VIS,
        cwd=mdir,
    )
    n = json.loads((mdir / "settings.json").read_text())["num_rows"]
    return n


def spread(rows):
    lo, hi = min(rows), max(rows)
    return {
        "min": lo,
        "max": hi,
        "spread_pct": round(100 * (hi - lo) / lo, 2),
        "logrows_min": logrows_for(lo),
        "logrows_max": logrows_for(hi),
    }


def main():
    ap = argparse.ArgumentParser(
        description=None, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--d-model", type=int, default=128)
    ap.add_argument("--t-max", type=int, default=16)
    ap.add_argument("--vocab", type=int, default=1024)
    ap.add_argument("--n", type=int, default=8, help="每个探针跑几个样本")
    ap.add_argument("--workdir", default=None)
    ap.add_argument(
        "--out", default=str(REPO / "results" / "incremental" / "p3_embed_outside")
    )
    args = ap.parse_args()

    wd = (
        Path(args.workdir)
        if args.workdir
        else Path(tempfile.mkdtemp(prefix="ezkl_sens_"))
    )
    wd.mkdir(parents=True, exist_ok=True)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cfg = [
        "--variant",
        "where",
        "--d-model",
        args.d_model,
        "--t-max",
        args.t_max,
        "--vocab",
        args.vocab,
        "--embed-outside",
    ]
    print(f"[workdir] {wd}\n[config] d={args.d_model} T={args.t_max} V={args.vocab}")

    print("\n=== A. 只换 x_t（喂不同的 token）===")
    mdir = wd / "probe_input"
    run(sys.executable, GEN, *cfg, "--out-dir", mdir, cwd=REPO)
    wte = json.loads((mdir / "wte.json").read_text())
    d = args.d_model
    base = json.loads((mdir / "input.json").read_text())
    a_rows = []
    for i in range(args.n):
        tok = (i * (args.vocab // args.n)) % args.vocab
        base["input_data"][0] = wte["data"][tok * d : (tok + 1) * d]
        (mdir / "input.json").write_text(json.dumps(base))
        n = gen_settings(mdir)
        a_rows.append({"token": tok, "num_rows": n, "logrows": logrows_for(n)})
        print(f"  token={tok:5d} -> rows={n:9d} k={logrows_for(n)}")
    a_stat = spread([r["num_rows"] for r in a_rows])
    print(
        f"  离散度 {a_stat['spread_pct']}%  (k {a_stat['logrows_min']}~{a_stat['logrows_max']})"
    )

    print("\n=== B. 换 --seed（整个模型重新抽）===")
    b_rows = []
    for s in range(args.n):
        sdir = wd / f"probe_seed{s}"
        run(sys.executable, GEN, *cfg, "--seed", s, "--out-dir", sdir, cwd=REPO)
        n = gen_settings(sdir)
        b_rows.append({"seed": s, "num_rows": n, "logrows": logrows_for(n)})
        print(f"  seed={s:5d} -> rows={n:9d} k={logrows_for(n)}")
    b_stat = spread([r["num_rows"] for r in b_rows])
    print(
        f"  离散度 {b_stat['spread_pct']}%  (k {b_stat['logrows_min']}~{b_stat['logrows_max']})"
    )

    rec = {
        "config": {
            "d_model": args.d_model,
            "t_max": args.t_max,
            "vocab": args.vocab,
            "variant": "embed_outside",
            "optimizations": "on",
        },
        "probe_A_input_only": {"samples": a_rows, **a_stat},
        "probe_B_full_instance": {"samples": b_rows, **b_stat},
    }
    (out / "row_sensitivity.json").write_text(json.dumps(rec, indent=2) + "\n")
    print(f"\n[wrote] {out/'row_sensitivity.json'}")
    print(f"[workdir kept] {wd}")


if __name__ == "__main__":
    main()
