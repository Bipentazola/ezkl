#!/usr/bin/env python3

import json
import os
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
EZKL = REPO / "target" / "release" / "ezkl"
MODEL = REPO / "models" / "seq8_d16_l1"
OUT = REPO / "results" / "wide_divisor_e2e"

CENSUS_RE = re.compile(r"divisor=(\d+) path=(\w+) limbs=(\d+) elems=(\d+)")
FUSION_RE = re.compile(r"rescale fusion: (.*)")

CONFIGS = [
    {
        "name": "scale7_both",
        "scale": 7,
        "env": {"EZKL_DIV_ROUND_NEAREST": "1", "EZKL_RESCALE_FUSION": "1"},
    },
    {
        "name": "scale13_both",
        "scale": 13,
        "env": {"EZKL_DIV_ROUND_NEAREST": "1", "EZKL_RESCALE_FUSION": "1"},
    },
    {"name": "scale7_divnew_only", "scale": 7, "env": {"EZKL_DIV_ROUND_NEAREST": "1"}},
]
DECOMP_BASE = 16384


def run(args, env, cwd=None, census=False):
    e = {**os.environ, **env}
    if census:
        e["RUST_LOG"] = "error,divisor_census=debug"
    r = subprocess.run(
        [str(a) for a in args], cwd=cwd, capture_output=True, text=True, env=e
    )
    return r


def census_of(text):
    c = Counter()
    for d, p, n, _elems in CENSUS_RE.findall(text):
        c[(int(d), p, int(n))] += 1
    return c


def main():
    if not EZKL.exists():
        sys.exit(f"缺少 {EZKL}")
    OUT.mkdir(parents=True, exist_ok=True)
    results = []

    for cfg in CONFIGS:
        name, env = cfg["name"], cfg["env"]
        wd = OUT / name
        wd.mkdir(parents=True, exist_ok=True)
        print(f"--- {name} (scale {cfg['scale']}, {env}) ---", flush=True)
        out = {"name": name, "scale": cfg["scale"], "env": env}

        s = wd / "settings.json"
        r = run(
            [
                EZKL,
                "gen-settings",
                "-M",
                MODEL / "model.onnx",
                "-O",
                s,
                "--input-scale",
                cfg["scale"],
                "--param-scale",
                cfg["scale"],
            ],
            {**env, "RUST_LOG": "error,ezkl::graph::rescale_fusion=debug"},
        )
        if r.returncode != 0:
            sys.exit(f"gen-settings failed:\n{r.stdout}\n{r.stderr}")
        hits = FUSION_RE.findall(r.stdout + r.stderr)
        out["fusion_report"] = hits[-1].strip() if hits else "(融合未启用)"
        st = json.loads(s.read_text())
        out["num_rows"] = st["num_rows"]
        k = max(1, (st["num_rows"] - 1).bit_length())
        out["logrows"] = k
        print(f"  num_rows={out['num_rows']:,} logrows={k}")
        print(f"  {out['fusion_report']}")

        steps = [
            (
                "compile",
                [
                    EZKL,
                    "compile-circuit",
                    "-M",
                    MODEL / "model.onnx",
                    "-S",
                    s,
                    "--compiled-circuit",
                    wd / "c.compiled",
                ],
                True,
            ),
            (
                "witness",
                [
                    EZKL,
                    "gen-witness",
                    "-M",
                    wd / "c.compiled",
                    "-D",
                    MODEL / "input.json",
                    "-O",
                    wd / "witness.json",
                ],
                True,
            ),
        ]
        srs = OUT / f"kzg{k}.srs"
        if not srs.exists():
            rr = run([EZKL, "get-srs", "--logrows", k, "--srs-path", srs], env)
            if rr.returncode != 0:
                sys.exit(f"get-srs failed:\n{rr.stdout}\n{rr.stderr}")
        steps += [
            (
                "setup",
                [
                    EZKL,
                    "setup",
                    "-M",
                    wd / "c.compiled",
                    "--srs-path",
                    srs,
                    "--vk-path",
                    wd / "vk.key",
                    "--pk-path",
                    wd / "pk.key",
                ],
                True,
            ),
            (
                "prove",
                [
                    EZKL,
                    "prove",
                    "-M",
                    wd / "c.compiled",
                    "-W",
                    wd / "witness.json",
                    "--pk-path",
                    wd / "pk.key",
                    "--proof-path",
                    wd / "proof.pf",
                    "--srs-path",
                    srs,
                ],
                False,
            ),
            (
                "verify",
                [
                    EZKL,
                    "verify",
                    "--proof-path",
                    wd / "proof.pf",
                    "--settings-path",
                    s,
                    "--vk-path",
                    wd / "vk.key",
                    "--srs-path",
                    srs,
                ],
                False,
            ),
        ]

        merged = Counter()
        for step, cmd, want_census in steps:
            r = run(cmd, env, census=want_census)
            ok = r.returncode == 0
            out[f"{step}_ok"] = ok
            log = r.stdout + r.stderr
            if want_census:
                c = census_of(log)
                merged.update(c)
                (wd / f"census_{step}.log").write_text(log)
            print(
                f"  {step:<8} {'OK' if ok else 'FAILED'}"
                + (f"  ({sum(c.values())} 次 div 调用)" if want_census else "")
            )
            if not ok:
                out["error"] = log[-800:]
                break

        sites = sorted({(d, p, n) for (d, p, n) in merged})
        out["divisors"] = [
            {
                "divisor": d,
                "log2": d.bit_length() - 1 if d & (d - 1) == 0 else None,
                "path": p,
                "limbs": n,
            }
            for d, p, n in sites
        ]
        out["n_wide"] = sum(1 for _, p, _ in sites if p == "wide")
        out["n_narrow"] = sum(1 for _, p, _ in sites if p == "narrow")
        out["widest_divisor"] = max((d for d, _, _ in sites), default=0)
        results.append(out)

        print(f"  除数集合（decomp_base = {DECOMP_BASE} = 2^14）：")
        for d, p, n in sites:
            lg = f"2^{d.bit_length()-1}" if d and d & (d - 1) == 0 else str(d)
            print(f"    {lg:>8}  {p:<7}" + (f"n={n} limbs" if p == "wide" else ""))

        pk = wd / "pk.key"
        if pk.exists():
            out["pk_bytes"] = pk.stat().st_size
            pk.unlink()

        (OUT / "summary.json").write_text(json.dumps(results, indent=2))

    print("\n=== 汇总 ===")
    print(f"{'配置':<22}{'rows':>9}{'宽':>4}{'窄':>4}  {'最宽除数':>10}  verify")
    for r in results:
        w = r["widest_divisor"]
        lg = f"2^{w.bit_length()-1}" if w and w & (w - 1) == 0 else str(w)
        print(
            f"{r['name']:<22}{r['num_rows']:>9,}{r['n_wide']:>4}{r['n_narrow']:>4}"
            f"  {lg:>10}  {r.get('verify_ok')}"
        )


if __name__ == "__main__":
    main()
