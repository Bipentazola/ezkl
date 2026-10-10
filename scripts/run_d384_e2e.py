#!/usr/bin/env python3

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
EZKL = REPO / "target" / "release" / "ezkl"
TIME_BIN = REPO / ".tools" / "time-pkg" / "usr" / "bin" / "time"
MODEL = REPO / "models_p" / "seq16_d384_l1"
OUT = REPO / "results" / "d384_e2e"

OPT_ON = {"EZKL_DIV_ROUND_NEAREST": "1", "EZKL_RESCALE_FUSION": "1"}

ROWS_N2_PUBLISHED = 2_801_714


def timed(args, cwd):
    env = {**os.environ, **OPT_ON}
    cmd = [str(TIME_BIN), "-v", *[str(a) for a in args]]
    t0 = time.time()
    r = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, env=env)
    wall = time.time() - t0
    m = re.search(r"Maximum resident set size \(kbytes\):\s*(\d+)", r.stderr)
    rss = int(m.group(1)) / 1024.0 if m else None
    killed = re.search(r"Command terminated by signal (\d+)", r.stderr)
    return (
        wall,
        rss,
        r.returncode == 0,
        (r.stdout + r.stderr),
        (int(killed.group(1)) if killed else None),
    )


def run_one(n: int, keep_pk: bool):
    wd = OUT / f"N{n}"
    wd.mkdir(parents=True, exist_ok=True)
    shutil.copy(MODEL / "model.onnx", wd / "model.onnx")
    shutil.copy(MODEL / "input.json", wd / "input.json")
    out = {"num_inner_cols": n}

    env = {**os.environ, **OPT_ON}
    for k_arg in ("26", None):
        k = k_arg or str(out["logrows"])
        subprocess.run(
            [
                str(EZKL),
                "gen-settings",
                "-M",
                "model.onnx",
                "-O",
                "settings.json",
                "-K",
                k,
                "-N",
                str(n),
            ],
            cwd=wd,
            capture_output=True,
            env=env,
            check=True,
        )
        if k_arg:
            rows = json.loads((wd / "settings.json").read_text())["num_rows"]
            out["num_rows"] = rows
            out["logrows"] = max(1, (rows - 1).bit_length())
    k = out["logrows"]
    print(f"  N={n}  num_rows={out['num_rows']:,}  logrows={k}", flush=True)

    srs = REPO / "results" / "d384_e2e" / f"kzg{k}.srs"
    if not srs.exists():
        r = subprocess.run(
            [str(EZKL), "get-srs", "--logrows", str(k), "--srs-path", str(srs)],
            capture_output=True,
            text=True,
            env=env,
        )
        if r.returncode != 0:
            out["status"] = "FAILED at get-srs"
            out["error"] = (r.stdout + r.stderr)[-700:]
            return out

    steps = [
        (
            "compile",
            [
                EZKL,
                "compile-circuit",
                "-M",
                "model.onnx",
                "-S",
                "settings.json",
                "--compiled-circuit",
                "c.compiled",
            ],
        ),
        (
            "witness",
            [
                EZKL,
                "gen-witness",
                "-M",
                "c.compiled",
                "-D",
                "input.json",
                "-O",
                "witness.json",
            ],
        ),
        (
            "setup",
            [
                EZKL,
                "setup",
                "-M",
                "c.compiled",
                "--srs-path",
                srs,
                "--vk-path",
                "vk.key",
                "--pk-path",
                "pk.key",
            ],
        ),
        (
            "prove",
            [
                EZKL,
                "prove",
                "-M",
                "c.compiled",
                "-W",
                "witness.json",
                "--pk-path",
                "pk.key",
                "--proof-path",
                "proof.pf",
                "--srs-path",
                srs,
            ],
        ),
        (
            "verify",
            [
                EZKL,
                "verify",
                "--proof-path",
                "proof.pf",
                "--settings-path",
                "settings.json",
                "--vk-path",
                "vk.key",
                "--srs-path",
                srs,
            ],
        ),
    ]
    for name, cmd in steps:
        wall, rss, ok, log, signal = timed(cmd, wd)
        out[f"{name}_s"] = round(wall, 1)
        out[f"{name}_rss_mb"] = round(rss, 1) if rss else None
        (wd / f"{name}.time").write_text(log)
        gb = f"{rss/1024:7.2f} GB" if rss else "      ? GB"
        if not ok:
            why = f"killed by signal {signal}" if signal else "nonzero exit"
            out["status"] = f"FAILED at {name} ({why})"
            out["killed_signal"] = signal
            out["error"] = log[-700:]
            print(f"  N={n} {name:<8} {wall:7.1f}s {gb}  <-- {why}", flush=True)
            break
        print(f"  N={n} {name:<8} {wall:7.1f}s {gb}", flush=True)
    else:
        out["status"] = "OK"
        out["proof_bytes"] = (wd / "proof.pf").stat().st_size

    pk = wd / "pk.key"
    if pk.exists():
        out["pk_bytes"] = pk.stat().st_size
        if not keep_pk:
            pk.unlink()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-values", type=int, nargs="+", default=[2, 4])
    ap.add_argument(
        "--keep-pk",
        action="store_true",
        help="保留 pk.key；默认在运行结束后删除以释放磁盘空间",
    )
    a = ap.parse_args()

    if not (MODEL / "model.onnx").exists():
        sys.exit(
            f"缺少 {MODEL}/model.onnx —— 先跑 "
            f"`python3 scripts/gen_p_measurement_models.py`（models_p/ 不入库）"
        )
    OUT.mkdir(parents=True, exist_ok=True)

    results = []
    for n in a.n_values:
        print(f"--- num_inner_cols = {n} ---", flush=True)
        r = run_one(n, a.keep_pk)
        results.append(r)
        (OUT / "summary.json").write_text(json.dumps(results, indent=2))

    print("\n=== d=384 端到端 ===")
    print(f"{'N':>3}{'num_rows':>12}{'k':>4}  {'setup':>16}  {'prove':>16}  status")
    for r in results:

        def cell(step):
            s, m = r.get(f"{step}_s"), r.get(f"{step}_rss_mb")
            return f"{s:7.1f}s {m/1024:6.1f}GB" if s and m else " " * 16

        print(
            f"{r['num_inner_cols']:>3}{r['num_rows']:>12,}{r['logrows']:>4}  "
            f"{cell('setup')}  {cell('prove')}  {r['status']}"
        )

    n2 = next((r for r in results if r["num_inner_cols"] == 2), None)
    if n2 and n2["num_rows"] != ROWS_N2_PUBLISHED:
        print(
            f"\n⚠️  N=2 行数 {n2['num_rows']:,} != 已发表的 "
            f"{ROWS_N2_PUBLISHED:,}（docs/recon/06 表 1）——先查口径再用这批数"
        )


if __name__ == "__main__":
    main()
