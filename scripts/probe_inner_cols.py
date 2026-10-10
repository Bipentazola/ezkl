#!/usr/bin/env python3

import argparse
import json
import math
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
EZKL = REPO / "target" / "release" / "ezkl"
GEN = REPO / "scripts" / "gen_decode_step_onnx.py"
TIME_BIN = REPO / ".tools" / "time-pkg" / "usr" / "bin" / "time"
VIS = [
    "--input-visibility",
    "polycommit",
    "--output-visibility",
    "polycommit",
    "--param-visibility",
    "fixed",
]
OPT_ON = {"EZKL_DIV_ROUND_NEAREST": "1", "EZKL_RESCALE_FUSION": "1"}


def timed(args, cwd, env_extra=None):
    import os

    env = dict(os.environ)
    for v in ("EZKL_DIV_ROUND_NEAREST", "EZKL_RESCALE_FUSION"):
        env.pop(v, None)
    env.update(env_extra or {})
    cmd = [str(TIME_BIN), "-v", *[str(a) for a in args]]
    t0 = time.time()
    r = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, env=env)
    wall = time.time() - t0
    rss = None
    m = re.search(r"Maximum resident set size \(kbytes\):\s*(\d+)", r.stderr)
    if m:
        rss = int(m.group(1)) / 1024.0
    return wall, rss, r.returncode == 0, r.stdout + r.stderr


def run_one(wd: Path, model: Path, n: int):
    wd.mkdir(parents=True, exist_ok=True)
    shutil.copy(model, wd / "model.onnx")
    shutil.copy(model.parent / "input.json", wd / "input.json")
    out = {"num_inner_cols": n}

    subprocess.run(
        [
            str(EZKL),
            "gen-settings",
            "-M",
            "model.onnx",
            "-O",
            "settings.json",
            "-K",
            "26",
            "-N",
            str(n),
            *VIS,
        ],
        cwd=wd,
        capture_output=True,
        env={**__import__("os").environ, **OPT_ON},
    )
    num_rows = json.loads((wd / "settings.json").read_text())["num_rows"]
    k = max(1, (num_rows - 1).bit_length())
    out.update(num_rows=num_rows, logrows=k)
    subprocess.run(
        [
            str(EZKL),
            "gen-settings",
            "-M",
            "model.onnx",
            "-O",
            "settings.json",
            "-K",
            str(k),
            "-N",
            str(n),
            *VIS,
        ],
        cwd=wd,
        capture_output=True,
        env={**__import__("os").environ, **OPT_ON},
    )

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
                "circuit.compiled",
            ],
        ),
        ("gensrs", [EZKL, "gen-srs", "--logrows", k, "--srs-path", f"kzg{k}.srs"]),
        (
            "setup",
            [
                EZKL,
                "setup",
                "-M",
                "circuit.compiled",
                "--srs-path",
                f"kzg{k}.srs",
                "--vk-path",
                "vk.key",
                "--pk-path",
                "pk.key",
            ],
        ),
        (
            "witness",
            [
                EZKL,
                "gen-witness",
                "-M",
                "circuit.compiled",
                "-D",
                "input.json",
                "-O",
                "witness.json",
                "--vk-path",
                "vk.key",
                "--srs-path",
                f"kzg{k}.srs",
            ],
        ),
        (
            "prove",
            [
                EZKL,
                "prove",
                "-W",
                "witness.json",
                "-M",
                "circuit.compiled",
                "--pk-path",
                "pk.key",
                "--proof-path",
                "proof.json",
                "--srs-path",
                f"kzg{k}.srs",
            ],
        ),
        (
            "verify",
            [
                EZKL,
                "verify",
                "--proof-path",
                "proof.json",
                "-S",
                "settings.json",
                "--vk-path",
                "vk.key",
                "--srs-path",
                f"kzg{k}.srs",
            ],
        ),
    ]
    for name, cmd in steps:
        wall, rss, ok, log = timed(cmd, wd, OPT_ON)
        out[f"{name}_s"] = round(wall, 1)
        out[f"{name}_rss_mb"] = round(rss, 1) if rss else None
        if not ok:
            out["status"] = f"FAILED at {name}"
            out["error"] = log[-700:]
            print(f"  N={n} FAILED at {name}: {log[-300:]}", flush=True)
            return out
        print(f"  N={n} {name:<8} {wall:7.1f}s  {rss/1024:7.2f} GB", flush=True)
    out["status"] = "OK"
    out["proof_bytes"] = (wd / "proof.json").stat().st_size
    out["pk_bytes"] = (wd / "pk.key").stat().st_size
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--d-model", type=int, default=128)
    ap.add_argument("--t-max", type=int, default=16)
    ap.add_argument("--vocab", type=int, default=1024)
    ap.add_argument("--n-values", type=int, nargs="+", default=[2, 4, 8])
    ap.add_argument(
        "--workdir", default=str(REPO / "models_vocab" / "_inner_cols_probe")
    )
    ap.add_argument("--out", default=str(REPO / "results" / "vocab_composition"))
    args = ap.parse_args()

    wd = Path(args.workdir)
    wd.mkdir(parents=True, exist_ok=True)
    mdir = wd / "model"
    print(
        f"generating d={args.d_model} T={args.t_max} V={args.vocab} (embed-outside) ..."
    )
    subprocess.run(
        [
            sys.executable,
            str(GEN),
            "--out-dir",
            str(mdir),
            "--d-model",
            str(args.d_model),
            "--t-max",
            str(args.t_max),
            "--vocab",
            str(args.vocab),
            "--embed-outside",
            "--variant",
            "where",
        ],
        capture_output=True,
        check=True,
    )

    recs = []
    for n in args.n_values:
        print(f"--- num_inner_cols = {n} ---", flush=True)
        recs.append(run_one(wd / f"N{n}", mdir / "model.onnx", n))
        Path(args.out, "inner_cols_probe.json").write_text(
            json.dumps({"config": vars(args), "runs": recs}, indent=2, default=str)
        )

    ok = [r for r in recs if r.get("status") == "OK"]
    if ok:
        print(
            f"\n{'N':>3}{'rows':>11}{'k':>4}{'setup_s':>9}{'setup_GB':>10}"
            f"{'prove_s':>9}{'prove_GB':>10}{'proof_B':>9}"
        )
        for r in ok:
            print(
                f"{r['num_inner_cols']:>3}{r['num_rows']:>11,}{r['logrows']:>4}"
                f"{r['setup_s']:>9.1f}{r['setup_rss_mb']/1024:>10.2f}"
                f"{r['prove_s']:>9.1f}{r['prove_rss_mb']/1024:>10.2f}"
                f"{r['proof_bytes']:>9,}"
            )


if __name__ == "__main__":
    main()
