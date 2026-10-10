#!/usr/bin/env python3

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
EZKL = REPO / "target" / "release" / "ezkl"
TIME = REPO / ".tools" / "time-pkg" / "usr" / "bin" / "time"
MODELS = REPO / "models_grid"

RESCUED = {
    "seq8_d512_l1": {"rows_base": 3627049, "rows_opt": 2083285},
    "seq16_d128_l2": {"rows_base": 3535860, "rows_opt": 2002652},
    "seq8_d256_l2": {"rows_base": 3497500, "rows_opt": 1968016},
    "seq8_d384_l1": {"rows_base": 2505218, "rows_opt": 1379546},
}

DEFAULT_CONFIGS = ["seq8_d512_l1", "seq16_d128_l2", "seq8_d256_l2"]
OPT_ENV = {"EZKL_DIV_ROUND_NEAREST": "1", "EZKL_RESCALE_FUSION": "1"}
LOGROWS = 21

BASELINE_ENV = {"EZKL_DIV_ROUND_NEAREST": "0", "EZKL_RESCALE_FUSION": "0"}
BASELINE_LOGROWS = 22

ACTIVE_ENV = OPT_ENV
ACTIVE_LOGROWS = LOGROWS


def sh(*args, cwd=None, check=True, timed=None):
    e = dict(os.environ)
    e.update(ACTIVE_ENV)
    argv = [str(a) for a in args]
    if timed:
        argv = [str(TIME), "-v"] + argv
    r = subprocess.run(argv, cwd=cwd, capture_output=True, text=True, env=e)
    if timed:
        Path(timed).write_text(r.stderr)
    if check and r.returncode != 0:
        logs = "\n".join(
            l
            for l in (r.stdout + r.stderr).splitlines()
            if "] - " in l or "error" in l.lower()
        )
        return None, (logs + "\n---\n" + r.stdout + r.stderr)
    return (r.stdout + r.stderr), None


def parse_time(path):
    t = Path(path).read_text()
    sig = re.search(r"Command terminated by signal (\d+)", t)
    rss = re.search(r"Maximum resident set size \(kbytes\): (\d+)", t)
    wall = re.search(r"Elapsed \(wall clock\) time[^\n]*?:\s*([\d:.]+)\s*$", t, re.M)
    secs = None
    if wall:
        parts = [float(p) for p in wall.group(1).split(":")]
        secs = sum(p * 60**i for i, p in enumerate(reversed(parts)))
    return {
        "wall_s": round(secs, 1) if secs else None,
        "peak_rss_gb": round(int(rss.group(1)) / 1024 / 1024, 1) if rss else None,
        "killed_signal": int(sig.group(1)) if sig else None,
    }


def run_one(name, wd, srs):
    m = MODELS / name
    if not (m / "model.onnx").exists():
        return {
            "config": name,
            "status": "MODEL_MISSING",
            "hint": "regenerate with scripts/run_feasibility_grid.py --reuse-models",
        }
    d = wd / name
    d.mkdir(parents=True, exist_ok=True)
    rec = {
        "config": name,
        **RESCUED[name],
        "logrows_target": ACTIVE_LOGROWS,
        "optimizations": "off" if ACTIVE_ENV is BASELINE_ENV else "on",
    }
    t0 = time.time()

    out, err = sh(
        EZKL,
        "gen-settings",
        "-M",
        m / "model.onnx",
        "-O",
        "settings.json",
        "-K",
        ACTIVE_LOGROWS,
        cwd=d,
    )
    if err:
        return {**rec, "status": "GEN_SETTINGS_FAILED", "error": err[:900]}
    st = json.loads((d / "settings.json").read_text())
    rec["num_rows"] = st["num_rows"]
    rec["logrows"] = st["run_args"]["logrows"]
    expect = RESCUED[name]["rows_base" if ACTIVE_ENV is BASELINE_ENV else "rows_opt"]
    rec["rows_match_grid"] = st["num_rows"] == expect

    for step, args in (
        (
            "compile",
            (
                EZKL,
                "compile-circuit",
                "-M",
                m / "model.onnx",
                "-S",
                "settings.json",
                "--compiled-circuit",
                "c.compiled",
            ),
        ),
        (
            "witness",
            (
                EZKL,
                "gen-witness",
                "-M",
                "c.compiled",
                "-D",
                m / "input.json",
                "-O",
                "witness.json",
            ),
        ),
    ):
        out, err = sh(*args, cwd=d)
        if err:
            return {**rec, "status": f"{step.upper()}_FAILED", "error": err[:900]}

    out, err = sh(
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
        cwd=d,
        timed=d / "setup.time",
    )
    rec["setup"] = parse_time(d / "setup.time")
    if err:
        return {**rec, "status": "SETUP_FAILED", "error": err[:900]}

    out, err = sh(
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
        cwd=d,
        timed=d / "prove.time",
    )
    rec["prove"] = parse_time(d / "prove.time")
    pk = d / "pk.key"
    if pk.exists():
        rec["pk_bytes_deleted"] = pk.stat().st_size
        pk.unlink()
    if err:
        return {**rec, "status": "PROVE_FAILED", "error": err[:900]}

    out, err = sh(
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
        cwd=d,
    )
    rec["verified"] = bool(out and "verified: true" in out.lower())
    rec["status"] = "OK" if rec["verified"] else "VERIFY_FAILED"
    rec["total_wall_s"] = round(time.time() - t0, 1)
    return rec


def main():
    ap = argparse.ArgumentParser(
        description=None, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--configs", nargs="+", default=DEFAULT_CONFIGS, choices=list(RESCUED)
    )
    ap.add_argument(
        "--baseline",
        action="store_true",
        help="跑关优化的一侧（logrows 22），预期 prove 被 OOM 杀",
    )
    ap.add_argument("--workdir", default=None)
    ap.add_argument("--out", default=str(REPO / "results" / "rescued_configs"))
    args = ap.parse_args()

    global ACTIVE_ENV, ACTIVE_LOGROWS
    if args.baseline:
        ACTIVE_ENV, ACTIVE_LOGROWS = BASELINE_ENV, BASELINE_LOGROWS
    json_name = (
        "rescued_configs_baseline.json" if args.baseline else "rescued_configs.json"
    )

    wd = (
        Path(args.workdir)
        if args.workdir
        else (
            REPO
            / "results"
            / "rescued_configs"
            / ("work_baseline" if args.baseline else "work")
        )
    )
    wd.mkdir(parents=True, exist_ok=True)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    print(
        f"[口径] optimizations={'off' if args.baseline else 'on'} "
        f"logrows={ACTIVE_LOGROWS} env={ACTIVE_ENV} -> {json_name}",
        flush=True,
    )

    srs = wd / f"kzg{ACTIVE_LOGROWS}.srs"
    if not srs.exists():
        print(f"[srs] fetching logrows={ACTIVE_LOGROWS} ...", flush=True)
        o, e = sh(
            EZKL, "get-srs", "--logrows", ACTIVE_LOGROWS, "--srs-path", srs, cwd=wd
        )
        if e:
            sys.exit(f"get-srs failed:\n{e}")
    print(f"[srs] {srs} ({srs.stat().st_size/1e6:.0f} MB)", flush=True)

    recs = []
    for name in args.configs:
        print(f"\n=== {name} ===", flush=True)
        rec = run_one(name, wd, srs.resolve())
        recs.append(rec)
        print(
            f"  status={rec['status']} rows={rec.get('num_rows')} k={rec.get('logrows')} "
            f"setup={rec.get('setup')} prove={rec.get('prove')} "
            f"verified={rec.get('verified')}",
            flush=True,
        )
        (out / json_name).write_text(
            json.dumps(recs, indent=2, ensure_ascii=False) + "\n"
        )

    print("\n=== 汇总 ===")
    for r in recs:
        print(
            f"{r['config']:15s} {r['status']:8s} rows={r.get('num_rows',0):>9,} "
            f"k={r.get('logrows','-')} "
            f"setup={r.get('setup',{}).get('wall_s','-')}s/{r.get('setup',{}).get('peak_rss_gb','-')}GB "
            f"prove={r.get('prove',{}).get('wall_s','-')}s/{r.get('prove',{}).get('peak_rss_gb','-')}GB"
        )
    print(f"\n[wrote] {out/json_name}")


if __name__ == "__main__":
    main()
