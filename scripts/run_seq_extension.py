#!/usr/bin/env python3

import csv
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

REPO_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = REPO_DIR / "models_p"
RESULTS_DIR = REPO_DIR / "results" / "p_measurement"
EZKL_BIN = REPO_DIR / "target" / "release" / "ezkl"
TIME_BIN = REPO_DIR / ".tools" / "time-pkg" / "usr" / "bin" / "time"
TIMEOUT_SECS = 6 * 3600

NEW_SEQ = (512, 1024)
SEQ_SCAN_D = 128
N_HEAD = 2

sys.path.insert(0, str(REPO_DIR))
import gen_models  # noqa: E402

gen_models.MODELS_DIR = MODELS_DIR


def parse_time_v(text: str):
    wall = rss_kb = None
    for m in re.finditer(r"Elapsed \(wall clock\) time.*?:\s*([0-9:.]+)", text):
        parts = [float(p) for p in m.group(1).strip().split(":")]
        while len(parts) < 3:
            parts.insert(0, 0.0)
        h, mnt, sec = parts[-3:]
        wall = h * 3600 + mnt * 60 + sec
    for m in re.finditer(r"Maximum resident set size \(kbytes\):\s*(\d+)", text):
        rss_kb = int(m.group(1))
    return wall, rss_kb


def export_if_missing(name: str, cfg: dict) -> dict:
    onnx_path = MODELS_DIR / name / "model.onnx"
    if onnx_path.exists():
        print(f"   onnx already present: {onnx_path}")
        return None
    row = gen_models.export_one(name, cfg, opset=10, seed=42)
    print(
        f"   exported params={row['params']:,} onnx_max_abs_diff={row['onnx_max_abs_diff']:.2e}"
    )
    return row


def append_to_manifest(rows: list):
    manifest_path = MODELS_DIR / "MANIFEST.csv"
    with open(manifest_path) as f:
        existing = list(csv.DictReader(f))
        fieldnames = list(existing[0].keys())
    have = {r["name"] for r in existing}
    added = [r for r in rows if r["name"] not in have]
    if not added:
        print("   MANIFEST.csv already lists both configs")
        return
    with open(manifest_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(existing)
        for r in added:
            writer.writerow({k: r.get(k, "") for k in fieldnames})
    print(f"   appended {len(added)} rows to {manifest_path}")


def run_gen_settings(name: str) -> dict:
    out_dir = RESULTS_DIR / name
    log_dir = out_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    settings_path = out_dir / "settings.json"
    log_path = log_dir / "gen_settings.log"
    onnx_path = MODELS_DIR / name / "model.onnx"

    env = os.environ.copy()
    env["RUST_LOG"] = "debug"
    cmd = [
        str(TIME_BIN),
        "-v",
        str(EZKL_BIN),
        "gen-settings",
        "-M",
        str(onnx_path),
        "-O",
        str(settings_path),
    ]

    t0 = time.monotonic()
    with open(log_path, "wb") as logf:
        proc = subprocess.Popen(
            cmd, stdout=logf, stderr=subprocess.STDOUT, env=env, start_new_session=True
        )
        try:
            rc = proc.wait(timeout=TIMEOUT_SECS)
            timed_out = False
        except subprocess.TimeoutExpired:
            os.killpg(os.getpgid(proc.pid), 9)
            rc = proc.wait()
            timed_out = True
    wall_elapsed = time.monotonic() - t0

    text = log_path.read_text(errors="replace")
    wall_s, rss_kb = parse_time_v(text)
    if wall_s is None:
        wall_s = wall_elapsed

    if timed_out:
        status = "TIMEOUT"
    elif rc == 0:
        status = "OK"
    elif rc in (-9, 137):
        status = "OOM_OR_KILLED"
    else:
        status = f"FAILED(rc={rc})"

    num_rows = None
    if status == "OK":
        num_rows = json.loads(settings_path.read_text())["num_rows"]

    return {
        "status": status,
        "wall_s": wall_s,
        "rss_mb": (rss_kb / 1024) if rss_kb else None,
        "num_rows": num_rows,
        "log_path": log_path,
    }


def main():
    wanted = [int(a) for a in sys.argv[1:]] or list(NEW_SEQ)
    manifest_rows = []

    for seq in wanted:
        cfg = dict(seq_len=seq, d_model=SEQ_SCAN_D, n_head=N_HEAD, n_layers=1)
        name = gen_models.cfg_name(cfg)
        print(f"\n== {name} ==", flush=True)
        row = export_if_missing(name, cfg)
        if row is not None:
            row["group"] = "seq_scan"
            row["n_head"] = N_HEAD
            manifest_rows.append(row)

    if manifest_rows:
        append_to_manifest(manifest_rows)

    results = []
    for seq in wanted:
        name = gen_models.cfg_name(
            dict(seq_len=seq, d_model=SEQ_SCAN_D, n_head=N_HEAD, n_layers=1)
        )
        print(f"\n== gen-settings {name} ==", flush=True)
        res = run_gen_settings(name)
        rss = f"{res['rss_mb']:.0f}MB" if res["rss_mb"] else "?"
        print(
            f"   status={res['status']} wall={res['wall_s']:.0f}s rss={rss} "
            f"num_rows={res['num_rows']}",
            flush=True,
        )
        results.append((name, res))

        if res["status"] == "OK":
            print(f"   parsing ledger...", flush=True)
            subprocess.run(
                [
                    sys.executable,
                    str(REPO_DIR / "scripts" / "parse_ledger.py"),
                    str(RESULTS_DIR / name),
                ],
                check=True,
            )

    print("\n=== summary ===")
    for name, res in results:
        rss = f"{res['rss_mb']:.0f}" if res["rss_mb"] else "?"
        print(
            f"  {name:20s} {res['status']:8s} wall={res['wall_s']:8.0f}s "
            f"rss={rss:>8s}MB num_rows={res['num_rows']}"
        )
    print("\nnext: python3 scripts/aggregate_p_measurement.py")


if __name__ == "__main__":
    main()
