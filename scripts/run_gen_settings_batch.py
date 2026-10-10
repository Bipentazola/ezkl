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
TIMEOUT_SECS = 600

sys.path.insert(0, str(REPO_DIR))
import gen_models  # noqa: E402


def parse_time_v(text: str):
    wall = None
    for m in re.finditer(r"Elapsed \(wall clock\) time.*?:\s*([0-9:.]+)", text):
        parts = [float(p) for p in m.group(1).strip().split(":")]
        while len(parts) < 3:
            parts.insert(0, 0.0)
        h, mnt, sec = parts[-3:]
        wall = h * 3600 + mnt * 60 + sec
    rss_kb = None
    for m in re.finditer(r"Maximum resident set size \(kbytes\):\s*(\d+)", text):
        rss_kb = int(m.group(1))
    return wall, rss_kb


def run_gen_settings(name: str, onnx_path: Path) -> dict:
    out_dir = RESULTS_DIR / name
    log_dir = out_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    settings_path = out_dir / "settings.json"
    log_path = log_dir / "gen_settings.log"

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
        status = "FAILED"

    return {
        "status": status,
        "wall_s": wall_s,
        "rss_mb": (rss_kb / 1024) if rss_kb else None,
        "settings_path": settings_path,
        "log_path": log_path,
    }


def main():
    manifest_path = MODELS_DIR / "MANIFEST.csv"
    with open(manifest_path) as f:
        configs = list(csv.DictReader(f))

    summary_rows = []
    for row in configs:
        name = row["name"]
        cfg = dict(
            seq_len=int(row["seq_len"]),
            d_model=int(row["d_model"]),
            n_head=int(row["n_head"]),
            n_layers=int(row["n_layers"]),
        )
        onnx_path = MODELS_DIR / name / "model.onnx"
        print(f"\n== {name}: {cfg} ==", flush=True)
        res = run_gen_settings(name, onnx_path)
        print(
            f"   status={res['status']} wall={res['wall_s']:.1f}s rss={res['rss_mb']}MB",
            flush=True,
        )

        used_fallback = False
        if res["status"] in ("TIMEOUT", "OOM_OR_KILLED") and cfg["seq_len"] != 8:
            print(
                f"   [fallback] retrying with seq_len=8 (was {cfg['seq_len']})",
                flush=True,
            )
            fb_cfg = {**cfg, "seq_len": 8}
            fb_name = f"{name}_seq8fallback"
            gen_models.MODELS_DIR = MODELS_DIR
            gen_models.export_one(fb_name, fb_cfg, opset=10, seed=42)
            fb_onnx = MODELS_DIR / fb_name / "model.onnx"
            fb_res = run_gen_settings(fb_name, fb_onnx)
            print(
                f"   [fallback] status={fb_res['status']} wall={fb_res['wall_s']:.1f}s",
                flush=True,
            )
            used_fallback = True

        num_rows = logrows = num_inner_cols = None
        if res["status"] == "OK":
            settings = json.loads(res["settings_path"].read_text())
            num_rows = settings["num_rows"]
            logrows = settings["run_args"]["logrows"]
            num_inner_cols = settings["run_args"]["num_inner_cols"]

        summary_rows.append(
            {
                "name": name,
                "group": row["group"],
                "seq_len": cfg["seq_len"],
                "d_model": cfg["d_model"],
                "n_head": cfg["n_head"],
                "status": res["status"],
                "wall_s": f"{res['wall_s']:.2f}",
                "rss_mb": f"{res['rss_mb']:.1f}" if res["rss_mb"] else "",
                "used_seq8_fallback": used_fallback,
                "num_rows": num_rows,
                "logrows_gen_settings_default": logrows,
                "num_inner_cols": num_inner_cols,
            }
        )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_csv = RESULTS_DIR / "gen_settings_run_summary.csv"
    fieldnames = list(summary_rows[0].keys())
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"\nwrote {out_csv}")

    n_ok = sum(1 for r in summary_rows if r["status"] == "OK")
    print(f"\n{n_ok}/{len(summary_rows)} configs OK")
    for r in summary_rows:
        if r["status"] != "OK":
            print(f"  NOT OK: {r['name']} status={r['status']}")


if __name__ == "__main__":
    main()
