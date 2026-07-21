#!/usr/bin/env python3
"""
bench_stages.py -- stage-level wall-clock + peak-memory benchmark for the
ezkl CLI pipeline, run against one models/<config>/ directory (model.onnx +
input.json).

Pipeline (fixed order, per spec):
    gen-settings -> calibrate-settings (--target resources) -> compile-circuit
    -> gen-srs (local dummy SRS, logrows read back from settings.json)
    -> setup -> gen-witness -> prove -> verify

Each stage is wrapped in GNU `time -v` (see NOTE below on where that binary
comes from) with RUST_LOG=debug; the combined stdout+stderr of every attempt
is saved in full to results/<config>/logs/<stage>[.repN|.warmup].log for
later per-op-line parsing. `setup` and `prove` run one untimed warm-up
(discarded, to exclude cold-cache effects) followed by --reps (default 3)
timed repetitions; the summary uses the median across the reps that
completed with status OK. A stage attempt that exceeds --timeout (default
1800s) or gets OOM-killed is killed as a whole process group and recorded
with an explicit status (TIMEOUT / OOM_OR_KILLED) rather than crashing the
run -- per spec, that is itself a valid data point.

One row per model config is appended to results/summary.csv, containing:
an `overall_status` column ("OK", or "FAILED(<stage>:<status>)" naming the
first stage that didn't complete cleanly -- OOM/timeout/error all show up
here at a glance in addition to the granular per-stage `<stage>_status`
columns), settings.json fields (logrows, num_rows/total_assignments,
lookup-related fields), per-stage median wall-seconds/max-RSS-MB/status, and
the proof file size.

`--calibrate-scales N` pins calibrate-settings' scale search to exactly N
(via its own `--scales` flag) instead of the default auto-search grid --
for a controlled-variable scale sweep (logrows/lookup-bits are still
calibrated normally, only the scale is held fixed). Also suffixes the
config name / results/<config>/ dir with `_scaleN` so it doesn't clobber
that model's unscaled run.

NOTE on `time -v`: this container has no `/usr/bin/time` (the `time` package
is not installed, and `perf` is separately unavailable/locked down -- see
.claude/ezkl_create_proof_flamegraph_scaling_notes.md from the earlier
flamegraph work). Installing it system-wide via `apt-get install time` would
write outside /data. Instead its .deb was downloaded and extracted with
`dpkg -x` into .tools/time-pkg/usr/bin/time (kept entirely inside /data) --
functionally identical to the system package, just not registered with dpkg
globally. If that path is missing, rerun:
    apt-get download time && dpkg -x time_*.deb .tools/time-pkg

Usage:
    python3 bench_stages.py models/seq32_d64_l1
    python3 bench_stages.py models/seq32_d64_l1 --reps 3 --timeout 1800
"""
import argparse
import json
import os
import re
import signal
import statistics
import subprocess
import sys
import time
from pathlib import Path

REPO_DIR = Path(__file__).resolve().parent
DEFAULT_EZKL_BIN = REPO_DIR / "target" / "release" / "ezkl"
DEFAULT_TIME_BIN = REPO_DIR / ".tools" / "time-pkg" / "usr" / "bin" / "time"

# CSV column order for the per-stage block (key name -> ezkl invocation uses
# hyphens; CSV/log-file names use underscores throughout for consistency)
STAGE_KEYS = [
    "gen_settings", "calibrate_settings", "compile_circuit", "gen_srs",
    "setup", "gen_witness", "prove", "verify",
]


def parse_wall_clock(s: str) -> float:
    parts = [float(p) for p in s.strip().split(":")]
    while len(parts) < 3:
        parts.insert(0, 0.0)
    h, m, sec = parts[-3:]
    return h * 3600 + m * 60 + sec


def parse_time_v_log(text: str):
    """Extract GNU time -v's own report fields from a log that may also
    contain ezkl's RUST_LOG=debug output interleaved before it."""
    wall = None
    for m in re.finditer(r"Elapsed \(wall clock\) time.*?:\s*([0-9:.]+)", text):
        wall = parse_wall_clock(m.group(1))
    max_rss_kb = None
    for m in re.finditer(r"Maximum resident set size \(kbytes\):\s*(\d+)", text):
        max_rss_kb = int(m.group(1))
    return wall, max_rss_kb


def run_once(label: str, cmd, log_path: Path, timeout_secs: float, time_bin: Path):
    """Run cmd under `time -v` + RUST_LOG=debug, combined output -> log_path.
    Returns a metrics dict; never raises on subprocess failure/timeout."""
    env = os.environ.copy()
    env["RUST_LOG"] = "debug"
    full_cmd = [str(time_bin), "-v"] + [str(c) for c in cmd]
    print(f"[bench] -- {label} --")
    print(f"[bench]    $ {' '.join(str(c) for c in cmd)}")

    t0 = time.monotonic()
    peak_rss_kb = 0
    timed_out = False
    with open(log_path, "wb") as logf:
        proc = subprocess.Popen(
            full_cmd, stdout=logf, stderr=subprocess.STDOUT, env=env,
            start_new_session=True,  # own process group -> reliable group-kill on timeout
        )
        pgid = os.getpgid(proc.pid)
        rc = None
        while rc is None:
            try:
                rc = proc.wait(timeout=1.0)
            except subprocess.TimeoutExpired:
                elapsed = time.monotonic() - t0
                for pid_dir in Path("/proc").glob("[0-9]*"):
                    try:
                        pid = int(pid_dir.name)
                        if os.getpgid(pid) != pgid:
                            continue
                        status_text = (pid_dir / "status").read_text()
                    except (OSError, ValueError):
                        continue
                    m = re.search(r"VmHWM:\s*(\d+)", status_text)
                    if m:
                        peak_rss_kb = max(peak_rss_kb, int(m.group(1)))
                if elapsed >= timeout_secs:
                    print(f"[bench]    TIMEOUT after {elapsed:.0f}s -- killing process group {pgid}")
                    try:
                        os.killpg(pgid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    timed_out = True
                    rc = proc.wait()
    wall_elapsed = time.monotonic() - t0

    text = log_path.read_text(errors="replace")
    wall_seconds, max_rss_kb_from_time = parse_time_v_log(text)
    if max_rss_kb_from_time is not None:
        peak_rss_kb = max_rss_kb_from_time  # authoritative when the run finished cleanly
    if wall_seconds is None:
        wall_seconds = wall_elapsed  # time -v never printed its report (killed) -- fall back

    if timed_out:
        status = "TIMEOUT"
    elif rc == 0:
        status = "OK"
    elif rc in (-9, 137):
        status = "OOM_OR_KILLED"
    else:
        status = "FAILED"

    rss_str = f"{peak_rss_kb/1024:.1f}MB" if peak_rss_kb else "n/a"
    print(f"[bench]    wall={wall_seconds:.2f}s  max_rss={rss_str}  status={status}  exit={rc}")
    if status != "OK":
        tail = "\n".join(text.splitlines()[-20:])
        print(f"[bench]    tail of {log_path.name}:\n{tail}")

    return {
        "wall_seconds": wall_seconds,
        "max_rss_kb": peak_rss_kb or None,
        "exit_code": rc,
        "status": status,
        "log": str(log_path),
    }


def run_stage(key: str, cmd, log_dir: Path, timeout_secs: float, time_bin: Path, reps: int = 1):
    """Runs `cmd`, once or (for reps>1) as 1 discarded warm-up + `reps` timed
    reps. Returns the aggregate dict used for the summary.csv row."""
    results = []
    if reps > 1:
        warm_log = log_dir / f"{key}.warmup.log"
        run_once(f"{key} (warm-up, discarded)", cmd, warm_log, timeout_secs, time_bin)
        for i in range(1, reps + 1):
            log_path = log_dir / f"{key}.rep{i}.log"
            r = run_once(f"{key} rep {i}/{reps}", cmd, log_path, timeout_secs, time_bin)
            results.append(r)
            if r["status"] != "OK":
                print(f"[bench]    rep {i} did not complete OK ({r['status']}) -- recorded, not retried")
    else:
        log_path = log_dir / f"{key}.log"
        results.append(run_once(key, cmd, log_path, timeout_secs, time_bin))

    ok = [r for r in results if r["status"] == "OK"]
    # median includes every rep with a real measured value, not just the OK
    # ones -- a stage that OOMs/times out still ran for a measured wall-clock
    # duration and hit a measured peak RSS before being killed, and that is
    # itself the data point (see run_stage docstring / spec).
    walls = [r["wall_seconds"] for r in results if r["wall_seconds"] is not None]
    rsss = [r["max_rss_kb"] for r in results if r["max_rss_kb"] is not None]
    agg = {
        "wall_seconds_median": statistics.median(walls) if walls else None,
        "max_rss_kb_median": statistics.median(rsss) if rsss else None,
        "n_reps": len(results),
        "n_ok": len(ok),
        "statuses": ";".join(r["status"] for r in results),
        "any_ok": len(ok) > 0,
    }
    (log_dir / f"{key}.metrics.json").write_text(json.dumps({"results": results, "aggregate": agg}, indent=2))
    return agg


def compute_overall_status(stage_metrics):
    """Single-glance pass/fail column: 'OK' if every stage that ran was OK,
    else 'FAILED(<stage>:<first-non-OK-status>)' naming the first stage that
    didn't cleanly succeed (a stage with mixed rep statuses, e.g.
    'OK;OOM_OR_KILLED;OK', also counts as failing here)."""
    for key in STAGE_KEYS:
        agg = stage_metrics.get(key)
        if agg is None:
            return f"FAILED({key}:NOT_RUN)"
        statuses = agg.get("statuses", "")
        if any(s != "OK" for s in statuses.split(";") if s):
            return f"FAILED({key}:{statuses})"
    return "OK"


def write_summary_row(config_name, model_dir, settings_path, proof_path, stage_metrics):
    settings = {}
    if settings_path.exists():
        settings = json.loads(settings_path.read_text())
    run_args = settings.get("run_args", {})

    row = {
        "config": config_name,
        "model_dir": str(model_dir),
        "overall_status": compute_overall_status(stage_metrics),
        "logrows": run_args.get("logrows"),
        "num_rows": settings.get("num_rows"),
        "total_assignments": settings.get("total_assignments"),
        "num_dynamic_lookups": settings.get("num_dynamic_lookups"),
        "num_shuffles": settings.get("num_shuffles"),
        "required_lookups": json.dumps(settings.get("required_lookups"), sort_keys=True) if settings.get("required_lookups") is not None else "",
        "lookup_range": json.dumps(run_args.get("lookup_range")) if run_args.get("lookup_range") is not None else "",
        "input_scale": run_args.get("input_scale"),
        "param_scale": run_args.get("param_scale"),
        "proof_size_bytes": proof_path.stat().st_size if proof_path.exists() else "",
    }
    for key in STAGE_KEYS:
        agg = stage_metrics.get(key, {})
        row[f"{key}_wall_s"] = f"{agg['wall_seconds_median']:.3f}" if agg.get("wall_seconds_median") is not None else ""
        row[f"{key}_max_rss_mb"] = f"{agg['max_rss_kb_median']/1024:.1f}" if agg.get("max_rss_kb_median") else ""
        row[f"{key}_status"] = agg.get("statuses", "NOT_RUN")

    summary_path = REPO_DIR / "results" / "summary.csv"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(row.keys())
    write_header = not summary_path.exists()
    import csv
    with open(summary_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)
    print(f"\n[bench] appended row to {summary_path}")
    return summary_path


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model_dir", help="path to a models/<config> dir containing model.onnx + input.json")
    ap.add_argument("--reps", type=int, default=3, help="timed reps for setup/prove (median); +1 discarded warm-up")
    ap.add_argument("--timeout", type=float, default=1800, help="per-stage-attempt timeout in seconds (default 30min)")
    ap.add_argument("--calibration-target", default="resources", choices=["resources", "accuracy"])
    ap.add_argument("--calibrate-scales", default=None,
                     help="pass-through to calibrate-settings --scales (e.g. '4'): pins the scale "
                          "search to exactly this value instead of the default auto-search grid, "
                          "for a controlled-variable scale sweep. Also suffixes the config/output "
                          "dir name with '_scaleN' so it doesn't clobber the unscaled run.")
    ap.add_argument("--ezkl-bin", default=str(DEFAULT_EZKL_BIN))
    ap.add_argument("--time-bin", default=str(DEFAULT_TIME_BIN))
    args = ap.parse_args()

    model_dir = Path(args.model_dir).resolve()
    config_name = model_dir.name
    if args.calibrate_scales is not None:
        config_name = f"{config_name}_scale{args.calibrate_scales}"
    model_onnx = model_dir / "model.onnx"
    input_json = model_dir / "input.json"
    ezkl_bin = Path(args.ezkl_bin)
    time_bin = Path(args.time_bin)

    for p, label in [(model_onnx, "model.onnx"), (input_json, "input.json")]:
        if not p.exists():
            sys.exit(f"ERROR: {label} not found at {p}")
    if not ezkl_bin.exists():
        sys.exit(f"ERROR: ezkl binary not found at {ezkl_bin}")
    if not time_bin.exists():
        sys.exit(
            f"ERROR: time binary not found at {time_bin}\n"
            f"  (extracted from the 'time' .deb into .tools/ -- see this script's module docstring. "
            f"Rerun: cd .tools && apt-get download time && dpkg -x time_*.deb time-pkg)"
        )

    out_dir = REPO_DIR / "results" / config_name
    log_dir = out_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    settings_json = out_dir / "settings.json"
    compiled = out_dir / "model.compiled"
    srs_path = out_dir / "kzg.srs"
    vk_path = out_dir / "vk.key"
    pk_path = out_dir / "pk.key"
    witness_json = out_dir / "witness.json"
    proof_json = out_dir / "proof.json"

    stage_metrics = {}

    def stage(key, cmd, reps=1):
        agg = run_stage(key, cmd, log_dir, args.timeout, time_bin, reps=reps)
        stage_metrics[key] = agg
        return agg["any_ok"]

    aborted = not stage("gen_settings", [ezkl_bin, "gen-settings", "-M", model_onnx, "-O", settings_json])

    if not aborted:
        calib_cmd = [ezkl_bin, "calibrate-settings", "-M", model_onnx,
                     "-D", input_json, "-O", settings_json,
                     "--target", args.calibration_target]
        if args.calibrate_scales is not None:
            calib_cmd += ["--scales", args.calibrate_scales]
        aborted = not stage("calibrate_settings", calib_cmd)
    if not aborted:
        aborted = not stage("compile_circuit", [ezkl_bin, "compile-circuit", "-M", model_onnx,
                                                  "--compiled-circuit", compiled, "-S", settings_json])
    if not aborted:
        logrows = json.loads(settings_json.read_text())["run_args"]["logrows"]
        aborted = not stage("gen_srs", [ezkl_bin, "gen-srs", "--srs-path", srs_path, "--logrows", logrows])

    if not aborted:
        aborted = not stage("setup", [ezkl_bin, "setup", "-M", compiled, "--srs-path", srs_path,
                                       "--vk-path", vk_path, "--pk-path", pk_path], reps=args.reps)
    if not aborted:
        aborted = not stage("gen_witness", [ezkl_bin, "gen-witness", "-D", input_json,
                                             "-M", compiled, "-O", witness_json])
    if not aborted:
        aborted = not stage("prove", [ezkl_bin, "prove", "-W", witness_json, "-M", compiled,
                                       "--pk-path", pk_path, "--proof-path", proof_json,
                                       "--srs-path", srs_path], reps=args.reps)
    if not aborted:
        stage("verify", [ezkl_bin, "verify", "-S", settings_json, "--proof-path", proof_json,
                          "--vk-path", vk_path, "--srs-path", srs_path])

    summary_path = write_summary_row(config_name, model_dir.relative_to(REPO_DIR), settings_json, proof_json, stage_metrics)

    if aborted:
        print(f"\n[bench] pipeline stopped early for {config_name} -- see per-stage status in {summary_path}")
        sys.exit(1)
    print(f"\n[bench] {config_name}: full pipeline completed")


if __name__ == "__main__":
    main()
