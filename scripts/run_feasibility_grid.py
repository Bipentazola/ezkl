#!/usr/bin/env python3

import csv
import json
import math
import os
import re
import subprocess
import sys
import time
from pathlib import Path

REPO_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = REPO_DIR / "models_grid"
RESULTS_DIR = REPO_DIR / "results" / "grid"
DOCS_DIR = REPO_DIR / "docs" / "recon"
EZKL_BIN = REPO_DIR / "target" / "release" / "ezkl"
TIME_BIN = REPO_DIR / ".tools" / "time-pkg" / "usr" / "bin" / "time"
TIMEOUT_SECS = 900
BUDGET_LOGROWS = 21

sys.path.insert(0, str(REPO_DIR))
import gen_models  # noqa: E402

D_MODELS = [128, 256, 384, 512, 768]
SEQ_LENS = [8, 16, 32, 64, 128]
LAYERS = [1, 2]
N_HEAD_BY_D = {128: 2, 256: 4, 384: 6, 512: 8, 768: 12}

COEF_NR = [3.55539968e-02, 9.95583146e-01, 7.59621744e02, 1.42661279e04]
COEF_WALL = [
    1.44929815e-07,
    3.36866214e-05,
    5.77914925e-04,
    -4.56474939e-02,
    7.20271152e00,
]
COEF_RSS = [
    1.95305714e-05,
    7.66668668e-04,
    -8.46204814e-04,
    -2.00378100e00,
    2.87746063e02,
]
LAYERS2_MULT_NR = 1.925
LAYERS2_MULT_WALL = 1.9

MIN_LOGROWS, RESERVED_BLINDING_ROWS, MAX_PUBLIC_SRS = 6, 8, 26


def logrows_est(num_rows: int) -> int:
    return min(
        max(MIN_LOGROWS, math.ceil(math.log2(num_rows + RESERVED_BLINDING_ROWS))), 32
    )


def predict_num_rows(d, s, layers):
    base = (
        COEF_NR[0] * s * d**2
        + COEF_NR[1] * s**2 * d
        + COEF_NR[2] * s * d
        + COEF_NR[3] * s
    )
    return base * (LAYERS2_MULT_NR if layers == 2 else 1.0)


def predict_wall(d, s, layers):
    x = [s * d**2, s**2 * d, s * d, s, 1]
    base = sum(c * v for c, v in zip(COEF_WALL, x))
    return max(base, 1.0) * (LAYERS2_MULT_WALL if layers == 2 else 1.0)


def predict_rss(d, s, layers):
    x = [s * d**2, s**2 * d, s * d, s, 1]
    base = sum(c * v for c, v in zip(COEF_RSS, x))
    return max(base, 20.0) * (LAYERS2_MULT_WALL if layers == 2 else 1.0)


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


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


def run_gen_settings(name: str, onnx_path: Path):
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
    return status, wall_s, (rss_kb / 1024 if rss_kb else None), settings_path, out_dir


CATEGORIES = ["matmul", "qk", "av", "softmax", "gelu", "layernorm", "rescale", "other"]


def build_grid():
    configs = []
    for layers in LAYERS:
        for d in D_MODELS:
            for s in SEQ_LENS:
                nr_pred = predict_num_rows(d, s, layers)
                lr_pred = logrows_est(nr_pred)
                w_pred = predict_wall(d, s, layers)
                r_pred = predict_rss(d, s, layers)
                skip = lr_pred > 27
                configs.append(
                    dict(
                        d_model=d,
                        seq_len=s,
                        n_layers=layers,
                        n_head=N_HEAD_BY_D[d],
                        name=f"seq{s}_d{d}_l{layers}",
                        num_rows_pred=nr_pred,
                        logrows_pred=lr_pred,
                        wall_s_pred=w_pred,
                        rss_mb_pred=r_pred,
                        skip_predicted=skip,
                    )
                )
    configs.sort(key=lambda c: c["num_rows_pred"])
    return configs


def parse_args():
    import argparse

    p = argparse.ArgumentParser(description=None)
    p.add_argument(
        "--out-dir",
        default=str(RESULTS_DIR),
        help="where grid.csv and per-config artifacts go (default: results/grid)",
    )
    p.add_argument(
        "--models-dir",
        default=str(MODELS_DIR),
        help="where the exported ONNX models go (default: models_grid)",
    )
    p.add_argument(
        "--reuse-models",
        action="store_true",
        help="skip re-exporting a config whose model.onnx already exists",
    )
    p.add_argument(
        "--no-report",
        action="store_true",
        help="skip rewriting docs/recon/04_feasibility_grid.md (use for variant runs, "
        "whose conclusions belong in a comparison doc instead)",
    )
    return p.parse_args()


def main():
    global RESULTS_DIR, MODELS_DIR
    args = parse_args()
    RESULTS_DIR = Path(args.out_dir)
    MODELS_DIR = Path(args.models_dir)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    gen_models.MODELS_DIR = MODELS_DIR
    for var in (
        "EZKL_DIV_ROUND_NEAREST",
        "EZKL_RESCALE_FUSION",
        "EZKL_RESCALE_FUSION_MAX_DIVISOR_BITS",
    ):
        if var in os.environ:
            log(f"optimization switch active: {var}={os.environ[var]}")

    configs = build_grid()
    n_skip = sum(1 for c in configs if c["skip_predicted"])
    n_run = len(configs) - n_skip
    total_pred = sum(c["wall_s_pred"] for c in configs if not c["skip_predicted"])
    log(
        f"=== feasibility grid: {len(configs)} configs, {n_run} to run, {n_skip} predicted-skipped "
        f"(logrows_pred>27), total predicted gen-settings wall time {total_pred/60:.1f} min ==="
    )
    for c in configs:
        if c["skip_predicted"]:
            log(
                f"SKIP_PREDICTED {c['name']} (L{c['n_layers']}): "
                f"predicted num_rows={c['num_rows_pred']:,.0f} logrows={c['logrows_pred']} > 27"
            )

    csv_path = RESULTS_DIR / "grid.csv"
    fieldnames = (
        [
            "name",
            "d_model",
            "seq_len",
            "n_layers",
            "n_head",
            "status",
            "num_rows_pred",
            "logrows_pred",
            "wall_s_pred",
            "rss_mb_pred",
            "num_rows",
            "logrows_est",
            "wall_s",
            "rss_mb",
        ]
        + [f"{c}_rows" for c in CATEGORIES]
        + [f"{c}_pct" for c in CATEGORIES]
        + ["matmul_family_pct", "nonlinear_pct"]
    )
    write_header = not csv_path.exists()
    csv_f = open(csv_path, "a", newline="")
    writer = csv.DictWriter(csv_f, fieldnames=fieldnames)
    if write_header:
        writer.writeheader()
        csv_f.flush()

    t_start = time.monotonic()
    n_done = 0
    for c in configs:
        row = {
            "name": c["name"],
            "d_model": c["d_model"],
            "seq_len": c["seq_len"],
            "n_layers": c["n_layers"],
            "n_head": c["n_head"],
            "num_rows_pred": round(c["num_rows_pred"]),
            "logrows_pred": c["logrows_pred"],
            "wall_s_pred": round(c["wall_s_pred"], 1),
            "rss_mb_pred": round(c["rss_mb_pred"], 1),
        }
        if c["skip_predicted"]:
            row["status"] = "SKIPPED_PREDICTED"
            writer.writerow(row)
            csv_f.flush()
            continue

        n_done += 1
        cfg = dict(
            seq_len=c["seq_len"],
            d_model=c["d_model"],
            n_head=c["n_head"],
            n_layers=c["n_layers"],
        )
        log(
            f"[{n_done}/{n_run}] {c['name']} L{c['n_layers']}: exporting ONNX "
            f"(predicted num_rows={c['num_rows_pred']:,.0f} logrows={c['logrows_pred']} "
            f"wall~{c['wall_s_pred']:.0f}s rss~{c['rss_mb_pred']:.0f}MB)"
        )
        try:
            existing = MODELS_DIR / c["name"] / "model.onnx"
            if args.reuse_models and existing.exists():
                log("    reusing existing ONNX")
            else:
                gen_models.export_one(c["name"], cfg, opset=10, seed=42)
        except Exception as e:
            log(f"    EXPORT_FAILED: {e}")
            row["status"] = "EXPORT_FAILED"
            writer.writerow(row)
            csv_f.flush()
            continue

        onnx_path = MODELS_DIR / c["name"] / "model.onnx"
        status, wall_s, rss_mb, settings_path, out_dir = run_gen_settings(
            c["name"], onnx_path
        )
        row["status"] = status
        row["wall_s"] = round(wall_s, 1) if wall_s else ""
        row["rss_mb"] = round(rss_mb, 1) if rss_mb else ""
        log(f"    gen-settings: status={status} wall={wall_s:.1f}s rss={rss_mb}MB")

        if status == "OK":
            settings = json.loads(settings_path.read_text())
            num_rows = settings["num_rows"]
            row["num_rows"] = num_rows
            row["logrows_est"] = logrows_est(num_rows)
            parse_res = subprocess.run(
                [
                    sys.executable,
                    str(REPO_DIR / "scripts" / "parse_ledger.py"),
                    str(out_dir),
                ],
                capture_output=True,
                text=True,
            )
            ledger_csv = out_dir / "op_ledger.csv"
            if parse_res.returncode == 0 and ledger_csv.exists():
                ledger = {r["category"]: r for r in csv.DictReader(open(ledger_csv))}
                total = sum(int(ledger[cat]["total_rows"]) for cat in CATEGORIES)
                for cat in CATEGORIES:
                    row[f"{cat}_rows"] = int(ledger[cat]["total_rows"])
                    row[f"{cat}_pct"] = (
                        round(100.0 * int(ledger[cat]["total_rows"]) / total, 3)
                        if total
                        else 0
                    )
                row["matmul_family_pct"] = round(
                    row["matmul_pct"] + row["qk_pct"] + row["av_pct"], 3
                )
                row["nonlinear_pct"] = round(row["softmax_pct"] + row["gelu_pct"], 3)
            else:
                log(
                    f"    WARNING: parse_ledger.py failed for {c['name']}: {parse_res.stderr[-500:]}"
                )

        writer.writerow(row)
        csv_f.flush()

        elapsed = time.monotonic() - t_start
        log(f"    done. {n_done}/{n_run} run so far, {elapsed/60:.1f} min elapsed")

    csv_f.close()
    log(
        f"=== grid run complete: {len(configs)} configs total ({n_run} run, {n_skip} skipped), "
        f"{(time.monotonic()-t_start)/60:.1f} min for the run phase ==="
    )

    if args.no_report:
        log(
            "=== --no-report: docs/recon/04_feasibility_grid.md left untouched -- DONE ==="
        )
    else:
        write_report(csv_path)
        log("=== docs/recon/04_feasibility_grid.md written -- DONE ===")


def write_report(csv_path: Path):
    rows = list(csv.DictReader(open(csv_path)))
    ok_rows = [r for r in rows if r["status"] == "OK"]

    in_budget = [
        r
        for r in ok_rows
        if r["logrows_est"] and int(r["logrows_est"]) <= BUDGET_LOGROWS
    ]
    in_budget.sort(key=lambda r: -int(r["num_rows"]))
    top5 = in_budget[:5]

    tier_candidates = []
    for r in ok_rows:
        nr = int(r["num_rows"])
        lr = int(r["logrows_est"])
        threshold = 2 ** (lr - 1)
        ratio = nr / threshold
        if ratio <= 1.20:
            would_cross_14pct = (nr * 0.86) <= threshold
            tier_candidates.append(
                (r["name"], r["n_layers"], nr, lr, threshold, ratio, would_cross_14pct)
            )
    tier_candidates.sort(key=lambda x: x[5])

    would_enter_budget = []
    for r in ok_rows:
        nr = int(r["num_rows"])
        lr = int(r["logrows_est"])
        if lr > BUDGET_LOGROWS:
            lr_after = logrows_est(int(nr * 0.86))
            if lr_after <= BUDGET_LOGROWS:
                would_enter_budget.append((r["name"], r["n_layers"], nr, lr, lr_after))
    would_enter_budget.sort(key=lambda x: x[2])

    n_ok = len(ok_rows)
    n_skip_pred = sum(1 for r in rows if r["status"] == "SKIPPED_PREDICTED")
    n_fail = len(rows) - n_ok - n_skip_pred

    lines = []
    lines.append(
        "# Feasibility grid: d_model x seq_len x n_layers row/logrows survey\n"
    )
    lines.append(
        f"ezkl commit: see `git rev-parse HEAD` at run time. Scale: gen-settings CLI "
        f"defaults (input_scale=7, param_scale=7), same as results/p_measurement/. "
        f"Only `gen-settings` was run: the results measure circuit row "
        f"counts for these settings, without establishing successful "
        f"setup/prove/verify or peak proving memory.\n"
    )
    lines.append(
        f"Grid: d_model in {D_MODELS} x seq_len in {SEQ_LENS} x n_layers in {LAYERS} "
        f"= {len(D_MODELS)*len(SEQ_LENS)*len(LAYERS)} configs. "
        f"{n_ok} measured OK, {n_skip_pred} skipped up front (predicted logrows>27, "
        f"clearly over the logrows<=26 MAX_PUBLIC_SRS ceiling), {n_fail} failed/timed out "
        f"during the actual run (see grid.csv `status` column for which).\n"
    )

    lines.append(
        "## Q1: largest real widths within the logrows<=21 setup/prove budget\n"
    )
    lines.append(
        f'Budget = logrows<=**{BUDGET_LOGROWS}** ("actually runnable for setup/prove on this '
        f"machine\" per the task's own framing). Top 5 by num_rows among all measured "
        f"configs meeting that budget:\n"
    )
    lines.append("| rank | config | n_layers | num_rows | logrows_est |")
    lines.append("|---|---|---|---|---|")
    for i, r in enumerate(top5, 1):
        lines.append(
            f"| {i} | {r['name']} | {r['n_layers']} | {int(r['num_rows']):,} | {r['logrows_est']} |"
        )
    if not top5:
        lines.append("| -- | (none found within budget) | | | |")
    lines.append("")

    lines.append(
        "## Q2: configs within 20% above a 2^k tier boundary (row-cut tier-crossing candidates)\n"
    )
    lines.append(
        f"threshold for a config at logrows=k is `2^(k-1)`; `ratio` = num_rows/threshold. "
        f"`would_cross_14pct` = TRUE means a 14% row-count cut (the div_round_nearest-scale "
        f"reduction) would drop it strictly below that threshold, i.e. buy a full logrows "
        f"tier -- not just close, but confirmed under the exact -14% figure.\n"
    )
    lines.append(
        "| config | n_layers | num_rows | logrows_est | tier threshold (2^(k-1)) | ratio | would_cross @ -14% |"
    )
    lines.append("|---|---|---|---|---|---|---|")
    for name, nl, nr, lr, thr, ratio, cross in tier_candidates:
        lines.append(
            f"| {name} | {nl} | {nr:,} | {lr} | {thr:,} | {ratio:.3f} | {'YES' if cross else 'no'} |"
        )
    if not tier_candidates:
        lines.append("| -- | (none found within 20% of a tier boundary) | | | | | |")
    lines.append("")

    lines.append(
        "## Q3: over budget now, would fall within logrows<=21 budget after a -14% row cut\n"
    )
    lines.append(
        "| config | n_layers | num_rows | logrows_est (now) | logrows_est (after -14%) |"
    )
    lines.append("|---|---|---|---|---|")
    for name, nl, nr, lr, lr_after in would_enter_budget:
        lines.append(f"| {name} | {nl} | {nr:,} | {lr} | {lr_after} |")
    if not would_enter_budget:
        lines.append("| -- | (none found) | | | |")
    lines.append("")

    lines.append("## Full data\n")
    lines.append(
        "See `results/grid/grid.csv` for every config (measured + predicted-skipped + "
        "any failed/timed-out), including per-category row percentages "
        "(matmul/qk/av/softmax/gelu/layernorm/rescale/other) for every successfully "
        "measured config.\n"
    )

    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    (DOCS_DIR / "04_feasibility_grid.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
