#!/usr/bin/env python3

import csv
import json
import math
from pathlib import Path

REPO_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_DIR / "results" / "p_measurement"
CATEGORIES = ["matmul", "qk", "av", "softmax", "gelu", "layernorm", "rescale", "other"]

MIN_LOGROWS = 6
RESERVED_BLINDING_ROWS = 8
MAX_PUBLIC_SRS = 26


def logrows_estimate(num_rows: int) -> int:
    est = max(MIN_LOGROWS, math.ceil(math.log2(num_rows + RESERVED_BLINDING_ROWS)))
    return min(est, MAX_PUBLIC_SRS)


def load_config(name: str, group: str) -> dict:
    cfg_dir = RESULTS_DIR / name
    settings = json.loads((cfg_dir / "settings.json").read_text())
    ledger = {}
    with open(cfg_dir / "op_ledger.csv") as f:
        for row in csv.DictReader(f):
            ledger[row["category"]] = row

    num_rows = settings["num_rows"]
    total_rows = sum(int(ledger[c]["total_rows"]) for c in CATEGORIES)
    row = {
        "name": name,
        "group": group,
        "seq_len": None,
        "d_model": None,
        "n_head": None,
        "num_rows": num_rows,
        "total_ledger_rows": total_rows,
        "logrows_est": logrows_estimate(num_rows),
        "num_inner_cols": settings["run_args"]["num_inner_cols"],
        "input_scale": settings["run_args"]["input_scale"],
        "param_scale": settings["run_args"]["param_scale"],
    }
    for c in CATEGORIES:
        row[f"{c}_rows"] = int(ledger[c]["total_rows"])
        row[f"{c}_pct"] = round(float(ledger[c]["pct_rows"]), 3)
    p = row["matmul_rows"] + row["qk_rows"] + row["av_rows"]
    row["matmul_family_rows"] = p
    row["p_matmul_family_pct"] = round(100.0 * p / total_rows, 3)
    p_nonlinear = row["softmax_rows"] + row["gelu_rows"]
    row["nonlinear_rows"] = p_nonlinear
    row["p_nonlinear_pct"] = round(100.0 * p_nonlinear / total_rows, 3)
    return row


def main():
    manifest = {
        r["name"]: r
        for r in csv.DictReader(open(REPO_DIR / "models_p" / "MANIFEST.csv"))
    }

    rows = []
    for cfg_dir in sorted(RESULTS_DIR.iterdir()):
        if not cfg_dir.is_dir() or cfg_dir.name not in manifest:
            continue
        name = cfg_dir.name
        m = manifest[name]
        row = load_config(name, m["group"])
        row["seq_len"] = int(m["seq_len"])
        row["d_model"] = int(m["d_model"])
        row["n_head"] = int(m["n_head"])
        row["params"] = int(m["params"])
        rows.append(row)

    fieldnames = (
        [
            "name",
            "group",
            "seq_len",
            "d_model",
            "n_head",
            "params",
            "num_rows",
            "total_ledger_rows",
            "logrows_est",
            "num_inner_cols",
            "input_scale",
            "param_scale",
        ]
        + [f"{c}_rows" for c in CATEGORIES]
        + [f"{c}_pct" for c in CATEGORIES]
        + [
            "matmul_family_rows",
            "p_matmul_family_pct",
            "nonlinear_rows",
            "p_nonlinear_pct",
        ]
    )

    width_rows = sorted(
        [r for r in rows if "width_scan" in r["group"]], key=lambda r: r["d_model"]
    )
    seq_rows = sorted(
        [r for r in rows if "seq_scan" in r["group"]], key=lambda r: r["seq_len"]
    )

    for out_name, out_rows in [
        ("p_by_width.csv", width_rows),
        ("p_by_seqlen.csv", seq_rows),
    ]:
        out_path = RESULTS_DIR / out_name
        with open(out_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in out_rows:
                writer.writerow({k: r[k] for k in fieldnames})
        print(f"wrote {out_path} ({len(out_rows)} rows)")

    all_path = RESULTS_DIR / "p_measurement_all.csv"
    with open(all_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in sorted(rows, key=lambda r: (r["group"], r["d_model"], r["seq_len"])):
            writer.writerow({k: r[k] for k in fieldnames})
    print(f"wrote {all_path} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
