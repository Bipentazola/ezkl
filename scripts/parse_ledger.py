#!/usr/bin/env python3

import csv
import json
import re
import sys
from pathlib import Path

LOG_CANDIDATES = [
    "setup.rep1.log",
    "setup.log",
    "setup.warmup.log",
    "gen_witness.log",
    "prove.rep1.log",
    "prove.log",
    "gen_settings.log",
]

LAYING_OUT_RE = re.compile(r"ezkl::graph::model\] - laying out (\d+): (.*)$")
REGION_RE = re.compile(
    r"ezkl::circuit::ops::region\] - \(rows=(\d+), coord=(\d+), constants=(\d+), "
    r"max_lookup_inputs=(-?\d+), min_lookup_inputs=(-?\d+), max_range_size=(-?\d+), "
    r"dynamic_lookup_col_coord=(\d+), shuffle_col_coord=(\d+), max_dynamic_input_len=(\d+)\)"
)
TABLE_RANGE_RE = re.compile(
    r"ezkl::circuit::table\] - table range: \((-?\d+), (-?\d+)\)"
)
RANGE_CHECK_RANGE_RE = re.compile(
    r"ezkl::circuit::table\] - range check range: \((-?\d+), (-?\d+)\)"
)
LAYING_OUT_TABLE_RE = re.compile(
    r"ezkl::circuit::ops::chip\] - laying out table for (\w+)\(([^)]*)\)"
)
EINSUM_RE = re.compile(r"EINSUM ([a-z]+),([a-z]+)->([a-z]+)")


def find_log_file(config_dir: Path) -> Path:
    for name in LOG_CANDIDATES:
        p = config_dir / "logs" / name
        if p.exists():
            return p
    raise SystemExit(
        f"no usable log found in {config_dir/'logs'} (looked for {LOG_CANDIDATES})"
    )


def first_pass_lines(log_path: Path):
    prev_idx = None
    with open(log_path, errors="replace") as f:
        for line in f:
            m = LAYING_OUT_RE.search(line)
            if m:
                idx = int(m.group(1))
                if prev_idx is not None and idx <= prev_idx:
                    return
                prev_idx = idx
            yield line


def classify(desc: str, state: dict) -> str:
    if "SOFTMAX" in desc:
        state["mode"] = "av" if state.get("mode", "qk") == "qk" else "qk"
        return "softmax"
    if "TANH" in desc:
        return "gelu"
    if (
        "MEANOFSQUARES" in desc
        or "RSQRT" in desc
        or "RESCALED INPUT (SUB)" in desc
        or "SUM (axes=" in desc
        or desc == "SUB"
    ):
        return "layernorm"
    m = EINSUM_RE.search(desc)
    if m:
        op1, op2, _out = m.groups()
        if min(len(op1), len(op2)) == 2:
            return "matmul"
        else:
            return state.get("mode", "qk")
    if "REBASED" in desc and re.search(r"\((MULT|DIV[^A-Z]*)\)\s*$", desc):
        return "rescale"
    return "other"


def parse_config(config_dir: Path):
    settings = json.loads((config_dir / "settings.json").read_text())
    num_rows_total = settings["num_rows"]
    log_path = find_log_file(config_dir)

    nodes = []
    tables = {"value_tables": {}, "range_check_tables": set()}
    state = {"mode": "qk"}

    for line in first_pass_lines(log_path):
        m = LAYING_OUT_TABLE_RE.search(line)
        if m:
            fn, params = m.groups()
            tables["value_tables"][fn] = params
            continue
        m = RANGE_CHECK_RANGE_RE.search(line)
        if m:
            tables["range_check_tables"].add((int(m.group(1)), int(m.group(2))))
            continue
        m = LAYING_OUT_RE.search(line)
        if m:
            idx, desc = int(m.group(1)), m.group(2)
            nodes.append([idx, desc, None])
            continue
        m = REGION_RE.search(line)
        if m and nodes and nodes[-1][2] is None:
            nodes[-1][2] = int(m.group(1))
            continue

    if not nodes:
        raise SystemExit(
            f"parsed zero 'laying out' nodes from {log_path} -- log format mismatch?"
        )

    rows_used = []
    for i, (idx, desc, rows_before) in enumerate(nodes):
        next_rows = nodes[i + 1][2] if i + 1 < len(nodes) else num_rows_total
        rows_used.append(max(0, next_rows - rows_before))

    records = []
    for (idx, desc, rows_before), delta in zip(nodes, rows_used):
        cat = classify(desc, state)
        uses_lookup_table = None
        if cat == "gelu":
            uses_lookup_table = "Tanh"
        elif cat == "softmax":
            uses_lookup_table = "Exp"
        elif cat == "rescale":
            uses_lookup_table = (
                f"range_check(0,{settings['run_args']['decomp_base']-1})"
            )
        records.append(
            {
                "node_idx": idx,
                "description": desc,
                "category": cat,
                "rows": delta,
                "uses_lookup_table": uses_lookup_table,
            }
        )

    return records, settings, tables, log_path


def write_op_ledger_csv(records, out_path: Path):
    categories = [
        "matmul",
        "qk",
        "av",
        "softmax",
        "gelu",
        "layernorm",
        "rescale",
        "other",
    ]
    total_rows = sum(r["rows"] for r in records)
    agg = {c: {"rows": 0, "lookup_op_instances": 0} for c in categories}
    for r in records:
        agg[r["category"]]["rows"] += r["rows"]
        if r["uses_lookup_table"] is not None:
            agg[r["category"]]["lookup_op_instances"] += 1

    total_lookup_instances = sum(a["lookup_op_instances"] for a in agg.values())
    rows_out = []
    for c in categories:
        rows_out.append(
            {
                "category": c,
                "total_rows": agg[c]["rows"],
                "pct_rows": 100.0 * agg[c]["rows"] / total_rows if total_rows else 0.0,
                "lookup_op_instances": agg[c]["lookup_op_instances"],
                "pct_lookup_op_instances": (
                    100.0 * agg[c]["lookup_op_instances"] / total_lookup_instances
                    if total_lookup_instances
                    else 0.0
                ),
            }
        )

    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "category",
                "total_rows",
                "pct_rows",
                "lookup_op_instances",
                "pct_lookup_op_instances",
            ],
        )
        writer.writeheader()
        writer.writerows(rows_out)
    return rows_out


def main():
    if len(sys.argv) != 2:
        print("Usage: python3 scripts/parse_ledger.py [arguments]")
        sys.exit(1)
    config_dir = Path(sys.argv[1]).resolve()
    records, settings, tables, log_path = parse_config(config_dir)

    out_csv = config_dir / "op_ledger.csv"
    rows_out = write_op_ledger_csv(records, out_csv)

    print(f"parsed {len(records)} op nodes from {log_path}", file=sys.stderr)
    print(f"value tables (from log): {tables['value_tables']}", file=sys.stderr)
    print(
        f"range-check tables (from log): {sorted(tables['range_check_tables'])}",
        file=sys.stderr,
    )
    for row in rows_out:
        print(
            f"  {row['category']:10s} rows={row['total_rows']:>8d} ({row['pct_rows']:5.1f}%)  "
            f"lookup_op_instances={row['lookup_op_instances']:>3d} ({row['pct_lookup_op_instances']:5.1f}%)",
            file=sys.stderr,
        )
    print(f"wrote {out_csv}", file=sys.stderr)

    raw_path = config_dir / "op_ledger_raw_nodes.csv"
    with open(raw_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "node_idx",
                "description",
                "category",
                "rows",
                "uses_lookup_table",
            ],
        )
        writer.writeheader()
        writer.writerows(records)
    print(f"wrote {raw_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
