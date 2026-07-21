#!/usr/bin/env python3
"""
parse_ledger.py -- per-operator row/lookup ledger for one results/<config>/
directory, built from ezkl's RUST_LOG=debug layout logs + settings.json.

## Log format this depends on (verified against results/seq8_d16_l1/logs/setup.rep1.log)

During circuit layout, `src/graph/model.rs::layout_nodes` prints, for every
graph node, in this exact order:

    [D] ..., ezkl::graph::model] - laying out {idx}: {op_description}
    [D] ..., ezkl::circuit::ops::region] - (rows={R}, coord={C}, constants=...,
        max_lookup_inputs=..., min_lookup_inputs=..., max_range_size=...,
        dynamic_lookup_col_coord=..., shuffle_col_coord=..., max_dynamic_input_len=...)
    [D] ..., ezkl::graph::model] - ------------ output node {idx}: ...
    [D] ..., ezkl::graph::model] - ------------ layout of {idx} took {duration}

IMPORTANT: the region snapshot is printed by `region.debug_report()` which is
called *before* node {idx} is laid out (see model.rs, right after the "laying
out" print, before any of that node's own `region.assign`/`region.increment`
calls) -- so `rows` at node idx's snapshot is the CUMULATIVE total consumed by
all *previous* nodes, not by node idx itself. Node idx's own row cost is
`rows_at(next node's snapshot) - rows_at(idx's snapshot)`; the last node in a
pass uses `settings.json["num_rows"]` as the "next" boundary.

A full `setup` invocation calls the model's `layout_nodes` at least 4 times
(empirically: 2x per keygen phase [vk, pk], and within each phase the node
0..N traversal itself repeats once more -- likely a measurement/dry-run pass
followed by the real assignment pass; both reach the same final row count, so
they're redundant, not additive). `layout_nodes` iterates `self.graph.nodes`,
a `BTreeMap`, so node `idx` is *guaranteed* strictly increasing within one
real traversal (verified: `laying out 0`, `1`, `3`, `4`, ... never decreases).
This script exploits that: it reads "laying out K: ..." lines until K would
be <= the previous node's idx, and stops there -- keeping exactly one
complete, real traversal and discarding every repeat, regardless of how many
there turn out to be for a given stage/config.

## What IS and ISN'T recoverable from this log (see also results/LEDGER_NOTES.md)

- Per-op row usage: exact (the delta above).
- Per-op TYPE/description: exact (the "laying out" string).
- Which static lookup/range-check tables exist + their value ranges: exact,
  but only as a GLOBAL list printed once per pass (`table range: (...)`,
  `range check range: (...)`, `laying out table for X`) -- NOT tied to a
  specific op instance in the log.
- Per-op-instance LOOKUP CALL COUNT (how many individual table lookups a
  given op invocation performs): NOT in the log at all. The log has no
  per-node counter for this. This script instead reports, per category, the
  count of op *instances* that reference a lookup-requiring table
  (`lookup_op_instances`) -- an instance count, not a per-element call count.
  See results/LEDGER_NOTES.md for where to add real instrumentation
  (`src/circuit/ops/layouts.rs`'s `nonlinearity()`/`decompose()`) if exact
  per-element call counts are needed.

## Category assignment (semantic buckets)

Assigned by regex/heuristic over each node's `laying out` description, in
this priority order:

1. `softmax`      -- "SOFTMAX" in description
2. `gelu`         -- "TANH" in description (this model's GELU is
                     tanh-approximated per `gen_models.py::new_gelu`; TANH
                     never appears standalone elsewhere in this architecture)
3. `layernorm`    -- "MEANOFSQUARES" / "RSQRT" / "RESCALED INPUT (SUB)" /
                     "SUM (axes=" -- all four are LayerNorm-internal steps
                     (mean, centering, variance, 1/sqrt(var+eps)) in this
                     architecture; there is no other reduce-sum in the graph.
4. `matmul`       -- EINSUM whose shorter operand has exactly 2 index
                     letters (a plain 2D weight matrix, e.g. "kn") -- QKV
                     projection / attention output projection / FFN fc1+fc2,
                     all grouped together per the requested taxonomy.
5. `qk` / `av`    -- EINSUM where both operands have >=3 index letters (an
                     activation x activation contraction with a shared
                     batch/head index) -- these are attention's Q.K^T and
                     A.V. Distinguishing the two needs a *sequence* heuristic
                     (equation shape alone can't tell you which): a small
                     state machine starts in "qk" state, flips to "av" on
                     the nearest preceding SOFTMAX node, and resets to "qk"
                     on the next `matmul`-category node (i.e. the next
                     Linear layer, which starts a new attention block or is
                     the output/FFN projection).
6. `rescale`      -- "REBASED" wrapping a bare MULT/DIV with no other named
                     op inside the parens -- i.e. a plain fixed-point
                     multiply immediately followed by its mandatory
                     scale-down division, with nothing else going on. This
                     is the model's "rescale after every multiply" pattern.
                     NOTE: REBASED wrapping EINSUM/SOFTMAX/MEANOFSQUARES also
                     performs a rescale, but that rescale's row/lookup cost
                     is NOT separable from the wrapped op's own cost in this
                     log (one node = one row-delta) -- it stays folded into
                     that op's category. See results/LEDGER_NOTES.md 4(b).
7. `other`        -- everything else (CONST, IDENTITY, RESHAPE, Input,
                     residual/bias ADD, anything unmatched).

Usage:
    python3 scripts/parse_ledger.py results/seq8_d16_l1
    # writes results/seq8_d16_l1/op_ledger.csv
"""
import csv
import json
import re
import sys
from pathlib import Path

LOG_CANDIDATES = [
    "setup.rep1.log", "setup.log", "setup.warmup.log",
    "gen_witness.log", "prove.rep1.log", "prove.log",
]

LAYING_OUT_RE = re.compile(r"ezkl::graph::model\] - laying out (\d+): (.*)$")
REGION_RE = re.compile(
    r"ezkl::circuit::ops::region\] - \(rows=(\d+), coord=(\d+), constants=(\d+), "
    r"max_lookup_inputs=(-?\d+), min_lookup_inputs=(-?\d+), max_range_size=(-?\d+), "
    r"dynamic_lookup_col_coord=(\d+), shuffle_col_coord=(\d+), max_dynamic_input_len=(\d+)\)"
)
TABLE_RANGE_RE = re.compile(r"ezkl::circuit::table\] - table range: \((-?\d+), (-?\d+)\)")
RANGE_CHECK_RANGE_RE = re.compile(r"ezkl::circuit::table\] - range check range: \((-?\d+), (-?\d+)\)")
LAYING_OUT_TABLE_RE = re.compile(r"ezkl::circuit::ops::chip\] - laying out table for (\w+)\(([^)]*)\)")
EINSUM_RE = re.compile(r"EINSUM ([a-z]+),([a-z]+)->([a-z]+)")


def find_log_file(config_dir: Path) -> Path:
    for name in LOG_CANDIDATES:
        p = config_dir / "logs" / name
        if p.exists():
            return p
    raise SystemExit(f"no usable log found in {config_dir/'logs'} (looked for {LOG_CANDIDATES})")


def first_pass_lines(log_path: Path):
    """Yields lines up through the end of the first complete `layout_nodes`
    traversal, then stops. Boundary = the first "laying out K: ..." line
    whose K is <= the previous node's idx (impossible within one real
    traversal, since layout_nodes iterates a BTreeMap in strictly increasing
    key order) -- i.e. the start of a repeat (dry-run/vk/pk) traversal."""
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
        # toggle, don't reset-on-matmul: an intervening weight matmul (e.g. V's
        # own projection, scheduled between softmax and the AV einsum by the
        # topological node order) must not flip attention-block state back.
        # Only a SOFTMAX genuinely marks a qk<->av transition (one softmax per
        # attention block -- this architecture batches all heads into a single
        # softmax call via the shared "b" einsum index, not one per head).
        state["mode"] = "av" if state.get("mode", "qk") == "qk" else "qk"
        return "softmax"
    if "TANH" in desc:
        return "gelu"
    if "MEANOFSQUARES" in desc or "RSQRT" in desc or "RESCALED INPUT (SUB)" in desc or "SUM (axes=" in desc:
        return "layernorm"
    m = EINSUM_RE.search(desc)
    if m:
        op1, op2, _out = m.groups()
        if min(len(op1), len(op2)) == 2:
            return "matmul"  # plain 2D weight operand -- does NOT touch qk/av state
        else:
            return state.get("mode", "qk")  # "qk" until a softmax is seen, then "av"
    if "REBASED" in desc and re.search(r"\((MULT|DIV[^A-Z]*)\)\s*$", desc):
        return "rescale"
    return "other"


def parse_config(config_dir: Path):
    settings = json.loads((config_dir / "settings.json").read_text())
    num_rows_total = settings["num_rows"]
    log_path = find_log_file(config_dir)

    nodes = []  # (idx, desc, rows_before)
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
        raise SystemExit(f"parsed zero 'laying out' nodes from {log_path} -- log format mismatch?")

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
            uses_lookup_table = f"range_check(0,{settings['run_args']['decomp_base']-1})"
        records.append({
            "node_idx": idx, "description": desc, "category": cat,
            "rows": delta, "uses_lookup_table": uses_lookup_table,
        })

    return records, settings, tables, log_path


def write_op_ledger_csv(records, out_path: Path):
    categories = ["matmul", "qk", "av", "softmax", "gelu", "layernorm", "rescale", "other"]
    total_rows = sum(r["rows"] for r in records)
    agg = {c: {"rows": 0, "lookup_op_instances": 0} for c in categories}
    for r in records:
        agg[r["category"]]["rows"] += r["rows"]
        if r["uses_lookup_table"] is not None:
            agg[r["category"]]["lookup_op_instances"] += 1

    total_lookup_instances = sum(a["lookup_op_instances"] for a in agg.values())
    rows_out = []
    for c in categories:
        rows_out.append({
            "category": c,
            "total_rows": agg[c]["rows"],
            "pct_rows": 100.0 * agg[c]["rows"] / total_rows if total_rows else 0.0,
            "lookup_op_instances": agg[c]["lookup_op_instances"],
            "pct_lookup_op_instances": (
                100.0 * agg[c]["lookup_op_instances"] / total_lookup_instances
                if total_lookup_instances else 0.0
            ),
        })

    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "category", "total_rows", "pct_rows",
            "lookup_op_instances", "pct_lookup_op_instances",
        ])
        writer.writeheader()
        writer.writerows(rows_out)
    return rows_out


def main():
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(1)
    config_dir = Path(sys.argv[1]).resolve()
    records, settings, tables, log_path = parse_config(config_dir)

    out_csv = config_dir / "op_ledger.csv"
    rows_out = write_op_ledger_csv(records, out_csv)

    print(f"parsed {len(records)} op nodes from {log_path}", file=sys.stderr)
    print(f"value tables (from log): {tables['value_tables']}", file=sys.stderr)
    print(f"range-check tables (from log): {sorted(tables['range_check_tables'])}", file=sys.stderr)
    for row in rows_out:
        print(
            f"  {row['category']:10s} rows={row['total_rows']:>8d} ({row['pct_rows']:5.1f}%)  "
            f"lookup_op_instances={row['lookup_op_instances']:>3d} ({row['pct_lookup_op_instances']:5.1f}%)",
            file=sys.stderr,
        )
    print(f"wrote {out_csv}", file=sys.stderr)

    # also dump the raw per-node records for auditability / debugging
    raw_path = config_dir / "op_ledger_raw_nodes.csv"
    with open(raw_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["node_idx", "description", "category", "rows", "uses_lookup_table"])
        writer.writeheader()
        writer.writerows(records)
    print(f"wrote {raw_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
