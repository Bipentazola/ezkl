#!/usr/bin/env python3

import csv
import re
import sys
from pathlib import Path

REPO_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_DIR / "scripts"))
import parse_ledger  # noqa: E402

ROW_RE = re.compile(r"^\s*\|\s*│(.*)│\s*$")


def parse_table_v2(table_path: Path):
    nodes = {}
    for line in open(table_path):
        m = ROW_RE.match(line)
        if not m:
            continue
        cells = [c.strip() for c in m.group(1).split("│")]
        if len(cells) != 5:
            continue
        idx_s, opkind, out_scale_s, inputs_s, out_dims_s = cells
        if idx_s in ("idx", "") or not idx_s.isdigit():
            continue
        idx = int(idx_s)
        out_scale = int(out_scale_s)
        inputs = [
            (int(a), int(b)) for a, b in re.findall(r"\((\d+),\s*(\d+)\)", inputs_s)
        ]
        out_dims = [int(x) for x in re.findall(r"\d+", out_dims_s)]
        n_elements = 1
        for d in out_dims:
            n_elements *= d
        nodes[idx] = dict(
            idx=idx,
            opkind=opkind,
            out_scale=out_scale,
            inputs=inputs,
            out_dims=out_dims,
            n_elements=n_elements,
        )
    return nodes


LOOKUP_OPNAMES = {
    "SOFTMAX",
    "TANH",
    "RSQRT",
    "SIGMOID",
    "EXP",
    "COS",
    "ACOS",
    "COSH",
    "ACOSH",
    "SIN",
    "ASIN",
    "SINH",
    "ASINH",
    "TAN",
    "ATAN",
    "ATANH",
    "ERF",
    "HARDSWISH",
    "LN",
    "POW",
    "ISODD",
    "POWERSOFTWO",
}
COMPARE_OPNAMES = {
    "GREATER",
    "LESS",
    "EQUALS",
    "GREATEREQUAL",
    "LESSEQUAL",
    "MAX",
    "MIN",
}
SAFE_OPNAMES = {"EINSUM", "ADD", "RESHAPE", "IDENTITY", "MULT", "DIV", "CONST", "INPUT"}


def core_opname(desc: str) -> str:
    d = desc.strip()
    while True:
        if d.startswith("REBASED"):
            split_pos = d.rfind(") (")
            if split_pos == -1:
                break
            d = d[split_pos + 2 :].strip()
            if d.startswith("(") and d.endswith(")"):
                d = d[1:-1]
            continue
        if d.startswith("RESCALED INPUT ("):
            d = (
                d[len("RESCALED INPUT (") : -1]
                if d.endswith(")")
                else d[len("RESCALED INPUT (") :]
            )
            continue
        break
    token = re.split(r"[\s(]", d, maxsplit=1)[0]
    return token.upper()


def classify_downstream(downstream_opkinds):
    if not downstream_opkinds:
        return (
            "C",
            "no downstream consumer found (graph output / dangling) -- cannot classify",
        )

    reasons = []
    for op in downstream_opkinds:
        name = core_opname(op)
        if name in LOOKUP_OPNAMES:
            return "B", f"downstream is a lookup-table op ({op})"
        if name in COMPARE_OPNAMES:
            return "B", f"downstream is a comparison op ({op})"
        if name == "SUB":
            reasons.append(
                f"downstream is SUB ({op}) -- requires_homogenous_input_scales (mechanism A), "
                "not literally lookup/compare/decompose but still a hard scale constraint"
            )
        elif name not in SAFE_OPNAMES:
            reasons.append(f"downstream op not in the confirmed A/B pattern set: {op}")

    if reasons:
        return "C", "; ".join(reasons)
    return (
        "A",
        "downstream op(s) have no hard scale requirement and are themselves either "
        "scale-agnostic (ADD/RESHAPE/IDENTITY) or produce their own separate rebase "
        "(EINSUM/MULT) -- this rescale's divisor is a candidate to defer/merge forward",
    )


def build_inventory(table_path: Path, config_dir: Path, config_name: str):
    nodes = parse_table_v2(table_path)
    records, settings, tables, log_path = parse_ledger.parse_config(config_dir)
    rows_by_idx = {r["node_idx"]: r["rows"] for r in records}
    cat_by_idx = {r["node_idx"]: r["category"] for r in records}

    consumers = {idx: [] for idx in nodes}
    for idx, n in nodes.items():
        for src, _slot in n["inputs"]:
            if src in consumers:
                consumers[src].append(idx)

    inventory = []
    for idx, n in sorted(nodes.items()):
        if not n["opkind"].startswith("REBASED"):
            continue
        m = re.search(r"div=([\d.]+)", n["opkind"])
        sigma = float(m.group(1)) if m else None
        split_pos = n["opkind"].rfind(") (")
        if split_pos != -1:
            inner_desc = n["opkind"][split_pos + 2 :].strip()
            if inner_desc.startswith("(") and inner_desc.endswith(")"):
                inner_desc = inner_desc[1:-1]
        else:
            inner_desc = ""
        is_pure = bool(re.search(r"^(MULT|DIV[^A-Z]*)$", inner_desc.strip()))

        upstream_ids = [src for (src, _slot) in n["inputs"]]
        upstream_ops = [nodes[u]["opkind"] for u in upstream_ids if u in nodes]
        downstream_ids = consumers.get(idx, [])
        downstream_ops = [nodes[d]["opkind"] for d in downstream_ids if d in nodes]

        out_scale = n["out_scale"]
        in_scale = out_scale + (
            round(__import__("math").log2(sigma)) if sigma and sigma > 0 else 0
        )

        cls, reason = classify_downstream(downstream_ops)

        inventory.append(
            dict(
                config=config_name,
                node_id=idx,
                upstream_op=(
                    "; ".join(upstream_ops) if upstream_ops else "(none / graph input)"
                ),
                downstream_op=(
                    "; ".join(downstream_ops)
                    if downstream_ops
                    else "(none / graph output)"
                ),
                input_scale=in_scale,
                output_scale=out_scale,
                sigma=sigma,
                rows=rows_by_idx.get(idx, ""),
                n_elements=n["n_elements"],
                is_pure_rescale=is_pure,
                wrapped_op=inner_desc,
                category_class=cls,
                classification_reason=reason,
            )
        )
    total_rows = sum(rows_by_idx.values())
    return inventory, total_rows, settings["num_rows"]


def main():
    if len(sys.argv) != 4:
        print("Usage: python3 scripts/build_rescale_inventory.py [arguments]")
        sys.exit(1)
    table_path, config_dir, out_csv = (
        Path(sys.argv[1]),
        Path(sys.argv[2]),
        Path(sys.argv[3]),
    )
    config_name = config_dir.name
    inventory, total_rows, num_rows = build_inventory(
        table_path, config_dir, config_name
    )

    fieldnames = [
        "config",
        "node_id",
        "upstream_op",
        "downstream_op",
        "input_scale",
        "output_scale",
        "sigma",
        "rows",
        "n_elements",
        "is_pure_rescale",
        "wrapped_op",
        "category_class",
        "classification_reason",
    ]
    write_header = not out_csv.exists()
    with open(out_csv, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            w.writeheader()
        w.writerows(inventory)

    print(
        f"{config_name}: {len(inventory)} rescale nodes, total circuit rows = {total_rows} "
        f"(settings.json num_rows = {num_rows})",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
