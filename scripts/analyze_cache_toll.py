#!/usr/bin/env python3

import argparse
import json
import re
import sys
from pathlib import Path

LAYING_OUT_RE = re.compile(r"ezkl::graph::model\] - laying out (\d+): (.*)$")
ROWS_RE = re.compile(r"ezkl::circuit::ops::region\] - \(rows=(\d+),")

GROUPS = [
    ("write", re.compile(r"^IFF|^GREATER|^MULTIBROADCASTTO")),
    ("attn_qk", re.compile(r"EINSUM bk,abnk->abmn")),
    ("attn_av", re.compile(r"EINSUM abmk,abkn->bn")),
    ("attn_soft", re.compile(r"SOFTMAX")),
    ("attn_mask", re.compile(r"RESCALED INPUT \(ADD\)|^SUB$|^MULT$|^IDENTITY")),
]

CACHE_INPUT_SLOPE_MIN = 1.0


def parse_nodes(d: Path):
    log = next(
        (p for p in (d / "gensettings.log", d / "setup.log") if p.exists()), None
    )
    if log is None:
        sys.exit(f"{d}: 找不到 gensettings.log / setup.log")
    num_rows = json.loads((d / "settings.json").read_text())["num_rows"]

    seq, pending = [], None
    for line in log.read_text(errors="replace").splitlines():
        m = LAYING_OUT_RE.search(line)
        if m:
            idx = int(m.group(1))
            if seq and idx <= seq[-1][0]:
                break
            pending = (idx, m.group(2).strip())
            continue
        if pending:
            r = ROWS_RE.search(line)
            if r:
                seq.append((pending[0], pending[1], int(r.group(1))))
                pending = None
    if not seq:
        sys.exit(f"{d}: 日志里没有 laying-out 记录（RUST_LOG=debug 了吗？）")

    out = {}
    for i, (idx, desc, rows) in enumerate(seq):
        nxt = seq[i + 1][2] if i + 1 < len(seq) else num_rows
        out[idx] = (desc, nxt - rows)
    return out, num_rows, seq[0][2]


def group_of(desc: str) -> str:
    for name, rx in GROUPS:
        if rx.search(desc):
            return name
    return "const"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+", type=Path)
    ap.add_argument("--d-model", type=int, default=128)
    ap.add_argument(
        "--out", type=Path, default=Path("results/cache_toll/decomposition.json")
    )
    a = ap.parse_args()

    runs = {}
    for d in sorted(a.dirs, key=lambda p: int(re.search(r"T(\d+)$", p.name).group(1))):
        t_max = int(re.search(r"T(\d+)$", d.name).group(1))
        nodes, num_rows, preamble = parse_nodes(d)
        total = sum(r for _, r in nodes.values()) + preamble
        if total != num_rows:
            sys.exit(
                f"{d}: 行数对不上（节点合计 {total:,} != settings {num_rows:,}）"
                f"——解析规则失效，先查日志再用这批数"
            )
        runs[t_max] = {"nodes": nodes, "num_rows": num_rows, "preamble": preamble}
        print(
            f"T_max={t_max:>4}  num_rows={num_rows:>9,}  节点 {len(nodes)} 个，行数对账通过"
        )

    ts = sorted(runs)
    if len(ts) < 2:
        sys.exit("至少要两个 T_max 才能求斜率")
    lo, hi = ts[0], ts[-1]
    d_elems = 2 * a.d_model * (hi - lo)

    per_group, per_node = {}, []
    keys = set(runs[lo]["nodes"]) | set(runs[hi]["nodes"])
    if set(runs[lo]["nodes"]) != set(runs[hi]["nodes"]):
        sys.exit("两档的节点集合不同，无法一一对应")
    for idx in sorted(keys):
        desc, r_lo = runs[lo]["nodes"][idx]
        _, r_hi = runs[hi]["nodes"][idx]
        slope = (r_hi - r_lo) / d_elems
        if desc == "Input":
            g = "cache_in" if slope >= CACHE_INPUT_SLOPE_MIN else "mask_in"
        else:
            g = group_of(desc)
        per_group[g] = per_group.get(g, 0.0) + slope
        if abs(r_hi - r_lo) > 0:
            per_node.append(
                {
                    "idx": idx,
                    "desc": desc,
                    "group": g,
                    "rows_lo": r_lo,
                    "rows_hi": r_hi,
                    "slope_rows_per_elem": round(slope, 4),
                }
            )

    pre_slope = (runs[hi]["preamble"] - runs[lo]["preamble"]) / d_elems
    if abs(pre_slope) > 1e-9:
        per_group["preamble"] = per_group.get("preamble", 0.0) + pre_slope

    total_slope = sum(per_group.values())
    measured = (runs[hi]["num_rows"] - runs[lo]["num_rows"]) / d_elems
    assert abs(total_slope - measured) < 1e-6, (total_slope, measured)

    write = per_group.get("write", 0.0)
    attn = sum(v for k, v in per_group.items() if k.startswith("attn_"))
    cache_in = per_group.get("cache_in", 0.0)

    print(f"\n=== 过路费分解（d={a.d_model}，T_max {lo} -> {hi}）===")
    print(
        f"总斜率 {total_slope:.4f} 行/元素"
        f"（docs/incremental/03 的代价模型报 8.148 / 8.266）\n"
    )
    print(f"{'组':<12}{'行/元素':>10}{'占比':>9}")
    for g, v in sorted(per_group.items(), key=lambda kv: -kv[1]):
        print(f"{g:<12}{v:>10.4f}{v/total_slope*100:>8.1f}%")
    print("-" * 31)
    print(
        f"装入 advice   {cache_in:>8.4f}{cache_in/total_slope*100:>8.1f}%  "
        f"注意力要读它，去不掉（且这**不是**承诺开销）"
    )
    print(f"注意力读缓冲  {attn:>8.4f}{attn/total_slope*100:>8.1f}%  计算本身，去不掉")
    print(
        f"缓存写入      {write:>8.4f}{write/total_slope*100:>8.1f}%  "
        f"<- 「只承诺 delta」唯一真能去掉的"
    )

    const = runs[lo]["num_rows"] - total_slope * 2 * a.d_model * lo
    print(f"\n=== 重算「只承诺 delta」的价码（常数项 {const:,.0f} 行）===")
    print(f"{'T_max':>6}{'现状':>12}{'去掉写入侧后':>14}{'省':>8}   logrows")
    revised = {}
    for t in (16, 32, 64, 128):
        elems = 2 * a.d_model * t
        now = const + total_slope * elems
        after = const + (total_slope - write) * elems
        k_now, k_aft = (max(1, (int(round(v)) - 1).bit_length()) for v in (now, after))
        revised[str(t)] = {
            "now": round(now),
            "after": round(after),
            "logrows_now": k_now,
            "logrows_after": k_aft,
        }
        print(
            f"{t:>6}{now:>12,.0f}{after:>14,.0f}{(now-after)/now*100:>7.1f}%"
            f"   {k_now} -> {k_aft}{'  (掉档)' if k_aft < k_now else '  (不掉档)'}"
        )

    result = {
        "d_model": a.d_model,
        "t_max_lo": lo,
        "t_max_hi": hi,
        "num_rows": {str(t): runs[t]["num_rows"] for t in ts},
        "total_slope_rows_per_elem": round(total_slope, 4),
        "by_group": {k: round(v, 4) for k, v in per_group.items()},
        "cache_assign_side": round(cache_in, 4),
        "write_side": round(write, 4),
        "attention_side": round(attn, 4),
        "const_rows": round(const),
        "delta_commit_projection": revised,
        "per_node": per_node,
    }
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(result, indent=2))
    print(f"\n写入 {a.out}")


if __name__ == "__main__":
    main()
