#!/usr/bin/env python3

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
EZKL = REPO / "target" / "release" / "ezkl"

ROW = re.compile(r"│\s*(\d+)\s*│\s*([^│]+?)\s*│\s*(\S*)\s*│")
DIV = re.compile(r"REBASED \(div=([0-9.]+)")


def table(model_onnx, scale, fusion):
    env = dict(os.environ, EZKL_DIV_ROUND_NEAREST="1", EZKL_RESCALE_FUSION=str(fusion))
    r = subprocess.run(
        [
            str(EZKL),
            "table",
            "-M",
            str(model_onnx),
            "--input-scale",
            str(scale),
            "--param-scale",
            str(scale),
        ],
        env=env,
        capture_output=True,
        text=True,
    )
    if r.returncode != 0:
        sys.exit(
            f"ezkl table failed (fusion={fusion}):\n{r.stdout[-2000:]}{r.stderr[-2000:]}"
        )
    nodes = {}
    for m in ROW.finditer(r.stdout):
        nodes[int(m.group(1))] = m.group(2).strip()
    return nodes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--scale", type=int, default=13)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    mdir = (
        REPO / args.model_dir
        if not Path(args.model_dir).is_absolute()
        else Path(args.model_dir)
    )
    a = table(mdir / "model.onnx", args.scale, 0)
    b = table(mdir / "model.onnx", args.scale, 1)
    assert set(a) == set(b), "融合不应增删节点，只应改写 RebaseScale 包裹"

    changed, dropped, widened = [], [], []
    for i in sorted(a):
        if a[i] == b[i]:
            continue
        d0, d1 = DIV.search(a[i]), DIV.search(b[i])
        rec = {
            "node": i,
            "before": a[i],
            "after": b[i],
            "div_before": float(d0.group(1)) if d0 else None,
            "div_after": float(d1.group(1)) if d1 else None,
        }
        changed.append(rec)
        (dropped if d1 is None else widened).append(rec)

    last = max(a)
    out = {
        "model_dir": str(mdir.relative_to(REPO)),
        "scale": args.scale,
        "n_nodes_listed": len(a),
        "max_node_index": last,
        "n_changed": len(changed),
        "n_rescale_dropped": len(dropped),
        "n_divisor_widened": len(widened),
        "widest_fused_divisor_log2": max(
            [int(round(__import__("math").log2(r["div_after"]))) for r in widened],
            default=None,
        ),
        "output_node": {
            "index": last,
            "op_unfused": a[last],
            "op_fused": b[last],
            "changed_by_fusion": a[last] != b[last],
        },
        "changed": changed,
    }
    p = Path(args.out)
    if not p.is_absolute():
        p = REPO / p
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(
        f"节点表列出 {out['n_nodes_listed']} 个（最大编号 {last}），融合改写 {out['n_changed']} 个"
        f"（{out['n_rescale_dropped']} 个不再 rescale、{out['n_divisor_widened']} 个除数变宽，"
        f"最宽 2^{out['widest_fused_divisor_log2']}）"
    )
    print(
        f"输出节点 {last}：{'被融合改写' if out['output_node']['changed_by_fusion'] else '**未被融合改写**'}"
    )
    print(f"  {a[last]}")
    print(f"写入 {p}")


if __name__ == "__main__":
    main()
