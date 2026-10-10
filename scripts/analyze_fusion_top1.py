#!/usr/bin/env python3

import csv
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent


def _sound(summary):
    v = summary.get("bound_sound_ulp")
    if v is None:
        raise SystemExit(
            "summary.json 缺少 bound_sound_ulp，无法计算有效的扰动界统计。\n"
            "使用 scripts/recount_fusion_bound.py 从权重重新计算界，或用\n"
            "scripts/run_fusion_top1.py 生成包含该字段的数据。\n"
            "bound_pair 仅覆盖原 top-1 和 top-2，不能替代全词表的扰动界。"
        )
    return v


def load(scale_dir):
    s = json.loads((scale_dir / "summary.json").read_text())
    with (scale_dir / "tokens.csv").open() as fh:
        rows = list(csv.DictReader(fh))
    cols = {k: np.array([float(r[k]) for r in rows]) for k in rows[0]}
    return s, cols


def main():
    roots = [Path(a) if Path(a).is_absolute() else REPO / a for a in sys.argv[1:]]
    if not roots:
        sys.exit("Usage: python3 scripts/analyze_fusion_top1.py [arguments]")
    entries = []
    for root in roots:
        dirs = [d for d in root.glob("scale*") if d.is_dir() and d.name[5:].isdigit()]
        for d in sorted(dirs, key=lambda p: int(p.name[5:])):
            if (d / "summary.json").exists() and d.joinpath("tokens.csv").exists():
                entries.append((root.name, *load(d)))

    print("\n### 表 A　top-1 一致率、困惑度与逐点误差\n")
    print(
        "| 模型 | scale | token 数 | top-1 一致（融合 vs 未融合） | 翻转数 | PPL 浮点 | PPL 未融合 | PPL 融合 | max\\|Δlogit\\| | mean\\|Δlogit\\| |"
    )
    print("|---|---|---|---|---|---|---|---|---|---|")
    for name, s, _ in entries:
        print(
            f"| {name} | {s['scale']} | {s['n_tokens']:,} | {s['top1_agree_fused_vs_unfused']*100:.2f}% | "
            f"{s['n_flip_fused_vs_unfused']} | {s['ppl_float']:.3f} | {s['ppl_unfused']:.3f} | "
            f"{s['ppl_fused']:.3f} | {s['max_abs_diff_fused_unfused']:.4f} | "
            f"{s['mean_abs_diff_fused_unfused']:.5f} |"
        )

    print("\n### 表 B　margin 分布与引理 3 单跳界（单位：输出 ulp）\n")
    print(
        "| 模型 | scale | margin p1 | p5 | 中位 | p95 | 单跳界（中位） | 单跳界（最大） | 判据覆盖率 | 实测翻转率 |"
    )
    print("|---|---|---|---|---|---|---|---|---|---|")
    for name, s, c in entries:
        m = c["margin_unfused_ulp"]
        print(
            f"| {name} | {s['scale']} | {np.percentile(m,1):.0f} | {np.percentile(m,5):.0f} | "
            f"{np.median(m):.0f} | {np.percentile(m,95):.0f} | {s['bound_pair_ulp_median']:.1f} | "
            f"{_sound(s):.1f} | {s['certified_frac_1hop']*100:.2f}% | "
            f"{(1-s['top1_agree_fused_vs_unfused'])*100:.2f}% |"
        )

    print("\n### 表 C　单跳判据 vs 实测：被判据「担保不翻」却翻了的 token\n")
    print(
        "| 模型 | scale | 判据担保的 token | 其中实际翻转 | 实测 max\\|Δlogit\\|(ulp) | 单跳界最大值(ulp) | 低估倍数 |"
    )
    print("|---|---|---|---|---|---|---|")
    for name, s, c in entries:
        cert = c["margin_unfused_ulp"] > _sound(s)
        flip = c["flip_fused_vs_unfused"] > 0
        d_ulp = c["max_abs_diff_fused_unfused"].max() * (2.0 ** s["scale"])
        print(
            f"| {name} | {s['scale']} | {int(cert.sum())}/{len(cert)} | {int((cert & flip).sum())} | "
            f"{d_ulp:,.0f} | {_sound(s):.1f} | {d_ulp/_sound(s):,.0f}x |"
        )

    print("\n### 表 D　把界换成实测扰动：标定/检验各半\n")
    print(
        "| 模型 | scale | 标定 Δmax(ulp) | 2Δmax vs margin 中位 | 检验集担保 | 担保内翻转 | 检验集总翻转 |"
    )
    print("|---|---|---|---|---|---|---|")
    for name, s, c in entries:
        n = len(c["margin_unfused_ulp"])
        half = n // 2
        u = 2.0 ** s["scale"]
        dmax_cal = c["max_abs_diff_fused_unfused"][:half].max() * u
        m_te = c["margin_unfused_ulp"][half:]
        f_te = c["flip_fused_vs_unfused"][half:] > 0
        cert = m_te > 2 * dmax_cal
        print(
            f"| {name} | {s['scale']} | {dmax_cal:,.0f} | "
            f"{2*dmax_cal/np.median(c['margin_unfused_ulp']):.1f}x | "
            f"{int(cert.sum())}/{len(cert)} | {int((cert & f_te).sum())} | {int(f_te.sum())} |"
        )

    print("\n### 表 E　扰动到底有多大：全词表最坏 vs 只看 top-2 两行（单位 ulp）\n")
    print(
        "| 模型 | scale | Δ 全词表中位 | Δ 全词表最大 | Δtop2 中位 | Δtop2 最大 | top2 样本 | margin 中位 |"
    )
    print("|---|---|---|---|---|---|---|---|")
    for name, s, c in entries:
        u = 2.0 ** s["scale"]
        dall = c["max_abs_diff_fused_unfused"] * u
        d = (
            REPO
            / "results"
            / "fusion_top1"
            / name
            / f"scale{s['scale']}"
            / "logits_topk.npz"
        )
        if not d.exists():
            continue
        z = np.load(d)
        uf, fu = z["unfused"].astype(np.float64), z["fused"].astype(np.float64)
        dt = np.abs(fu[:, :2] - uf[:, :2]).max(axis=1) * u
        print(
            f"| {name} | {s['scale']} | {np.median(dall):,.0f} | {dall.max():,.0f} | "
            f"{np.median(dt):,.0f} | {dt.max():,.0f} | {uf.shape[0]} | "
            f"{np.median(c['margin_unfused_ulp']):,.0f} |"
        )

    print("\n### 补充：融合是否让电路更接近浮点\n")
    print(
        "| 模型 | scale | max\\|未融合−浮点\\| | max\\|融合−浮点\\| | 翻转 vs 浮点（未融合） | 翻转 vs 浮点（融合） | Δ困惑度（融合−未融合） |"
    )
    print("|---|---|---|---|---|---|---|")
    for name, s, _ in entries:
        print(
            f"| {name} | {s['scale']} | {s['max_abs_diff_unfused_float']:.4f} | "
            f"{s['max_abs_diff_fused_float']:.4f} | {s['n_flip_unfused_vs_float']} | "
            f"{s['n_flip_fused_vs_float']} | {s['ppl_fused']-s['ppl_unfused']:+.4f} |"
        )

    print("\n### 行数（同一份 ONNX，只切融合开关）\n")
    print("| 模型 | scale | 未融合行数 | 融合行数 | 削幅 | 融合报告 |")
    print("|---|---|---|---|---|---|")
    for name, s, _ in entries:
        cut = (s["num_rows_fused"] - s["num_rows_unfused"]) / s["num_rows_unfused"]
        print(
            f"| {name} | {s['scale']} | {s['num_rows_unfused']:,} | {s['num_rows_fused']:,} | "
            f"{cut*100:+.1f}% | {s['fusion_report']} |"
        )


if __name__ == "__main__":
    main()
