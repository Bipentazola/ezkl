#!/usr/bin/env python3

import csv
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
CASES = [
    ("1M", "tinystories1m_seq32", [12, 13, 16]),
    ("8M", "tinystories8m_seq16", [11, 13, 15]),
]


def quantize(w: np.ndarray, scale: int) -> np.ndarray:
    x = w * (2.0**scale)
    return np.where(x >= 0, np.floor(x + 0.5), np.ceil(x - 0.5))


def lm_head_rows(model_dir: Path) -> np.ndarray:
    z = np.load(model_dir / "embed.npz")
    for k in z.keys():
        a = z[k]
        if a.ndim == 2 and a.shape[0] > 50_000:
            return a.astype(np.float64)
    raise SystemExit(
        f"{model_dir}: embed.npz 里找不到 V×d 的嵌入表，键={list(z.keys())}"
    )


def main() -> int:
    print("表 5-3（sound 判据：阈值取遍全部 V 行）\n")
    print(
        "| 模型 | scale | 单跳界 2Δ_max | 判据担保的 token | 其中实际翻转 | "
        "实测最大扰动 | 低估倍数 |"
    )
    print("|---|---|---|---|---|---|---|")
    for label, d, scales in CASES:
        wte = lm_head_rows(REPO / "models_pretrained" / d)
        for scale in scales:
            W = quantize(wte, scale)
            bound_rows = np.abs(W).sum(axis=1) / (2.0 * (2.0**scale)) + 1.0
            thr = 2.0 * float(bound_rows.max())

            csv_path = REPO / "results/fusion_top1" / d / f"scale{scale}" / "tokens.csv"
            rows = list(csv.DictReader(open(csv_path)))
            margin = np.array([float(r["margin_unfused_ulp"]) for r in rows])
            flip = np.array(
                [r["flip_fused_vs_unfused"] in ("True", "true", "1") for r in rows]
            )
            pert_ulp = np.array(
                [float(r["max_abs_diff_fused_unfused"]) for r in rows]
            ) * (2.0**scale)

            covered = margin > thr
            print(
                f"| {label} | {scale} | {thr:.1f} | {covered.sum():,}/{len(rows):,} | "
                f"{int(flip[covered].sum())} | {pert_ulp.max():,.0f} | "
                f"{pert_ulp.max()/thr:,.0f}x |"
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
