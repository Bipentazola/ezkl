#!/usr/bin/env python3

import csv
import sys
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

CATEGORY_ORDER = [
    "msm",
    "fft",
    "lookup",
    "permutation",
    "quotient",
    "synthesize",
    "other",
]
CATEGORY_LABELS = {
    "msm": "MSM /\ncommit",
    "fft": "FFT/NTT",
    "lookup": "lookup\nargument",
    "permutation": "permutation\nargument",
    "quotient": "quotient /\ngate eval",
    "synthesize": "witness /\nassign",
    "other": "other",
}
RUN_ORDER = ["seq4_1thread", "seq4_nthread", "base_nthread"]
RUN_LABELS = {
    "seq4_1thread": "seq4_d16_l1, RAYON_NUM_THREADS=1",
    "seq4_nthread": "seq4_d16_l1, default threads (64)",
    "base_nthread": "seq32_d64_l1 (base), default threads (64)",
}
RUN_COLORS = {
    "seq4_1thread": "#2a78d6",
    "seq4_nthread": "#eb6834",
    "base_nthread": "#1baf7a",
}

INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
BASELINE = "#c3c2b7"
SURFACE = "#fcfcfb"


def main():
    in_csv = sys.argv[1] if len(sys.argv) > 1 else "results/prove_breakdown.csv"
    out_png = sys.argv[2] if len(sys.argv) > 2 else "results/prove_breakdown.png"

    data = defaultdict(dict)
    with open(in_csv) as f:
        for row in csv.DictReader(f):
            data[row["run"]][row["category"]] = float(row["pct"])

    plt.rcParams["font.sans-serif"] = [
        "Noto Sans CJK SC",
        "Noto Sans CJK",
        "WenQuanYi Zen Hei",
        "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False

    fig, ax = plt.subplots(figsize=(11, 6), dpi=160)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    n_runs = len(RUN_ORDER)
    n_cats = len(CATEGORY_ORDER)
    bar_w = 0.24
    group_gap = 0.02
    x = range(n_cats)

    for i, run in enumerate(RUN_ORDER):
        offset = (i - (n_runs - 1) / 2) * (bar_w + group_gap)
        heights = [data[run].get(cat, 0.0) for cat in CATEGORY_ORDER]
        xs = [xi + offset for xi in x]
        bars = ax.bar(
            xs,
            heights,
            width=bar_w,
            color=RUN_COLORS[run],
            label=RUN_LABELS[run],
            edgecolor=SURFACE,
            linewidth=0.6,
            zorder=3,
        )
        for b, h in zip(bars, heights):
            if h >= 4:
                ax.text(
                    b.get_x() + b.get_width() / 2,
                    h + 0.8,
                    f"{h:.0f}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    color=INK_SECONDARY,
                )

    ax.set_xticks(list(x))
    ax.set_xticklabels(
        [CATEGORY_LABELS[c] for c in CATEGORY_ORDER], fontsize=10, color=INK_PRIMARY
    )
    ax.set_ylabel("sample share (%)", fontsize=10, color=INK_SECONDARY)
    ax.set_ylim(0, max(50, max(v for d in data.values() for v in d.values()) + 8))

    ax.yaxis.grid(True, color=GRIDLINE, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for spine in ["top", "right", "left"]:
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color(BASELINE)
    ax.tick_params(axis="y", colors=INK_MUTED, labelsize=9)
    ax.tick_params(axis="x", length=0)

    ax.set_title(
        "ezkl prove: 6-bucket hotspot breakdown (pprof sampling)",
        fontsize=13,
        color=INK_PRIMARY,
        loc="left",
        pad=14,
    )
    legend = ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.16),
        ncol=1,
        frameon=False,
        fontsize=9,
        labelcolor=INK_SECONDARY,
    )

    fig.tight_layout()
    fig.savefig(out_png, facecolor=SURFACE, bbox_inches="tight")
    print(f"wrote {out_png}")


if __name__ == "__main__":
    main()
