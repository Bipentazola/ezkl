#!/usr/bin/env python3

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

REPO_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_DIR / "results" / "p_measurement"

CATEGORY_STYLE = {
    "rescale": ("#2a78d6", "o"),
    "matmul": ("#eb6834", "s"),
    "layernorm": ("#1baf7a", "^"),
    "softmax": ("#eda100", "D"),
    "qk": ("#e87ba4", "v"),
    "av": ("#008300", "P"),
    "gelu": ("#4a3aa7", "X"),
    "other": ("#e34948", "*"),
}
CATEGORY_ORDER = list(CATEGORY_STYLE.keys())

INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
SURFACE = "#fcfcfb"


def style_axes(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(BASELINE)
    ax.tick_params(colors=INK_MUTED, labelsize=9)
    ax.yaxis.label.set_color(INK_SECONDARY)
    ax.xaxis.label.set_color(INK_SECONDARY)


def plot_trend(
    csv_path: Path, x_field: str, x_label: str, title: str, out_path: Path, log_x: bool
):
    rows = list(csv.DictReader(open(csv_path)))
    rows.sort(key=lambda r: float(r[x_field]))
    xs = [float(r[x_field]) for r in rows]

    fig, ax = plt.subplots(figsize=(8, 5.2), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    style_axes(ax)

    end_vals = {cat: float(rows[-1][f"{cat}_pct"]) for cat in CATEGORY_ORDER}
    sorted_ends = sorted(end_vals.items(), key=lambda kv: kv[1], reverse=True)
    label_cats = {
        cat
        for cat, v in sorted_ends
        if all(cat == c2 or abs(v - v2) > 4.0 for c2, v2 in sorted_ends)
    }

    for cat in CATEGORY_ORDER:
        color, marker = CATEGORY_STYLE[cat]
        ys = [float(r[f"{cat}_pct"]) for r in rows]
        ax.plot(
            xs,
            ys,
            color=color,
            marker=marker,
            markersize=5.5,
            linewidth=1.8,
            zorder=3,
            label=cat,
        )
        if cat in label_cats:
            ax.annotate(
                cat,
                (xs[-1], ys[-1]),
                xytext=(6, 0),
                textcoords="offset points",
                fontsize=8,
                color=color,
                va="center",
                fontweight="medium",
            )

    if log_x:
        ax.set_xscale("log", base=2)
        ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
        ax.set_xticks(xs)
        ax.get_xaxis().set_tick_params(which="minor", size=0)

    ax.set_xlabel(x_label)
    ax.set_ylabel("share of circuit rows (%)")
    ax.set_title(
        title, color=INK_PRIMARY, fontsize=12, fontweight="bold", loc="left", pad=12
    )
    ax.set_ylim(0, max(float(r["rescale_pct"]) for r in rows) * 1.12)
    right_pad = (max(xs) - min(xs)) * 0.18 if not log_x else max(xs) ** 0.12
    ax.set_xlim(
        min(xs) * (0.85 if log_x else 1) - (0 if log_x else right_pad * 0.05),
        max(xs) * (1.55 if log_x else 1) + (0 if log_x else right_pad),
    )

    legend = ax.legend(
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        frameon=False,
        fontsize=8.5,
        labelcolor=INK_SECONDARY,
        handlelength=1.5,
    )
    fig.tight_layout()
    fig.savefig(out_path, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


def main():
    plot_trend(
        RESULTS_DIR / "p_by_width.csv",
        "d_model",
        "d_model (log scale)",
        "Circuit row share by operator category vs. d_model (seq_len=16, n_layers=1)",
        RESULTS_DIR / "trend_vs_d_model.png",
        log_x=True,
    )
    plot_trend(
        RESULTS_DIR / "p_by_seqlen.csv",
        "seq_len",
        "seq_len",
        "Circuit row share by operator category vs. seq_len (d_model=128, n_layers=1)",
        RESULTS_DIR / "trend_vs_seq_len.png",
        log_x=False,
    )


if __name__ == "__main__":
    main()
