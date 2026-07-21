#!/usr/bin/env python3
"""
plot_figures.py -- the four report figures for results/REPORT.md, written to
results/figures/. Reads results/summary.csv (+ models/MANIFEST.csv for
seq_len lookups). No CLI args; run from repo root:

    python3 scripts/plot_figures.py

Figures (see results/REPORT.md for how each answers a specific question):
  fig_a_prove_vs_seqlen.png   -- prove wall-time vs seq_len, log-log + fitted
                                 slope (attention is O(seq_len^2); slope near
                                 2 would confirm it empirically)
  fig_b_memory_vs_logrows.png -- peak RSS (setup stage) vs logrows k, log2(y)
                                 vs k with a fitted slope (slope near 1 would
                                 confirm "+1 k doubles memory")
  fig_c_stage_stacked.png     -- median wall-seconds per stage, stacked bar,
                                 one bar per config
  fig_d_scale_sweep.png       -- prove wall-time and total rows vs quantization
                                 scale (2 side-by-side panels, not dual-axis --
                                 the dataviz "one axis" rule forbids that)
"""
import csv
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_DIR = Path(__file__).resolve().parent.parent
SUMMARY_CSV = REPO_DIR / "results" / "summary.csv"
MANIFEST_CSV = REPO_DIR / "models" / "MANIFEST.csv"
FIG_DIR = REPO_DIR / "results" / "figures"

# palette.md categorical slots 1/2/3 (validated: light-mode, worst adjacent
# CVD dE 9.2, normal-vision dE 27.6 -- see results/README.md's dataviz note)
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK_PRIMARY, INK_SECONDARY, INK_MUTED = "#0b0b0b", "#52514e", "#898781"
GRIDLINE, BASELINE, SURFACE = "#e1e0d9", "#c3c2b7", "#fcfcfb"

STAGE_KEYS = ["gen_settings", "calibrate_settings", "compile_circuit", "gen_srs",
              "setup", "gen_witness", "prove", "verify"]
STAGE_COLORS = ["#cde2fb", "#9ec5f4", "#6da7ec", "#2a78d6",
                "#eb6834", "#f0a988", "#1baf7a", "#a8ded0"]


def load_summary():
    with open(SUMMARY_CSV) as f:
        return list(csv.DictReader(f))


def load_seq_len_map():
    with open(MANIFEST_CSV) as f:
        return {r["name"]: int(r["seq_len"]) for r in csv.DictReader(f)}


def style_axes(ax):
    ax.set_facecolor(SURFACE)
    ax.yaxis.grid(True, color=GRIDLINE, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    for spine in ["left", "bottom"]:
        ax.spines[spine].set_color(BASELINE)
    ax.tick_params(colors=INK_MUTED, labelsize=9)


def fig_a_prove_vs_seqlen(rows, seq_len_of):
    seq_scan = ["seq4_d16_l1", "seq8_d16_l1", "seq16_d16_l1", "seq32_d16_l1"]
    xs, ys = [], []
    for r in rows:
        if r["config"] in seq_scan and r["prove_wall_s"]:
            xs.append(seq_len_of[r["config"]])
            ys.append(float(r["prove_wall_s"]))
    order = np.argsort(xs)
    xs, ys = np.array(xs)[order], np.array(ys)[order]

    log_x, log_y = np.log2(xs), np.log2(ys)
    slope, intercept = np.polyfit(log_x, log_y, 1)

    fig, ax = plt.subplots(figsize=(7, 5.5), dpi=160)
    fig.patch.set_facecolor(SURFACE)
    style_axes(ax)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log", base=2)

    ax.scatter(xs, ys, color=BLUE, s=70, zorder=4, label="measured (median of 3 reps)")
    fit_x = np.array([xs.min(), xs.max()])
    fit_y = 2 ** (intercept + slope * np.log2(fit_x))
    ax.plot(fit_x, fit_y, color=ORANGE, linewidth=2, zorder=3,
            label=f"fit: prove_time ~ seq_len^{slope:.2f}")
    ref_y = ys[0] * (fit_x / xs[0]) ** 2
    ax.plot(fit_x, ref_y, color=INK_MUTED, linewidth=1.5, linestyle="--", zorder=2,
            label="reference: seq_len^2 (quadratic)")

    for x, y in zip(xs, ys):
        ax.annotate(f"{y:.1f}s", (x, y), textcoords="offset points", xytext=(6, 6),
                    fontsize=8, color=INK_SECONDARY)

    ax.set_xticks(xs)
    ax.set_xticklabels([str(int(x)) for x in xs])
    ax.set_xlabel("seq_len (log2 scale)", fontsize=10, color=INK_SECONDARY)
    ax.set_ylabel("prove wall time, seconds (log2 scale)", fontsize=10, color=INK_SECONDARY)
    ax.set_title("prove time vs seq_len (d_model=16, 1 layer)", fontsize=13,
                 color=INK_PRIMARY, loc="left", pad=12)
    ax.legend(frameon=False, fontsize=9, labelcolor=INK_SECONDARY, loc="upper left")
    fig.tight_layout()
    out = FIG_DIR / "fig_a_prove_vs_seqlen.png"
    fig.savefig(out, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out} (fitted slope={slope:.3f}, i.e. prove_time ~ seq_len^{slope:.2f})")
    return slope


def fig_b_memory_vs_logrows(rows):
    xs, ys, labels = [], [], []
    for r in rows:
        if r["logrows"] and r["setup_max_rss_mb"]:
            xs.append(int(r["logrows"]))
            ys.append(float(r["setup_max_rss_mb"]))
            labels.append(r["config"])
    xs, ys, labels = np.array(xs), np.array(ys), np.array(labels)
    order = np.argsort(xs)
    xs, ys, labels = xs[order], ys[order], labels[order]

    log2_y = np.log2(ys)
    slope, intercept = np.polyfit(xs, log2_y, 1)

    fig, ax = plt.subplots(figsize=(7.5, 5.5), dpi=160)
    fig.patch.set_facecolor(SURFACE)
    style_axes(ax)
    ax.set_yscale("log", base=2)

    ax.scatter(xs, ys, color=BLUE, s=60, zorder=4, label="measured (setup peak RSS)")
    fit_x = np.array([xs.min(), xs.max()])
    fit_y = 2 ** (intercept + slope * fit_x)
    ax.plot(fit_x, fit_y, color=ORANGE, linewidth=2, zorder=3,
            label=f"fit: log2(RSS) slope={slope:.2f} per +1 k")
    ref_y = ys[0] * 2.0 ** (fit_x - xs[0])
    ax.plot(fit_x, ref_y, color=INK_MUTED, linewidth=1.5, linestyle="--", zorder=2,
            label="reference: exact doubling per +1 k")

    ax.set_xlabel("logrows (k)", fontsize=10, color=INK_SECONDARY)
    ax.set_ylabel("setup stage peak RSS, MB (log2 scale)", fontsize=10, color=INK_SECONDARY)
    ax.set_title("peak memory vs logrows (all benchmarked configs)", fontsize=13,
                 color=INK_PRIMARY, loc="left", pad=12)
    ax.legend(frameon=False, fontsize=9, labelcolor=INK_SECONDARY, loc="upper left")
    fig.tight_layout()
    out = FIG_DIR / "fig_b_memory_vs_logrows.png"
    fig.savefig(out, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out} (fitted slope={slope:.3f} log2-MB per +1 k; 1.0 = exact doubling)")
    return slope


def fig_c_stage_stacked(rows):
    # main sweep + reference configs, in a fixed, readable order (excludes
    # the scale-sweep variants and the OOM config, which has no full stage set)
    order = ["seq4_d16_l1", "seq8_d16_l1", "seq16_d16_l1", "seq32_d16_l1",
             "seq8_d32_l1", "seq8_d64_l1", "seq8_d16_l2", "seq8_d16_l4", "seq32_d64_l1"]
    by_config = {r["config"]: r for r in rows}

    fig, ax = plt.subplots(figsize=(11, 6.5), dpi=160)
    fig.patch.set_facecolor(SURFACE)
    style_axes(ax)

    bottoms = np.zeros(len(order))
    for stage, color in zip(STAGE_KEYS, STAGE_COLORS):
        heights = []
        for cfg in order:
            r = by_config.get(cfg, {})
            v = r.get(f"{stage}_wall_s", "")
            heights.append(float(v) if v else 0.0)
        heights = np.array(heights)
        ax.bar(order, heights, bottom=bottoms, color=color, edgecolor=SURFACE,
               linewidth=0.6, label=stage, zorder=3)
        bottoms += heights

    for i, cfg in enumerate(order):
        ax.text(i, bottoms[i] + max(bottoms) * 0.01, f"{bottoms[i]:.0f}s",
                ha="center", va="bottom", fontsize=8, color=INK_SECONDARY)

    ax.set_ylabel("wall time, seconds (median of 3 reps for setup/prove)", fontsize=10, color=INK_SECONDARY)
    ax.set_title("per-stage wall time, stacked by config", fontsize=13,
                 color=INK_PRIMARY, loc="left", pad=12)
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right", fontsize=9, color=INK_PRIMARY)
    ax.legend(frameon=False, fontsize=8, labelcolor=INK_SECONDARY, ncol=4,
              loc="upper center", bbox_to_anchor=(0.5, -0.22))
    fig.tight_layout()
    out = FIG_DIR / "fig_c_stage_stacked.png"
    fig.savefig(out, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


def fig_d_scale_sweep(rows):
    scales = [4, 7, 10, 13]  # 13 = the unscaled seq8_d16_l1 run's actual param_scale
    prove_t, num_rows = [], []
    by_config = {r["config"]: r for r in rows}
    for s in scales:
        cfg = f"seq8_d16_l1_scale{s}" if s != 13 else "seq8_d16_l1"
        r = by_config.get(cfg)
        prove_t.append(float(r["prove_wall_s"]) if r and r["prove_wall_s"] else np.nan)
        num_rows.append(int(r["num_rows"]) if r and r["num_rows"] else np.nan)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5), dpi=160)
    for ax in (ax1, ax2):
        fig.patch.set_facecolor(SURFACE)
        style_axes(ax)

    ax1.plot(scales, prove_t, marker="o", color=BLUE, linewidth=2, markersize=8, zorder=3)
    for x, y in zip(scales, prove_t):
        ax1.annotate(f"{y:.1f}s", (x, y), textcoords="offset points", xytext=(0, 8),
                     fontsize=8, color=INK_SECONDARY, ha="center")
    ax1.set_xlabel("quantization scale (input_scale = param_scale)", fontsize=9, color=INK_SECONDARY)
    ax1.set_ylabel("prove wall time, seconds", fontsize=9, color=INK_SECONDARY)
    ax1.set_title("prove time vs scale", fontsize=12, color=INK_PRIMARY, loc="left")
    ax1.set_xticks(scales)

    ax2.plot(scales, num_rows, marker="o", color=AQUA, linewidth=2, markersize=8, zorder=3)
    for x, y in zip(scales, num_rows):
        ax2.annotate(f"{y:,}", (x, y), textcoords="offset points", xytext=(0, 8),
                     fontsize=8, color=INK_SECONDARY, ha="center")
    ax2.set_xlabel("quantization scale (input_scale = param_scale)", fontsize=9, color=INK_SECONDARY)
    ax2.set_ylabel("total circuit rows (num_rows)", fontsize=9, color=INK_SECONDARY)
    ax2.set_title("total rows vs scale", fontsize=12, color=INK_PRIMARY, loc="left")
    ax2.set_xticks(scales)

    fig.suptitle("seq8_d16_l1 quantization-scale sweep (calibrate-settings --scales pinned; logrows/lookup-bits still calibrated)",
                 fontsize=10, color=INK_SECONDARY, y=1.02)
    fig.tight_layout()
    out = FIG_DIR / "fig_d_scale_sweep.png"
    fig.savefig(out, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    rows = load_summary()
    seq_len_of = load_seq_len_map()
    fig_a_prove_vs_seqlen(rows, seq_len_of)
    fig_b_memory_vs_logrows(rows)
    fig_c_stage_stacked(rows)
    fig_d_scale_sweep(rows)


if __name__ == "__main__":
    main()
