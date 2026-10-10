#!/usr/bin/env python3

import csv
import json
import re
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
RUNS_DIR = REPO / "scaling_runs"
RESULTS_DIR = RUNS_DIR / "results"
RAW_CSV = RESULTS_DIR / "all_runs_raw.csv"

CATEGORIES = ["msm", "fft", "lookup", "permutation", "quotient", "synthesize", "other"]

CONFIGS = {
    "baseline": {
        "dir": "baseline",
        "label": "baseline (all params at minimum)",
        "params": {
            "num_layers": 1,
            "hidden_size": 8,
            "num_heads": 1,
            "intermediate_size": 8,
            "vocab_size": 8,
            "seq_len": 4,
        },
    },
    "large_n": {
        "dir": "large_n",
        "label": "large-n (seq_len 4→64, arch@min)",
        "params": {
            "num_layers": 1,
            "hidden_size": 8,
            "num_heads": 1,
            "intermediate_size": 8,
            "vocab_size": 8,
            "seq_len": 64,
        },
    },
    "large_d": {
        "dir": "large_d",
        "label": "large-d (hidden 8→128, arch@min)",
        "params": {
            "num_layers": 1,
            "hidden_size": 128,
            "num_heads": 1,
            "intermediate_size": 8,
            "vocab_size": 8,
            "seq_len": 4,
        },
    },
    "large_scale": {
        "dir": "large_scale",
        "label": "large-scale (param_scale 12→13, arch@min)",
        "params": {
            "num_layers": 1,
            "hidden_size": 8,
            "num_heads": 1,
            "intermediate_size": 8,
            "vocab_size": 8,
            "seq_len": 4,
        },
    },
}


def mean(xs):
    return sum(xs) / len(xs) if xs else 0.0


def stdev(xs):
    if len(xs) < 2:
        return 0.0
    m = mean(xs)
    return (sum((x - m) ** 2 for x in xs) / (len(xs) - 1)) ** 0.5


def parse_prove_times(config_dir: Path):
    times = []
    report = config_dir / "profile_report.md"
    if report.exists():
        text = report.read_text()
        m = re.search(r"\|\s*09_prove\s*\|\s*([\d.]+)s", text)
        if m:
            times.append(float(m.group(1)))
    return times


def main():
    runs = defaultdict(dict)
    with open(RAW_CSV) as f:
        for row in csv.DictReader(f):
            runs[row["run"]][row["category"]] = float(row["pct"])

    summary_rows = []
    summary_json = {"categories": CATEGORIES, "configs": []}

    for config_key, meta in CONFIGS.items():
        rep_runs = [r for r in runs if r.startswith(config_key + "_rep")]
        rep_runs.sort()
        if not rep_runs:
            print(f"WARNING: no runs found for config {config_key}")
            continue

        per_cat_values = {
            cat: [runs[r].get(cat, 0.0) for r in rep_runs] for cat in CATEGORIES
        }
        means = {cat: mean(vals) for cat, vals in per_cat_values.items()}
        stds = {cat: stdev(vals) for cat, vals in per_cat_values.items()}

        config_dir = RUNS_DIR / meta["dir"]
        settings_path = config_dir / "settings.json"
        run_args = {}
        if settings_path.exists():
            run_args = json.load(open(settings_path))["run_args"]

        prove_times = parse_prove_times(config_dir)

        row = {
            "config": config_key,
            "label": meta["label"],
            "n_reps": len(rep_runs),
            "logrows": run_args.get("logrows"),
            "input_scale": run_args.get("input_scale"),
            "param_scale": run_args.get("param_scale"),
            **meta["params"],
        }
        for cat in CATEGORIES:
            row[f"{cat}_pct_mean"] = round(means[cat], 2)
            row[f"{cat}_pct_std"] = round(stds[cat], 2)
        summary_rows.append(row)

        summary_json["configs"].append(
            {
                "config": config_key,
                "label": meta["label"],
                "params": meta["params"],
                "logrows": run_args.get("logrows"),
                "input_scale": run_args.get("input_scale"),
                "param_scale": run_args.get("param_scale"),
                "n_reps": len(rep_runs),
                "prove_time_s_rep1": prove_times[0] if prove_times else None,
                "categories_pct_mean": {
                    cat: round(means[cat], 3) for cat in CATEGORIES
                },
                "categories_pct_std": {cat: round(stds[cat], 3) for cat in CATEGORIES},
            }
        )

        print(
            f"{config_key}: logrows={row['logrows']} scale=({row['input_scale']},{row['param_scale']}) "
            + " ".join(f"{cat}={means[cat]:.1f}%±{stds[cat]:.1f}" for cat in CATEGORIES)
        )

    fieldnames = list(summary_rows[0].keys())
    with open(RESULTS_DIR / "summary.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)

    with open(RESULTS_DIR / "summary.json", "w") as f:
        json.dump(summary_json, f, indent=2)

    print(f"\nwrote {RESULTS_DIR / 'summary.csv'}")
    print(f"wrote {RESULTS_DIR / 'summary.json'}")

    write_paper_tables(summary_json)


CATEGORY_LABELS = {
    "msm": "MSM",
    "fft": "FFT/NTT",
    "lookup": "Lookup",
    "permutation": "Perm.\\ $Z$",
    "quotient": "Quotient",
    "synthesize": "Synth.",
    "other": "Other",
}
CATEGORY_LABELS_MD = {
    "msm": "MSM",
    "fft": "FFT/NTT",
    "lookup": "Lookup",
    "permutation": "Perm. Z",
    "quotient": "Quotient",
    "synthesize": "Synth.",
    "other": "Other",
}


def write_paper_tables(summary_json):
    configs = summary_json["configs"]
    cats = summary_json["categories"]

    md = []
    md.append(
        "# create_proof phase breakdown (mean % of sampled CPU time, n=3 reps, ± std)\n"
    )
    header = "| Config | " + " | ".join(CATEGORY_LABELS_MD[c] for c in cats) + " |"
    sep = "|---|" + "|".join(["---"] * len(cats)) + "|"
    md.append(header)
    md.append(sep)
    for cfg in configs:
        cells = [
            f"{cfg['categories_pct_mean'][c]:.1f} ± {cfg['categories_pct_std'][c]:.1f}"
            for c in cats
        ]
        md.append(f"| {cfg['label']} | " + " | ".join(cells) + " |")
    md.append("")
    md.append("## Config / calibration metadata\n")
    md.append(
        "| Config | Varied param | logrows | input_scale | param_scale | prove time (s, rep 1) |"
    )
    md.append("|---|---|---|---|---|---|")
    for cfg in configs:
        md.append(
            f"| {cfg['config']} | {', '.join(f'{k}={v}' for k, v in cfg['params'].items())} "
            f"| {cfg['logrows']} | {cfg['input_scale']} | {cfg['param_scale']} "
            f"| {cfg['prove_time_s_rep1']} |"
        )
    (RESULTS_DIR / "paper_table.md").write_text("\n".join(md) + "\n")

    tex = []
    tex.append("% Requires \\usepackage{booktabs}")
    tex.append("\\begin{table}[t]")
    tex.append("  \\centering")
    tex.append(
        "  \\caption{Share of sampled CPU time in \\texttt{create\\_proof} by phase (mean \\% $\\pm$ std over 3 reps).}"
    )
    tex.append("  \\label{tab:create-proof-phase-breakdown}")
    col_spec = "l" + "c" * len(cats)
    tex.append(f"  \\begin{{tabular}}{{{col_spec}}}")
    tex.append("    \\toprule")
    tex.append("    Config & " + " & ".join(CATEGORY_LABELS[c] for c in cats) + " \\\\")
    tex.append("    \\midrule")
    for cfg in configs:
        cells = [
            f"{cfg['categories_pct_mean'][c]:.1f} $\\pm$ {cfg['categories_pct_std'][c]:.1f}"
            for c in cats
        ]
        name = cfg["config"].replace("_", "\\_")
        tex.append(f"    {name} & " + " & ".join(cells) + " \\\\")
    tex.append("    \\bottomrule")
    tex.append("  \\end{tabular}")
    tex.append("\\end{table}")
    tex.append("")
    tex.append("\\begin{table}[t]")
    tex.append("  \\centering")
    tex.append(
        "  \\caption{Configuration and calibrated settings for each scale point.}"
    )
    tex.append("  \\label{tab:create-proof-scale-configs}")
    tex.append("  \\begin{tabular}{lp{4.2cm}ccc}")
    tex.append("    \\toprule")
    tex.append(
        "    Config & Varied parameter & logrows & param\\_scale & prove (s, rep 1) \\\\"
    )
    tex.append("    \\midrule")
    for cfg in configs:
        name = cfg["config"].replace("_", "\\_")
        varied = ", ".join(f"{k}={v}" for k, v in cfg["params"].items()).replace(
            "_", "\\_"
        )
        tex.append(
            f"    {name} & {varied} & {cfg['logrows']} & {cfg['param_scale']} & {cfg['prove_time_s_rep1']} \\\\"
        )
    tex.append("    \\bottomrule")
    tex.append("  \\end{tabular}")
    tex.append("\\end{table}")
    (RESULTS_DIR / "paper_table.tex").write_text("\n".join(tex) + "\n")

    print(f"wrote {RESULTS_DIR / 'paper_table.md'}")
    print(f"wrote {RESULTS_DIR / 'paper_table.tex'}")


if __name__ == "__main__":
    main()
