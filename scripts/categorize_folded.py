#!/usr/bin/env python3
"""
Categorize pprof `.folded` collapsed-stack files (produced by ezkl's
pprof-profiling instrumentation around `create_proof`) into six Halo2
create_proof phases:

  msm          - best_multiexp (MSM / polynomial commitments)
  fft          - best_fft (FFT/NTT, domain conversion)
  lookup       - mv_lookup::/plonk::lookup:: (lookup argument construction)
  permutation  - plonk::permutation::prover (Z polynomial construction)
  quotient     - evaluate_h / h_poly / plonk::vanishing (gate constraint eval + h(X))
  synthesize   - Circuit::synthesize / Layouter::assign_*/tract:: (witness/circuit assignment)
  other        - anything not matched by the above

Each unique stack ("frame1;frame2;...;frameN count") is assigned to exactly
one category by scanning frames from the leaf (end of the line, where the
CPU actually was at sample time) back towards the root, taking the first
frame that matches any category pattern, checked in priority order
(msm > fft > lookup > permutation > quotient > synthesize). This correctly
attributes e.g. an FFT called from inside lookup construction to "fft", not
"lookup" -- self-time by nearest identifiable phase.

Usage:
    python3 scripts/categorize_folded.py OUT.csv config=name rep=1 file1.folded [config=name rep=2 file2.folded ...]

or simpler, just:
    python3 scripts/categorize_folded.py OUT.csv label1=path1.folded label2=path2.folded ...

Each input is "label=path" where label is a free-form string (e.g.
"baseline_rep1") recorded verbatim in the output CSV's `run` column; pass
--config-of to control how `run` maps to `config` (see below), or just
post-process the CSV yourself. Kept deliberately simple: one row per
(run, category) with percentage of total samples in that run.
"""
import re
import sys
import csv
from collections import Counter
from pathlib import Path

PATTERNS = [
    ("msm", re.compile(r"best_multiexp|halo2curves::msm::")),
    ("fft", re.compile(r"best_fft|halo2_proofs::fft::")),
    ("lookup", re.compile(r"mv_lookup::|plonk::lookup::")),
    ("permutation", re.compile(r"plonk::permutation::")),
    ("quotient", re.compile(r"evaluate_h|h_poly|plonk::vanishing::")),
    ("synthesize", re.compile(r"synthesize|::assign|tract::")),
]
CATEGORIES = [p[0] for p in PATTERNS] + ["other"]


def categorize_stack(frames):
    """frames: list of frame name strings, root-first (leaf is frames[-1])."""
    for frame in reversed(frames):
        for cat, pat in PATTERNS:
            if pat.search(frame):
                return cat
    return "other"


def parse_folded(path):
    """Yields (category, count, leaf) for every stack line in a .folded file."""
    with open(path, "r", errors="replace") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            # format: "frame1;frame2;...;frameN count"
            try:
                stack_part, count_str = line.rsplit(" ", 1)
                count = int(count_str)
            except ValueError:
                continue
            frames = stack_part.split(";")
            cat = categorize_stack(frames)
            yield cat, count, frames[-1]


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)

    out_csv = sys.argv[1]
    entries = []  # (run_label, path)
    for arg in sys.argv[2:]:
        if "=" not in arg:
            print(f"skipping malformed arg (expected label=path.folded): {arg}", file=sys.stderr)
            continue
        label, path = arg.split("=", 1)
        entries.append((label, path))

    rows = []
    other_top_lines = []
    for label, path in entries:
        if not Path(path).exists():
            print(f"WARNING: missing folded file for {label}: {path}", file=sys.stderr)
            continue
        totals = {c: 0 for c in CATEGORIES}
        other_leaf_counts = Counter()
        grand_total = 0
        for cat, count, leaf in parse_folded(path):
            totals[cat] += count
            grand_total += count
            if cat == "other":
                other_leaf_counts[leaf] += count

        if grand_total == 0:
            print(f"WARNING: zero samples in {path} (run={label})", file=sys.stderr)
            continue

        for cat in CATEGORIES:
            rows.append({
                "run": label,
                "category": cat,
                "samples": totals[cat],
                "total_samples": grand_total,
                "pct": 100.0 * totals[cat] / grand_total,
            })
        print(f"{label}: total_samples={grand_total} " +
              " ".join(f"{c}={100.0*totals[c]/grand_total:.1f}%" for c in CATEGORIES),
              file=sys.stderr)

        other_pct = 100.0 * totals["other"] / grand_total
        if other_pct > 15.0:
            print(f"  {label}: other={other_pct:.1f}% > 15%, top symbols in 'other':", file=sys.stderr)
            for leaf, c in other_leaf_counts.most_common(15):
                pct = 100.0 * c / grand_total
                print(f"    {pct:5.2f}%  {leaf}", file=sys.stderr)
                other_top_lines.append({"run": label, "pct_of_total": pct, "symbol": leaf})

    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["run", "category", "samples", "total_samples", "pct"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {out_csv}", file=sys.stderr)

    if other_top_lines:
        other_csv = str(Path(out_csv).with_name(Path(out_csv).stem + "_other_top_symbols.csv"))
        with open(other_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["run", "pct_of_total", "symbol"])
            writer.writeheader()
            writer.writerows(other_top_lines)
        print(f"wrote {other_csv}", file=sys.stderr)


if __name__ == "__main__":
    main()
