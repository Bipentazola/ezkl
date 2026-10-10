#!/usr/bin/env python3

import csv
import json
import math
from pathlib import Path

REPO_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_DIR / "results" / "p_measurement"
OUT_DIR = REPO_DIR / "results" / "seq_crossover"
CATEGORIES = ["matmul", "qk", "av", "softmax", "gelu", "layernorm", "rescale", "other"]
FIT_MAX_SEQ = 256


def load_seq_rows():
    path = RESULTS_DIR / "p_by_seqlen.csv"
    rows = []
    for r in csv.DictReader(open(path)):
        rec = {
            "seq_len": int(r["seq_len"]),
            "num_rows": int(r["num_rows"]),
            "d_model": int(r["d_model"]),
            "n_head": int(r["n_head"]),
        }
        for c in CATEGORIES:
            rec[c] = int(r[f"{c}_rows"])
        rows.append(rec)
    return sorted(rows, key=lambda r: r["seq_len"])


def fit_coefficients(rows):
    fit_rows = [r for r in rows if r["seq_len"] <= FIT_MAX_SEQ]
    anchor = fit_rows[-1]
    pts = fit_rows[:3]
    coef = {}
    for c in CATEGORIES:
        (s1, s2, s3) = (p["seq_len"] for p in pts)
        (r1, r2, r3) = (p[c] for p in pts)
        det = (
            s1 * s1 * (s2 - s3)
            - s1 * (s2 * s2 - s3 * s3)
            + (s2 * s2 * s3 - s3 * s3 * s2)
        )
        a = (r1 * (s2 - s3) - s1 * (r2 - r3) + (r2 * s3 - r3 * s2)) / det
        b = (
            s1 * s1 * (r2 - r3)
            - r1 * (s2 * s2 - s3 * s3)
            + (s2 * s2 * r3 - s3 * s3 * r2)
        ) / det
        d = (
            s1 * s1 * (s2 * r3 - s3 * r2)
            - s1 * (s2 * s2 * r3 - s3 * s3 * r2)
            + r1 * (s2 * s2 * s3 - s3 * s3 * s2)
        ) / det
        exact = all(
            abs(a * r["seq_len"] ** 2 + b * r["seq_len"] + d - r[c]) < 0.5
            for r in fit_rows
        )
        coef[c] = {
            "a_quadratic": a,
            "b_linear": b,
            "d_const": d,
            "exact_on_fitted": exact,
        }
    return coef, anchor


def exponents(rows):
    out = {}
    for c in CATEGORIES:
        v = [(r["seq_len"], r[c]) for r in rows if r[c] > 0]
        out[c] = [round(math.log2(v[i + 1][1] / v[i][1]), 3) for i in range(len(v) - 1)]
    return out


def predict(coef, s):
    per_cat = {
        c: coef[c]["a_quadratic"] * s * s + coef[c]["b_linear"] * s + coef[c]["d_const"]
        for c in CATEGORIES
    }
    total = sum(per_cat.values())
    return per_cat, total


def _sum_coef(coef, cats):
    return (
        sum(coef[c]["a_quadratic"] for c in cats),
        sum(coef[c]["b_linear"] for c in cats),
        sum(coef[c]["d_const"] for c in cats),
    )


def crossover(coef):
    a_nl, b_nl, d_nl = _sum_coef(coef, ["softmax", "gelu"])

    def solve(cats):
        a_r, b_r, d_r = _sum_coef(coef, cats)
        da, db, dd = a_nl - a_r, b_nl - b_r, d_nl - d_r
        if da <= 0:
            return float("inf")
        disc = db * db - 4 * da * dd
        return (-db + disc**0.5) / (2 * da)

    return solve(["rescale"]), solve(["matmul", "qk", "av"])


def main():
    rows = load_seq_rows()
    coef, anchor = fit_coefficients(rows)
    held_out = [r for r in rows if r["seq_len"] > FIT_MAX_SEQ]

    print(
        f"fitted on seq <= {FIT_MAX_SEQ} (anchor seq={anchor['seq_len']}), "
        f"d_model={anchor['d_model']}, n_head={anchor['n_head']}"
    )
    print("\nmeasured per-doubling exponents (all rows, incl. held-out):")
    for c, ex in exponents(rows).items():
        print(f"  {c:>10s}: " + " ".join(f"{e:5.2f}" for e in ex))

    print("\nfitted coefficients (rows = a*seq^2 + b*seq):")
    for c in CATEGORIES:
        k = coef[c]
        print(
            f"  {c:>10s}: a={k['a_quadratic']:>10.4f}  b={k['b_linear']:>12.4f}"
            f"  d={k['d_const']:>8.4f}   exact: {k['exact_on_fitted']}"
        )

    s_res, s_mm = crossover(coef)
    print(
        f"\npredicted crossover (nonlinear overtakes rescale):      seq* = {s_res:.2f}"
    )
    print(f"predicted crossover (nonlinear overtakes matmul family): seq* = {s_mm:.2f}")

    check = []
    if held_out:
        print(f"\nheld-out check on {len(held_out)} config(s) never used to fit:")
        hdr = f"  {'seq':>5s} {'category':>10s} {'predicted':>12s} {'measured':>12s} {'err%':>7s}"
        print(hdr)
        for r in held_out:
            per_cat, total = predict(coef, r["seq_len"])
            for c in CATEGORIES + ["TOTAL"]:
                p = total if c == "TOTAL" else per_cat[c]
                m = r["num_rows"] if c == "TOTAL" else r[c]
                err = 100.0 * (p - m) / m if m else float("nan")
                print(
                    f"  {r['seq_len']:>5d} {c:>10s} {p:>12,.0f} {m:>12,d} {err:>7.2f}"
                )
                check.append(
                    {
                        "seq_len": r["seq_len"],
                        "category": c,
                        "predicted": round(p, 1),
                        "measured": m,
                        "err_pct": round(err, 3),
                    }
                )
            print()

    shares = []
    for r in rows:
        tot = sum(r[c] for c in CATEGORIES)
        nl = (r["softmax"] + r["gelu"]) / tot * 100
        rs = r["rescale"] / tot * 100
        mm = (r["matmul"] + r["qk"] + r["av"]) / tot * 100
        shares.append(
            {
                "seq_len": r["seq_len"],
                "num_rows": r["num_rows"],
                "logrows_est": max(6, math.ceil(math.log2(r["num_rows"] + 8))),
                "rescale_pct": round(rs, 2),
                "nonlinear_pct": round(nl, 2),
                "matmul_family_pct": round(mm, 2),
                "held_out": r["seq_len"] > FIT_MAX_SEQ,
            }
        )
    print("shares:")
    print(
        f"  {'seq':>5s} {'num_rows':>13s} {'k':>3s} {'rescale%':>9s} {'nonlin%':>8s} {'matmul%':>8s}"
    )
    for s in shares:
        mark = " *" if s["held_out"] else ""
        print(
            f"  {s['seq_len']:>5d} {s['num_rows']:>13,d} {s['logrows_est']:>3d} "
            f"{s['rescale_pct']:>9.2f} {s['nonlinear_pct']:>8.2f} "
            f"{s['matmul_family_pct']:>8.2f}{mark}"
        )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = {
        "fit_max_seq": FIT_MAX_SEQ,
        "fit_anchor_seq": anchor["seq_len"],
        "d_model": anchor["d_model"],
        "n_head": anchor["n_head"],
        "coefficients": coef,
        "crossover_seq_nonlinear_over_rescale": round(s_res, 1),
        "crossover_seq_nonlinear_over_matmul_family": round(s_mm, 1),
        "held_out_check": check,
        "shares": shares,
    }
    (OUT_DIR / "crossover.json").write_text(json.dumps(out, indent=2) + "\n")
    print(f"\nwrote {OUT_DIR / 'crossover.json'}")

    with open(OUT_DIR / "shares_by_seqlen.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(shares[0].keys()))
        w.writeheader()
        w.writerows(shares)
    print(f"wrote {OUT_DIR / 'shares_by_seqlen.csv'}")


if __name__ == "__main__":
    main()
