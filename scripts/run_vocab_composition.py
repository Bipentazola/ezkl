#!/usr/bin/env python3

import argparse
import csv
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
EZKL = REPO / "target" / "release" / "ezkl"
GEN = REPO / "scripts" / "gen_decode_step_onnx.py"
PARSE = REPO / "scripts" / "parse_ledger.py"

OPT_ON = {"EZKL_DIV_ROUND_NEAREST": "1", "EZKL_RESCALE_FUSION": "1"}
POLYCOMMIT_VIS = [
    "--input-visibility",
    "polycommit",
    "--output-visibility",
    "polycommit",
    "--param-visibility",
    "fixed",
]

MAIN_D, MAIN_TMAX = 384, 128
MAIN_VOCABS = [0, 1024, 4096, 16384, 50257]
CHECK_D, CHECK_TMAX = 128, 16
CHECK_VOCABS = [0, 1024, 16384]

CATEGORIES = ["matmul", "qk", "av", "softmax", "gelu", "layernorm", "rescale", "other"]
TIMEOUT = 3600


def logrows_for(num_rows: int) -> int:
    return max(6, math.ceil(math.log2(num_rows + 8)))


def gen_model(out_dir: Path, d_model: int, t_max: int, vocab: int):
    cmd = [
        sys.executable,
        str(GEN),
        "--out-dir",
        str(out_dir),
        "--d-model",
        str(d_model),
        "--t-max",
        str(t_max),
        "--variant",
        "where",
    ]
    if vocab:
        cmd += ["--vocab", str(vocab)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        return None, (r.stdout + r.stderr)[-800:]
    return json.loads(r.stdout[r.stdout.index("{") :]), None


def gen_settings_debug(model_dir: Path, optimized: bool):
    env = dict(os.environ)
    for var in ("EZKL_DIV_ROUND_NEAREST", "EZKL_RESCALE_FUSION"):
        env.pop(var, None)
    if optimized:
        env.update(OPT_ON)
    env["RUST_LOG"] = "debug"
    settings = model_dir / "settings.json"
    cmd = [
        str(EZKL),
        "gen-settings",
        "-M",
        str(model_dir / "model.onnx"),
        "-O",
        str(settings),
        "-K",
        "26",
    ] + POLYCOMMIT_VIS
    t0 = time.time()
    try:
        r = subprocess.run(
            cmd, capture_output=True, text=True, env=env, timeout=TIMEOUT
        )
    except subprocess.TimeoutExpired:
        return None, None, f"TIMEOUT after {TIMEOUT}s"
    wall = time.time() - t0
    (model_dir / "logs").mkdir(exist_ok=True)
    (model_dir / "logs" / "gen_settings.log").write_text(r.stdout + r.stderr)
    if r.returncode != 0 or not settings.exists():
        return None, wall, (r.stdout + r.stderr)[-800:]
    return json.loads(settings.read_text()), wall, None


def parse_ledger(model_dir: Path):
    r = subprocess.run(
        [sys.executable, str(PARSE), str(model_dir)], capture_output=True, text=True
    )
    ledger_csv = model_dir / "op_ledger.csv"
    if r.returncode != 0 or not ledger_csv.exists():
        return None, (r.stdout + r.stderr)[-800:]
    out = {}
    with open(ledger_csv) as f:
        for row in csv.DictReader(f):
            out[row["category"]] = {
                "rows": int(row["total_rows"]),
                "pct": float(row["pct_rows"]),
            }
    return out, None


def top_nodes(model_dir: Path, n=8):
    raw = model_dir / "op_ledger_raw_nodes.csv"
    if not raw.exists():
        return []
    with open(raw) as f:
        recs = sorted(csv.DictReader(f), key=lambda r: -int(r["rows"]))
    return [
        {
            "node_idx": r["node_idx"],
            "rows": int(r["rows"]),
            "category": r["category"],
            "description": r["description"][:110],
        }
        for r in recs[:n]
    ]


def run_one(workdir: Path, d_model: int, t_max: int, vocab: int, optimized: bool):
    variant = "optimized" if optimized else "baseline"
    name = f"d{d_model}_T{t_max}_V{vocab}"
    mdir = workdir / f"{name}_{variant}"
    mdir.mkdir(parents=True, exist_ok=True)
    base = {
        "name": name,
        "variant": variant,
        "d_model": d_model,
        "t_max": t_max,
        "vocab": vocab,
    }

    print(f"[{name}/{variant}] generating model ...", flush=True)
    meta, err = gen_model(mdir, d_model, t_max, vocab)
    if err:
        print(f"[{name}/{variant}] MODEL FAILED: {err}", flush=True)
        return {**base, "status": "MODEL_FAILED", "error": err}

    print(f"[{name}/{variant}] gen-settings (RUST_LOG=debug) ...", flush=True)
    settings, wall, err = gen_settings_debug(mdir, optimized)
    if err:
        print(f"[{name}/{variant}] GEN-SETTINGS FAILED: {err}", flush=True)
        return {**base, "status": "SETTINGS_FAILED", "error": err, "wall_s": wall}

    num_rows = settings["num_rows"]
    ledger, err = parse_ledger(mdir)
    if err:
        print(f"[{name}/{variant}] LEDGER FAILED: {err}", flush=True)
        return {
            **base,
            "status": "LEDGER_FAILED",
            "error": err,
            "num_rows": num_rows,
            "wall_s": wall,
        }

    rec = {
        **base,
        "status": "OK",
        "num_rows": num_rows,
        "logrows": logrows_for(num_rows),
        "wall_s": round(wall, 1),
        "ledger_total_rows": sum(v["rows"] for v in ledger.values()),
    }
    for c in CATEGORIES:
        rec[f"{c}_rows"] = ledger.get(c, {}).get("rows", 0)
        rec[f"{c}_pct"] = round(ledger.get(c, {}).get("pct", 0.0), 3)
    rec["top_nodes"] = top_nodes(mdir)

    print(
        f"[{name}/{variant}] rows={num_rows:,} k={rec['logrows']} "
        f"rescale={rec['rescale_pct']:.1f}% matmul={rec['matmul_pct']:.1f}% "
        f"({rec['wall_s']}s)",
        flush=True,
    )
    return rec


def gen_model_embedout(out_dir: Path, d_model: int, t_max: int, vocab: int):
    cmd = [
        sys.executable,
        str(GEN),
        "--out-dir",
        str(out_dir),
        "--d-model",
        str(d_model),
        "--t-max",
        str(t_max),
        "--variant",
        "where",
        "--vocab",
        str(vocab),
        "--embed-outside",
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        return None, (r.stdout + r.stderr)[-800:]
    return json.loads(r.stdout[r.stdout.index("{") :]), None


def n_sweep(model_dir: Path, ns=(2, 4, 8, 16)):
    env = dict(os.environ)
    env.update(OPT_ON)
    out = []
    for n in ns:
        s = model_dir / f"settings_N{n}.json"
        cmd = [
            str(EZKL),
            "gen-settings",
            "-M",
            str(model_dir / "model.onnx"),
            "-O",
            str(s),
            "-K",
            "26",
            "-N",
            str(n),
        ] + POLYCOMMIT_VIS
        r = subprocess.run(
            cmd, capture_output=True, text=True, env=env, timeout=TIMEOUT
        )
        if r.returncode != 0 or not s.exists():
            out.append({"num_inner_cols": n, "status": "FAILED"})
            continue
        rows = json.loads(s.read_text())["num_rows"]
        out.append(
            {
                "num_inner_cols": n,
                "num_rows": rows,
                "logrows": logrows_for(rows),
                "status": "OK",
            }
        )
        print(f"    N={n:<3} rows={rows:>12,} k={logrows_for(rows)}", flush=True)
    return out


def embed_outside_study(workdir: Path, outdir: Path):
    d, t, v = MAIN_D, MAIN_TMAX, 50257
    rec = {"d_model": d, "t_max": t, "vocab": v, "onehot": {}, "embed_outside": {}}

    for label, gen in (("onehot", gen_model), ("embed_outside", gen_model_embedout)):
        mdir = workdir / f"d{d}_T{t}_V{v}_{label}"
        mdir.mkdir(parents=True, exist_ok=True)
        print(f"[{label}] generating ...", flush=True)
        meta, err = gen(mdir, d, t, v) if label == "onehot" else gen(mdir, d, t, v)
        if err:
            rec[label] = {"status": "MODEL_FAILED", "error": err}
            continue
        for optimized in (False, True):
            key = "optimized" if optimized else "baseline"
            settings, wall, err = gen_settings_debug(mdir, optimized)
            if err:
                rec[label][key] = {"status": "FAILED", "error": err}
                continue
            rows = settings["num_rows"]
            entry = {
                "num_rows": rows,
                "logrows": logrows_for(rows),
                "wall_s": round(wall, 1),
                "status": "OK",
            }
            if not optimized:
                ledger, lerr = parse_ledger(mdir)
                if ledger:
                    entry["ledger_pct"] = {
                        c: round(ledger.get(c, {}).get("pct", 0.0), 2)
                        for c in CATEGORIES
                    }
                entry["top_nodes"] = top_nodes(mdir, 4)
            rec[label][key] = entry
            print(f"  [{label}/{key}] rows={rows:,} k={entry['logrows']}", flush=True)
        print(f"  [{label}] num_inner_cols sweep:", flush=True)
        rec[label]["n_sweep"] = n_sweep(mdir)

    ob = rec["onehot"].get("baseline", {}).get("num_rows")
    eb = rec["embed_outside"].get("baseline", {}).get("num_rows")
    if ob and eb:
        rec["row_cut_pct_baseline"] = round(100 * (ob - eb) / ob, 2)
    oo = rec["onehot"].get("optimized", {}).get("num_rows")
    eo = rec["embed_outside"].get("optimized", {}).get("num_rows")
    if oo and eo:
        rec["row_cut_pct_optimized"] = round(100 * (oo - eo) / oo, 2)

    (outdir / "embed_outside.json").write_text(
        json.dumps(rec, indent=2, ensure_ascii=False)
    )
    print(f"\nwrote {outdir}/embed_outside.json")
    if ob and eb:
        print(f"  one-hot      {ob:>12,} rows (k={logrows_for(ob)})")
        print(
            f"  embed_outside{eb:>12,} rows (k={logrows_for(eb)})  "
            f"cut={rec['row_cut_pct_baseline']}%"
        )
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="冒烟：只跑小 V")
    ap.add_argument(
        "--embed-outside-study",
        action="store_true",
        help="只跑嵌入移出电路的对照和 num_inner_cols 扫描",
    )
    ap.add_argument("--out", default=str(REPO / "results" / "vocab_composition"))
    ap.add_argument("--workdir", default=str(REPO / "models_vocab"))
    args = ap.parse_args()

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    workdir = Path(args.workdir)
    workdir.mkdir(parents=True, exist_ok=True)

    if args.embed_outside_study:
        embed_outside_study(workdir, outdir)
        return

    if args.quick:
        configs = [(MAIN_D, MAIN_TMAX, v) for v in (0, 1024)]
    else:
        configs = [(MAIN_D, MAIN_TMAX, v) for v in MAIN_VOCABS] + [
            (CHECK_D, CHECK_TMAX, v) for v in CHECK_VOCABS
        ]

    records = []
    for d, t, v in configs:
        for optimized in (False, True):
            records.append(run_one(workdir, d, t, v, optimized))
            (outdir / "vocab_composition.json").write_text(
                json.dumps(records, indent=2, ensure_ascii=False)
            )

    fields = [
        "name",
        "variant",
        "d_model",
        "t_max",
        "vocab",
        "status",
        "num_rows",
        "logrows",
        "wall_s",
        "ledger_total_rows",
    ] + [f"{c}_{s}" for c in CATEGORIES for s in ("rows", "pct")]
    with open(outdir / "vocab_composition.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in records:
            w.writerow(r)

    print(f"\nwrote {outdir}/vocab_composition.{{json,csv}}")
    ok = [r for r in records if r.get("status") == "OK"]
    if ok:
        print(
            f"\n{'config':<20}{'variant':<11}{'rows':>13}{'k':>4}"
            f"{'rescale':>10}{'matmul':>9}{'other':>8}"
        )
        for r in ok:
            print(
                f"{r['name']:<20}{r['variant']:<11}{r['num_rows']:>13,}"
                f"{r['logrows']:>4}{r['rescale_pct']:>9.1f}%"
                f"{r['matmul_pct']:>8.1f}%{r['other_pct']:>7.1f}%"
            )


if __name__ == "__main__":
    main()
