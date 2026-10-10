#!/usr/bin/env python3

import argparse
import csv
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
EZKL = REPO / "target" / "release" / "ezkl"
GEN = REPO / "scripts" / "gen_decode_step_onnx.py"

WRITE_VARIANTS = ["mask", "where", "scatter"]
WRITE_TMAX = [16, 32, 64, 128, 256]
WRITE_D = 384

STEP_TMAX = [16, 32, 64, 128]
STEP_D = [128, 384]

OPT_ON = {"EZKL_DIV_ROUND_NEAREST": "1", "EZKL_RESCALE_FUSION": "1"}


def gen_model(out_dir, **kw):
    cmd = [sys.executable, str(GEN), "--out-dir", str(out_dir)]
    for k, v in kw.items():
        flag = "--" + k.replace("_", "-")
        if v is True:
            cmd.append(flag)
        elif v is not False and v is not None:
            cmd += [flag, str(v)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        return None, (r.stdout + r.stderr)[-600:]
    return json.loads(r.stdout[r.stdout.index("{") :]), None


POLYCOMMIT_VIS = [
    "--input-visibility",
    "polycommit",
    "--output-visibility",
    "polycommit",
    "--param-visibility",
    "fixed",
]


def gen_settings(model_dir, env_extra, logrows=24, polycommit=False, tag=""):
    env = dict(os.environ)
    for var in ("EZKL_DIV_ROUND_NEAREST", "EZKL_RESCALE_FUSION"):
        env.pop(var, None)
    env.update(env_extra)
    settings = Path(model_dir) / f"settings{tag}.json"
    cmd = [
        str(EZKL),
        "gen-settings",
        "-M",
        str(Path(model_dir) / "model.onnx"),
        "-O",
        str(settings),
        "-K",
        str(logrows),
    ]
    if polycommit:
        cmd += POLYCOMMIT_VIS
    r = subprocess.run(cmd, capture_output=True, text=True, env=env)
    if r.returncode != 0 or not settings.exists():
        return None, (r.stdout + r.stderr)[-600:]
    s = json.loads(settings.read_text())
    return s, None


def logrows_for(num_rows):
    k = 1
    while (1 << k) < num_rows:
        k += 1
    return k


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=["write", "step"], default=None)
    ap.add_argument("--out", default=str(REPO / "results" / "incremental" / "p1_rows"))
    ap.add_argument("--workdir", default=None)
    args = ap.parse_args()

    if not EZKL.exists():
        sys.exit(f"missing {EZKL}")
    wd = (
        Path(args.workdir)
        if args.workdir
        else Path(tempfile.mkdtemp(prefix="ezkl_p1_"))
    )
    wd.mkdir(parents=True, exist_ok=True)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    print(f"[workdir] {wd}")

    if args.only != "step":
        rows = []
        for variant in WRITE_VARIANTS:
            for t in WRITE_TMAX:
                d = wd / f"write_{variant}_T{t}"
                info, err = gen_model(
                    d, write_only=True, variant=variant, d_model=WRITE_D, t_max=t
                )
                if info is None:
                    rows.append(
                        dict(
                            variant=variant, t_max=t, status=f"EXPORT_FAIL: {err[:120]}"
                        )
                    )
                    print(f"  {variant:8s} T={t:4d}  EXPORT FAILED")
                    continue
                s, err = gen_settings(d, OPT_ON)
                if s is None:
                    rows.append(
                        dict(
                            variant=variant,
                            t_max=t,
                            cache_elems=info["cache_elems"],
                            status=f"GEN_SETTINGS_FAIL: {err[:120]}",
                        )
                    )
                    print(f"  {variant:8s} T={t:4d}  GEN-SETTINGS FAILED")
                    continue
                n = s["num_rows"]
                ce = info["cache_elems"]
                rows.append(
                    dict(
                        variant=variant,
                        t_max=t,
                        cache_elems=ce,
                        num_rows=n,
                        rows_per_cache_elem=round(n / ce, 3),
                        num_lookups=len(s.get("required_lookups", [])),
                        status="OK",
                    )
                )
                print(
                    f"  {variant:8s} T={t:4d}  cache={ce:8d}  rows={n:9d}  "
                    f"rows/elem={n/ce:7.3f}"
                )
        with (out / "p1_write_variants.csv").open("w", newline="") as f:
            cols = [
                "variant",
                "t_max",
                "cache_elems",
                "num_rows",
                "rows_per_cache_elem",
                "num_lookups",
                "status",
            ]
            w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)
        print(f"[wrote] {out/'p1_write_variants.csv'}")

    if args.only != "write":
        rows = []
        for d_model in STEP_D:
            for t in STEP_TMAX:
                mdir = wd / f"step_where_d{d_model}_T{t}"
                info, err = gen_model(mdir, variant="where", d_model=d_model, t_max=t)
                if info is None:
                    print(f"  d={d_model} T={t}  EXPORT FAILED: {err[:200]}")
                    rows.append(dict(d_model=d_model, t_max=t, status="EXPORT_FAIL"))
                    continue
                rec = dict(
                    d_model=d_model,
                    n_head=info["n_head"],
                    t_max=t,
                    cache_elems=info["cache_elems"],
                    status="OK",
                )
                for vis, pc in (("default", False), ("polycommit", True)):
                    for label, env in (("baseline", {}), ("optimized", OPT_ON)):
                        key = f"{vis}_{label}"
                        s, err = gen_settings(mdir, env, polycommit=pc, tag=f"_{key}")
                        if s is None:
                            rec[f"num_rows_{key}"] = None
                            rec["status"] = f"GEN_SETTINGS_FAIL({key})"
                            continue
                        rec[f"num_rows_{key}"] = s["num_rows"]
                        rec[f"logrows_{key}"] = logrows_for(s["num_rows"])
                        rec[f"num_tables_{key}"] = len(s.get("required_lookups", []))
                    b = rec.get(f"num_rows_{vis}_baseline")
                    o = rec.get(f"num_rows_{vis}_optimized")
                    if b and o:
                        rec[f"row_cut_pct_{vis}"] = round(100 * (b - o) / b, 2)
                        rec[f"tables_unchanged_{vis}"] = (
                            rec[f"num_tables_{vis}_baseline"]
                            == rec[f"num_tables_{vis}_optimized"]
                        )
                rows.append(rec)
                print(
                    f"  d={d_model:4d} T={t:4d}  "
                    f"default opt={rec.get('num_rows_default_optimized')}"
                    f"(k={rec.get('logrows_default_optimized')})  "
                    f"polycommit base={rec.get('num_rows_polycommit_baseline')}"
                    f"(k={rec.get('logrows_polycommit_baseline')}) "
                    f"opt={rec.get('num_rows_polycommit_optimized')}"
                    f"(k={rec.get('logrows_polycommit_optimized')}) "
                    f"cut={rec.get('row_cut_pct_polycommit')}% "
                    f"tables_same={rec.get('tables_unchanged_polycommit')}"
                )
        cols = ["d_model", "n_head", "t_max", "cache_elems"]
        for vis in ("default", "polycommit"):
            for label in ("baseline", "optimized"):
                cols += [
                    f"num_rows_{vis}_{label}",
                    f"logrows_{vis}_{label}",
                    f"num_tables_{vis}_{label}",
                ]
            cols += [f"row_cut_pct_{vis}", f"tables_unchanged_{vis}"]
        cols.append("status")
        with (out / "p1_decode_step.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)
        print(f"[wrote] {out/'p1_decode_step.csv'}")

        fit = {}
        for d_model in STEP_D:
            pts = [
                (int(r["cache_elems"]), r.get("num_rows_polycommit_optimized"))
                for r in rows
                if r.get("d_model") == d_model
                and r.get("num_rows_polycommit_optimized")
            ]
            if len(pts) < 2:
                continue
            (x0, y0), (x1, y1) = pts[0], pts[-1]
            s = (y1 - y0) / (x1 - x0)
            c = y0 - s * x0
            slopes = [
                round((pts[i + 1][1] - pts[i][1]) / (pts[i + 1][0] - pts[i][0]), 4)
                for i in range(len(pts) - 1)
            ]
            budget = 1 << 21
            fit[str(d_model)] = dict(
                rows_per_cache_elem=round(s, 4),
                fixed_rows=round(c),
                pairwise_slopes=slopes,
                exactly_linear=(len(set(slopes)) == 1),
                t_max_ceiling_at_logrows_21=int((budget - c) // (s * 2 * d_model)),
            )
        (out / "p1_cost_model.json").write_text(json.dumps(fit, indent=2) + "\n")
        print(f"[wrote] {out/'p1_cost_model.json'}")

    print(f"[workdir kept] {wd}")


if __name__ == "__main__":
    main()
