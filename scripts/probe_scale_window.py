#!/usr/bin/env python3

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
EZKL = REPO / "target" / "release" / "ezkl"
ERR = re.compile(r"\[E\].*")


def run(cmd, env):
    return subprocess.run(
        [str(c) for c in cmd],
        env=dict(os.environ, **env),
        cwd=str(REPO),
        capture_output=True,
        text=True,
    )


def probe(model_onnx, data, scale, fusion, wd):
    env = {"EZKL_DIV_ROUND_NEAREST": "1", "EZKL_RESCALE_FUSION": str(fusion)}
    rec = {"scale": scale, "fusion": fusion}
    s, c, w = wd / "s.json", wd / "c.compiled", wd / "w.json"
    r = run(
        [
            EZKL,
            "gen-settings",
            "-M",
            model_onnx,
            "-O",
            s,
            "--input-scale",
            scale,
            "--param-scale",
            scale,
        ],
        env,
    )
    if r.returncode != 0:
        rec.update(stage="gen-settings", ok=False, error=first_err(r))
        return rec
    rec["num_rows"] = json.loads(s.read_text())["num_rows"]
    r = run(
        [EZKL, "compile-circuit", "-M", model_onnx, "-S", s, "--compiled-circuit", c],
        env,
    )
    if r.returncode != 0:
        rec.update(stage="compile-circuit", ok=False, error=first_err(r))
        return rec
    r = run([EZKL, "gen-witness", "-M", c, "-D", data, "-O", w], env)
    rec.update(stage="gen-witness", ok=(r.returncode == 0 and w.exists()))
    if not rec["ok"]:
        rec["error"] = first_err(r)
    for p in (c, w):
        p.unlink(missing_ok=True)
    return rec


def first_err(r):
    for line in (r.stdout + r.stderr).splitlines():
        m = ERR.search(line)
        if m:
            return m.group(0).strip()[:240]
    return f"rc={r.returncode}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--scales", default="7,10,11,12,13,14,15,16,17,18")
    ap.add_argument("--out", required=True)
    ap.add_argument("--temp-dir", help="临时目录的父目录；默认使用系统临时目录")
    args = ap.parse_args()

    mdir = (
        REPO / args.model_dir
        if not Path(args.model_dir).is_absolute()
        else Path(args.model_dir)
    )
    recs = []
    if args.temp_dir:
        Path(args.temp_dir).mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="ezkl_scale_", dir=args.temp_dir) as tmp:
        wd = Path(tmp)
        for scale in [int(x) for x in args.scales.split(",")]:
            for fusion in (1, 0):
                rec = probe(mdir / "model.onnx", mdir / "input.json", scale, fusion, wd)
                recs.append(rec)
                print(
                    f"scale={scale} fusion={fusion} {rec['stage']} "
                    f"ok={rec.get('ok')} {rec.get('error','')}",
                    flush=True,
                )

    out = Path(args.out)
    if not out.is_absolute():
        out = REPO / out
    out.parent.mkdir(parents=True, exist_ok=True)
    ok = {
        s
        for s in {r["scale"] for r in recs}
        if all(r["ok"] for r in recs if r["scale"] == s)
    }
    same = all(
        {r["ok"] for r in recs if r["scale"] == s}
        == {next(x["ok"] for x in recs if x["scale"] == s)}
        for s in {r["scale"] for r in recs}
    )
    out.write_text(
        json.dumps(
            {
                "model_dir": str(mdir.relative_to(REPO)),
                "usable_window": sorted(ok),
                "fusion_changes_the_window": not same,
                "records": recs,
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    print(f"\n可用窗口 = {sorted(ok)}；融合是否改变窗口 = {not same}\n写入 {out}")


if __name__ == "__main__":
    main()
