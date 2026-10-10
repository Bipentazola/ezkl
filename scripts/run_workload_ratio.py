#!/usr/bin/env python3

import argparse
import json
import os
import resource
import subprocess
import sys
import tempfile
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
EZKL = REPO / "target" / "release" / "ezkl"
GEN = REPO / "scripts" / "gen_decode_step_onnx.py"
OPT = {"EZKL_DIV_ROUND_NEAREST": "1", "EZKL_RESCALE_FUSION": "1"}


def cpu_of_children():
    r = resource.getrusage(resource.RUSAGE_CHILDREN)
    return r.ru_utime + r.ru_stime, r.ru_maxrss / 1e6


def run(*args, cwd, check=True):
    c0, _ = cpu_of_children()
    t0 = time.time()
    r = subprocess.run(
        [str(a) for a in args],
        cwd=str(cwd),
        capture_output=True,
        text=True,
        env=dict(os.environ, **OPT),
    )
    c1, peak = cpu_of_children()
    if check and r.returncode != 0:
        sys.exit(
            f"failed: {' '.join(map(str, args))}\n{r.stdout[-2000:]}\n{r.stderr[-2000:]}"
        )
    return r.stdout + r.stderr, c1 - c0, time.time() - t0, peak


def logrows(settings):
    n = json.loads(Path(settings).read_text())["num_rows"]
    return n, max(1, (n - 1).bit_length())


def build_and_prove(tag, model_onnx, inputs, wd, max_k_bump=4):
    wd.mkdir(parents=True, exist_ok=True)
    s = wd / "settings.json"
    run(
        EZKL,
        "gen-settings",
        "-M",
        model_onnx,
        "-O",
        s,
        "-K",
        24,
        "--input-scale",
        7,
        "--param-scale",
        7,
        cwd=wd,
    )
    rows, k = logrows(s)
    out = {"tag": tag, "num_rows": rows, "logrows_from_num_rows": k}

    srs = None
    for bump in range(max_k_bump + 1):
        kk = k + bump
        run(
            EZKL,
            "gen-settings",
            "-M",
            model_onnx,
            "-O",
            s,
            "-K",
            kk,
            "--input-scale",
            7,
            "--param-scale",
            7,
            cwd=wd,
        )
        run(
            EZKL,
            "compile-circuit",
            "-M",
            model_onnx,
            "-S",
            s,
            "--compiled-circuit",
            wd / "c.compiled",
            cwd=wd,
        )
        srs = wd / f"kzg{kk}.srs"
        run(EZKL, "get-srs", "--logrows", kk, "--srs-path", srs, cwd=wd)
        o, c, w, _ = run(
            EZKL,
            "setup",
            "-M",
            wd / "c.compiled",
            "--srs-path",
            srs,
            "--vk-path",
            wd / "vk.key",
            "--pk-path",
            wd / "pk.key",
            cwd=wd,
            check=False,
        )
        if "too small for the given circuit" not in o:
            if not (wd / "pk.key").exists():
                sys.exit(f"[{tag}] setup 失败（k={kk}）：\n{o[-2000:]}")
            out["logrows"] = kk
            out["k_bumped_by"] = bump
            out["setup_cpu_s"], out["setup_wall_s"] = round(c, 2), round(w, 2)
            break
        print(f"    [{tag}] k={kk} 太小（查找表下限），+1 重来", flush=True)
    else:
        sys.exit(f"[{tag}] k 连加 {max_k_bump} 次仍太小")

    steps = []
    for i, inp in enumerate(inputs):
        run(
            EZKL,
            "gen-witness",
            "-M",
            wd / "c.compiled",
            "-D",
            inp,
            "-O",
            wd / f"w{i}.json",
            cwd=wd,
        )
        _, c, w, peak = run(
            EZKL,
            "prove",
            "-W",
            wd / f"w{i}.json",
            "-M",
            wd / "c.compiled",
            "--pk-path",
            wd / "pk.key",
            "--proof-path",
            wd / f"p{i}.json",
            "--srs-path",
            srs,
            cwd=wd,
        )
        steps.append(
            {
                "step": i,
                "prove_cpu_s": round(c, 2),
                "prove_wall_s": round(w, 2),
                "peak_rss_gb": round(peak, 2),
            }
        )
        print(f"    [{tag}] step {i}: prove {c:.1f}s CPU / {w:.1f}s wall", flush=True)
    ok, _, _, _ = run(
        EZKL,
        "verify",
        "--proof-path",
        wd / "p0.json",
        "-S",
        s,
        "--vk-path",
        wd / "vk.key",
        "--srs-path",
        srs,
        cwd=wd,
        check=False,
    )
    out["verify_first_proof"] = "verified: true" in ok
    out["steps"] = steps
    out["prove_cpu_total_s"] = round(sum(x["prove_cpu_s"] for x in steps), 2)
    (wd / "pk.key").unlink(missing_ok=True)
    return out


def masks_for(t, t_max):
    return (
        [1.0 if i == t else 0.0 for i in range(t_max)],
        [1.0 if i <= t else 0.0 for i in range(t_max)],
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--configs",
        default="16x16,16x32",
        help="逗号分隔的 d x T，例如 16x16 表示 d_model=16、T=16",
    )
    ap.add_argument("--workdir", help="工作目录；默认在系统临时目录下创建独立目录")
    ap.add_argument("--out", default="results/incremental/workload_ratio")
    args = ap.parse_args()

    root = (
        Path(args.workdir)
        if args.workdir
        else Path(tempfile.mkdtemp(prefix="ezkl_workload_ratio_"))
    )
    print(f"[workdir] {root.resolve()}", flush=True)
    results = []
    for cfg in args.configs.split(","):
        d, T = (int(x) for x in cfg.lower().split("x"))
        print(f"=== d={d} T={T} ===", flush=True)
        wd = root / cfg
        full = REPO / "models" / f"seq{T}_d{d}_l1"
        if not full.exists():
            sys.exit(f"缺 {full}——先用 gen_models.py 生成，或换一个 --configs")
        unroll = build_and_prove(
            "unroll", full / "model.onnx", [full / "input.json"], wd / "unroll"
        )

        dec = wd / "decode"
        dec.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [
                sys.executable,
                str(GEN),
                "--variant",
                "where",
                "--d-model",
                str(d),
                "--t-max",
                str(T),
                "--position",
                "0",
                "--out-dir",
                str(dec),
            ],
            cwd=str(REPO),
            capture_output=True,
            text=True,
            check=True,
        )
        base = json.loads((dec / "input.json").read_text())
        shapes = base["input_shapes"]
        cache_len = 1
        for x in shapes[1]:
            cache_len *= x
        inputs = []
        for t in range(T):
            mw, ma = masks_for(t, T)
            p = dec / f"in_{t}.json"
            p.write_text(
                json.dumps(
                    {
                        "input_data": [
                            base["input_data"][0],
                            [0.0] * cache_len,
                            [0.0] * cache_len,
                            mw,
                            ma,
                        ],
                        "input_shapes": shapes,
                    }
                )
            )
            inputs.append(p)
        step = build_and_prove("step", dec / "model.onnx", inputs, wd / "step")

        proxy = T * 2 ** step["logrows"] / 2 ** unroll["logrows"]
        measured = step["prove_cpu_total_s"] / unroll["steps"][0]["prove_cpu_s"]
        rec = {
            "d_model": d,
            "T": T,
            "unroll": unroll,
            "step": step,
            "proxy_ratio": round(proxy, 3),
            "measured_ratio": round(measured, 3),
            "measured_over_proxy": round(measured / proxy, 3),
            "tier_gap": unroll["logrows"] - step["logrows"],
        }
        print(
            f"  -> 代理比 {proxy:.2f}x  实测比 {measured:.2f}x  "
            f"（实测/代理 = {measured / proxy:.2f}）",
            flush=True,
        )
        results.append(rec)

    out = Path(args.out)
    if not out.is_absolute():
        out = REPO / out
    out.mkdir(parents=True, exist_ok=True)
    (out / "workload_ratio.json").write_text(
        json.dumps(
            {
                "note": "代价代理 = T·2^k_step / 2^k_unroll（§7.5.2 表 7-5 的最后一列）；"
                "实测比 = T 次单步 prove 的 CPU 时间之和 / 全序列 prove 的 CPU 时间。"
                "两侧同二进制、同优化开关、同（默认）可见性。**这不验证 d=384 上的 "
                "8×**——那一侧仍跑不完；它验证的是这个代理在跨 tier 时的准确度。",
                "configs": results,
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    print(f"\n写入 {out / 'workload_ratio.json'}")


if __name__ == "__main__":
    main()
