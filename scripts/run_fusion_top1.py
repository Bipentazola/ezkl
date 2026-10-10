#!/usr/bin/env python3

import argparse
import csv
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
EZKL = REPO / "target" / "release" / "ezkl"
os.environ.setdefault("HF_HOME", "/data/hf_cache")
sys.path.insert(0, str(REPO / "scripts"))


def _quantize(w, scale):
    x = w * (2.0**scale)
    return np.where(x >= 0, np.floor(x + 0.5), np.ceil(x - 0.5))


def run(cmd, env=None, cwd=REPO):
    e = dict(os.environ)
    if env:
        e.update(env)
    return subprocess.run(
        [str(c) for c in cmd], env=e, cwd=str(cwd), capture_output=True, text=True
    )


def load_windows(model_id, corpus_path, seq_len, n_tokens):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_id)
    text = Path(corpus_path).read_text(errors="ignore")
    text = text.split("\n", 1)[1] if "\n" in text else text
    ids = tok(text)["input_ids"]
    n_win = min(n_tokens // seq_len, (len(ids) - 1) // seq_len)
    wins = [
        (
            ids[i * seq_len : (i + 1) * seq_len],
            ids[i * seq_len + 1 : (i + 1) * seq_len + 1],
        )
        for i in range(n_win)
    ]
    return tok, wins


def float_reference(model_id, seq_len, wins):
    import torch
    from gen_pretrained_lm_onnx import build, embed

    _, m, cfg, wte, wpe = build(model_id, seq_len)
    m.eval()
    h0s, logits = [], []
    with torch.no_grad():
        for inp, _ in wins:
            ids = torch.tensor([inp])
            h = embed(wte, wpe, ids)
            h0s.append(h.numpy().astype(np.float64)[0])
            logits.append(m(h).numpy().astype(np.float64)[0])
    return cfg, wte.numpy().astype(np.float64), np.array(h0s), np.array(logits)


def prepare_variant(model_onnx, workdir, scale, fusion):
    env = {
        "EZKL_DIV_ROUND_NEAREST": "1",
        "EZKL_RESCALE_FUSION": str(fusion),
        "RUST_LOG": "error,ezkl::graph::rescale_fusion=debug",
    }
    workdir.mkdir(parents=True, exist_ok=True)
    s = workdir / "settings.json"
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
        sys.exit(
            f"gen-settings failed (fusion={fusion}, scale={scale}):\n{r.stdout[-2000:]}\n{r.stderr[-2000:]}"
        )
    rows = json.loads(s.read_text())["num_rows"]
    c = workdir / "c.compiled"
    r = run(
        [EZKL, "compile-circuit", "-M", model_onnx, "-S", s, "--compiled-circuit", c],
        env,
    )
    if r.returncode != 0:
        sys.exit(
            f"compile failed (fusion={fusion}):\n{r.stdout[-2000:]}\n{r.stderr[-2000:]}"
        )
    report = ""
    for line in (r.stdout + r.stderr).splitlines():
        if "rescale fusion:" in line and "delayed" in line:
            report = line.split("rescale fusion:")[-1].strip()
    return {
        "settings": s,
        "compiled": c,
        "num_rows": rows,
        "fusion_report": report or ("(融合关闭)" if not fusion else "(未报告)"),
        "env": {k: v for k, v in env.items() if k.startswith("EZKL")},
    }


class WitnessFailure(RuntimeError):
    pass


def witness_logits(variant, h0, seq_len, vocab, workdir, tag):
    inp = workdir / f"in_{tag}.json"
    inp.write_text(
        json.dumps(
            {
                "input_data": [h0.reshape(-1).tolist()],
                "input_shapes": [[1, seq_len, h0.shape[-1]]],
            }
        )
    )
    out = workdir / f"w_{tag}.json"
    env = dict(variant["env"])
    r = run([EZKL, "gen-witness", "-M", variant["compiled"], "-D", inp, "-O", out], env)
    if r.returncode == -9:
        time.sleep(20)
        r = run(
            [EZKL, "gen-witness", "-M", variant["compiled"], "-D", inp, "-O", out], env
        )
    if r.returncode != 0 or not out.exists():
        msg = ""
        for line in (r.stdout + r.stderr).splitlines():
            if "[E]" in line:
                msg = line.strip()[:200]
                break
        inp.unlink(missing_ok=True)
        raise WitnessFailure(msg or f"gen-witness rc={r.returncode}")
    w = json.loads(out.read_text())
    vals = w["pretty_elements"]["rescaled_outputs"][0]
    lg = np.array([float(x) for x in vals], dtype=np.float64).reshape(seq_len, vocab)
    inp.unlink()
    out.unlink()
    return lg


def top2(x):
    i1 = int(np.argmax(x))
    v1 = x[i1]
    x2 = x.copy()
    x2[i1] = -np.inf
    return i1, v1, float(np.max(x2))


def logprob(x, target):
    m = x.max()
    return float(x[target] - (m + np.log(np.exp(x - m).sum())))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument(
        "--corpus", default="models_pretrained/corpus/tinystories_valid_head.txt"
    )
    ap.add_argument("--scales", default="13")
    ap.add_argument("--tokens", type=int, default=1024)
    ap.add_argument(
        "--dump-logits",
        type=int,
        default=64,
        help="留存 top-K logits 明细的 token 数（见 --dump-topk）",
    )
    ap.add_argument(
        "--dump-topk",
        type=int,
        default=64,
        help="每个留存 token 存多少个词表项。全词表 50257 项存下来一个 "
        "scale 就要 30 MB，而派生量只用得到前几项，故只存 top-K",
    )
    ap.add_argument("--jobs", type=int, default=5, help="并行的 gen-witness 进程数")
    ap.add_argument("--out", default=None)
    ap.add_argument("--workdir", help="工作目录；默认在系统临时目录下创建独立目录")
    args = ap.parse_args()

    mdir = (
        REPO / args.model_dir
        if not Path(args.model_dir).is_absolute()
        else Path(args.model_dir)
    )
    meta = json.loads((mdir / "meta.json").read_text())
    seq_len, vocab, model_id = meta["seq_len"], meta["vocab_size"], meta["model"]
    out_root = (
        Path(args.out) if args.out else REPO / "results" / "fusion_top1" / mdir.name
    )
    work = (
        Path(args.workdir)
        if args.workdir
        else Path(tempfile.mkdtemp(prefix="ezkl_fusion_top1_"))
    )
    work.mkdir(parents=True, exist_ok=True)
    print(f"[workdir] {work.resolve()}", flush=True)

    tok, wins = load_windows(model_id, REPO / args.corpus, seq_len, args.tokens)
    print(
        f"[corpus] {len(wins)} windows x {seq_len} tokens = {len(wins)*seq_len} tokens",
        flush=True,
    )
    cfg, wte, h0s, ref_logits = float_reference(model_id, seq_len, wins)
    print(f"[float] reference logits done, shape {ref_logits.shape}", flush=True)

    for scale in [int(s) for s in args.scales.split(",")]:
        t0 = time.time()
        outdir = out_root / f"scale{scale}"
        outdir.mkdir(parents=True, exist_ok=True)
        variants = {}
        for fusion in (0, 1):
            variants[fusion] = prepare_variant(
                mdir / "model.onnx", work / f"s{scale}_f{fusion}", scale, fusion
            )
            print(
                f"[scale {scale}] fusion={fusion}: num_rows={variants[fusion]['num_rows']:,}  "
                f"{variants[fusion]['fusion_report']}",
                flush=True,
            )

        W_int = _quantize(wte, scale)
        l1_rows = np.abs(W_int).sum(axis=1)
        bound_rows_ulp = l1_rows / (2.0 * (2.0**scale)) + 1.0
        bound_max_ulp = float(bound_rows_ulp.max())

        rows_csv = []
        dumps = {"float": [], "unfused": [], "fused": []}
        n_dump = 0
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=args.jobs) as pool:
            futs = {
                (wi, f): pool.submit(
                    witness_logits,
                    variants[f],
                    h0s[wi],
                    seq_len,
                    vocab,
                    work,
                    f"s{scale}f{f}w{wi}",
                )
                for wi in range(len(wins))
                for f in (0, 1)
            }
            failures = []
            for wi, ((inp, tgt), ref) in enumerate(zip(wins, ref_logits)):
                try:
                    lg = {f: futs[(wi, f)].result() for f in (0, 1)}
                except WitnessFailure as e:
                    failures.append({"window": wi, "error": str(e)})
                    print(f"  [scale {scale}] window {wi} 剔除：{e}", flush=True)
                    continue
                if n_dump < args.dump_logits:
                    take = min(seq_len, args.dump_logits - n_dump)
                    dumps["float"].append(ref[:take])
                    dumps["unfused"].append(lg[0][:take])
                    dumps["fused"].append(lg[1][:take])
                    n_dump += take
                for p in range(seq_len):
                    rf, r0, r1 = ref[p], lg[0][p], lg[1][p]
                    a_f, v1_f, v2_f = top2(rf)
                    a_0, v1_0, v2_0 = top2(r0)
                    a_1, v1_1, v2_1 = top2(r1)
                    d10 = np.abs(r1 - r0)
                    rows_csv.append(
                        dict(
                            window=wi,
                            pos=p,
                            token=int(inp[p]),
                            target=int(tgt[p]),
                            argmax_float=a_f,
                            argmax_unfused=a_0,
                            argmax_fused=a_1,
                            flip_fused_vs_unfused=int(a_1 != a_0),
                            flip_unfused_vs_float=int(a_0 != a_f),
                            flip_fused_vs_float=int(a_1 != a_f),
                            margin_float=v1_f - v2_f,
                            margin_unfused=v1_0 - v2_0,
                            margin_unfused_ulp=(v1_0 - v2_0) * (2.0**scale),
                            bound_top1_ulp=float(bound_rows_ulp[a_0]),
                            bound_pair_ulp=float(
                                bound_rows_ulp[a_0]
                                + bound_rows_ulp[int(np.argsort(r0)[-2])]
                            ),
                            bound_max_ulp=2.0 * bound_max_ulp,
                            max_abs_diff_fused_unfused=float(d10.max()),
                            mean_abs_diff_fused_unfused=float(d10.mean()),
                            max_abs_diff_unfused_float=float(np.abs(r0 - rf).max()),
                            max_abs_diff_fused_float=float(np.abs(r1 - rf).max()),
                            logprob_float=logprob(rf, tgt[p]),
                            logprob_unfused=logprob(r0, tgt[p]),
                            logprob_fused=logprob(r1, tgt[p]),
                        )
                    )
                el = time.time() - t0
                print(
                    f"  [scale {scale}] {(wi+1)*seq_len}/{len(wins)*seq_len} tokens  "
                    f"({el:.0f}s elapsed)",
                    flush=True,
                )

        if not rows_csv:
            (outdir / "summary.json").write_text(
                json.dumps(
                    {
                        "model": model_id,
                        "scale": scale,
                        "n_tokens": 0,
                        "n_windows": len(wins),
                        "n_windows_dropped": len(failures),
                        "dropped": failures,
                        "note": "该 scale 下所有窗口的 gen-witness 都失败，无数据",
                    },
                    indent=2,
                    ensure_ascii=False,
                )
            )
            print(f"[scale {scale}] 全部 {len(wins)} 个窗口被剔除，跳过", flush=True)
            continue

        keys = list(rows_csv[0].keys())
        with (outdir / "tokens.csv").open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=keys)
            w.writeheader()
            w.writerows(rows_csv)
        if dumps["float"]:
            cat = {k: np.concatenate(v)[: args.dump_logits] for k, v in dumps.items()}
            idx = np.argsort(cat["unfused"], axis=1)[:, ::-1][:, : args.dump_topk]
            np.savez_compressed(
                outdir / "logits_topk.npz",
                topk_index=idx.astype(np.int32),
                **{
                    k: np.take_along_axis(v, idx, axis=1).astype(np.float32)
                    for k, v in cat.items()
                },
            )

        arr = {k: np.array([r[k] for r in rows_csv], dtype=np.float64) for k in keys}
        n = len(rows_csv)
        summary = dict(
            model=model_id,
            model_dir=str(mdir.relative_to(REPO)),
            seq_len=seq_len,
            vocab=vocab,
            d_model=cfg["d_model"],
            n_layers=cfg["n_layers"],
            scale=scale,
            n_tokens=n,
            n_windows=len(wins),
            n_windows_dropped=len(failures),
            dropped=failures,
            num_rows_unfused=variants[0]["num_rows"],
            num_rows_fused=variants[1]["num_rows"],
            fusion_report=variants[1]["fusion_report"],
            top1_agree_fused_vs_unfused=1 - arr["flip_fused_vs_unfused"].mean(),
            top1_agree_unfused_vs_float=1 - arr["flip_unfused_vs_float"].mean(),
            top1_agree_fused_vs_float=1 - arr["flip_fused_vs_float"].mean(),
            n_flip_fused_vs_unfused=int(arr["flip_fused_vs_unfused"].sum()),
            n_flip_unfused_vs_float=int(arr["flip_unfused_vs_float"].sum()),
            n_flip_fused_vs_float=int(arr["flip_fused_vs_float"].sum()),
            ppl_float=float(np.exp(-arr["logprob_float"].mean())),
            ppl_unfused=float(np.exp(-arr["logprob_unfused"].mean())),
            ppl_fused=float(np.exp(-arr["logprob_fused"].mean())),
            max_abs_diff_fused_unfused=float(arr["max_abs_diff_fused_unfused"].max()),
            mean_abs_diff_fused_unfused=float(
                arr["mean_abs_diff_fused_unfused"].mean()
            ),
            max_abs_diff_unfused_float=float(arr["max_abs_diff_unfused_float"].max()),
            max_abs_diff_fused_float=float(arr["max_abs_diff_fused_float"].max()),
            margin_ulp_median=float(np.median(arr["margin_unfused_ulp"])),
            margin_ulp_p05=float(np.percentile(arr["margin_unfused_ulp"], 5)),
            margin_ulp_min=float(arr["margin_unfused_ulp"].min()),
            bound_pair_ulp_median=float(np.median(arr["bound_pair_ulp"])),
            bound_pair_ulp_max=float(arr["bound_pair_ulp"].max()),
            bound_sound_ulp=2.0 * bound_max_ulp,
            certified_frac_1hop=float(
                (arr["margin_unfused_ulp"] > 2.0 * bound_max_ulp).mean()
            ),
            certified_frac_1hop_pair_LEGACY=float(
                (arr["margin_unfused_ulp"] > arr["bound_pair_ulp"]).mean()
            ),
            elapsed_s=time.time() - t0,
        )
        (outdir / "summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False)
        )
        print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
