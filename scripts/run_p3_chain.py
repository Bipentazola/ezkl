#!/usr/bin/env python3

import argparse
import json
import math
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
EZKL = REPO / "target" / "release" / "ezkl"
GEN = REPO / "scripts" / "gen_decode_step_onnx.py"
CHECKER = REPO / "scripts" / "verify_chain.py"
COMMIT_BIN = REPO / "target" / "release" / "examples" / "kzg_commit_vector"
POINT = 64

P = 21888242871839275222246405745257275088548364400416034343698204186575808495617

LAYOUT = {
    "cache_in_idx": [1, 2],
    "cache_out_idx": [6, 7],
    "mask_idx": {"m_write": 3, "m_attn": 4},
    "unchecked_input_idx": [0],
}
TOK_IDX, LOGITS_IDX = 0, 8

VOCAB_SCAN = [0, 64, 256, 1024]
OPT_ENV = {"EZKL_DIV_ROUND_NEAREST": "1", "EZKL_RESCALE_FUSION": "1"}


def run(*args, cwd, check=True, env=None):
    import os

    e = dict(os.environ)
    for v in ("EZKL_DIV_ROUND_NEAREST", "EZKL_RESCALE_FUSION"):
        e.pop(v, None)
    e.update(OPT_ENV if env is None else env)
    r = subprocess.run(
        [str(a) for a in args], cwd=cwd, capture_output=True, text=True, env=e
    )
    if check and r.returncode != 0:
        sys.exit(f"command failed: {' '.join(map(str, args))}\n{r.stdout}\n{r.stderr}")
    return r.stdout + r.stderr


def felt_to_int(h):
    v = int.from_bytes(bytes.fromhex(h), "little")
    return v - P if v > P // 2 else v


def int_to_felt(v):
    return (v % P).to_bytes(32, "little").hex()


def quantize(x, scale):
    y = float(x) * (2.0**scale)
    return math.floor(y + 0.5) if y >= 0 else math.ceil(y - 0.5)


def commitments(path, n):
    b = bytes(json.loads(Path(path).read_text())["proof"])
    return [b[i * POINT : (i + 1) * POINT] for i in range(n)]


def masks_for(t, t_max):
    mw = [0.0] * t_max
    mw[t] = 1.0
    return mw, [1.0 if i <= t else 0.0 for i in range(t_max)]


def logrows_for(n):
    return max(1, (n - 1).bit_length())


def measure_rows(wd, d_model, t_max, vocab, embed_outside=False):
    tag = "_embedout" if embed_outside else ""
    mdir = wd / f"rows_d{d_model}_T{t_max}_V{vocab}{tag}"
    extra = ["--embed-outside"] if embed_outside else []
    run(
        sys.executable,
        GEN,
        "--variant",
        "where",
        "--d-model",
        d_model,
        "--t-max",
        t_max,
        "--vocab",
        vocab,
        *extra,
        "--out-dir",
        mdir,
        cwd=REPO,
    )
    run(
        EZKL,
        "gen-settings",
        "-M",
        "model.onnx",
        "-O",
        "settings.json",
        "-K",
        24,
        "--input-visibility",
        "polycommit",
        "--output-visibility",
        "polycommit",
        "--param-visibility",
        "fixed",
        cwd=mdir,
    )
    n = json.loads((mdir / "settings.json").read_text())["num_rows"]
    return {
        "d_model": d_model,
        "t_max": t_max,
        "vocab": vocab,
        "embed_outside": embed_outside,
        "num_rows": n,
        "logrows": logrows_for(n),
    }


def main():
    ap = argparse.ArgumentParser(
        description=None, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--d-model", type=int, default=128)
    ap.add_argument("--t-max", type=int, default=16)
    ap.add_argument("--vocab", type=int, default=256)
    ap.add_argument("--skip-rows", action="store_true")
    ap.add_argument(
        "--embed-outside",
        action="store_true",
        help="把嵌入查表移出电路。T1 随之换成 T1a+T1b，"
        "行数扫描换成两个变体的 A/B 对照",
    )
    ap.add_argument("--workdir", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    assert args.steps <= args.t_max
    if args.out is None:
        args.out = str(
            REPO
            / "results"
            / "incremental"
            / ("p3_embed_outside" if args.embed_outside else "p3_chain")
        )

    for b in (EZKL, COMMIT_BIN):
        if not b.exists():
            sys.exit(
                f"missing {b}\n(build with: cargo build --release --example kzg_commit_vector)"
            )
    wd = (
        Path(args.workdir)
        if args.workdir
        else Path(tempfile.mkdtemp(prefix="ezkl_p3_"))
    )
    wd.mkdir(parents=True, exist_ok=True)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    print(f"[workdir] {wd}")

    rows = []
    if not args.skip_rows:
        print("\n=== A. vocab 扫描（gen-settings only）===")
        for v in VOCAB_SCAN:
            r = measure_rows(wd, args.d_model, args.t_max, v)
            rows.append(r)
            print(
                f"  d={r['d_model']} T={r['t_max']} V={v:5d} one-hot     -> "
                f"rows={r['num_rows']:9d} k={r['logrows']}"
            )
            if args.embed_outside and v:
                e = measure_rows(wd, args.d_model, args.t_max, v, embed_outside=True)
                e["delta_vs_onehot"] = e["num_rows"] / r["num_rows"] - 1.0
                rows.append(e)
                print(
                    f"  d={e['d_model']} T={e['t_max']} V={v:5d} embed-out   -> "
                    f"rows={e['num_rows']:9d} k={e['logrows']}  "
                    f"({e['delta_vs_onehot']*100:+.1f}%)"
                )
        if not args.embed_outside:
            for v in (0, args.vocab):
                r = measure_rows(wd, 384, 128, v)
                rows.append(r)
                print(
                    f"  d=384 T=128 V={v:5d} -> rows={r['num_rows']:9d} k={r['logrows']}"
                )
        (out / "p3_vocab_rows.json").write_text(json.dumps(rows, indent=2) + "\n")

    print(
        f"\n=== B. {args.steps} 步链（d={args.d_model}, T_max={args.t_max}, V={args.vocab}）==="
    )
    cdir = wd / "chain"
    gen_extra = ["--embed-outside"] if args.embed_outside else []
    run(
        sys.executable,
        GEN,
        "--variant",
        "where",
        "--d-model",
        args.d_model,
        "--t-max",
        args.t_max,
        "--vocab",
        args.vocab,
        *gen_extra,
        "--out-dir",
        cdir,
        cwd=REPO,
    )
    shapes = json.loads((cdir / "input.json").read_text())["input_shapes"]

    vis = [
        "--input-visibility",
        "polycommit",
        "--output-visibility",
        "polycommit",
        "--param-visibility",
        "fixed",
    ]
    run(
        EZKL,
        "gen-settings",
        "-M",
        "model.onnx",
        "-O",
        "settings.json",
        "-K",
        24,
        *vis,
        cwd=cdir,
    )
    n_rows = json.loads((cdir / "settings.json").read_text())["num_rows"]
    k = logrows_for(n_rows)
    run(
        EZKL,
        "gen-settings",
        "-M",
        "model.onnx",
        "-O",
        "settings.json",
        "-K",
        k,
        *vis,
        cwd=cdir,
    )
    run(
        EZKL,
        "compile-circuit",
        "-M",
        "model.onnx",
        "-S",
        "settings.json",
        "--compiled-circuit",
        "circuit.compiled",
        cwd=cdir,
    )
    srs = f"kzg{k}.srs"
    run(EZKL, "gen-srs", "--logrows", k, "--srs-path", srs, cwd=cdir)
    run(
        EZKL,
        "setup",
        "-M",
        "circuit.compiled",
        "--srs-path",
        srs,
        "--vk-path",
        "vk.key",
        "--pk-path",
        "pk.key",
        cwd=cdir,
    )
    print(f"[circuit] num_rows={n_rows} logrows={k}; single pk for all steps")

    wte = None
    if args.embed_outside:
        w = json.loads((cdir / "wte.json").read_text())
        assert w["shape"] == [args.vocab, args.d_model], w["shape"]
        wte = [
            w["data"][i * args.d_model : (i + 1) * args.d_model]
            for i in range(args.vocab)
        ]

    def token_tensor(tok):
        tok %= args.vocab
        if args.embed_outside:
            return list(wte[tok])
        v = [0.0] * args.vocab
        v[tok] = 1.0
        return v

    def write_input(path, tok, kc, vc, t):
        mw, ma = masks_for(t, args.t_max)
        (cdir / path).write_text(
            json.dumps(
                {
                    "input_data": [token_tensor(tok), kc, vc, mw, ma],
                    "input_shapes": shapes,
                }
            )
        )

    cache_len = 1
    for d in shapes[1]:
        cache_len *= d
    zeros = [0.0] * cache_len

    kc, vc = zeros, zeros
    tokens, steps, logits_seq = [7], [], []
    for t in range(args.steps):
        write_input(f"input_{t}.json", tokens[t], kc, vc, t)
        run(
            EZKL,
            "gen-witness",
            "-M",
            "circuit.compiled",
            "-D",
            f"input_{t}.json",
            "-O",
            f"witness_{t}.json",
            "--vk-path",
            "vk.key",
            "--srs-path",
            srs,
            cwd=cdir,
        )
        w = json.loads((cdir / f"witness_{t}.json").read_text())
        run(
            EZKL,
            "prove",
            "-W",
            f"witness_{t}.json",
            "-M",
            "circuit.compiled",
            "--pk-path",
            "pk.key",
            "--proof-path",
            f"proof_{t}.json",
            "--srs-path",
            srs,
            cwd=cdir,
        )
        kc, vc = w["outputs"][1], w["outputs"][2]
        lg = w["outputs"][3]
        logits_seq.append(lg)
        (cdir / f"logits_{t}.json").write_text(json.dumps(lg))
        nxt = max(range(len(lg)), key=lambda i: felt_to_int(lg[i]))
        tokens.append(nxt)
        steps.append({"t": t, "proof": f"proof_{t}.json", "input": f"input_{t}.json"})
        print(f"[step {t}] tok={tokens[t]:4d} -> argmax(logits)={nxt:4d}")

    write_input("input_anchor.json", tokens[0], zeros, zeros, 0)
    mask_inputs, tok_inputs = {}, {}
    for t in range(args.steps):
        write_input(f"input_mask_{t}.json", tokens[0], zeros, zeros, t)
        mask_inputs[str(t)] = f"input_mask_{t}.json"
        write_input(f"input_tok_{t}.json", tokens[t], zeros, zeros, 0)
        tok_inputs[str(t)] = f"input_tok_{t}.json"
    (cdir / "manifest.json").write_text(
        json.dumps(
            {
                "settings": "settings.json",
                "vk": "vk.key",
                "srs": srs,
                "compiled": "circuit.compiled",
                "layout": LAYOUT,
                "steps": steps,
                "zero_anchor_input": "input_anchor.json",
                "mask_inputs": mask_inputs,
            },
            indent=2,
        )
    )

    print("\n--- verify_chain (P2 的四项) ---")
    rep = run(
        sys.executable,
        CHECKER,
        "--manifest",
        "manifest.json",
        "--json-out",
        str(out / "p3_chain_report.json"),
        cwd=cdir,
        check=False,
    )
    print(rep.strip())
    base_ok = json.loads((out / "p3_chain_report.json").read_text())["all_passed"]

    print("\n--- token 绑定 ---")

    def expected_input_commit(ref_input, idx, tag):
        run(
            EZKL,
            "gen-witness",
            "-M",
            "circuit.compiled",
            "-D",
            ref_input,
            "-O",
            f"_exp_{tag}.json",
            "--vk-path",
            "vk.key",
            "--srs-path",
            srs,
            cwd=cdir,
        )
        w = json.loads((cdir / f"_exp_{tag}.json").read_text())
        flat = [h for g in w["processed_inputs"]["polycommit"] for h in g]
        b = bytearray(bytes.fromhex(flat[idx])[::-1])
        b[0] &= 0x7F
        return bytes(b)

    def commit_vector(vec_path):
        o = run(
            COMMIT_BIN,
            "--settings",
            "settings.json",
            "--vk",
            "vk.key",
            "--srs",
            srs,
            "--input",
            vec_path,
            cwd=cdir,
        )
        return [bytes.fromhex(l.strip()) for l in o.strip().splitlines() if l.strip()]

    st = json.loads((cdir / "settings.json").read_text())
    in_scale = st.get("model_input_scales") or [st["run_args"]["input_scale"]]
    in_scale = in_scale[0]

    def direct_commit(tok, tag, witness_felts):
        felts = [int_to_felt(quantize(x, in_scale)) for x in token_tensor(tok)]
        p = cdir / f"_tokvec_{tag}.json"
        p.write_text(json.dumps(felts))
        pts = commit_vector(p.name)
        return pts, len(felts), felts == list(witness_felts)

    t1, t1b, t2 = [], [], []
    for i, s in enumerate(steps):
        c = commitments(cdir / s["proof"], LOGITS_IDX + 1)
        exp = expected_input_commit(tok_inputs[str(s["t"])], TOK_IDX, f"tok{s['t']}")
        t1.append(
            {"t": s["t"], "token": tokens[s["t"]], "match": c[TOK_IDX][:32] == exp}
        )
        wf = json.loads((cdir / f"witness_{s['t']}.json").read_text())["inputs"][0]
        pts, n_elems, quant_ok = direct_commit(tokens[s["t"]], f"tok{s['t']}", wf)
        t1b.append(
            {
                "t": s["t"],
                "token": tokens[s["t"]],
                "verifier_elements": n_elems,
                "quantization_matches_witness": quant_ok,
                "match": len(pts) == 1 and pts[0] == c[TOK_IDX],
            }
        )
        wrong = (tokens[s["t"]] + 1) % args.vocab
        w_pts, _, _ = direct_commit(wrong, f"wrong{s['t']}", [])
        t1b[-1]["wrong_token_rejected"] = not (
            len(w_pts) == 1 and w_pts[0] == c[TOK_IDX]
        )
        pts = commit_vector(f"logits_{s['t']}.json")
        bound = len(pts) == 1 and pts[0] == c[LOGITS_IDX]
        greedy = tokens[s["t"] + 1] == max(
            range(len(logits_seq[i])), key=lambda j: felt_to_int(logits_seq[i][j])
        )
        t2.append(
            {
                "t": s["t"],
                "logits_commitment_matches": bound,
                "next_token_is_argmax": greedy,
                "next_token": tokens[s["t"] + 1],
            }
        )

    t1_ok = all(r["match"] for r in t1)
    t1b_ok = all(
        r["match"] and r["quantization_matches_witness"] and r["wrong_token_rejected"]
        for r in t1b
    )
    t2_ok = all(
        r["logits_commitment_matches"] and r["next_token_is_argmax"] for r in t2
    )
    kind = "wte[tok]" if args.embed_outside else "one-hot"
    print(
        f"  T1a {kind:8s} token 绑定 : {sum(r['match'] for r in t1)}/{len(t1)} -> "
        f"{'PASS' if t1_ok else 'FAIL'}"
    )
    print(
        f"  T1b 验证方直算 Com(x_t)  : "
        f"{sum(r['match'] for r in t1b)}/{len(t1b)} 承诺吻合, "
        f"{sum(r['quantization_matches_witness'] for r in t1b)}/{len(t1b)} 量化吻合, "
        f"{sum(r['wrong_token_rejected'] for r in t1b)}/{len(t1b)} 错 token 被拒, "
        f"每步 {t1b[0]['verifier_elements']} 个元素 -> {'PASS' if t1b_ok else 'FAIL'}"
    )
    print(
        f"  T2 logits 承诺 + 贪心 : "
        f"{sum(r['logits_commitment_matches'] for r in t2)}/{len(t2)} 承诺吻合, "
        f"{sum(r['next_token_is_argmax'] for r in t2)}/{len(t2)} argmax 吻合 -> "
        f"{'PASS' if t2_ok else 'FAIL'}"
    )

    summary = {
        "config": {
            "d_model": args.d_model,
            "t_max": args.t_max,
            "vocab": args.vocab,
            "steps": args.steps,
            "num_rows": n_rows,
            "logrows": k,
            "single_pk_reused": True,
            "embed_outside": args.embed_outside,
            "input_scale": in_scale,
        },
        "token_sequence": tokens,
        "chain_checks_passed": base_ok,
        "token_binding": {
            "T1a_token_tensor_commit": {"kind": kind, "detail": t1, "passed": t1_ok},
            "T1b_verifier_direct_commit": {"detail": t1b, "passed": t1b_ok},
            "T2_logits_commit_and_greedy": {"detail": t2, "passed": t2_ok},
        },
        "vocab_rows": rows,
        "all_passed": bool(base_ok and t1_ok and t1b_ok and t2_ok),
    }
    (out / "p3_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"\n[wrote] {out/'p3_summary.json'}  all_passed={summary['all_passed']}")
    print(f"[workdir kept] {wd}")
    return 0 if summary["all_passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
