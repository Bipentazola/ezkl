#!/usr/bin/env python3

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
EZKL = REPO / "target" / "release" / "ezkl"
GEN = REPO / "scripts" / "gen_decode_step_onnx.py"
CHECKER = REPO / "scripts" / "verify_chain.py"
POINT = 64

LAYOUT = {
    "cache_in_idx": [1, 2],
    "cache_out_idx": [6, 7],
    "mask_idx": {"m_write": 3, "m_attn": 4},
    "unchecked_input_idx": [0],
}


def run(*args, cwd, check=True, env=None):
    r = subprocess.run(
        [str(a) for a in args], cwd=cwd, capture_output=True, text=True, env=env
    )
    if check and r.returncode != 0:
        sys.exit(f"command failed: {' '.join(map(str, args))}\n{r.stdout}\n{r.stderr}")
    return r.stdout + r.stderr


def verified(out):
    if "verified: true" in out:
        return True
    if "verified: false" in out:
        return False
    return None


def masks_for(t, t_max):
    m_write = [0.0] * t_max
    m_write[t] = 1.0
    m_attn = [1.0 if i <= t else 0.0 for i in range(t_max)]
    return m_write, m_attn


def main():
    ap = argparse.ArgumentParser(
        description=None, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--steps", type=int, default=3)
    ap.add_argument("--d-model", type=int, default=128)
    ap.add_argument("--t-max", type=int, default=16)
    ap.add_argument("--workdir", default=None)
    ap.add_argument("--out", default=str(REPO / "results" / "incremental" / "p2_chain"))
    args = ap.parse_args()
    assert args.steps <= args.t_max

    if not EZKL.exists():
        sys.exit(f"missing {EZKL}")
    wd = (
        Path(args.workdir)
        if args.workdir
        else Path(tempfile.mkdtemp(prefix="ezkl_p2_"))
    )
    wd.mkdir(parents=True, exist_ok=True)
    print(f"[workdir] {wd}")

    run(
        sys.executable,
        GEN,
        "--variant",
        "where",
        "--d-model",
        args.d_model,
        "--t-max",
        args.t_max,
        "--position",
        0,
        "--out-dir",
        wd,
        cwd=REPO,
    )
    base = json.loads((wd / "input.json").read_text())
    shapes = base["input_shapes"]

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
        cwd=wd,
    )
    num_rows = json.loads((wd / "settings.json").read_text())["num_rows"]
    k = max(1, (num_rows - 1).bit_length())
    print(f"[circuit] num_rows={num_rows} -> logrows={k}")
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
        cwd=wd,
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
        cwd=wd,
    )
    srs = f"kzg{k}.srs"
    run(EZKL, "gen-srs", "--logrows", k, "--srs-path", srs, cwd=wd)
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
        cwd=wd,
    )
    print("[setup] done -- this single pk is reused by every step")

    def witness(data, out):
        run(
            EZKL,
            "gen-witness",
            "-M",
            "circuit.compiled",
            "-D",
            data,
            "-O",
            out,
            "--vk-path",
            "vk.key",
            "--srs-path",
            srs,
            cwd=wd,
        )
        return json.loads((wd / out).read_text())

    def prove(wit, out):
        run(
            EZKL,
            "prove",
            "-W",
            wit,
            "-M",
            "circuit.compiled",
            "--pk-path",
            "pk.key",
            "--proof-path",
            out,
            "--srs-path",
            srs,
            cwd=wd,
        )

    def verify(proof):
        return verified(
            run(
                EZKL,
                "verify",
                "--proof-path",
                proof,
                "-S",
                "settings.json",
                "--vk-path",
                "vk.key",
                "--srs-path",
                srs,
                cwd=wd,
                check=False,
            )
        )

    def write_input(path, x_t, k_cache, v_cache, t):
        mw, ma = masks_for(t, args.t_max)
        (wd / path).write_text(
            json.dumps(
                {
                    "input_data": [x_t, k_cache, v_cache, mw, ma],
                    "input_shapes": shapes,
                }
            )
        )

    def zeros(n):
        return [0.0] * n

    cache_len = base["input_shapes"][1]
    cache_len = 1
    for d in shapes[1]:
        cache_len *= d

    x_seq = [base["input_data"][0]]
    kc, vc = zeros(cache_len), zeros(cache_len)
    steps = []
    for t in range(args.steps):
        write_input(f"input_{t}.json", x_seq[t], kc, vc, t)
        w = witness(f"input_{t}.json", f"witness_{t}.json")
        prove(f"witness_{t}.json", f"proof_{t}.json")
        kc, vc = w["outputs"][1], w["outputs"][2]
        if t + 1 < args.steps:
            x_seq.append([v * 0.5 for v in x_seq[t]])
        steps.append({"t": t, "proof": f"proof_{t}.json", "input": f"input_{t}.json"})
        print(f"[step {t}] witness + proof done")

    write_input("input_anchor.json", x_seq[0], zeros(cache_len), zeros(cache_len), 0)
    mask_inputs = {}
    for t in range(args.steps):
        write_input(
            f"input_mask_{t}.json", x_seq[0], zeros(cache_len), zeros(cache_len), t
        )
        mask_inputs[str(t)] = f"input_mask_{t}.json"

    manifest = {
        "settings": "settings.json",
        "vk": "vk.key",
        "srs": srs,
        "compiled": "circuit.compiled",
        "layout": LAYOUT,
        "steps": steps,
        "zero_anchor_input": "input_anchor.json",
        "mask_inputs": mask_inputs,
    }
    (wd / "manifest.json").write_text(json.dumps(manifest, indent=2))

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    print("\n=== verify_chain ===")
    rep = run(
        sys.executable,
        CHECKER,
        "--manifest",
        "manifest.json",
        "--json-out",
        str(out / "p2_chain_report.json"),
        cwd=wd,
        check=False,
    )
    print(rep.strip())
    chain_ok = json.loads((out / "p2_chain_report.json").read_text())["all_passed"]

    print("\n=== negative tests ===")
    neg = {}

    def commits(p):
        b = bytes(json.loads((wd / p).read_text())["proof"])
        return [b[i * POINT : (i + 1) * POINT] for i in range(8)]

    w0 = json.loads((wd / "witness_0.json").read_text())
    bad_kc = list(w0["outputs"][1])
    bad_kc[0] = ("01" if not bad_kc[0].startswith("01") else "02") + bad_kc[0][2:]
    write_input("input_1_forged.json", x_seq[1], bad_kc, w0["outputs"][2], 1)
    witness("input_1_forged.json", "witness_1_forged.json")
    prove("witness_1_forged.json", "proof_1_forged.json")
    n1_self = verify("proof_1_forged.json")
    n1_link = commits("proof_0.json")[6] == commits("proof_1_forged.json")[1]
    neg["N1_forged_cache"] = {
        "forged_proof_self_verifies": n1_self,
        "link_check_accepts": n1_link,
        "passed": bool(n1_self) and not n1_link,
    }
    print(
        f"  N1 断链    伪造证明自身 verify={n1_self}, 链接检查接受={n1_link} "
        f"-> {'PASS' if neg['N1_forged_cache']['passed'] else 'FAIL'}"
    )

    p = json.loads((wd / "proof_1_forged.json").read_text())
    b = bytearray(p["proof"])
    c0 = commits("proof_0.json")
    b[1 * POINT : 2 * POINT] = c0[6]
    b[2 * POINT : 3 * POINT] = c0[7]
    p["proof"] = list(b)
    p["hex_proof"] = "0x" + bytes(b).hex()
    (wd / "proof_1_spliced.json").write_text(json.dumps(p))
    n2_link = bytes(b)[1 * POINT : 2 * POINT] == c0[6]
    n2_verify = verify("proof_1_spliced.json")
    neg["N2_spliced_commitment"] = {
        "link_check_now_accepts": n2_link,
        "verify": n2_verify,
        "passed": n2_link and n2_verify is False,
    }
    print(
        f"  N2 拼接    链接检查被骗过={n2_link}, verify={n2_verify} "
        f"-> {'PASS' if neg['N2_spliced_commitment']['passed'] else 'FAIL'}"
    )

    w = json.loads((wd / "witness_0.json").read_text())
    w["outputs"][1] = list(w["outputs"][1])
    w["outputs"][1][0] = (
        "01" if not w["outputs"][1][0].startswith("01") else "02"
    ) + w["outputs"][1][0][2:]
    (wd / "witness_0_badout.json").write_text(json.dumps(w))
    r = subprocess.run(
        [
            str(EZKL),
            "prove",
            "-W",
            "witness_0_badout.json",
            "-M",
            "circuit.compiled",
            "--pk-path",
            "pk.key",
            "--proof-path",
            "proof_0_badout.json",
            "--srs-path",
            srs,
        ],
        cwd=wd,
        capture_output=True,
        text=True,
    )
    n3_prove_ok = r.returncode == 0
    n3_verify = verify("proof_0_badout.json") if n3_prove_ok else None
    n3_same = (
        commits("proof_0.json") == commits("proof_0_badout.json")
        if n3_prove_ok
        else None
    )
    neg["N3_tampered_output_witness"] = {
        "prove_succeeded": n3_prove_ok,
        "verify": n3_verify,
        "all_commitments_unchanged": n3_same,
        "prove_stderr_tail": (r.stdout + r.stderr)[-300:] if not n3_prove_ok else None,
        "passed": (not n3_prove_ok) or bool(n3_same),
    }
    print(
        f"  N3 输出侧  prove 成功={n3_prove_ok}, 承诺全部未变={n3_same} "
        f"-> {'PASS' if neg['N3_tampered_output_witness']['passed'] else 'FAIL'}"
    )

    summary = {
        "config": {
            "d_model": args.d_model,
            "t_max": args.t_max,
            "steps": args.steps,
            "num_rows": num_rows,
            "logrows": k,
            "single_pk_reused": True,
        },
        "chain_check_passed": chain_ok,
        "negative_tests": neg,
        "all_passed": bool(chain_ok) and all(v["passed"] for v in neg.values()),
    }
    (out / "p2_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"\n[wrote] {out/'p2_summary.json'}  all_passed={summary['all_passed']}")
    print(f"[workdir kept] {wd}")
    return 0 if summary["all_passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
