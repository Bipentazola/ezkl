#!/usr/bin/env python3

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
EZKL = REPO / "target" / "release" / "ezkl"
MODEL = REPO / "examples" / "onnx" / "1l_sigmoid"

POINT = 64
IN_COMMIT = slice(0, POINT)
OUT_COMMIT = slice(POINT, 2 * POINT)


def run(*args, cwd, check=True):
    r = subprocess.run([str(a) for a in args], cwd=cwd, capture_output=True, text=True)
    if check and r.returncode != 0:
        sys.exit(f"command failed: {' '.join(map(str, args))}\n{r.stdout}\n{r.stderr}")
    return r.stdout + r.stderr


def verified(out: str) -> bool:
    if "verified: true" in out:
        return True
    if "verified: false" in out:
        return False
    sys.exit(f"could not parse verify output:\n{out}")


def proof_bytes(p: Path) -> bytes:
    return bytes(json.loads(p.read_text())["proof"])


def witness_commits(p: Path, key: str) -> list:
    w = json.loads(p.read_text())
    return [h for grp in w[key]["polycommit"] for h in grp]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--workdir",
        default=None,
        help="放 srs/pk/proof 等大文件；默认用临时目录并在结束时保留路径",
    )
    ap.add_argument("--out", default=str(REPO / "results" / "incremental" / "p0_link"))
    args = ap.parse_args()

    if not EZKL.exists():
        sys.exit(f"missing {EZKL}; build with `cargo build --release` first")

    wd = (
        Path(args.workdir)
        if args.workdir
        else Path(tempfile.mkdtemp(prefix="ezkl_p0_"))
    )
    wd.mkdir(parents=True, exist_ok=True)
    shutil.copy(MODEL / "network.onnx", wd / "network.onnx")
    shutil.copy(MODEL / "input.json", wd / "input_0.json")
    print(f"[workdir] {wd}")

    vis = [
        "--input-visibility",
        "polycommit",
        "--output-visibility",
        "polycommit",
        "--param-visibility",
        "fixed",
    ]
    results = {}

    run(
        EZKL,
        "gen-settings",
        "-M",
        "network.onnx",
        "-O",
        "settings.json",
        "-K",
        17,
        *vis,
        cwd=wd,
    )
    run(
        EZKL,
        "compile-circuit",
        "-M",
        "network.onnx",
        "-S",
        "settings.json",
        "--compiled-circuit",
        "network.compiled",
        cwd=wd,
    )
    run(EZKL, "gen-srs", "--logrows", 17, "--srs-path", "kzg17.srs", cwd=wd)
    run(
        EZKL,
        "setup",
        "-M",
        "network.compiled",
        "--srs-path",
        "kzg17.srs",
        "--vk-path",
        "vk.key",
        "--pk-path",
        "pk.key",
        cwd=wd,
    )

    def witness(data, out):
        run(
            EZKL,
            "gen-witness",
            "-M",
            "network.compiled",
            "-D",
            data,
            "-O",
            out,
            "--vk-path",
            "vk.key",
            "--srs-path",
            "kzg17.srs",
            cwd=wd,
        )

    def prove(wit, out):
        run(
            EZKL,
            "prove",
            "-W",
            wit,
            "-M",
            "network.compiled",
            "--pk-path",
            "pk.key",
            "--proof-path",
            out,
            "--srs-path",
            "kzg17.srs",
            cwd=wd,
        )

    def verify(proof, settings="settings.json", vk="vk.key", srs="kzg17.srs"):
        return verified(
            run(
                EZKL,
                "verify",
                "--proof-path",
                proof,
                "-S",
                settings,
                "--vk-path",
                vk,
                "--srs-path",
                srs,
                cwd=wd,
                check=False,
            )
        )

    witness("input_0.json", "witness_0.json")
    prove("witness_0.json", "proof_0.json")

    w0 = json.loads((wd / "witness_0.json").read_text())
    (wd / "input_1.json").write_text(json.dumps(dict(input_data=w0["outputs"])))
    witness("input_1.json", "witness_1.json")
    prove("witness_1.json", "proof_1.json")

    b0, b1 = proof_bytes(wd / "proof_0.json"), proof_bytes(wd / "proof_1.json")

    c1 = witness_commits(wd / "witness_0.json", "processed_outputs") == witness_commits(
        wd / "witness_1.json", "processed_inputs"
    )

    v0, v1 = verify("proof_0.json"), verify("proof_1.json")
    c2 = (b0[OUT_COMMIT] == b1[IN_COMMIT]) and v0 and v1

    def x_of(compressed_hex: str) -> bytes:
        b = bytearray(bytes.fromhex(compressed_hex)[::-1])
        b[0] &= 0x7F
        return bytes(b)

    x_in = x_of(witness_commits(wd / "witness_0.json", "processed_inputs")[0])
    x_out = x_of(witness_commits(wd / "witness_0.json", "processed_outputs")[0])
    r3 = (b0[0:32] == x_in) and (b0[POINT : POINT + 32] == x_out)

    prove("witness_0.json", "proof_0_rerun.json")
    b0r = proof_bytes(wd / "proof_0_rerun.json")
    diff = [i for i in range(min(len(b0), len(b0r))) if b0[i] != b0r[i]]
    c3 = (
        (b0[: 2 * POINT] == b0r[: 2 * POINT]) and len(diff) > 0 and diff[0] >= 2 * POINT
    )

    wrong = [row[:] for row in w0["outputs"]]
    wrong[0][0] = "41" + wrong[0][0][2:]
    (wd / "input_1_wrong.json").write_text(json.dumps(dict(input_data=wrong)))
    witness("input_1_wrong.json", "witness_1_wrong.json")
    prove("witness_1_wrong.json", "proof_1_wrong.json")
    bw = proof_bytes(wd / "proof_1_wrong.json")
    c4_self_valid = verify("proof_1_wrong.json")
    c4 = c4_self_valid and (b0[OUT_COMMIT] != bw[IN_COMMIT])

    p = json.loads((wd / "proof_1_wrong.json").read_text())
    tampered = bytearray(p["proof"])
    tampered[IN_COMMIT] = b0[OUT_COMMIT]
    p["proof"] = list(tampered)
    p["hex_proof"] = "0x" + bytes(tampered).hex()
    (wd / "proof_1_tampered.json").write_text(json.dumps(p))
    bt = proof_bytes(wd / "proof_1_tampered.json")
    c5_link_now_passes = b0[OUT_COMMIT] == bt[IN_COMMIT]
    c5 = c5_link_now_passes and (verify("proof_1_tampered.json") is False)

    run(
        EZKL,
        "gen-settings",
        "-M",
        "network.onnx",
        "-O",
        "settings_k18.json",
        "-K",
        18,
        *vis,
        cwd=wd,
    )
    run(
        EZKL,
        "compile-circuit",
        "-M",
        "network.onnx",
        "-S",
        "settings_k18.json",
        "--compiled-circuit",
        "network_k18.compiled",
        cwd=wd,
    )
    run(EZKL, "gen-srs", "--logrows", 18, "--srs-path", "kzg18.srs", cwd=wd)
    run(
        EZKL,
        "setup",
        "-M",
        "network_k18.compiled",
        "--srs-path",
        "kzg18.srs",
        "--vk-path",
        "vk18.key",
        "--pk-path",
        "pk18.key",
        cwd=wd,
    )
    run(
        EZKL,
        "gen-witness",
        "-M",
        "network_k18.compiled",
        "-D",
        "input_0.json",
        "-O",
        "witness_0_k18.json",
        "--vk-path",
        "vk18.key",
        "--srs-path",
        "kzg18.srs",
        cwd=wd,
    )
    c17 = witness_commits(wd / "witness_0.json", "processed_inputs")
    c18 = witness_commits(wd / "witness_0_k18.json", "processed_inputs")
    c6 = c17 != c18

    (wd / "input_zero.json").write_text(json.dumps(dict(input_data=[[0.0, 0.0, 0.0]])))
    witness("input_zero.json", "witness_zero.json")
    anchor = witness_commits(wd / "witness_zero.json", "processed_inputs")

    results = {
        "model": "examples/onnx/1l_sigmoid (shape [3]->[3], non-idempotent)",
        "logrows": 17,
        "point_bytes": POINT,
        "checks": {
            "C1_link_witness_level": c1,
            "C2_link_proof_bytes_and_both_verify": c2,
            "C3_commitment_deterministic_rest_is_not": c3,
            "C4_wrong_successor_rejected_by_link": c4,
            "C5_spliced_commitment_fails_verification": c5,
            "C6_same_tensor_different_logrows_differs": c6,
            "R3_commitments_are_the_proof_prefix": r3,
        },
        "evidence": {
            "proof_len_bytes": len(b0),
            "step0_in_commit": b0[IN_COMMIT].hex(),
            "step0_out_commit": b0[OUT_COMMIT].hex(),
            "step1_in_commit": b1[IN_COMMIT].hex(),
            "step1_out_commit": b1[OUT_COMMIT].hex(),
            "rerun_first_differing_byte": diff[0] if diff else None,
            "rerun_num_differing_bytes": len(diff),
            "wrong_successor_in_commit": bw[IN_COMMIT].hex(),
            "wrong_successor_self_verifies": c4_self_valid,
            "tampered_passes_link_check": c5_link_now_passes,
            "commit_x0_k17": c17,
            "commit_x0_k18": c18,
            "anchor_commit_zero_tensor_k17": anchor,
        },
    }

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "p0_results.json").write_text(json.dumps(results, indent=2) + "\n")

    print()
    for k, v in results["checks"].items():
        print(f"  {'PASS' if v else 'FAIL'}  {k}")
    print(f"\n[wrote] {out / 'p0_results.json'}")
    print(f"[workdir kept] {wd}")
    return 0 if all(results["checks"].values()) else 1


if __name__ == "__main__":
    sys.exit(main())
