#!/usr/bin/env python3

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
EZKL = REPO / "target" / "release" / "ezkl"
MODEL = REPO / "models" / "seq8_d16_l1"
RPC = "http://127.0.0.1:8545"
PK = "ac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80"

VARIANTS = {
    "base": ("0", "0"),
    "div": ("1", "0"),
    "fusion": ("0", "1"),
    "both": ("1", "1"),
}


def sh(*args, cwd=None, check=True, env_extra=None):
    e = dict(os.environ)
    e.setdefault("NO_PROXY", "localhost,127.0.0.1")
    e.setdefault("no_proxy", "localhost,127.0.0.1")
    e["PATH"] = "/data/foundry/bin:" + e.get("PATH", "")
    for v in ("EZKL_DIV_ROUND_NEAREST", "EZKL_RESCALE_FUSION"):
        e.pop(v, None)
    if env_extra:
        e.update(env_extra)
    r = subprocess.run(
        [str(a) for a in args], cwd=cwd, capture_output=True, text=True, env=e
    )
    if check and r.returncode != 0:
        sys.exit(f"failed: {' '.join(map(str, args))}\n{r.stdout}\n{r.stderr}")
    return (r.stdout + r.stderr).strip()


def frame_gas(txhash, cwd):
    out = sh("cast", "run", txhash, "--rpc-url", RPC, cwd=cwd, check=False)
    for line in out.split("\n"):
        m = re.search(r"\[(\d+)\]\s+0x[0-9a-fA-F]{40}::", line)
        if m:
            return int(m.group(1))
    return None


def build_variant(name, wd, srs_src):
    d, f = VARIANTS[name]
    env = {"EZKL_DIV_ROUND_NEAREST": d, "EZKL_RESCALE_FUSION": f}
    vd = wd / name
    vd.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    sh(
        EZKL,
        "gen-settings",
        "-M",
        MODEL / "model.onnx",
        "-O",
        "settings.json",
        "--input-scale",
        13,
        "--param-scale",
        13,
        cwd=vd,
        env_extra=env,
    )
    st = json.loads((vd / "settings.json").read_text())
    rows, k = st["num_rows"], st["run_args"]["logrows"]

    srs = vd / f"kzg{k}.srs"
    src = srs_src.get(k)
    if src and Path(src).exists():
        os.link(src, srs) if not srs.exists() else None
    else:
        sh(EZKL, "gen-srs", "--logrows", k, "--srs-path", srs, cwd=vd, env_extra=env)
        srs_src[k] = str(srs)

    sh(
        EZKL,
        "compile-circuit",
        "-M",
        MODEL / "model.onnx",
        "-S",
        "settings.json",
        "--compiled-circuit",
        "circuit.compiled",
        cwd=vd,
        env_extra=env,
    )
    sh(
        EZKL,
        "gen-witness",
        "-M",
        "circuit.compiled",
        "-D",
        MODEL / "input.json",
        "-O",
        "witness.json",
        cwd=vd,
        env_extra=env,
    )
    sh(
        EZKL,
        "setup",
        "-M",
        "circuit.compiled",
        "--srs-path",
        srs.name,
        "--vk-path",
        "vk.key",
        "--pk-path",
        "pk.key",
        cwd=vd,
        env_extra=env,
    )
    sh(
        EZKL,
        "prove",
        "-W",
        "witness.json",
        "-M",
        "circuit.compiled",
        "--pk-path",
        "pk.key",
        "--proof-path",
        "proof.json",
        "--srs-path",
        srs.name,
        cwd=vd,
        env_extra=env,
    )
    off = sh(
        EZKL,
        "verify",
        "--proof-path",
        "proof.json",
        "--settings-path",
        "settings.json",
        "--vk-path",
        "vk.key",
        "--srs-path",
        srs.name,
        cwd=vd,
        env_extra=env,
    )

    sh(
        EZKL,
        "create-evm-verifier",
        "-R",
        "-S",
        "settings.json",
        "--vk-path",
        "vk.key",
        "--srs-path",
        srs.name,
        "--sol-code-path",
        "verifier.sol",
        "--abi-path",
        "verifier_abi.json",
        cwd=vd,
        env_extra=env,
    )
    sh(
        EZKL,
        "create-evm-vka",
        "-S",
        "settings.json",
        "--vk-path",
        "vk.key",
        "--srs-path",
        srs.name,
        "--vka-path",
        "vka.bytes",
        cwd=vd,
        env_extra=env,
    )
    sh(
        EZKL,
        "encode-evm-calldata",
        "--proof-path",
        "proof.json",
        "--calldata-path",
        "calldata.bytes",
        "--vka-path",
        "vka.bytes",
        cwd=vd,
        env_extra=env,
    )

    pk = vd / "pk.key"
    pk_bytes = pk.stat().st_size
    pk.unlink()

    return {
        "variant": name,
        "switches": {"div_round_nearest": d, "rescale_fusion": f},
        "num_rows": rows,
        "logrows": k,
        "offchain_verified": "verified: true" in off.lower(),
        "verifier_sol_bytes": (vd / "verifier.sol").stat().st_size,
        "vka_bytes": (vd / "vka.bytes").stat().st_size,
        "calldata_bytes": (vd / "calldata.bytes").stat().st_size,
        "pk_bytes_deleted": pk_bytes,
        "build_wall_s": round(time.time() - t0, 1),
    }, vd


def measure_on_chain(name, vd, rec):
    sh(
        EZKL,
        "deploy-evm",
        "--sol-code-path",
        "verifier.sol",
        "-U",
        RPC,
        "--addr-path",
        "addr.txt",
        "-P",
        PK,
        "-C",
        "verifier/reusable",
        cwd=vd,
    )
    addr = (vd / "addr.txt").read_text().strip()
    rec["deployed_bytecode_bytes"] = (
        len(sh("cast", "code", addr, "--rpc-url", RPC, cwd=vd)) // 2
    )

    reg = sh(
        EZKL,
        "register-vka",
        "-U",
        RPC,
        "--addr-verifier",
        addr,
        "--vka-path",
        "vka.bytes",
        "--vka-digest-path",
        "vka_digest.txt",
        "-P",
        PK,
        cwd=vd,
    )
    m = re.search(r"(\d+)", reg.split("gas")[-1]) if "gas" in reg else None
    rec["vka_registration_gas_reported"] = int(m.group(1)) if m else None

    o = sh(
        EZKL,
        "verify-evm",
        "--proof-path",
        "proof.json",
        "--addr-verifier",
        addr,
        "-U",
        RPC,
        "--vka-path",
        "vka.bytes",
        cwd=vd,
    )
    rec["on_chain_verified"] = "verification result: true" in o
    m = re.search(r"estimated verify gas cost: (\d+)", o)
    rec["verify_gas_ezkl_estimate"] = int(m.group(1)) if m else None

    calldata = "0x" + (vd / "calldata.bytes").read_bytes().hex()
    send = sh(
        "cast",
        "send",
        addr,
        calldata,
        "--rpc-url",
        RPC,
        "--private-key",
        "0x" + PK,
        "--gas-limit",
        "30000000",
        "--json",
        cwd=vd,
        check=False,
    )
    try:
        j = json.loads(send)
        txh, status = j.get("transactionHash"), j.get("status")
        rec["tx_status"] = status
        rec["receipt_gas_used"] = (
            int(j.get("gasUsed", "0x0"), 16)
            if isinstance(j.get("gasUsed"), str)
            else j.get("gasUsed")
        )
        rec["verify_gas_executed"] = frame_gas(txh, vd)
    except Exception as e:  # noqa: BLE001
        rec["tx_error"] = f"{type(e).__name__}: {e}"
        rec["raw_send_output"] = send[:800]
    return rec


def main():
    ap = argparse.ArgumentParser(
        description=None, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--variants", nargs="+", default=list(VARIANTS), choices=list(VARIANTS)
    )
    ap.add_argument("--workdir", default=None)
    ap.add_argument("--out", default=str(REPO / "results" / "fusion_evm_gas"))
    args = ap.parse_args()

    if not sh("cast", "block-number", "--rpc-url", RPC, check=False).strip().isdigit():
        sys.exit("anvil 不可达。先起：NO_PROXY=localhost,127.0.0.1 anvil --port 8545")

    wd = (
        Path(args.workdir)
        if args.workdir
        else Path(tempfile.mkdtemp(prefix="ezkl_gas_"))
    )
    wd.mkdir(parents=True, exist_ok=True)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    print(f"[workdir] {wd}")

    srs_src = {}
    existing = REPO / "results" / "seq8_d16_l1_div_new" / "kzg.srs"
    if existing.exists():
        srs_src[17] = str(existing)
        print(f"[srs] reusing {existing} for k=17")

    recs = []
    for name in args.variants:
        print(f"\n=== {name} ===", flush=True)
        try:
            rec, vd = build_variant(name, wd, srs_src)
        except SystemExit as e:
            print(f"  BUILD FAILED: {str(e)[:300]}", flush=True)
            recs.append(
                {"variant": name, "status": "BUILD_FAILED", "error": str(e)[-1500:]}
            )
            (out / "fusion_evm_gas.json").write_text(
                json.dumps(recs, indent=2, ensure_ascii=False) + "\n"
            )
            continue
        print(
            f"  rows={rec['num_rows']:,} k={rec['logrows']} "
            f"offchain_verified={rec['offchain_verified']} ({rec['build_wall_s']}s)",
            flush=True,
        )
        rec = measure_on_chain(name, vd, rec)
        print(
            f"  on-chain verified={rec.get('on_chain_verified')} "
            f"estimate={rec.get('verify_gas_ezkl_estimate')} "
            f"executed={rec.get('verify_gas_executed')} "
            f"vka={rec['vka_bytes']}B",
            flush=True,
        )
        recs.append(rec)
        (out / "fusion_evm_gas.json").write_text(
            json.dumps(recs, indent=2, ensure_ascii=False) + "\n"
        )

    print("\n=== 汇总 ===")
    hdr = f"{'variant':8s} {'rows':>9s} {'k':>3s} {'vka B':>7s} {'verify gas(exec)':>17s} {'vs base':>8s}"
    print(hdr)
    b = next((r for r in recs if r["variant"] == "base"), None)
    for r in recs:
        if r.get("status") == "BUILD_FAILED":
            print(
                f"{r['variant']:8s} {'BUILD_FAILED — 见 fusion_evm_gas.json 的 error 字段':>50s}"
            )
            continue
        g = r.get("verify_gas_executed")
        rel = (
            f"{(g / b['verify_gas_executed'] - 1) * 100:+.1f}%"
            if b and g and b.get("verify_gas_executed")
            else "-"
        )
        print(
            f"{r['variant']:8s} {r['num_rows']:9,d} {r['logrows']:3d} {r['vka_bytes']:7,d} "
            f"{(g if g else 0):17,d} {rel:>8s}"
        )
    print(f"\n[wrote] {out/'fusion_evm_gas.json'}")
    print(f"[workdir kept] {wd}")


if __name__ == "__main__":
    main()
