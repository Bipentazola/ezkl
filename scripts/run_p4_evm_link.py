#!/usr/bin/env python3

import argparse
import json
import re
import struct
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
EZKL = REPO / "target" / "release" / "ezkl"
CONTRACT = REPO / "contracts" / "ChainLinkVerifier.sol"
RPC = "http://127.0.0.1:8545"
PK = "ac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80"
CACHE_OUT, CACHE_IN = "[6,7]", "[1,2]"


def sh(*args, cwd=None, check=True):
    r = subprocess.run([str(a) for a in args], cwd=cwd, capture_output=True, text=True)
    if check and r.returncode != 0:
        sys.exit(f"failed: {' '.join(map(str, args))}\n{r.stdout}\n{r.stderr}")
    return (r.stdout + r.stderr).strip()


def frame_gas(txhash, cwd):
    out = sh("cast", "run", txhash, "--rpc-url", RPC, cwd=cwd, check=False)
    for line in out.split("\n"):
        m = re.search(r"\[(\d+)\]\s+0x[0-9a-fA-F]{40}::", line)
        if m:
            return int(m.group(1)), out
    return None, out


def proofs_arg(chain, n):
    return (
        "["
        + ",".join(
            json.loads((chain / f"proof_{t}.json").read_text())["hex_proof"]
            for t in range(n)
        )
        + "]"
    )


def main():
    ap = argparse.ArgumentParser(
        description=None, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--chain-dir", required=True, help="P3/P2 的 chain 工作目录")
    ap.add_argument("--srs", default=None, help="默认从 chain-dir 里找 kzg*.srs")
    ap.add_argument("--steps", type=int, default=8, help="链接检查用多少份证明")
    ap.add_argument(
        "--verify-chain-steps",
        type=int,
        default=3,
        help="verifyChain 一笔交易验几份（gas 随 T 线性增长）",
    )
    ap.add_argument("--out", default=str(REPO / "results" / "incremental" / "p4_evm"))
    args = ap.parse_args()

    chain = Path(args.chain_dir).resolve()
    srs = args.srs or next(chain.glob("kzg*.srs")).name
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    R = {}

    if sh("cast", "block-number", "--rpc-url", RPC, check=False).isdigit() is False:
        sys.exit(
            f"anvil not reachable at {RPC} (start it, and set NO_PROXY=localhost,127.0.0.1)"
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
        srs,
        "--sol-code-path",
        "verifier.sol",
        "--abi-path",
        "verifier_abi.json",
        cwd=chain,
    )
    sh(
        EZKL,
        "create-evm-vka",
        "-S",
        "settings.json",
        "--vk-path",
        "vk.key",
        "--srs-path",
        srs,
        "--vka-path",
        "vka.bytes",
        cwd=chain,
    )
    sh(
        EZKL,
        "deploy-evm",
        "--sol-code-path",
        "verifier.sol",
        "-U",
        RPC,
        "--addr-path",
        "addr_verifier.txt",
        "-P",
        PK,
        "-C",
        "verifier/reusable",
        cwd=chain,
    )
    verifier = (chain / "addr_verifier.txt").read_text().strip()
    sh(
        EZKL,
        "register-vka",
        "-U",
        RPC,
        "--addr-verifier",
        verifier,
        "--vka-path",
        "vka.bytes",
        "--vka-digest-path",
        "vka_digest.txt",
        "-P",
        PK,
        cwd=chain,
    )

    blob = (chain / "vka.bytes").read_bytes()
    nwords = struct.unpack("<Q", blob[:8])[0]
    vka_words = (
        "["
        + ",".join(
            "0x" + blob[8 + i * 32 : 8 + (i + 1) * 32].hex() for i in range(nwords)
        )
        + "]"
    )

    R["sizes_bytes"] = {
        "reusable_verifier_bytecode": len(
            sh("cast", "code", verifier, "--rpc-url", RPC, cwd=chain)
        )
        // 2,
        "vka": len(blob),
        "eip170_limit": 24576,
    }

    o = sh(
        EZKL,
        "verify-evm",
        "--proof-path",
        "proof_0.json",
        "--addr-verifier",
        verifier,
        "-U",
        RPC,
        "--vka-path",
        "vka.bytes",
        cwd=chain,
    )
    R["single_proof_on_chain"] = {
        "result": "true" if "verification result: true" in o else "FAILED",
        "ezkl_estimated_tx_gas": int(
            re.search(r"estimated verify gas cost: (\d+)", o).group(1)
        ),
    }

    proj = chain / "_fproj"
    (proj / "src").mkdir(parents=True, exist_ok=True)
    (proj / "src" / "ChainLinkVerifier.sol").write_text(CONTRACT.read_text())
    (proj / "foundry.toml").write_text(
        '[profile.default]\nsrc = "src"\nout = "out"\nlibs = []\n'
        "optimizer = true\noptimizer_runs = 200\n"
    )
    sh("forge", "build", cwd=proj)
    dep = sh(
        "forge",
        "create",
        "src/ChainLinkVerifier.sol:ChainLinkVerifier",
        "--rpc-url",
        RPC,
        "--private-key",
        "0x" + PK,
        "--broadcast",
        cwd=proj,
    )
    link_addr = re.search(r"Deployed to: (0x[0-9a-fA-F]{40})", dep).group(1)
    R["sizes_bytes"]["chain_link_verifier_bytecode"] = (
        len(sh("cast", "code", link_addr, "--rpc-url", RPC, cwd=chain)) // 2
    )

    def send(sig, *a, gas_limit=None):
        cmd = [
            "cast",
            "send",
            link_addr,
            sig,
            *a,
            "--rpc-url",
            RPC,
            "--private-key",
            "0x" + PK,
            "--json",
        ]
        if gas_limit:
            cmd += ["--gas-limit", str(gas_limit)]
        return json.loads(sh(*cmd, cwd=chain))["transactionHash"]

    def call(sig, *a):
        return sh("cast", "call", link_addr, sig, *a, "--rpc-url", RPC, cwd=chain)

    P = proofs_arg(chain, args.steps)
    SIG = "checkLinks(bytes[],uint256[],uint256[])"
    NOOP = "noopBaseline(bytes[],uint256[],uint256[])"

    R["check_links"] = {
        "positive_returns_true": call(SIG, P, CACHE_OUT, CACHE_IN).endswith("1"),
        "negative_returns_false": call(SIG, P, CACHE_OUT, CACHE_OUT).endswith("0"),
        "n_proofs": args.steps,
        "n_comparisons": (args.steps - 1) * 2,
    }
    g_full, _ = frame_gas(send(SIG, P, CACHE_OUT, CACHE_IN), chain)
    g_short, _ = frame_gas(send(SIG, P, CACHE_OUT, CACHE_OUT), chain)
    g_noop, _ = frame_gas(send(NOOP, P, CACHE_OUT, CACHE_IN), chain)
    ncmp = (args.steps - 1) * 2
    R["check_links"].update(
        {
            "gas_full": g_full,
            "gas_short_circuit_1_cmp": g_short,
            "gas_noop_0_cmp": g_noop,
            "gas_per_comparison": round((g_full - g_short) / max(1, ncmp - 1), 1),
        }
    )

    p0 = json.loads((chain / "proof_0.json").read_text())
    b0 = bytes(p0["proof"])
    anchor = (
        "["
        + ",".join(
            "0x" + b0[i * 64 + j * 32 : i * 64 + (j + 1) * 32].hex()
            for i in (1, 2)
            for j in (0, 1)
        )
        + "]"
    )
    R["check_anchor_returns_true"] = call(
        "checkAnchor(bytes,uint256[],bytes32[])", p0["hex_proof"], CACHE_IN, anchor
    ).endswith("1")

    vc = []
    for n in range(1, args.verify_chain_steps + 1):
        Pn = proofs_arg(chain, n)
        In = "[" + ",".join(["[]"] * n) + "]"
        sig = "verifyChain(address,bytes[],uint256[][],bytes32[],uint256[],uint256[])"
        h = send(
            sig, verifier, Pn, In, vka_words, CACHE_OUT, CACHE_IN, gas_limit=30_000_000
        )
        g, trace = frame_gas(h, chain)
        rets = re.findall(r"\u2190 \[Return\] (0x[0-9a-fA-F]+)", trace)
        vc.append(
            {
                "T": n,
                "exec_gas": g,
                "verifier_calls": len(
                    re.findall(r"\u251c\u2500 \[\d+\] " + verifier[:8], trace, re.I)
                ),
                "ecpairing_calls": trace.lower().count("precompiles::ecpairing"),
                "returned_true": bool(rets) and int(rets[-1], 16) == 1,
            }
        )
        print(
            f"  verifyChain T={n}: exec_gas={g} calls={vc[-1]['verifier_calls']} "
            f"pairings={vc[-1]['ecpairing_calls']} true={vc[-1]['returned_true']}"
        )
    R["verify_chain"] = vc
    if len(vc) >= 2:
        R["verify_chain_marginal_gas_per_proof"] = round(
            (vc[-1]["exec_gas"] - vc[0]["exec_gas"]) / (vc[-1]["T"] - vc[0]["T"])
        )
    R["verify_chain_negative_returns_false"] = call(
        "verifyChain(address,bytes[],uint256[][],bytes32[],uint256[],uint256[])",
        verifier,
        proofs_arg(chain, args.verify_chain_steps),
        "[" + ",".join(["[]"] * args.verify_chain_steps) + "]",
        vka_words,
        CACHE_OUT,
        CACHE_OUT,
    ).endswith("0")

    (out / "p4_results.json").write_text(json.dumps(R, indent=2) + "\n")
    print("\n" + json.dumps(R, indent=2))
    print(f"\n[wrote] {out / 'p4_results.json'}")


if __name__ == "__main__":
    main()
