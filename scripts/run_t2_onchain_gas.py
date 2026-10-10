#!/usr/bin/env python3

import argparse
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
CONTRACT = REPO / "contracts" / "T2OnChainGas.sol"
FOUNDRY = "/data/foundry/bin"
RPC = "http://127.0.0.1:8545"
PK = "ac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80"
MAINNET_BLOCK_GAS = 30_000_000
REAL_VOCAB = 50257
EST = {
    "calldata_gas_per_nonzero_byte": 16,
    "msm_gas_per_point": 6150,
    "argmax_gas_per_compare": 60,
}


def sh(*args, cwd=None, check=True, env=None):
    e = dict(
        os.environ,
        PATH=FOUNDRY + ":" + os.environ["PATH"],
        NO_PROXY="localhost,127.0.0.1",
        no_proxy="localhost,127.0.0.1",
    )
    if env:
        e.update(env)
    r = subprocess.run(
        [str(a) for a in args], cwd=cwd, capture_output=True, text=True, env=e
    )
    if check and r.returncode != 0:
        sys.exit(
            f"failed: {' '.join(map(str, args))}\n{r.stdout[-3000:]}\n{r.stderr[-3000:]}"
        )
    return (r.stdout + r.stderr).strip()


ACCT = "0xf39Fd6e51aad88F6F4ce6aB8827279cffFb92266"


_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def rpc(method, params):
    req = urllib.request.Request(
        RPC,
        data=json.dumps(
            {"jsonrpc": "2.0", "id": 1, "method": method, "params": params}
        ).encode(),
        headers={"Content-Type": "application/json"},
    )
    with _OPENER.open(req, timeout=1800) as r:
        d = json.loads(r.read())
    if "error" in d:
        return None, d["error"].get("message", str(d["error"]))
    return d["result"], None


def wait_receipt(txh, timeout=3600, interval=2):
    t0 = time.time()
    while time.time() - t0 < timeout:
        r, _ = rpc("eth_getTransactionReceipt", [txh])
        if r:
            return r
        time.sleep(interval)
    return None


def encode_call(selector, xs, tail=()):
    head = [32 * (1 + len(tail))]
    words = [head[0]] + list(tail) + [len(xs)] + list(xs)
    body = "".join(f"{w & (2**256 - 1):064x}" for w in words)
    return selector + body


def scalars(n, seed=1):
    return [
        (seed + i)
        * 0x0101010101010101010101010101010101010101010101010101010101
        % (2**254)
        | 1
        for i in range(n)
    ]


def fit_slope(ns, gas):
    k = len(ns)
    sx, sy = sum(ns), sum(gas)
    sxx = sum(n * n for n in ns)
    sxy = sum(n * g for n, g in zip(ns, gas))
    b = (k * sxy - sx * sy) / (k * sxx - sx * sx)
    return b, (sy - b * sx) / k


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", default="16,64,256,384,1024,4096")
    ap.add_argument(
        "--skip-real-vocab",
        action="store_true",
        help="跳过 V=50,257 那一档（它最慢，MSM 要几分钟）",
    )
    ap.add_argument("--out", default="results/incremental/t2_onchain_gas")
    ap.add_argument("--temp-dir", help="临时构建目录的父目录；默认使用系统临时目录")
    ap.add_argument("--anvil-gas-limit", type=int, default=2_000_000_000)
    ap.add_argument(
        "--hardforks",
        default="cancun,prague",
        help="cancun = 论文假设的 calldata 定价；prague = 含 EIP-7623 地板价的当前定价",
    )
    args = ap.parse_args()

    if not shutil.which("anvil", path=FOUNDRY):
        sys.exit(f"anvil not found under {FOUNDRY}")
    out = Path(args.out)
    if not out.is_absolute():
        out = REPO / out
    out.mkdir(parents=True, exist_ok=True)
    all_results = {}
    if (out / "gas.json").exists():
        try:
            all_results = json.loads((out / "gas.json").read_text()).get(
                "by_hardfork", {}
            )
        except Exception:
            all_results = {}
    if args.temp_dir:
        Path(args.temp_dir).mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="ezkl_t2_gas_", dir=args.temp_dir) as tmp:
        work = Path(tmp)
        (work / "src").mkdir()
        shutil.copy(CONTRACT, work / "src" / CONTRACT.name)
        (work / "foundry.toml").write_text(
            '[profile.default]\nsrc = "src"\nout = "out"\nlibs = []\n'
            "optimizer = true\noptimizer_runs = 200\n"
        )
        for hf in args.hardforks.split(","):
            all_results[hf] = run_one(hf, args, work)
            write_out(out, all_results)


def write_out(out, all_results):
    (out / "gas.json").write_text(
        json.dumps(
            {
                "contract": str(CONTRACT.relative_to(REPO)),
                "rpc": "anvil (local, no block competition)",
                "mainnet_block_gas_limit": MAINNET_BLOCK_GAS,
                "real_vocab": REAL_VOCAB,
                "paper_estimates_ss7_6_1": EST,
                "by_hardfork": all_results,
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    print(f"\n写入 {out / 'gas.json'}", flush=True)


def calldata_theory(hexdata):
    b = bytes.fromhex(hexdata[2:] if hexdata.startswith("0x") else hexdata)
    z = sum(1 for c in b if c == 0)
    nz = len(b) - z
    standard = 21000 + z * 4 + nz * 16
    floor = 21000 + 10 * (z + 4 * nz)
    return {
        "bytes": len(b),
        "zero_bytes": z,
        "nonzero_bytes": nz,
        "eip2028_standard_gas": standard,
        "eip7623_floor_gas": floor,
    }


def wait_port_free(port=8545, timeout=60):
    import socket

    t0 = time.time()
    while time.time() - t0 < timeout:
        s = socket.socket()
        try:
            s.bind(("127.0.0.1", port))
            s.close()
            return True
        except OSError:
            s.close()
            time.sleep(1)
    return False


def run_one(hardfork, args, work):
    wait_port_free()
    anvil = subprocess.Popen(
        [
            f"{FOUNDRY}/anvil",
            "--port",
            "8545",
            "--gas-limit",
            str(args.anvil_gas_limit),
            "--hardfork",
            hardfork,
            "--code-size-limit",
            "100000",
            "--no-request-size-limit",
            "--silent",
        ],
        env=dict(os.environ, PATH=FOUNDRY + ":" + os.environ["PATH"]),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    print(f"\n===== hardfork = {hardfork} =====", flush=True)
    try:
        for _ in range(90):
            if sh("cast", "block-number", "--rpc-url", RPC, check=False).isdigit():
                break
            time.sleep(1)
        else:
            sys.exit(f"anvil 起不来（hardfork={hardfork}）")

        sh("forge", "build", cwd=work)
        dep = sh(
            "forge",
            "create",
            f"src/{CONTRACT.name}:T2OnChainGas",
            "--rpc-url",
            RPC,
            "--private-key",
            "0x" + PK,
            "--broadcast",
            cwd=work,
        )
        addr = re.search(r"Deployed to: (0x[0-9a-fA-F]{40})", dep).group(1)
        print(f"[deployed] {addr}", flush=True)
        selectors = {
            sig: sh("cast", "sig", sig).strip()
            for sig in (
                "receiveLogits(uint256[])",
                "argmaxOf(uint256[])",
                "msmWithGeneratorBase(uint256[])",
                "t2Step(uint256[],uint256,uint256)",
            )
        }

        sizes = [int(x) for x in args.sizes.split(",")]
        if not args.skip_real_vocab:
            sizes.append(REAL_VOCAB)
        rows = []
        for n in sizes:
            xs = scalars(n)
            rec = {"n": n}
            t0 = time.time()
            for name, sig, extra in (
                ("calldata_only", "receiveLogits(uint256[])", ()),
                ("argmax", "argmaxOf(uint256[])", ()),
                ("msm", "msmWithGeneratorBase(uint256[])", ()),
                ("t2_step", "t2Step(uint256[],uint256,uint256)", (0, 0)),
            ):
                data = encode_call(selectors[sig], xs, extra)
                call = {
                    "from": ACCT,
                    "to": addr,
                    "data": data,
                    "gas": hex(args.anvil_gas_limit),
                }
                txh, err = rpc("eth_sendTransaction", [call])
                tx_gas = None
                if txh:
                    rcpt = wait_receipt(txh)
                    if rcpt and rcpt.get("status") == "0x1":
                        tx_gas = int(rcpt["gasUsed"], 16)
                    elif rcpt:
                        err = f"reverted (status {rcpt.get('status')})"
                rec[name] = {"tx_gas": tx_gas, "sent": bool(txh)}
                if name == "calldata_only":
                    rec[name]["theory"] = calldata_theory(data)
                print(
                    f"  n={n:6d} {name:14s} tx_gas={tx_gas}"
                    + (f"  [{err}]" if err else ""),
                    flush=True,
                )
            rec["seconds"] = round(time.time() - t0, 1)
            rows.append(rec)

        fit = {}
        for name, est_key, unit, subtract in (
            ("calldata_only", "calldata_gas_per_nonzero_byte", 32, False),
            ("argmax", "argmax_gas_per_compare", 1, True),
            ("msm", "msm_gas_per_point", 1, True),
            ("t2_step", None, 1, True),
        ):
            pts = []
            for r in rows:
                g = r[name].get("tx_gas")
                base = r["calldata_only"].get("tx_gas")
                if g is None or (subtract and base is None):
                    continue
                pts.append((r["n"], g - base if subtract else g))
            if len(pts) < 2:
                continue
            b, a = fit_slope([p[0] for p in pts], [p[1] for p in pts])
            e = {
                "measured_gas_per_element": round(b, 1),
                "intercept": round(a, 1),
                "basis": (
                    "整笔交易 gas"
                    if not subtract
                    else "整笔 gas − 同 n 的 calldata_only"
                ),
                "extrapolated_to_V50257": round(a + b * REAL_VOCAB),
            }
            if est_key:
                e["paper_estimate_per_element"] = EST[est_key] * unit
                e["measured_over_estimate"] = round(b / (EST[est_key] * unit), 3)
            fit[name] = e
        print("\n" + json.dumps(fit, indent=2, ensure_ascii=False), flush=True)
        return {
            "hardfork": hardfork,
            "address": addr,
            "anvil_gas_limit": args.anvil_gas_limit,
            "fit": fit,
            "measurements": rows,
        }
    finally:
        anvil.send_signal(signal.SIGTERM)
        anvil.wait(timeout=20)


if __name__ == "__main__":
    main()
