#!/usr/bin/env python3

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
EZKL = REPO / "target" / "release" / "ezkl"
POINT = 64


def run(*args, cwd=None, check=True):
    r = subprocess.run([str(a) for a in args], cwd=cwd, capture_output=True, text=True)
    if check and r.returncode != 0:
        sys.exit(f"command failed: {' '.join(map(str, args))}\n{r.stdout}\n{r.stderr}")
    return r.stdout + r.stderr


def commitments(proof_path: Path, n: int) -> list:
    b = bytes(json.loads(proof_path.read_text())["proof"])
    if len(b) < n * POINT:
        sys.exit(f"{proof_path}: proof shorter than {n} commitment points")
    return [b[i * POINT : (i + 1) * POINT] for i in range(n)]


def verify_one(proof, m, cwd):
    out = run(
        EZKL,
        "verify",
        "--proof-path",
        proof,
        "-S",
        m["settings"],
        "--vk-path",
        m["vk"],
        "--srs-path",
        m["srs"],
        cwd=cwd,
        check=False,
    )
    if "verified: true" in out:
        return True
    if "verified: false" in out:
        return False
    sys.exit(f"could not parse verify output for {proof}:\n{out}")


def expected_input_commitments(m, input_json, cwd, tag):
    wpath = f"_expected_{tag}.json"
    run(
        EZKL,
        "gen-witness",
        "-M",
        m["compiled"],
        "-D",
        input_json,
        "-O",
        wpath,
        "--vk-path",
        m["vk"],
        "--srs-path",
        m["srs"],
        cwd=cwd,
    )
    w = json.loads((Path(cwd) / wpath).read_text())
    out = []
    for grp in w["processed_inputs"]["polycommit"]:
        for h in grp:
            out.append(bytes.fromhex(h))
    return out


def n_input_commitments(m, cwd) -> int:
    wpath = "_layout_probe.json"
    ref = m.get("zero_anchor_input") or m["steps"][0]["input"]
    run(
        EZKL,
        "gen-witness",
        "-M",
        m["compiled"],
        "-D",
        ref,
        "-O",
        wpath,
        "--vk-path",
        m["vk"],
        "--srs-path",
        m["srs"],
        cwd=cwd,
    )
    w = json.loads((Path(cwd) / wpath).read_text())
    return sum(len(g) for g in w["processed_inputs"]["polycommit"])


BN254_P = 21888242871839275222246405745257275088696311157297823662689037894645226208583


def compress_proof_point(point: bytes) -> bytes:
    if len(point) != 64:
        raise ValueError(f"expected a 64-byte uncompressed point, got {len(point)}")
    x = int.from_bytes(point[:32], "big")
    y = int.from_bytes(point[32:], "big")
    if x >= BN254_P or y >= BN254_P:
        raise ValueError("point coordinate out of field")
    if (y * y - (x * x * x + 3)) % BN254_P != 0:
        raise ValueError(f"point is not on the BN254 G1 curve: x={x:#x}")
    b = bytearray(x.to_bytes(32, "little"))
    if y & 1:
        b[31] |= 0x80
    return bytes(b)


def point_matches(proof_point: bytes, witness_compressed: bytes) -> tuple:
    got = compress_proof_point(proof_point)
    if got == witness_compressed:
        return True, ""
    x_same = got[:31] == witness_compressed[:31] and (got[31] & 0x7F) == (
        witness_compressed[31] & 0x7F
    )
    if x_same:
        return False, (
            "x matches but the y sign bit does not: "
            f"proof={got[31] >> 7} witness={witness_compressed[31] >> 7} "
            "— the two points are P and -P, which is exactly what the "
            "old x-only comparison could not tell apart"
        )
    return False, "x coordinates differ"


def main():
    ap = argparse.ArgumentParser(
        description=None, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()

    mpath = Path(args.manifest).resolve()
    m = json.loads(mpath.read_text())
    cwd = mpath.parent
    layout = m["layout"]
    cin, cout = layout["cache_in_idx"], layout["cache_out_idx"]
    masks = layout.get("mask_idx", {})
    n_points = max(cout + cin + list(masks.values())) + 1
    steps = m["steps"]

    report = {"n_steps": len(steps), "checks": {}, "detail": {}}
    ok = True

    structural = []
    n_inputs = n_input_commitments(m, cwd)
    declared_unchecked = set(layout.get("unchecked_input_idx", []))
    named_inputs = set(cin) | set(masks.values())
    unaccounted = set(range(n_inputs)) - named_inputs - declared_unchecked
    if unaccounted:
        structural.append(
            f"电路有 {n_inputs} 个输入承诺，但索引 {sorted(unaccounted)} 既未出现在 "
            "layout（cache_in_idx / mask_idx）中，也未在 layout.unchecked_input_idx "
            "里声明为不检查。缩小 layout 就是缩小检查范围，必须显式声明"
        )
    stray = (named_inputs | declared_unchecked) - set(range(n_inputs))
    if stray:
        structural.append(
            f"layout 里的输入侧索引 {sorted(stray)} 超出电路的 "
            f"{n_inputs} 个输入承诺"
        )
    if set(cout) & set(range(n_inputs)):
        structural.append(
            f"cache_out_idx {sorted(set(cout) & set(range(n_inputs)))} 落在输入承诺区间内"
        )
    report["detail"]["n_input_commitments"] = n_inputs
    if declared_unchecked:
        report["detail"]["declared_unchecked_input_idx"] = sorted(declared_unchecked)
    if not cin or not cout:
        structural.append("layout.cache_in_idx / cache_out_idx 不得为空")
    if len(cin) != len(cout):
        structural.append(
            f"cache_in_idx({len(cin)}) 与 cache_out_idx({len(cout)}) 长度不等，"
            "无法建立一一对应的链接关系"
        )
    if len(set(cin)) != len(cin) or len(set(cout)) != len(cout):
        structural.append("cache 索引有重复")
    if not masks:
        structural.append(
            "layout.mask_idx 缺失或为空：本协议的第 4 项（公开量绑定）依赖它，"
            "没有它就无法验证喂入的掩码确是验证方按 t 算出的那个"
        )
    ts = [s_["t"] for s_ in steps]
    if ts != list(range(len(ts))):
        structural.append(f"步号必须从 0 起连续递增，实际为 {ts}")
    report["checks"]["0_layout_complete"] = not structural
    if structural:
        report["detail"]["layout_problems"] = structural
        ok = False

    per_proof = [verify_one(s["proof"], m, cwd) for s in steps]
    report["checks"]["1_each_proof_verifies"] = all(per_proof)
    report["detail"]["per_proof_verify"] = per_proof
    ok &= all(per_proof)

    commits = [commitments(cwd / s["proof"], n_points) for s in steps]

    links, link_ok = [], True
    for i in range(len(steps) - 1):
        for a, b in zip(cout, cin):
            same = commits[i][a] == commits[i + 1][b]
            links.append(
                {
                    "from_step": steps[i]["t"],
                    "to_step": steps[i + 1]["t"],
                    "out_idx": a,
                    "in_idx": b,
                    "match": same,
                    "commitment": commits[i][a].hex(),
                }
            )
            link_ok &= same
    report["checks"]["2_cache_links"] = link_ok if len(steps) > 1 else None
    report["detail"]["links"] = links
    ok &= link_ok or len(steps) < 2

    missing = []
    if "zero_anchor_input" in m:
        exp = expected_input_commitments(m, m["zero_anchor_input"], cwd, "anchor")
        results, diags = [], []
        if not cin:
            results.append(False)
            diags.append({"idx": None, "reason": "cache_in_idx 为空，没有可锚定的输入"})
        for i in cin:
            same, why = point_matches(commits[0][i], exp[i])
            results.append(same)
            if why:
                diags.append({"idx": i, "reason": why})
        anchor_ok = all(results)
        report["checks"]["3_zero_anchor"] = anchor_ok
        report["detail"]["anchor_expected"] = [exp[i].hex() for i in cin]
        report["detail"]["anchor_actual"] = [
            compress_proof_point(commits[0][i]).hex() for i in cin
        ]
        if diags:
            report["detail"]["anchor_mismatch"] = diags
        ok &= anchor_ok
    else:
        report["checks"]["3_zero_anchor"] = False
        missing.append("zero_anchor_input")
        ok = False

    if masks and m.get("mask_inputs"):
        mask_results = []
        for s in steps:
            ref = m["mask_inputs"][str(s["t"])]
            exp = expected_input_commitments(m, ref, cwd, f"mask_t{s['t']}")
            for name, idx in masks.items():
                same, why = point_matches(commits[steps.index(s)][idx], exp[idx])
                rec = {"t": s["t"], "tensor": name, "idx": idx, "match": same}
                if why:
                    rec["reason"] = why
                mask_results.append(rec)
        mask_ok = all(r["match"] for r in mask_results)
        report["checks"]["4_public_mask_binding"] = mask_ok
        report["detail"]["masks"] = mask_results
        ok &= mask_ok
    elif masks:
        report["checks"]["4_public_mask_binding"] = False
        missing.append("mask_inputs")
        ok = False

    if missing:
        report["missing_required_inputs"] = missing
    report["all_passed"] = bool(ok)
    text = json.dumps(report, indent=2)
    if args.json_out:
        Path(args.json_out).write_text(text + "\n")
    for k, v in report["checks"].items():
        print(f"  {'PASS' if v else ('SKIP' if v is None else 'FAIL')}  {k}")
    print(f"\nall_passed = {report['all_passed']}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
