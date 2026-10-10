#!/usr/bin/env python3

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
EZKL = REPO / "target" / "release" / "ezkl"
TIME = REPO / ".tools" / "time-pkg" / "usr" / "bin" / "time"
GEN = REPO / "scripts" / "gen_decode_step_onnx.py"
DEFAULT_SRS = REPO / "results" / "rescued_configs" / "work" / "kzg21.srs"

POLYCOMMIT_VIS = [
    "--input-visibility",
    "polycommit",
    "--output-visibility",
    "polycommit",
    "--param-visibility",
    "fixed",
]
OPT_ON = {"EZKL_DIV_ROUND_NEAREST": "1", "EZKL_RESCALE_FUSION": "1"}
OPT_OFF = {"EZKL_DIV_ROUND_NEAREST": "0", "EZKL_RESCALE_FUSION": "0"}

EXPECTED_ROWS_OPT = {
    (384, 128): 1851517,
    (384, 64): 1451005,
    (384, 32): 1250749,
    (384, 16): 1150621,
    (128, 128): 432381,
    (128, 64): 296957,
    (128, 32): 229245,
    (128, 16): 195389,
}

ACTIVE_ENV = OPT_ON
TASKSET = ["taskset", "-c", "0-47"]


def sh(*args, cwd=None, timed=None, taskset=False):
    e = dict(os.environ)
    for var in ("EZKL_DIV_ROUND_NEAREST", "EZKL_RESCALE_FUSION"):
        e.pop(var, None)
    e.update(ACTIVE_ENV)
    argv = [str(a) for a in args]
    if taskset and TASKSET:
        argv = TASKSET + argv
    if timed:
        argv = [str(TIME), "-v"] + argv
    r = subprocess.run(argv, cwd=cwd, capture_output=True, text=True, env=e)
    if timed:
        Path(timed).write_text(r.stderr)
    if r.returncode != 0:
        logs = "\n".join(
            l
            for l in (r.stdout + r.stderr).splitlines()
            if "] - " in l or "error" in l.lower()
        )
        return None, (logs + "\n---\n" + r.stdout + r.stderr)
    return (r.stdout + r.stderr), None


def parse_time(path):
    t = Path(path).read_text()
    sig = re.search(r"Command terminated by signal (\d+)", t)
    rss = re.search(r"Maximum resident set size \(kbytes\): (\d+)", t)
    wall = re.search(r"Elapsed \(wall clock\) time[^\n]*?:\s*([\d:.]+)\s*$", t, re.M)
    secs = None
    if wall:
        parts = [float(p) for p in wall.group(1).split(":")]
        secs = sum(p * 60**i for i, p in enumerate(reversed(parts)))
    return {
        "wall_s": round(secs, 1) if secs else None,
        "peak_rss_gb": round(int(rss.group(1)) / 1024 / 1024, 1) if rss else None,
        "killed_signal": int(sig.group(1)) if sig else None,
    }


def cgroup_quota():
    q = {}
    try:
        c = Path("/sys/fs/cgroup/cpu.max").read_text().split()
        q["cpus"] = None if c[0] == "max" else round(int(c[0]) / int(c[1]), 2)
        m = Path("/sys/fs/cgroup/memory.max").read_text().strip()
        q["mem_gib"] = None if m == "max" else round(int(m) / 1024**3, 1)
    except Exception as exc:
        q["error"] = str(exc)
    return q


def write_env(path, wd, label, phase, extra=""):
    q = cgroup_quota()
    with Path(path).open("a") as f:
        f.write(f"[{phase}] {time.strftime('%Y-%m-%dT%H:%M:%S%z')}\n")
        f.write(f"  config           {label}\n")
        f.write(f"  workdir          {wd}\n")
        f.write(
            f"  cgroup cpu.max   {Path('/sys/fs/cgroup/cpu.max').read_text().strip()}"
            f"   -> {q.get('cpus')} cores\n"
        )
        f.write(
            f"  cgroup memory.max {Path('/sys/fs/cgroup/memory.max').read_text().strip()}"
            f"  -> {q.get('mem_gib')} GiB\n"
        )
        f.write(f"  nproc (host view) {os.cpu_count()}  (**不是配额**)\n")
        f.write(f"  taskset          {' '.join(TASKSET) or '(none)'}\n")
        f.write(
            f"  RAYON_NUM_THREADS {os.environ.get('RAYON_NUM_THREADS', '(unset)')}\n"
        )
        f.write(f"  loadavg          {Path('/proc/loadavg').read_text().strip()}\n")
        if extra:
            f.write(f"  {extra}\n")
        f.write("\n")


def logrows_for(n):
    return max(1, (n - 1).bit_length())


def main():
    ap = argparse.ArgumentParser(
        description=None, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--d-model", type=int, default=384)
    ap.add_argument("--t-max", type=int, default=128)
    ap.add_argument(
        "--position",
        type=int,
        default=None,
        help="本步写入的 cache 槽位，默认交给生成器（t_max//2），与 P1 网格一致",
    )
    ap.add_argument(
        "--baseline", action="store_true", help="关两项优化（行数更高，预期更难）"
    )
    ap.add_argument(
        "--srs-path",
        default=str(DEFAULT_SRS),
        help="复用已有 SRS；不存在则报错退出（本脚本不 gen-srs）",
    )
    ap.add_argument(
        "--no-taskset", action="store_true", help="不锁核（会破坏与 results/ 的可比性）"
    )
    ap.add_argument(
        "--keep-pk", action="store_true", help="跑完不删 pk.key（默认删，几十 GB）"
    )
    ap.add_argument("--workdir", default=None)
    ap.add_argument(
        "--out", default=str(REPO / "results" / "incremental" / "p1_polycommit_e2e")
    )
    args = ap.parse_args()

    global ACTIVE_ENV, TASKSET
    if args.baseline:
        ACTIVE_ENV = OPT_OFF
    if args.no_taskset:
        TASKSET = []

    if not EZKL.exists():
        sys.exit(f"missing {EZKL}")
    srs = Path(args.srs_path).resolve()
    if not srs.exists():
        sys.exit(
            f"missing SRS {srs}\n本脚本不生成 SRS（gen-srs 是随机的，会让已提交的证明不可验证）。"
            f"\n请指向仓库已有的一份，或用 `ezkl get-srs` 取标准 SRS。"
        )

    label = f"d{args.d_model}_T{args.t_max}" + (
        "_baseline" if args.baseline else "_optimized"
    )
    wd = Path(args.workdir) if args.workdir else Path(args.out) / "work" / label
    wd.mkdir(parents=True, exist_ok=True)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    rec = {
        "config": label,
        "d_model": args.d_model,
        "t_max": args.t_max,
        "position": args.position,
        "visibility": "polycommit",
        "optimizations": "off" if args.baseline else "on",
        "srs_path": (
            str(srs.relative_to(REPO)) if srs.is_relative_to(REPO) else str(srs)
        ),
        "srs_bytes": srs.stat().st_size,
        "taskset": " ".join(TASKSET) or None,
        "cgroup": cgroup_quota(),
    }
    print(
        f"[口径] {label} vis=polycommit env={ACTIVE_ENV} srs={srs.name} "
        f"taskset={rec['taskset']} cgroup={rec['cgroup']}",
        flush=True,
    )
    envfile = out / f"{label}.env.txt"
    write_env(
        envfile,
        wd,
        label,
        "start",
        extra=f"srs {rec['srs_path']} ({rec['srs_bytes']} bytes)",
    )
    t0 = time.time()

    def finish(status, **kw):
        rec.update(status=status, total_wall_s=round(time.time() - t0, 1), **kw)
        write_env(envfile, wd, label, "end", extra=f"status {status}")
        (out / f"{label}.json").write_text(
            json.dumps(rec, indent=2, ensure_ascii=False) + "\n"
        )
        print(f"\n[status] {status}")
        print(f"[wrote] {out/f'{label}.json'}")
        print(f"[workdir kept] {wd}")
        return 0 if status == "OK" else 1

    gen_cmd = [
        sys.executable,
        GEN,
        "--variant",
        "where",
        "--d-model",
        args.d_model,
        "--t-max",
        args.t_max,
        "--out-dir",
        wd,
    ]
    if args.position is not None:
        gen_cmd += ["--position", args.position]
    o, err = sh(*gen_cmd, cwd=REPO)
    if err:
        return finish("EXPORT_FAILED", error=err[-900:])

    o, err = sh(
        EZKL,
        "gen-settings",
        "-M",
        "model.onnx",
        "-O",
        "settings.json",
        "-K",
        24,
        *POLYCOMMIT_VIS,
        cwd=wd,
    )
    if err:
        return finish("GEN_SETTINGS_FAILED", error=err[-900:])
    num_rows = json.loads((wd / "settings.json").read_text())["num_rows"]
    k = logrows_for(num_rows)
    rec.update(num_rows=num_rows, logrows=k)
    exp = EXPECTED_ROWS_OPT.get((args.d_model, args.t_max))
    rec["rows_match_grid"] = (
        None if (exp is None or args.baseline) else (num_rows == exp)
    )
    print(
        f"[circuit] num_rows={num_rows:,} -> logrows={k}  "
        f"rows_match_grid={rec['rows_match_grid']}",
        flush=True,
    )
    if rec["rows_match_grid"] is False:
        return finish(
            "ROWS_MISMATCH",
            error=f"expected {exp} from p1_decode_step.csv, got {num_rows}",
        )
    srs_k = max(1, (srs.stat().st_size // 128).bit_length() - 1)
    if k > srs_k:
        return finish(
            "SRS_TOO_SMALL",
            error=f"circuit needs logrows={k}, {srs.name} looks like <= {srs_k}",
        )
    o, err = sh(
        EZKL,
        "gen-settings",
        "-M",
        "model.onnx",
        "-O",
        "settings.json",
        "-K",
        k,
        *POLYCOMMIT_VIS,
        cwd=wd,
    )
    if err:
        return finish("GEN_SETTINGS_FAILED", error=err[-900:])

    for step, a in (
        (
            "compile",
            (
                EZKL,
                "compile-circuit",
                "-M",
                "model.onnx",
                "-S",
                "settings.json",
                "--compiled-circuit",
                "circuit.compiled",
            ),
        ),
        (
            "witness",
            (
                EZKL,
                "gen-witness",
                "-M",
                "circuit.compiled",
                "-D",
                "input.json",
                "-O",
                "witness.json",
            ),
        ),
    ):
        o, err = sh(*a, cwd=wd, timed=wd / f"{step}.time", taskset=True)
        rec[step] = parse_time(wd / f"{step}.time")
        print(f"[{step}] {rec[step]}", flush=True)
        if err:
            return finish(f"{step.upper()}_FAILED", error=err[-900:])

    o, err = sh(
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
        timed=wd / "setup.time",
        taskset=True,
    )
    rec["setup"] = parse_time(wd / "setup.time")
    pk = wd / "pk.key"
    rec["pk_bytes"] = pk.stat().st_size if pk.exists() else None
    print(f"[setup] {rec['setup']}  pk={rec['pk_bytes']}", flush=True)
    if err:
        return finish("SETUP_FAILED", error=err[-900:])

    o, err = sh(
        EZKL,
        "prove",
        "-M",
        "circuit.compiled",
        "-W",
        "witness.json",
        "--pk-path",
        "pk.key",
        "--proof-path",
        "proof.pf",
        "--srs-path",
        srs,
        cwd=wd,
        timed=wd / "prove.time",
        taskset=True,
    )
    rec["prove"] = parse_time(wd / "prove.time")
    print(f"[prove] {rec['prove']}", flush=True)
    if pk.exists() and not args.keep_pk:
        rec["pk_bytes_deleted"] = pk.stat().st_size
        pk.unlink()
    if err:
        return finish("PROVE_FAILED", error=err[-900:])
    proof = wd / "proof.pf"
    rec["proof_bytes"] = proof.stat().st_size if proof.exists() else None

    o, err = sh(
        EZKL,
        "verify",
        "--proof-path",
        "proof.pf",
        "--settings-path",
        "settings.json",
        "--vk-path",
        "vk.key",
        "--srs-path",
        srs,
        cwd=wd,
        timed=wd / "verify.time",
    )
    rec["verify"] = parse_time(wd / "verify.time")
    rec["verified"] = bool(o and "verified: true" in o.lower())
    if err:
        return finish("VERIFY_FAILED", error=err[-900:])
    return finish("OK" if rec["verified"] else "VERIFY_FALSE")


if __name__ == "__main__":
    sys.exit(main())
