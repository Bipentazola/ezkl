#!/usr/bin/env python3

import json
import statistics as st
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
REPS = REPO / "results" / "incremental" / "workload_ratio" / "reps"


def load(name):
    f = REPS / name / "workload_ratio.json"
    if not f.exists():
        return None
    d = json.loads(f.read_text())
    env = REPS / name / "env.txt"
    d["_env"] = (
        dict(l.split("=", 1) for l in env.read_text().split("\n") if "=" in l)
        if env.exists()
        else {}
    )
    return d


def key(c):
    return f"d={c['d_model']}, T={c['T']}"


def spread(xs):
    return (max(xs) - min(xs)) / st.median(xs) if len(xs) > 1 and st.median(xs) else 0.0


def main():
    reps = [d for d in (load(f"nt1_rep{i}") for i in (1, 2, 3)) if d]
    sweep = [(nt, d) for nt in (2, 4, 8, 16) for d in [load(f"nt{nt}")] if d]
    if not reps:
        raise SystemExit(
            "还没有单线程重复数据；先跑 scripts/run_workload_ratio_reps.sh"
        )

    cfgs = [key(c) for c in reps[0]["configs"]]
    print(
        f"# 工作负载比汇总（单线程 {len(reps)} 次重复，并行度扫描 {len(sweep)} 档）\n"
    )

    env = reps[0]["_env"]
    if env:
        print(
            "**测量口径**："
            + "，".join(
                f"`{k}={env[k]}`"
                for k in ("RAYON_NUM_THREADS", "cpu.max", "memory.max")
                if k in env
            )
            + "\n"
        )

    print("## 单线程测量\n")
    print(
        "| 配置 | 全序列 rows / k | 单步 rows / k | tier 差 | 代理比 | "
        "实测比（中位） | 极差/中位 | 实测/代理 |"
    )
    print("|---|---|---|---|---|---|---|---|")
    summary = {}
    for i, name in enumerate(cfgs):
        cs = [r["configs"][i] for r in reps]
        c0 = cs[0]
        meas = [c["measured_ratio"] for c in cs]
        med = st.median(meas)
        summary[name] = {
            "proxy": c0["proxy_ratio"],
            "measured": meas,
            "median": med,
            "spread": spread(meas),
            "over_proxy": med / c0["proxy_ratio"],
            "unroll_k": c0["unroll"]["logrows"],
            "step_k": c0["step"]["logrows"],
            "unroll_rows": c0["unroll"]["num_rows"],
            "step_rows": c0["step"]["num_rows"],
            "step_cpu": [c["step"]["prove_cpu_total_s"] for c in cs],
            "unroll_cpu": [c["unroll"]["steps"][0]["prove_cpu_s"] for c in cs],
        }
        s = summary[name]
        print(
            f"| {name} | {s['unroll_rows']:,} / {s['unroll_k']} | {s['step_rows']:,} / {s['step_k']} "
            f"| {c0['tier_gap']} | {s['proxy']:.2f}× | **{med:.2f}×** | {s['spread']:.1%} "
            f"| {s['over_proxy']:.2f} |"
        )

    devs = [abs(s["over_proxy"] - 1) for s in summary.values()]
    lo = min(s["over_proxy"] for s in summary.values()) - 1
    hi = max(s["over_proxy"] for s in summary.values()) - 1
    print(
        f"\n**代理准确度（单线程口径）**：{lo:+.0%} ~ {hi:+.0%}"
        f"（最大偏离 {max(devs):.0%}）\n"
    )

    print("## 重复测量明细\n")
    print(
        "| 配置 | "
        + " | ".join(f"rep{i+1} 实测比" for i in range(len(reps)))
        + " | 极差/中位 |"
    )
    print("|---|" + "---|" * (len(reps) + 1))
    for name, s in summary.items():
        print(
            f"| {name} | "
            + " | ".join(f"{m:.3f}×" for m in s["measured"])
            + f" | {s['spread']:.1%} |"
        )

    print("\n**prove CPU 秒（单步侧合计 ÷ 全序列侧）**\n")
    print("| 配置 | " + " | ".join(f"rep{i+1}" for i in range(len(reps))) + " |")
    print("|---|" + "---|" * len(reps))
    for name, s in summary.items():
        print(
            f"| {name} | "
            + " | ".join(
                f"{a:,.1f} ÷ {b:,.1f}" for a, b in zip(s["step_cpu"], s["unroll_cpu"])
            )
            + " |"
        )

    if sweep:
        print("\n## 并行度扫描：实测比与代理比随线程数的变化\n")
        print(
            "| `RAYON_NUM_THREADS` | "
            + " | ".join(f"{n} 的实测/代理" for n in cfgs)
            + " |"
        )
        print("|---|" + "---|" * len(cfgs))
        print(
            "| **1（主口径）** | "
            + " | ".join(f"**{summary[n]['over_proxy']:.2f}**" for n in cfgs)
            + " |"
        )
        for nt, d in sweep:
            print(
                f"| {nt} | "
                + " | ".join(
                    f"{d['configs'][i]['measured_over_proxy']:.2f}"
                    for i in range(len(cfgs))
                )
                + " |"
            )
        base = [summary[n]["over_proxy"] for n in cfgs]
        top = [
            sweep[-1][1]["configs"][i]["measured_over_proxy"] for i in range(len(cfgs))
        ]
        print(
            "\n从 nt=1 到 nt=%d，比值移动了 %s。"
            % (
                sweep[-1][0],
                " 与 ".join(f"{(t/b-1)*100:+.1f}%" for b, t in zip(base, top)),
            )
        )

    print("\n## 不同线程设置下的历史测量对照\n")
    print("线程数未记录的测量不用于单线程统计。\n")
    print("| 配置 | 代理比 | 首次测量（线程数未记录） | 48 线程诊断 | **单线程重复** |")
    print("|---|---|---|---|---|")
    OLD = {"d=16, T=16": (2.00, 1.05), "d=16, T=32": (4.00, 0.84)}
    diag = load("rep1")
    for i, name in enumerate(cfgs):
        d48 = f"{diag['configs'][i]['measured_over_proxy']:.2f}" if diag else "—"
        o = OLD.get(name)
        print(
            f"| {name} | {summary[name]['proxy']:.2f}× | {o[1] if o else '—'} | {d48} "
            f"| **{summary[name]['over_proxy']:.2f}** |"
        )


if __name__ == "__main__":
    main()
