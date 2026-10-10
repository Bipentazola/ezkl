#!/usr/bin/env python3

import argparse
import json
import math
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def load(paths):
    out = []
    for p in paths:
        d = json.loads(Path(p).read_text())
        cfg = d["config"]
        tag = f"d{cfg['d_model']}_T{cfg['t_max']}_V{cfg['vocab']}"
        for r in d["runs"]:
            if not isinstance(r, dict):
                continue
            out.append({"config": tag, **r})
    return out


def get(r, stage, field):
    if field == "wall_s":
        return r.get(f"{stage}_s")
    if field == "peak_gb":
        mb = r.get(f"{stage}_rss_mb")
        return mb / 1024 if mb else None
    return None


def successful_prove(r):
    wall_s = get(r, "prove", "wall_s")
    return (
        r.get("status") == "OK"
        and isinstance(wall_s, (int, float))
        and math.isfinite(wall_s)
        and wall_s > 0
    )


def main():
    ap = argparse.ArgumentParser(
        description=None, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("paths", nargs="+")
    ap.add_argument(
        "--out", default=str(REPO / "results" / "inner_cols_2d" / "summary.json")
    )
    args = ap.parse_args()

    recs = load(args.paths)
    if not recs:
        sys.exit("没有可解析的记录；先确认 probe 的输出结构")

    configs = sorted({r["config"] for r in recs})
    print(f"{len(recs)} 条记录，{len(configs)} 个配置\n")

    best = {}
    print(
        f"{'config':22s} {'N':>3s} {'rows':>10s} {'k':>3s} "
        f"{'prove s':>9s} {'prove GB':>9s} {'proof B':>9s} status"
    )
    for c in configs:
        rows = sorted(
            (r for r in recs if r["config"] == c),
            key=lambda r: r.get("num_inner_cols") or 0,
        )
        for r in rows:
            n = r.get("num_inner_cols")
            print(
                f"{c:22s} {n:3d} {r.get('num_rows', 0):10,d} "
                f"{r.get('logrows', 0):3d} "
                f"{(get(r,'prove','wall_s') or 0):9.1f} "
                f"{(get(r,'prove','peak_gb') or 0):9.2f} "
                f"{r.get('proof_bytes', 0):9,d} {r.get('status', 'UNKNOWN')}"
            )
        ok = [r for r in rows if successful_prove(r)]
        if ok:
            b = min(ok, key=lambda r: get(r, "prove", "wall_s"))
            best[c] = b.get("num_inner_cols")
        print()

    print("=== 各配置在成功样本中的最优 N（按 prove 墙钟）===")
    for c in configs:
        print(f"  {c:22s} -> N = {best.get(c, '?')}")
    if not best:
        print("  没有成功的端到端样本，无法比较最优 N。\n")
    elif len(best) < 2:
        print("  只有一个配置有成功样本，无法比较最优 N 是否随配置变化。\n")
    else:
        relation = "相同" if len(set(best.values())) == 1 else "不同"
        print(
            f"  {len(best)}/{len(configs)} 个配置有成功样本；"
            f"这些配置在已测档位中的最优 N {relation}。\n"
        )

    print("=== 固定 N 时，成功样本的代价 ~ 2^logrows 拟合 ===")
    fits = {}
    for n in sorted({r["num_inner_cols"] for r in recs if successful_prove(r)}):
        pts = [
            (2 ** r["logrows"], get(r, "prove", "wall_s"), get(r, "prove", "peak_gb"))
            for r in recs
            if successful_prove(r)
            and r.get("num_inner_cols") == n
            and r.get("logrows")
            and get(r, "prove", "wall_s")
        ]
        if len(pts) < 2:
            continue
        for i, label in ((1, "prove_s"), (2, "prove_gb")):
            xs = [p[0] for p in pts]
            ys = [
                p[i]
                for p in pts
                if isinstance(p[i], (int, float)) and math.isfinite(p[i]) and p[i] > 0
            ]
            if len(ys) != len(xs):
                continue
            a = sum(x * y for x, y in zip(xs, ys)) / sum(x * x for x in xs)
            resid = [abs(y - a * x) / y for x, y in zip(xs, ys)]
            worst = max(resid) * 100
            fits[f"N={n}/{label}"] = round(worst, 1)
            print(f"  N={n:2d} {label:9s} 最大相对残差 {worst:5.1f}%  ({len(xs)} 点)")
    if fits:
        by_n = {}
        for k, v in fits.items():
            n = int(k.split("/")[0][2:])
            by_n.setdefault(n, []).append(v)
        good = sorted(n for n, vs in by_n.items() if max(vs) < 15)
        bad = sorted(n for n, vs in by_n.items() if max(vs) >= 15)
        print()
        if good:
            print(
                f"  N ∈ {good}：最大残差 {max(max(by_n[n]) for n in good):.1f}% —— "
                "低于 15% 的残差阈值"
            )
        if bad:
            print(
                f"  N ∈ {bad}：最大残差 {max(max(by_n[n]) for n in bad):.1f}% —— "
                "达到或超过 15% 的残差阈值"
            )
        print("  拟合仅覆盖上述已测 N 和配置，不推断未测档位的适用区间。")
    else:
        print("  成功样本不足，无法拟合。")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "records": recs,
                "best_n_by_config": best,
                "fixed_n_fit_worst_resid_pct": fits,
            },
            indent=2,
            ensure_ascii=False,
            default=str,
        )
        + "\n"
    )
    print(f"\n[wrote] {out}")


if __name__ == "__main__":
    main()
