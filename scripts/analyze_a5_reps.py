#!/usr/bin/env python3

import json, statistics as st, sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_rescued_configs import parse_time  # noqa: E402  (复用同一个 time -v 解析)

REPS = sorted((REPO / "results/rescued_configs/reps").glob("rep*/rescued_configs.json"))
PUB = json.loads((REPO / "results/rescued_configs/rescued_configs.json").read_text())
LIMIT_GIB = 162.0

pub = {r["config"]: r for r in PUB}

DEMO = REPO / "results" / "tier_crossing_demo"
if "seq8_d384_l1" not in pub and (DEMO / "setup.time").exists():
    pub["seq8_d384_l1"] = {
        "setup": parse_time(DEMO / "setup.time"),
        "prove": parse_time(DEMO / "prove.time"),
        "_source": "results/tier_crossing_demo/",
    }
runs = {}
for f in REPS:
    for r in json.loads(f.read_text()):
        runs.setdefault(r["config"], []).append((f.parent.name, r))

print("## 核验一：峰值 RSS 是否都在 162 GiB 安全线内（线 = %.1f GiB）\n" % LIMIT_GIB)
worst = 0.0
bad = []
for cfg, lst in runs.items():
    for rep, r in lst:
        for stage in ("setup", "prove"):
            v = r[stage]["peak_rss_gb"]
            worst = max(worst, v)
            if v >= LIMIT_GIB:
                bad.append((cfg, rep, stage, v))
print(
    "   最高一次峰值：%.1f GiB（安全线 %.1f GiB，余量 %.1f GiB = %.0f%%）"
    % (worst, LIMIT_GIB, LIMIT_GIB - worst, 100 * (LIMIT_GIB - worst) / LIMIT_GIB)
)
print("   越线的运行：%s\n" % ("无" if not bad else bad))

print("## 状态：三轮 × 三配置是否都 verified\n")
allok = all(
    r["status"] == "OK" and r.get("verified") for l in runs.values() for _, r in l
)
for cfg, lst in sorted(runs.items()):
    print(
        "   %-15s %s" % (cfg, " ".join("%s:%s" % (rep, r["status"]) for rep, r in lst))
    )
print("   全部 verified：%s\n" % ("是" if allok else "否"))

print("## 噪声底（3 次重复，极差/中位）与已发表单次值的对照\n")
for cfg in sorted(runs):
    src = pub.get(cfg, {}).get("_source")
    if src:
        print(
            "   注：%s 的已发表基准取自 %s（同一台一号机、同一口径，驱动脚本不同）\n"
            % (cfg, src)
        )
print(
    "| 配置 | 指标 | 三次值 | 中位 | 极差/中位 | 已发表(一号机) | 已发表偏离中位 | 落在带内 |"
)
print("|---|---|---|---|---|---|---|---|")
summary = {}
for cfg, lst in sorted(runs.items()):
    for stage in ("setup", "prove"):
        for key, unit, fmt in (("wall_s", "s", "%.1f"), ("peak_rss_gb", "GB", "%.1f")):
            vals = [r[stage][key] for _, r in lst]
            med = st.median(vals)
            rng = max(vals) - min(vals)
            rel = 100 * rng / med
            p = pub[cfg][stage][key]
            dev = 100 * (p - med) / med
            inband = min(vals) <= p <= max(vals)
            summary.setdefault((stage, key), []).append(rel)
            print(
                "| %s | %s %s | %s | %s | %.2f%% | %s | %+.2f%% | %s |"
                % (
                    cfg,
                    stage,
                    key.replace("wall_s", "时间").replace("peak_rss_gb", "内存"),
                    ", ".join(fmt % v for v in vals),
                    fmt % med,
                    rel,
                    fmt % p,
                    dev,
                    "是" if inband else "否",
                )
            )

print("\n## 汇总的噪声底\n")
for (stage, key), rels in summary.items():
    lab = "时间" if key == "wall_s" else "内存"
    print(
        "   %-6s %s ：极差/中位 中位 %.2f%%、最大 %.2f%%"
        % (stage, lab, st.median(rels), max(rels))
    )
