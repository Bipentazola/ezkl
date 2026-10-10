#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
CONFIGS=${CONFIGS:-16x16,16x32}
OUTROOT=results/incremental/workload_ratio/reps
mkdir -p "$OUTROOT"
WORKDIR_ROOT=${WORKDIR_ROOT:-${TMPDIR:-/tmp}}
mkdir -p "$WORKDIR_ROOT"

while pgrep -f "run_workload_ratio.py" >/dev/null; do sleep 15; done

run_one() {
    local nt=$1 name=$2
    local wd
    wd=$(mktemp -d "$WORKDIR_ROOT/ezkl_wlr_${name}.XXXXXX")
    echo "=== nt=$nt -> $OUTROOT/$name ($(date -Is)) ===" >&2
    RAYON_NUM_THREADS=$nt python3 scripts/run_workload_ratio.py \
        --configs "$CONFIGS" --workdir "$wd" --out "$OUTROOT/$name" \
        2>&1 | tee "$OUTROOT/$name.log"
    echo "$nt" > "$OUTROOT/$name/threads.txt"
    { echo "RAYON_NUM_THREADS=$nt"
      echo "started=$started"
      echo "finished=$(date -Is)"
      echo "cpu.max=$(cat /sys/fs/cgroup/cpu.max)"
      echo "memory.max=$(cat /sys/fs/cgroup/memory.max)"
      echo "nproc_host_view=$(nproc)"
      echo "loadavg_at_start=$loadavg"
      echo "loadavg_at_end=$(cut -d' ' -f1-3 /proc/loadavg)"
      echo "ezkl_binary=$(ls -l --time-style=long-iso target/release/ezkl | awk '{print $5, $6, $7}')"
    } > "$OUTROOT/$name/env.txt"
    rm -rf "$wd"
}

for i in 1 2 3; do
    started=$(date -Is); loadavg=$(cut -d' ' -f1-3 /proc/loadavg)
    run_one 1 "nt1_rep$i"
done
for nt in 2 4 8 16; do
    started=$(date -Is); loadavg=$(cut -d' ' -f1-3 /proc/loadavg)
    run_one "$nt" "nt$nt"
done

echo "=== 全部完成 ===" >&2
