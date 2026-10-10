#!/usr/bin/env bash
set -u
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPS="${1:-3}"
shift || true
CFG=("$@")
SUFFIX=""
ARGS=()
if [ "${#CFG[@]}" -gt 0 ]; then
  SUFFIX="_$(IFS=-; echo "${CFG[*]}")"
  ARGS=(--configs "${CFG[@]}")
fi
cd "$REPO"

for r in $(seq 1 "$REPS"); do
  wd="results/rescued_configs/reps/work_rep${r}${SUFFIX}"
  out="results/rescued_configs/reps/rep${r}${SUFFIX}"
  mkdir -p "$wd" "$out"
  [ -e "$wd/kzg21.srs" ] || ln -s "$REPO/results/rescued_configs/work/kzg21.srs" "$wd/kzg21.srs"
  echo "########## rep $r / $REPS  start $(date -Is) ##########"
  taskset -c 0-47 python3 scripts/run_rescued_configs.py --workdir "$wd" --out "$out" ${ARGS[@]+"${ARGS[@]}"}
  echo "########## rep $r / $REPS  done  $(date -Is) ##########"
done
echo "########## ALL DONE $(date -Is) ##########"
