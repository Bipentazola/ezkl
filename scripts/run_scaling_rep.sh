#!/usr/bin/env bash
set -uo pipefail
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

EZKL_BIN="${EZKL_BIN:-$REPO_DIR/target/release/ezkl}"
MODEL="${MODEL:?set MODEL to the onnx model path}"
OUT_DIR="${OUT_DIR:?set OUT_DIR}"
SEQ_LEN="${SEQ_LEN:?set SEQ_LEN}"
BATCH_SIZE="${BATCH_SIZE:-1}"
REPS="${REPS:-3}"
CALIBRATE_SCALES="${CALIBRATE_SCALES:-}"
export RAYON_NUM_THREADS="${RAYON_NUM_THREADS:-8}"
INPUT_SCALE="${INPUT_SCALE:-4}"
PARAM_SCALE="${PARAM_SCALE:-4}"

mkdir -p "$OUT_DIR"
PPROF_DIR="$OUT_DIR/pprof"
mkdir -p "$PPROF_DIR"

echo "[scaling] model=$MODEL seq_len=$SEQ_LEN out_dir=$OUT_DIR reps=$REPS calibrate_scales='${CALIBRATE_SCALES}'"

export EZKL_PPROF_OUT="$PPROF_DIR/rep1"
EZKL_BIN="$EZKL_BIN" MODEL="$MODEL" OUT_DIR="$OUT_DIR" BATCH_SIZE="$BATCH_SIZE" SEQ_LEN="$SEQ_LEN" \
  INPUT_SCALE="$INPUT_SCALE" PARAM_SCALE="$PARAM_SCALE" \
  CALIBRATE_SCALES="$CALIBRATE_SCALES" RUN_PROVE=1 SRS_MODE=dummy \
  "$REPO_DIR/scripts/profile_ezkl.sh"
rc=$?
unset EZKL_PPROF_OUT
if [ $rc -ne 0 ]; then
  echo "[scaling] profile_ezkl.sh failed (rc=$rc) -- aborting extra reps for $OUT_DIR"
  exit $rc
fi

if [ ! -f "$PPROF_DIR/rep1.folded" ]; then
  echo "[scaling] WARNING: rep1 produced no .folded file at $PPROF_DIR/rep1.folded -- was the binary built with --features pprof-profiling and EZKL_PPROF_OUT respected?"
fi

COMPILED_CIRCUIT="$OUT_DIR/model.compiled"
WITNESS_JSON="$OUT_DIR/witness.json"
PK_PATH="$OUT_DIR/pk.key"
SRS_PATH="$OUT_DIR/kzg.srs"

for i in $(seq 2 "$REPS"); do
  echo "[scaling] === extra prove rep $i / $REPS ==="
  t0=$(date +%s.%N)
  EZKL_PPROF_OUT="$PPROF_DIR/rep${i}" "$EZKL_BIN" prove \
    -W "$WITNESS_JSON" -M "$COMPILED_CIRCUIT" --pk-path "$PK_PATH" \
    --proof-path "$OUT_DIR/proof_rep${i}.json" --srs-path "$SRS_PATH" \
    > "$OUT_DIR/prove_rep${i}.log" 2>&1
  rc=$?
  t1=$(date +%s.%N)
  dt=$(awk -v a="$t0" -v b="$t1" 'BEGIN{printf "%.2f", b-a}')
  echo "[scaling] rep $i rc=$rc time=${dt}s (log: $OUT_DIR/prove_rep${i}.log, folded: $PPROF_DIR/rep${i}.folded)"
  if [ $rc -ne 0 ]; then
    tail -n 20 "$OUT_DIR/prove_rep${i}.log"
  fi
done

echo "[scaling] done. folded files:"
ls -la "$PPROF_DIR"/*.folded 2>/dev/null
