#!/usr/bin/env bash
set -uo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

EZKL_BIN="${EZKL_BIN:-$REPO_DIR/target/release/ezkl}"
MODEL="${MODEL:-$REPO_DIR/models/BertForSequenceClassification/onnx/model.onnx}"
CONFIG_JSON="${CONFIG_JSON:-$(dirname "$MODEL")/../config.json}"
OUT_DIR="${OUT_DIR:-$REPO_DIR/ezkl_profile_out}"

BATCH_SIZE="${BATCH_SIZE:-1}"
SEQ_LEN="${SEQ_LEN:-128}"
SEED="${SEED:-42}"

INPUT_SCALE="${INPUT_SCALE:-4}"
PARAM_SCALE="${PARAM_SCALE:-4}"
CALIBRATION_TARGET="${CALIBRATION_TARGET:-resources}"
MAX_LOGROWS="${MAX_LOGROWS:-}"
CALIBRATE_SCALES="${CALIBRATE_SCALES:-}"
CALIBRATE_REBASE_MULTIPLIERS="${CALIBRATE_REBASE_MULTIPLIERS:-}"

MASK_CONSTANT_THRESHOLD="${MASK_CONSTANT_THRESHOLD:-1e6}"
MASK_REPLACEMENT="${MASK_REPLACEMENT:--1e4}"

RUN_PROVE="${RUN_PROVE:-1}"

SRS_MODE="${SRS_MODE:-dummy}"

PYTHON_BIN="${PYTHON_BIN:-python3}"

mkdir -p "$OUT_DIR"
REPORT="$OUT_DIR/profile_report.md"
STAGE_TABLE="$OUT_DIR/.stage_rows.tmp"
: > "$STAGE_TABLE"

log() { printf '[profile] %s\n' "$*"; }
die() { printf '[profile] ERROR: %s\n' "$*" >&2; exit 1; }

[ -x "$EZKL_BIN" ] || die "ezkl binary not found/executable at $EZKL_BIN"
[ -f "$MODEL" ] || die "onnx model not found at $MODEL"

run_stage() {
  local name="$1"; shift
  local logfile="$OUT_DIR/${name}.log"
  local t0 t1 dt rc pid maxrss_kb=0 sample

  log "== ${name} =="
  t0=$(date +%s.%N)

  "$@" >"$logfile" 2>&1 &
  pid=$!
  while kill -0 "$pid" 2>/dev/null; do
    if [ -r "/proc/$pid/status" ]; then
      sample=$(awk '/VmHWM/{print $2}' "/proc/$pid/status" 2>/dev/null)
      [ -n "$sample" ] && [ "$sample" -gt "$maxrss_kb" ] 2>/dev/null && maxrss_kb=$sample
    fi
    sleep 0.2
  done
  wait "$pid"
  rc=$?

  t1=$(date +%s.%N)
  dt=$(awk -v a="$t0" -v b="$t1" 'BEGIN{printf "%.2f", b-a}')

  local maxrss="n/a"
  [ "$maxrss_kb" -gt 0 ] 2>/dev/null && maxrss=$(awk -v k="$maxrss_kb" 'BEGIN{printf "%.1f MB", k/1024}')

  local status="ok"
  [ "$rc" -ne 0 ] && status="FAILED (exit $rc)"

  printf '| %s | %ss | %s | %s |\n' "$name" "$dt" "$maxrss" "$status" >> "$STAGE_TABLE"
  log "   time=${dt}s  peak_rss=${maxrss}  status=${status}  (log: $logfile)"

  if [ "$rc" -ne 0 ]; then
    log "   --- tail of $logfile ---"
    tail -n 20 "$logfile" | sed 's/^/[profile]   /'
  fi
  return $rc
}

PATCHED_MODEL="$OUT_DIR/model.patched.onnx"

patch_onnx() {
  "$PYTHON_BIN" - "$MODEL" "$PATCHED_MODEL" "$MASK_CONSTANT_THRESHOLD" "$MASK_REPLACEMENT" <<'PYEOF'
import sys
import onnx
from onnx import numpy_helper
import numpy as np

src, dst, threshold, replacement = sys.argv[1], sys.argv[2], float(sys.argv[3]), float(sys.argv[4])

m = onnx.load(src)
patched = 0
for node in m.graph.node:
    if node.op_type != "Constant":
        continue
    for attr in node.attribute:
        if attr.name != "value":
            continue
        arr = numpy_helper.to_array(attr.t)
        if arr.size and np.abs(arr).max() > threshold:
            new_arr = np.where(
                arr > threshold, abs(replacement),
                np.where(arr < -threshold, -abs(replacement), arr),
            ).astype(arr.dtype)
            new_tensor = numpy_helper.from_array(new_arr, attr.t.name)
            attr.t.CopyFrom(new_tensor)
            patched += 1

onnx.save(m, dst)
print(f"patched {patched} constant node(s) with |value| > {threshold} -> {replacement}")
if patched == 0:
    print("note: no oversized constants found; model was used as-is")
PYEOF
}

INPUT_JSON="$OUT_DIR/input.json"

gen_input() {
  "$PYTHON_BIN" - "$MODEL" "$INPUT_JSON" "$BATCH_SIZE" "$SEQ_LEN" "$SEED" "$CONFIG_JSON" <<'PYEOF'
import sys, json, random
import onnx

model_path, out_path, batch_size, seq_len, seed, config_path = sys.argv[1:7]
batch_size, seq_len, seed = int(batch_size), int(seq_len), int(seed)

vocab_size, type_vocab_size = 30522, 2
try:
    with open(config_path) as f:
        cfg = json.load(f)
    vocab_size = cfg.get("vocab_size", vocab_size)
    type_vocab_size = cfg.get("type_vocab_size", type_vocab_size)
except (FileNotFoundError, json.JSONDecodeError):
    pass

m = onnx.load(model_path)
rng = random.Random(seed)
n = batch_size * seq_len

def values_for(name):
    lname = name.lower()
    if "input_ids" in lname:
        return [float(rng.randint(0, vocab_size - 1)) for _ in range(n)]
    if "token_type" in lname:
        return [float(rng.randint(0, max(type_vocab_size - 1, 0))) for _ in range(n)]
    if "attention_mask" in lname or "mask" in lname:
        return [1.0 for _ in range(n)]
    return [float(rng.randint(0, 99)) for _ in range(n)]

input_data = [values_for(inp.name) for inp in m.graph.input]

with open(out_path, "w") as f:
    json.dump({"input_data": input_data}, f)

names = [inp.name for inp in m.graph.input]
print(f"wrote {out_path}: inputs={names} batch_size={batch_size} sequence_length={seq_len}")
PYEOF
}

VARIABLES="batch_size->${BATCH_SIZE},sequence_length->${SEQ_LEN}"
SETTINGS_JSON="$OUT_DIR/settings.json"
COMPILED_CIRCUIT="$OUT_DIR/model.compiled"
WITNESS_JSON="$OUT_DIR/witness.json"
SRS_PATH="$OUT_DIR/kzg.srs"
VK_PATH="$OUT_DIR/vk.key"
PK_PATH="$OUT_DIR/pk.key"
PROOF_JSON="$OUT_DIR/proof.json"

log "model:        $MODEL"
log "patched onnx: $PATCHED_MODEL"
log "out dir:      $OUT_DIR"
log "batch_size=${BATCH_SIZE} sequence_length=${SEQ_LEN} input_scale=${INPUT_SCALE} param_scale=${PARAM_SCALE}"
echo

overall_rc=0

run_stage "00_patch_onnx" patch_onnx || overall_rc=1
run_stage "01_gen_input" gen_input || overall_rc=1

[ $overall_rc -eq 0 ] && run_stage "02_gen_settings" \
  "$EZKL_BIN" gen-settings \
    -M "$PATCHED_MODEL" \
    -O "$SETTINGS_JSON" \
    -S "$INPUT_SCALE" --param-scale "$PARAM_SCALE" \
    -V "$VARIABLES" \
  || overall_rc=1

if [ $overall_rc -eq 0 ]; then
  calib_args=(calibrate-settings -M "$PATCHED_MODEL" -D "$INPUT_JSON" -O "$SETTINGS_JSON" --target "$CALIBRATION_TARGET")
  [ -n "$MAX_LOGROWS" ] && calib_args+=(--max-logrows "$MAX_LOGROWS")
  [ -n "$CALIBRATE_SCALES" ] && calib_args+=(--scales "$CALIBRATE_SCALES")
  [ -n "$CALIBRATE_REBASE_MULTIPLIERS" ] && calib_args+=(--scale-rebase-multiplier "$CALIBRATE_REBASE_MULTIPLIERS")
  run_stage "03_calibrate_settings" "$EZKL_BIN" "${calib_args[@]}" || overall_rc=1
fi

[ $overall_rc -eq 0 ] && run_stage "04_compile_circuit" \
  "$EZKL_BIN" compile-circuit \
    -M "$PATCHED_MODEL" \
    --compiled-circuit "$COMPILED_CIRCUIT" \
    -S "$SETTINGS_JSON" \
  || overall_rc=1

[ $overall_rc -eq 0 ] && run_stage "05_gen_witness" \
  "$EZKL_BIN" gen-witness \
    -D "$INPUT_JSON" \
    -M "$COMPILED_CIRCUIT" \
    -O "$WITNESS_JSON" \
  || overall_rc=1

[ $overall_rc -eq 0 ] && run_stage "06_mock" \
  "$EZKL_BIN" mock \
    -W "$WITNESS_JSON" \
    -M "$COMPILED_CIRCUIT" \
  || overall_rc=1

if [ $overall_rc -eq 0 ] && [ "$RUN_PROVE" = "1" ]; then
  logrows=$("$PYTHON_BIN" -c "import json; print(json.load(open('$SETTINGS_JSON'))['run_args']['logrows'])" 2>/dev/null)
  log "logrows selected by calibration: ${logrows:-unknown}"

  if [ "$SRS_MODE" = "download" ]; then
    run_stage "07_get_srs" "$EZKL_BIN" get-srs --srs-path "$SRS_PATH" -S "$SETTINGS_JSON" || overall_rc=1
  else
    [ -n "$logrows" ] || die "could not read logrows from $SETTINGS_JSON"
    run_stage "07_gen_srs" "$EZKL_BIN" gen-srs --srs-path "$SRS_PATH" --logrows "$logrows" || overall_rc=1
  fi

  [ $overall_rc -eq 0 ] && run_stage "08_setup" \
    "$EZKL_BIN" setup \
      -M "$COMPILED_CIRCUIT" \
      --srs-path "$SRS_PATH" \
      --vk-path "$VK_PATH" \
      --pk-path "$PK_PATH" \
    || overall_rc=1

  [ $overall_rc -eq 0 ] && run_stage "09_prove" \
    "$EZKL_BIN" prove \
      -W "$WITNESS_JSON" \
      -M "$COMPILED_CIRCUIT" \
      --pk-path "$PK_PATH" \
      --proof-path "$PROOF_JSON" \
      --srs-path "$SRS_PATH" \
    || overall_rc=1

  [ $overall_rc -eq 0 ] && run_stage "10_verify" \
    "$EZKL_BIN" verify \
      --proof-path "$PROOF_JSON" \
      --settings-path "$SETTINGS_JSON" \
      --vk-path "$VK_PATH" \
      --srs-path "$SRS_PATH" \
    || overall_rc=1
else
  [ "$RUN_PROVE" != "1" ] && log "RUN_PROVE=0: skipping gen-srs/setup/prove/verify"
fi

{
  echo "# ezkl profiling report"
  echo
  echo "- model: \`$MODEL\`"
  echo "- patched onnx: \`$PATCHED_MODEL\`"
  echo "- batch_size=${BATCH_SIZE} sequence_length=${SEQ_LEN}"
  echo "- input_scale=${INPUT_SCALE} param_scale=${PARAM_SCALE} calibration_target=${CALIBRATION_TARGET}"
  echo "- generated: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo
  if [ -f "$SETTINGS_JSON" ]; then
    logrows=$("$PYTHON_BIN" -c "import json; d=json.load(open('$SETTINGS_JSON'))['run_args']; print(f\"logrows={d['logrows']} input_scale={d['input_scale']} param_scale={d['param_scale']}\")" 2>/dev/null)
    [ -n "$logrows" ] && echo "- calibrated settings: $logrows"
    echo
  fi
  echo "| stage | wall time | peak RSS | status |"
  echo "|---|---|---|---|"
  cat "$STAGE_TABLE"
} > "$REPORT"
rm -f "$STAGE_TABLE"

echo
log "report written to $REPORT"
cat "$REPORT"

exit $overall_rc
