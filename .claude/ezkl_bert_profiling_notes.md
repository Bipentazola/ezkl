# ezkl profiling notes: BertForSequenceClassification

## TL;DR

**A full end-to-end proof (`setup` → `prove` → `verify`) cannot be completed
for this model in any tested environment, up to ~130 GiB.** This is not a
tunable-parameter problem: `logrows=22` is the smallest valid circuit for
this model (floor set by the `exp`/`tanh`/`erf` lookup-table domain; any
lower scale fails Halo2 synthesis outright), and even at that floor,
`setup`'s proving-key (PK) generation gets OOM-killed past 130 GB right
after the verifying key (VK) finishes (~206s). `prove` was never reached.
Only the pre-proving pipeline — `patch_onnx → gen_input → gen_settings →
calibrate_settings → compile_circuit → gen_witness → mock` — completes
reliably (peak ~17 GB, `mock` itself ~280s). Time breakdown for that part is
in "Smallest config that actually runs clean" below. To get real
`setup`/`prove`/`verify` numbers would need a much larger box (256 GB+,
untested) or a smaller/pruned model — not further ezkl parameter tuning.

Context: profiling `models/BertForSequenceClassification/onnx/model.onnx` with the
compiled `target/release/ezkl` binary. Reusable script lives at
`scripts/profile_ezkl.sh` (run `./scripts/profile_ezkl.sh --help`-style via env vars,
see header of the script for all options).

## Problems hit and how they were solved

### 1. `-3.4028235e+38` mask constant breaks quantization

HF's ONNX export of BERT bakes the attention mask math as
`(1 - attention_mask) * torch.finfo(float32).min`, i.e. a literal
`-3.4028235e+38` `Constant` node feeding into the attention-score add. ezkl's
fixed-point quantizer uses `i128` internally (`IntegerRep = i128`,
`src/fieldutils.rs:6`), whose max (~1.7e38) is smaller than `|float32::MIN|`
(~3.4e38) even at `scale=0`, so `gen-settings`/`table` always fail with:

```
[graph] [tensor] significant bit truncation when instantiating, try lowering the scale
```

Lowering `--input-scale`/`--param-scale` does **not** fix this — the constant
is unrepresentable at any scale.

**Fix**: patch the ONNX graph before running ezkl, replacing any scalar
`Constant` node whose `|value| > threshold` (default `1e6`) with a
sign-preserving, numerically-equivalent-after-softmax replacement (default
`-1e4`). Implemented as stage `00_patch_onnx` in `scripts/profile_ezkl.sh`.
Confirmed only one such node exists in this model
(`/bert/Constant_2`).

### 2. `ezkl gen-random-data` doesn't produce valid inputs for this graph

`gen-random-data` applies a single `--min`/`--max` to every input tensor. For
BERT's three inputs (`input_ids`, `attention_mask`, `token_type_ids`) this
produces out-of-range `token_type_ids` (this model's config has
`type_vocab_size=16`, `vocab_size=1124`), which crashes tract with an
out-of-bounds `Gather` during `calibrate-settings`:

```
[graph] [tract] Evaluating #5 "/bert/embeddings/token_type_embeddings/Gather" Gather
```

**Fix**: stage `01_gen_input` generates input.json manually per input name:
`input_ids` ∈ `[0, vocab_size)`, `token_type_ids` ∈ `[0, type_vocab_size)`,
`attention_mask` = all `1.0`. Reads `vocab_size`/`type_vocab_size` from the
model's `config.json` when available, falls back to generic ints otherwise.

### 3. `mock`/`setup`/`prove` OOM in this container regardless of scale/seq_len

This container's cgroup memory limit is **~17 GiB**
(`/sys/fs/cgroup/memory.max` = 18253611008 bytes), even though the host has
500 GiB RAM. `calibrate-settings --target resources` requires
`param_scale ∈ [11,14]` for this model — anything ≤ 7 fails Halo2 synthesis
outright (not a memory issue, a correctness one: softmax/layernorm need that
much fixed-point precision for this network). But scale ≥ 11 forces the
`exp`/`tanh`/`erf` lookup tables to span a domain of ~4.19M entries
(≈ 2^22), which sets a floor on circuit size **independent of
`sequence_length`**.

Empirically tested (all `batch_size=1`):

| sequence_length | calibrate scale search | picked (input/param scale, logrows) | `mock` result |
|---|---|---|---|
| 128 | default (11-13 grid) | 0 / 13, logrows=**26** (max allowed) | OOM-killed, peak ~16.4 GB, in 14s |
| 32  | default (11-13 grid) | 0 / 13, logrows=**22** | OOM-killed, peak ~16.4 GB, in 206s |
| 32  | forced `2,4,7`        | **calibration itself fails** (`General synthesis error` at every candidate) | n/a |
| 32  | forced `8,9,10,11`   | 0 / 11, logrows=**22** (same floor) | OOM-killed, peak ~16.4 GB, in 246s |

Conclusion: shrinking `sequence_length` reduces `gen-settings`/`gen-witness`
cost nicely, but does **not** shrink the `mock`/`setup`/`prove` circuit below
the container's memory ceiling for this model, because the lookup-table
floor is set by required precision, not sequence length. To get real
`setup`/`prove`/`verify` numbers for this model, rerun
`scripts/profile_ezkl.sh` (unmodified) in an environment with a higher
memory cgroup limit — 64 GB+ is a reasonable starting guess given the
observed ~16.4 GB and rising trend before being killed.

**Update (2026-07-18, container with `/sys/fs/cgroup/memory.max` ≈ 130 GiB):**
re-ran `scripts/profile_ezkl.sh` unmodified (`SEQ_LEN=128`, `RUN_PROVE=1`).
Stages through `gen-witness` succeeded (same calibrated settings:
`logrows=26 input_scale=0 param_scale=13`). `mock` was **still OOM-killed**
(exit 137), but this time peak RSS reached **~132 GB** before being killed
(vs. ~16.4 GB under the ~17 GiB limit) — i.e. the process keeps consuming
whatever headroom it's given rather than plateauing, so the true requirement
is well above 130 GB, not the 64 GB originally guessed. Because the script
gates `gen-srs`/`setup`/`prove`/`verify` on `mock` succeeding
(`overall_rc -eq 0`, see `scripts/profile_ezkl.sh:266-308`), none of those
downstream stages ran again — still no real prove/verify numbers for this
model at `logrows=26`.

### Smallest config that actually runs clean (no OOM kill)

Found by shrinking `SEQ_LEN` and forcing the calibration grid to the
already-known-good scale range:

```bash
OUT_DIR=ezkl_profile_out_small SEQ_LEN=32 CALIBRATE_SCALES="11,12" \
  RUN_PROVE=1 SRS_MODE=dummy ./scripts/profile_ezkl.sh
```

Result: `logrows=22 input_scale=0 param_scale=12` (same floor as every
previous seq_len=32 attempt — `logrows=22` is a hard floor for this model,
set by the `exp`/`tanh`/`erf` lookup-table domain, not reducible via scale
or sequence length; scales below ~8 fail Halo2 synthesis outright, so 22 is
as small as this model's circuit gets). At this floor:

| stage | wall time | peak RSS | status |
|---|---|---|---|
| gen_settings | 15.8s | 114 MB | ok |
| calibrate_settings | 129s | 166 MB | ok |
| gen_witness | 17.5s | 116 MB | ok |
| **mock** | 280s | **17.0 GB** | **ok** |
| gen_srs (dummy) | 106s | 1.3 GB | ok |
| setup | 695s | **131.7 GB** | **FAILED, OOM-killed (exit 137)** |

`mock` now reliably passes (confirmed twice — once inside the full script
run, once standalone via `ezkl mock -M model.compiled -W witness.json`
reusing cached lookup tables, ~2 min, exit 0, `succeeded`). So **the
smallest test that runs to completion without being killed is the pipeline
up through `mock`** (`RUN_PROVE=0`, `SEQ_LEN=32`, `CALIBRATE_SCALES="11,12"`)
— `patch_onnx` → `gen_input` → `gen_settings` → `calibrate_settings` →
`compile_circuit` → `gen_witness` → `mock`, all comfortably under budget
(peak ~17 GB).

`setup` is a different story: its log shows the verifying key (VK) finishes
fine (~206s), then proving-key (PK) generation starts a second `model
layout...` pass and gets OOM-killed climbing past 130 GB. This confirms PK
generation is intrinsically far more memory-hungry than `mock`'s
`MockProver`, **even at the smallest valid circuit size for this model**
(`logrows=22`). So `setup`/`prove`/`verify` are not reachable for
`BertForSequenceClassification` in any container up to ~130 GiB regardless
of `SEQ_LEN` — getting real numbers for those stages needs either a
much larger box (256 GB+, untested) or a smaller/pruned model (fewer
layers/heads), not just ezkl parameter tuning — the ezkl-tunable
parameters are already at their floor.

## Clean results obtained (stages that do work in this container)

From the seq_len=32 runs (representative; seq_len=128 numbers are similar in
kind, just ~5-8x slower on `gen-settings`/`calibrate-settings`/`gen-witness`):

| stage | wall time | peak RSS |
|---|---|---|
| patch onnx | 0.4s | ~2 MB |
| gen random (valid) input | 0.4s | ~2 MB |
| gen-settings | ~12s | ~110 MB |
| calibrate-settings (default grid) | ~130-145s | ~150-180 MB |
| compile-circuit | ~0.2s | a few MB |
| gen-witness | ~12-14s | ~105-125 MB |
| mock | fails (OOM) in this container | >16.4 GB and rising |

At seq_len=128, `gen-settings` takes ~78s (602 MB) and `calibrate-settings`
(full default 11-14 x rebase-multiplier grid) takes ~19 minutes (768 MB) —
worth knowing if re-running with the full sequence length, budget time
accordingly.

## Script usage cheatsheet

```bash
# quick shape/settings-only profile, skip the memory-heavy stages
RUN_PROVE=0 SEQ_LEN=32 ./scripts/profile_ezkl.sh

# full pipeline incl. setup/prove/verify (needs a bigger-memory box for this model)
SEQ_LEN=128 ./scripts/profile_ezkl.sh

# narrow calibrate-settings' scale search to speed up repeated runs
CALIBRATE_SCALES="11,12" SEQ_LEN=32 ./scripts/profile_ezkl.sh
```

Key env vars: `MODEL`, `OUT_DIR`, `BATCH_SIZE`, `SEQ_LEN`, `INPUT_SCALE`,
`PARAM_SCALE`, `CALIBRATION_TARGET`, `MAX_LOGROWS`, `CALIBRATE_SCALES`,
`CALIBRATE_REBASE_MULTIPLIERS`, `RUN_PROVE` (0/1), `SRS_MODE` (`dummy`/
`download`), `MASK_CONSTANT_THRESHOLD`, `MASK_REPLACEMENT`.

Output per run: `$OUT_DIR/profile_report.md` (summary table) plus one
`$OUT_DIR/<NN_stage>.log` per stage with full ezkl stdout/stderr.
