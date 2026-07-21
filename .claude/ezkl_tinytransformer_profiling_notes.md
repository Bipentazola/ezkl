# ezkl profiling notes: single-layer TinyTransformer

## TL;DR

Generator script: `scripts/gen_transformer_onnx.py` builds a configurable
transformer encoder (any depth/width) and exports a clean ONNX graph using
only ops already known to work with ezkl/tract (manually decomposed
LayerNorm, erf-based GELU, Softmax, MatMul/Gemm — same op set as the working
BERT export, see `.claude/ezkl_bert_profiling_notes.md`). It writes `<out>/onnx/model.onnx`
+ `<out>/config.json` in the same layout `scripts/profile_ezkl.sh` already
expects, so no script changes were needed to profile it.

Generated a 1-layer transformer (hidden=64, heads=4, ffn=256, vocab=128,
seq_len=16, ~60K params) and ran it through `scripts/profile_ezkl.sh`.
`mock`/`setup` worked immediately; `prove` initially failed with
`[halo2] General synthesis error` — **this turned out to be a real bug in
the pinned halo2 fork, now found, fixed, and confirmed working end-to-end**
(`prove` + `verify` both succeed). See "Root cause and fix" below.

**Do not repeat the fork-bug hunt** — the fix is a one-line `Cargo.toml`
pin bump (see below). If `prove` fails again after a `cargo update`, check
whether `Cargo.toml`'s `[patch.'https://github.com/zkonduit/halo2']` rev
got reverted or whether a *newer* fork commit reintroduced something
similar, rather than re-deriving all of this from scratch.

## Root cause and fix (found 2026-07-19)

**Symptom**: `prove` (never `mock`, never `setup`) fails with
`[halo2] General synthesis error` for *any* circuit with 2 or more
matmul/Gemm ops anywhere in the graph — confirmed via minimal isolated
repros (see "Isolation process" below), independent of whether they're
chained/dependent, independent of any lookup op (softmax/erf/relu) being
present or absent, independent of `logrows`. This is not
transformer-specific and not specific to this generator script — it broke
`prove` for essentially any non-trivial model (even a plain 2-linear-layer
MLP) on this ezkl build.

**Root cause**: `src/graph/model.rs:1104` — whenever a compiled circuit's
`settings.einsum_params.total_einsum_col_size > 0` (roughly: the graph has
enough/large-enough matmuls that ezkl can't use its simplest single-matmul
"base ops" path), `configure_einsums()` →
`src/circuit/ops/chip/einsum/mod.rs`'s `configure_universal()` allocates
**second-phase (`Phase::Second`) advice columns** for a Freivalds-argument
random-linear-combination (RLC) gate that probabilistically verifies matrix
products. This makes the circuit genuinely two-phase, so real `create_proof`
does two full `model layout...` synthesis passes (visible in logs) — the
*second* one, which fills in the real challenge-dependent RLC columns, is
where it crashes. `mock` and `setup`'s keygen never do a real
challenge-dependent second pass (dummy/placeholder assignments only), which
is why they never hit it. Note: `settings.json`'s `disable_freivalds` field
looked like a relevant escape hatch but is a red herring — it's only wired
into `dummy_layout` (`src/graph/model.rs:1498`, used solely for
calibration's constraint-count estimate), **not** into the real
`configure()` used by `compile-circuit`/`setup`/`prove`, so toggling it in
settings.json and recompiling has zero effect on the real circuit.

The actual bug was one level down, in the pinned halo2 fork itself:
`Cargo.toml`'s `[patch.'https://github.com/zkonduit/halo2']` pins
`halo2_proofs` to commit `1dd2090741f006fd031a07da7f3c9dfce5e0015e`
("fix: mvlookup", 2025-08-12) from the local mirror at
`/root/.cache/ezkl-halo2-mirror` (branch `ac/conditional-compilation-icicle2`).
The very next commit on that same branch —
`01c88842679b4308e43ae5ed91c4183e861669bd`, "feat: allow assignment of
values from previous phase to next phase (#26)" (2025-08-25) — is a 4-line
fix to `halo2_proofs/src/plonk/prover.rs`:

```diff
-            // Ignore assignment of advice column in different phase than current one.
-            if self.current_phase != column.column_type().phase {
+            // Ignore assignment of advice column in later phase than current one.
+            // Okay: Assigning Phase 1 value to Phase 2 column
+            // Not okay: Assigning Phase 2 value to Phase 1 column
+            if self.current_phase < column.column_type().phase {
                 return Ok(());
             }
```

Before this fix: during the second-phase re-synthesis pass, any attempt to
*write* a value into an earlier-phase column was silently dropped (the old
`!=` check matched and skipped it), leaving that cell as `Value::unknown()`
forever. Later code reading that cell via `.ok_or(Error::Synthesis)`
(`halo2_proofs/src/circuit/value.rs`) then throws exactly the observed
error. The einsum RLC gate's `first_phase_inputs`/`second_phase_inputs`
split (`src/circuit/ops/chip/einsum/mod.rs:669`) needs exactly this
cross-phase write, which is why it broke specifically (and only) for
circuits that trigger the einsum/Freivalds path.

**Fix applied**: bumped the pin in `Cargo.toml` from `1dd2090...` to
`01c88842...` (one commit forward, same branch, from the same local
mirror — no other changes needed), then `cargo update -p halo2_proofs`
to refresh `Cargo.lock`.

**False lead, corrected**: while validating the fix with a debug build, I
first reused the *old* `model.compiled` (compiled by the pre-fix binary)
with the new patched binary for `setup`, which failed with
`k = 20 is too small for the given circuit`. This led to a wrong
conclusion — that the fix "increases the true row requirement" and
calibration now undercounts. **That was an artifact of mixing a
stale/incompatible compiled-circuit file with an upgraded binary, not a
real property of the fix.** A full fresh run — `gen-settings` →
`calibrate-settings` → `compile-circuit` → `gen-witness` → `mock` →
`gen-srs` → `setup` → `prove` → `verify`, all with the *same* fixed
release binary throughout (`scripts/profile_ezkl.sh` end to end, no manual
steps) — succeeds completely at the originally-calibrated **`logrows=20`**,
no bump needed. **Lesson: after changing the ezkl/halo2 binary, always
recompile the circuit from scratch (`compile-circuit`) rather than reusing
an old `model.compiled`/`witness.json` — the compiled circuit format can
silently encode binary-version-dependent assumptions.**

**Verified end-to-end** (`models/TinyTransformer1L`, logrows=20, release
`target/release/ezkl`, full fresh `scripts/profile_ezkl.sh` run):

| stage | wall time | peak RSS | status |
|---|---|---|---|
| patch_onnx | 0.42s | 2.0 MB | ok |
| gen_input | 0.42s | 1.4 MB | ok |
| gen_settings | 2.46s | 58 MB | ok |
| calibrate_settings | 29.4s | 81 MB | ok → logrows=20, input_scale=0, param_scale=13 |
| compile_circuit | 0.21s | 1.4 MB | ok |
| gen_witness | 2.88s | 51 MB | ok |
| mock | 45.7s | 3.9 GB | ok |
| gen_srs (dummy) | 26.9s | 329 MB | ok |
| setup | 210s | 34.0 GB | ok — vk.key 7.9 MB, pk.key 20.3 GB |
| **prove** | **169s** | 49.2 GB | **ok** |
| **verify** | **0.6s** | 1.4 GB | **ok — `verified: true`** |

(For comparison, the same `prove` step took ~1519s / 25min on an
unoptimized debug build used earlier just to validate the fix quickly —
release is ~9x faster, as expected.) `pk.key` (20.3 GB) was deleted after
recording its size; everything else was kept in
`ezkl_profile_out_tinytransformer_release/`.

### Isolation process (how the 2+-matmul trigger was found)

Empirically bisected with tiny hand-built ONNX models (logrows=15, full
`setup`+`prove` cycle takes seconds each, so this was fast):

| model | matmuls | lookup ops | prove result (pre-fix) |
|---|---|---|---|
| `1l_linear` (repo example) | 1 (Gemm) | none | ok |
| `1l_erf` (repo example) | 0 | 1 (erf) | ok |
| `1l_mlp` (repo example) | 1 (Gemm) | 1 (relu) | ok |
| `1l_softmax` (repo example) | 0 | 1 (softmax/exp) | ok |
| minimal Q·Kᵀ→softmax→·V attention | 5 | 1 (softmax) | **FAILS** |
| 2 independent (unrelated) `nn.Linear` on the same input, summed | 2 | 0 | **FAILS** |
| 2 chained matmuls, no softmax at all | 5 | 0 | **FAILS** |
| `examples/onnx/multihead_attention` (repo example) | many | 1 (softmax) | **FAILS** |
| `TinyTransformer1L` (this generator) | many | 2 (softmax+erf) | **FAILS** |

The one clean split: **≤1 matmul always worked, ≥2 always failed**,
regardless of lookups. That pointed straight at the einsum/Freivalds
second-phase code path rather than anything lookup- or
attention-specific.

**Process note for next time**: don't conclude "the library/repo has a
bug" from one failing complex case — first confirm the simplest possible
known-good case in the same environment/build actually works (here:
`1l_linear`, `1l_erf`), *then* binary-search a minimal repro varying one
property at a time. Going straight from "my model" or "one other complex
model" to "upstream is broken" is an overclaim the user correctly pushed
back on mid-session — see memory `feedback-verify-before-blaming-deps`.

## Results table

See the definitive, clean numbers under "Root cause and fix" above
(release build, logrows=20, full fresh pipeline, `prove`+`verify` both
succeed). Earlier numbers gathered pre-fix and mid-fix-validation (debug
build, manually-bumped logrows=21) are superseded — the fix does not
actually require a logrows bump when the circuit is compiled fresh with
the fixed binary throughout; see the "False lead, corrected" note above.
`ezkl_profile_out_tinytransformer/` (pre-fix + debug-validation artifacts)
and `ezkl_profile_out_tinytransformer_release/` (final clean release-build
run) both still exist on disk if you need to compare logs directly.

## Practical takeaway

`mock` + `setup` give real, trustworthy resource numbers (time, peak RSS,
key sizes) fast even without a working `prove`, so lean on those for quick
iteration. Now that `prove`/`verify` work too (after the halo2 pin bump
above), a full profiling pass just works — `scripts/profile_ezkl.sh`
end-to-end, no manual logrows bump needed, **as long as the whole pipeline
(gen-settings through verify) is run with the same binary** — never mix a
`model.compiled`/`witness.json` produced by one ezkl build with a
`setup`/`prove` run from a different build. Always rebuild **release**,
not debug, before trusting `prove`/`verify` timing numbers (release was
~9x faster than debug for `prove` on this model).

## Disk-space / persistence gotchas hit while profiling

The container's root filesystem (`/`, including all of `/tmp` **and**
`/root`) is a small ~45 GB overlay that **does not persist across
sessions**, separate from the ~89 GB `/data` volume which does persist.
`pk.key` files for even a *tiny* (60K-param) transformer at logrows≈20 are
tens of GB (19–38 GB here) — writing one to `/tmp` instead of `/data`
filled the root fs completely and broke every tool that needs to write
anywhere under `/tmp` (including this harness's own per-command output
capture), because `/tmp` isn't a separate mount. Always point large ezkl
artifacts (`--srs-path`, `--pk-path`, `OUT_DIR`) at `/data`, never `/tmp`,
on this host.

This also bit the Rust toolchain: this container had no `cargo`/`rustc` at
all (a prior session's toolchain, installed to the default `~/.cargo`
under `/root`, was gone). Reinstalled with `rustup-init.sh` but with
`RUSTUP_HOME=/data/rust/.rustup CARGO_HOME=/data/rust/.cargo` so it
survives future sessions — add `/data/rust/.cargo/bin` to `PATH` (and
re-export those two env vars) in any future session that needs to build
this repo. The project pins `nightly-2025-12-01` via `rust-toolchain.toml`,
which rustup picks up automatically once `cargo`/`rustc` are on `PATH`.

## Script usage cheatsheet

```bash
# generate a 1-layer toy transformer
python3 scripts/gen_transformer_onnx.py \
  --output-dir models/TinyTransformer1L \
  --num-layers 1 --hidden-size 64 --num-heads 4 \
  --intermediate-size 256 --vocab-size 128 --max-seq-len 32 --seq-len 16

# profile it (writes into /data/coding/ezkl/<OUT_DIR>, never /tmp)
OUT_DIR=ezkl_profile_out_tinytransformer \
  MODEL=models/TinyTransformer1L/onnx/model.onnx \
  BATCH_SIZE=1 SEQ_LEN=16 RUN_PROVE=1 SRS_MODE=dummy \
  ./scripts/profile_ezkl.sh
```

Key generator flags: `--num-layers`, `--hidden-size`, `--num-heads`,
`--intermediate-size`, `--vocab-size`, `--max-seq-len`, `--seq-len`,
`--batch-size`, `--num-classes`, `--activation` (gelu/relu), `--opset`,
`--no-verify` (skip the onnxruntime-vs-torch cross-check).
