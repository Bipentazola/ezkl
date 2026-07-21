# ezkl `create_proof` flamegraph scaling experiment: notes

## Goal

Get a function-level flamegraph of `create_proof` and break it into six phases
(MSM/commitments, FFT/NTT, lookup argument, permutation Z poly, quotient/h_poly,
synthesize/witness), then see how the % breakdown shifts as one architecture
axis at a time is scaled up from an all-minimum baseline. Not a real-BERT-scale
benchmark — see `ezkl_bert_profiling_notes.md` for why that's infeasible in
this container (OOMs past 130GB before `prove` is even reached).

## Problem 1: `perf` is unusable in this sandbox

`perf_event_paranoid=4` (sysctl read-only, can't lower it) and no
`linux-tools-<kernel>` package available via apt. **Fix**: profile with the
`pprof` crate instead (signal-based CPU sampling via `SIGPROF`/`setitimer`,
no `perf_event_open` needed). Added as an optional dependency gated by a new
`pprof-profiling` Cargo feature, instrumented around the `create_proof_circuit`
call in `src/execute.rs::prove()`, triggered only when `EZKL_PPROF_OUT` is set
(zero effect on normal builds/usage). Writes `<prefix>.folded` (our own
collapsed-stack text, root→leaf per line) and `<prefix>.svg`.

## Problem 2: default pprof backend segfaults under this workload

First build (default DWARF/libunwind-based backtrace) segfaulted every time,
seconds into `prove`, right after the profiler signal handler armed. Root
cause: DWARF/libunwind-style unwinding inside a signal handler is not
guaranteed async-signal-safe, and this is a heavily multi-threaded (rayon),
heavily-generic (monomorphized halo2) binary under LTO — exactly the
conditions that trigger it.

**Fix**: switch to pprof's `frame-pointer` feature (walks `rbp` manually,
validates the pointer before dereferencing, no libunwind calls — much safer
under signal delivery), and add `-C force-frame-pointers=yes` to
`[profile.release] rustflags` (needs `cargo-features = ["profile-rustflags"]`,
already enabled in this repo's `Cargo.toml`) so the *whole* dependency graph
keeps frame pointers, not just the `pprof` crate itself. This requires a full
rebuild (rustflags change invalidates every crate's fingerprint) — budget
~25 min per full release rebuild in this environment (`lto="fat"`,
`codegen-units=1`, now also `debug=true` for symbol names — all three are
needed simultaneously: LTO+1CGU for realistic timings, debug info for
symbolication, frame pointers for safe unwinding).

**Don't skip straight to a rebuild after any pprof-related code change** —
`cargo check --release --features pprof-profiling` catches type errors in
under 2 minutes; save the ~25 min full build for after that passes.

## Problem 3: naive function-name matching misses most MSM/FFT time

`halo2_proofs::arithmetic::best_multiexp` / `best_fft` are thin wrapper
functions — under `lto=fat` + `-O3` they get inlined away at most call sites,
so grepping stack frames for the literal substrings `best_multiexp`/`best_fft`
initially caught only ~5-15% of what was actually MSM/FFT work; the rest
landed in a leaf frame from the *real* implementation module
(`halo2curves::msm::{Bucket,Schedule,msm_best,batch_add}` /
`halo2_proofs::fft::parallel::{serial_fft,parallel_fft,split_radix_fft,...}`)
with an ancestor chain that also doesn't mention `best_multiexp`/`best_fft`
(that frame is simply gone from the stack). **Fix**: match on the
*implementation module path* instead of the wrapper name:
`halo2curves::msm::` and `halo2_proofs::fft::` (kept the wrapper-name patterns
too, as a cheap union, in case a given call site doesn't get inlined).

Similarly, `permutation::prover` / `vanishing::prover` (assuming the physical
`.rs` submodule shows up in the symbol path) never matched — Rust's symbol
path for an `impl` follows the *type's declaring module*, not the file the
`impl` block happens to be written in. `Argument::commit` for permutation is
physically in `permutation/prover.rs` but demangles as
`halo2_proofs::plonk::permutation::Argument::commit` (no `::prover::`
anywhere). **Fix**: match the broader `plonk::permutation::` /
`plonk::vanishing::` instead of assuming the prover submodule appears
literally.

**Lesson for next time**: don't assume a wrapper/module name in the source
survives to the compiled symbol — grep a *real* `.folded` sample's actual
leaf-frame names (`cut -d' ' -f... | sort | uniq -c | sort -rn` style) and
adjust regexes to what's actually there, before trusting any category %.

## Problem 4: rayon's default thread count turns "idle" into fake "other"

With rayon's default pool size (= logical CPUs = 64 in this container), most
threads sit parked/spinning for these tiny circuits, and SIGPROF sampling
picks up that idle machinery (`crossbeam_epoch::default::pin`,
`rayon_core::registry::WorkerThread::{wait_until,steal}`,
`rayon_core::sleep::Sleep::no_work_found`, ...) as if it were real work —
it isn't part of any of the six phases, but it swamped the "other" bucket
(~44-67% before the fix). **Fix**: `export RAYON_NUM_THREADS=8` before
running the profiled `prove` (added to `scripts/run_scaling_rep.sh`). Dropped
"other" to a much more honest ~13-17% (genuine misc: batch inversion,
allocation, transcript hashing, ambiguous curve-point arithmetic with no
identifiable caller). ezkl/halo2 never override rayon's thread count
themselves (confirmed via grep — no `RAYON_NUM_THREADS`/`ThreadPoolBuilder`
anywhere in `src/`), so the env var is respected as-is.

## Problem 5: `calibrate-settings` scale has a narrow feasible ceiling

Pushing `param_scale` to 14 or 16 (via `--scales`) fails outright —
`General synthesis error` (rebase=1) / `significant bit truncation when
instantiating` (rebase=10) — this is the fixed-point (`i128`) representation
overflowing, not a tunable-precision tradeoff. For this tiny architecture the
ceiling is **13** (baseline auto-picks 12; `large-scale` config forces 13).
Same shape of ceiling as documented for full BERT in
`ezkl_bert_profiling_notes.md` (there the floor for correctness was ~11, here
the *ceiling* for representability is ~13) — scale has both a floor (need
enough precision for erf/softmax) and a ceiling (must fit `i128`), and the
gap between them can be narrow.

## Categorization method (final, in `scripts/categorize_folded.py`)

For each unique stack line (`frame1;frame2;...;frameN count`, root-first),
scan frames **leaf→root** and assign the sample to the first category whose
pattern matches, checked in this priority order:

1. `msm`: `best_multiexp|halo2curves::msm::`
2. `fft`: `best_fft|halo2_proofs::fft::`
3. `lookup`: `mv_lookup::|plonk::lookup::`
4. `permutation`: `plonk::permutation::`
5. `quotient`: `evaluate_h|plonk::vanishing::`
6. `synthesize`: `synthesize`
7. everything else → `other`

Leaf-first (not root-first) matters: it attributes self-time to the
*nearest* identifiable phase, so e.g. an FFT called from inside lookup
construction counts as `fft`, not `lookup`.

## Experiment design actually used

Control-variable sweep, not a realistic-transformer-size sweep (per explicit
instruction): one minimal baseline, then one axis at a time pushed up while
everything else stays at its minimum.

| config | num_layers | hidden | heads | ffn | vocab | seq_len | param_scale | logrows |
|---|---|---|---|---|---|---|---|---|
| baseline | 1 | 8 | 1 | 8 | 8 | 4 | 12 (auto) | 15 |
| large_n | 1 | 8 | 1 | 8 | 8 | **64** | 12 (auto) | 20 |
| large_d | 1 | **128** | 1 | 8 | 8 | 4 | 13 (auto) | 18 |
| large_scale | 1 | 8 | 1 | 8 | 8 | 4 | **13 (forced)** | 16 |

3 `prove` reps per config (pipeline/setup run once, pk+witness reused across
reps) for variance reduction — std devs across reps were small (mostly
<1 percentage point, largest was large_d's fft/quotient at ~1.7pp).

## Reproducing / rerunning

```bash
export RUSTUP_HOME=/data/rust/.rustup CARGO_HOME=/data/rust/.cargo PATH=/data/rust/.cargo/bin:$PATH
# one-time (already done): cargo build --release --features pprof-profiling --bin ezkl

MODEL=models/ScaleBaseline/onnx/model.onnx SEQ_LEN=4 OUT_DIR=/data/coding/ezkl/scaling_runs/baseline \
  REPS=3 CALIBRATE_SCALES="" ./scripts/run_scaling_rep.sh

python3 scripts/categorize_folded.py OUT.csv label1=path1.folded ...
python3 scripts/aggregate_scaling_results.py   # reads scaling_runs/*, writes scaling_runs/results/{summary.csv,summary.json,paper_table.md,paper_table.tex}
```

Key files:
- `src/execute.rs` — pprof instrumentation (search `pprof-profiling`)
- `scripts/run_scaling_rep.sh` — one scale-point driver (pipeline once + N prove reps)
- `scripts/categorize_folded.py` — folded-stack → category % (regexes above)
- `scripts/aggregate_scaling_results.py` — per-config mean/std + paper-ready table export
- `models/Scale{Baseline,LargeN,LargeD}/` — the 3 generated ONNX models (large_scale reuses ScaleBaseline's onnx)
- `scaling_runs/{baseline,large_n,large_d,large_scale}/` — full pipeline artifacts per config (settings.json, logs, `pprof/repN.{folded,svg}`)
- `scaling_runs/results/` — final aggregated CSV/JSON + paper table

## Caveats / what this data does and doesn't show

- Absolute % values are specific to this tiny synthetic model + this
  container's CPU; only the *direction and rough magnitude of the shift*
  across configs is the meaningful result, not the precise numbers.
- `other` (~13-17%) is real overhead (allocation, transcript, ambiguous
  curve-point ops with no attributable caller after inlining) — it is not
  zero-able by better regexes, some of it is genuinely not attributable
  post-inlining without recompiling with `-C inline-threshold=0` or similar
  (not attempted — would change the timing characteristics being measured).
- This does **not** cover `setup` (keygen) or `mock`, only the `prove`
  step's `create_proof_circuit` call, matching what the user asked for.
