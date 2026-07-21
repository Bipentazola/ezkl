# prove flamegraph / breakdown — what's what

Generated 2026-07-21 by profiling `ezkl prove` with the repo's `pprof-profiling`
Cargo feature (SIGPROF sampling; `perf`/`cargo flamegraph` don't work in this
sandbox — `perf_event_paranoid=4`, no `linux-tools` package). Binary used:
`target/release/ezkl` (release, debug symbols, frame-pointer unwinding,
already built with `--features pprof-profiling`, not rebuilt for this run).

## Run labels (used as the `run` column in prove_breakdown*.csv and as file
## name prefixes in results/pprof/)

| label | model config | RAYON_NUM_THREADS | prove wall time | pprof samples | artifacts |
|---|---|---|---|---|---|
| `seq4_1thread` | `models/seq4_d16_l1` (seq_len=4, d_model=16, 1 layer, 3312 params, logrows=16) | `1` (forced) | 94.4s | 23,626 | `results/pprof/seq4_1thread.{folded,svg,log}` → also copied to `results/flamegraph_1thread.svg` |
| `seq4_nthread` | same model, same compiled circuit/witness/pk | unset (default = 64, all logical CPUs) | 10.3s | 2,832 | `results/pprof/seq4_nthread.{folded,svg,log}` → also copied to `results/flamegraph_nthread.svg` |
| `base_nthread` | `models/seq32_d64_l1` (seq_len=32, d_model=64, 1 layer, logrows=21, ~2M rows — this is the "base" config from the earlier stage-timing sweep, `results/summary.csv`) | unset (default = 64) | 342.6s | 102,476 | `results/pprof/base_nthread.{folded,svg,log}` → also copied to `results/flamegraph_base_nthread.svg` |

`seq4_1thread` and `seq4_nthread` are the same circuit/witness/pk (only
`RAYON_NUM_THREADS` differs) — that's the intended 1-thread-vs-default-thread
comparison. `base_nthread` is a separate, much larger circuit run only at
default thread count, included as a "does this hold at real scale" reference
(see caveat below — it does, `other` stays large).

Each `.folded` file is the raw pprof collapsed-stack sample dump (one line
per unique call stack, `frame1;frame2;...;frameN <sample_count>`) for that
run; `.svg` is the rendered flamegraph; `.log` is the full `ezkl prove`
stdout/stderr for that invocation.

## Bucket definitions (the `category` column)

Regex patterns live in `scripts/categorize_folded.py::PATTERNS`. A stack is
assigned to the first bucket (in this priority order) whose pattern matches
scanning frames **leaf → root** (so e.g. an FFT called from inside lookup
construction counts as `fft`, not `lookup` — nearest identifiable phase wins):

| category (CSV value) | matches | meaning |
|---|---|---|
| `msm` | `best_multiexp`, `halo2curves::msm::` | MSM / polynomial commitments |
| `fft` | `best_fft`, `halo2_proofs::fft::` | FFT/NTT, domain conversion |
| `lookup` | `mv_lookup::`, `plonk::lookup::` | lookup argument construction |
| `permutation` | `plonk::permutation::` | permutation (Z polynomial) argument |
| `quotient` | `evaluate_h`, `h_poly`, `plonk::vanishing::` | quotient polynomial / gate constraint evaluation |
| `synthesize` | `synthesize`, `::assign`, `tract::` | witness / circuit assignment |
| `other` | (none of the above matched) | everything else — see caveat |

## Caveat: `other` is mostly rayon idle-thread noise in the *_nthread runs, not real work

`seq4_1thread`'s `other` is 3.4% — clean. Both `*_nthread` runs have `other`
in the 36–47% range; `results/prove_breakdown_other_top_symbols.csv` shows why:
the top entries are `crossbeam_epoch::default::pin`,
`rayon_core::registry::WorkerThread::{wait_until,steal}`,
`rayon_core::sleep::Sleep::no_work_found`, etc. — SIGPROF sampling idle/parked
worker threads in the default 64-thread rayon pool, not genuine proving cost.
This reproduces even on the 2M-row `base_nthread` circuit, so it isn't just a
"circuit too small" artifact — `prove` has enough serial/sync windows that
64 threads don't stay uniformly busy at any circuit size tested here.
**Treat `seq4_1thread`'s percentages as the reliable per-bucket attribution;
treat the `*_nthread` `other` numbers as "idle-thread contamination," not a
real bottleneck.**

## Files in this directory

- `prove_breakdown.csv` — bucket × run, `samples`/`total_samples`/`pct`
- `prove_breakdown_other_top_symbols.csv` — top leaf symbols inside `other`, for every run where `other` > 15%
- `prove_breakdown.png` — grouped bar chart of the CSV above
- `flamegraph_1thread.svg`, `flamegraph_nthread.svg` — the seq4 1-thread/default-thread comparison pair
- `flamegraph_base_nthread.svg` — base-model reference flamegraph
- `pprof/` — raw `.folded`/`.svg`/`.log` per run (source data behind everything above)
