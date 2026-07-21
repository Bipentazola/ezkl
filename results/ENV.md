# Experiment environment

Captured 2026-07-22, same container/session all benchmark and profiling data
in `results/` was produced on (the `target/release/ezkl` binary and
`results/*` artifacts have not moved between machines).

## Hardware / OS

| | |
|---|---|
| CPU | 64 logical cores (`nproc`) |
| Memory | 251 GiB total (`free -h`), ~172 GiB free at capture time |
| OS | Ubuntu 22.04.4 LTS (jammy), kernel 5.19.0-50-generic x86_64 |
| Disk | `/` : 45 GB overlay, **ephemeral, wiped between sessions** (3.1 GB used); `/data` : 1.2 TB, **persistent** (158 GB used) -- everything under this repo lives on `/data`. See `models/README.md`/`results/README.md` for what's generated where. |
| `perf_event_paranoid` | `4` (read-only, can't lower) -- `perf`/`cargo flamegraph` do not work in this sandbox; profiling used the `pprof` crate instead (SIGPROF sampling). See `results/PPROF_README.md`. |

## Toolchain

| | |
|---|---|
| Rust | `rustc 1.93.0-nightly (b84478a1c 2025-11-30)`, `cargo 1.93.0-nightly` -- installed to `/data/rust/.rustup` + `/data/rust/.cargo` (not the default `~/.rustup`/`~/.cargo`, which don't survive a session restart on this host) |
| Python | 3.10.20, `matplotlib` (installed via `pip` for this task's figures), `torch`/`onnx`/`onnxruntime`/`transformers` (for `gen_models.py`/`gen_transformer_onnx.py`) |
| `ezkl` binary | `target/release/ezkl`, release profile: `debug = true`, `lto = "fat"`, `codegen-units = 1`, `rustflags = ["-C relocation-model=pic", "-C force-frame-pointers=yes"]`, built with `--features pprof-profiling` (see `Cargo.toml` `[profile.release]` + the `pprof` optional dependency). Frame pointers + debug symbols are kept in the release build specifically so `pprof`'s signal-safe unwinder and symbolication work without a debug build's slowdown. |
| `halo2_proofs` fork pin | bumped one commit forward on `ac/conditional-compilation-icicle2` (commit `01c88842679b4308e43ae5ed91c4183e861669bd`) to fix a real prove failure on any 2+-matmul circuit -- see `.claude/ezkl_tinytransformer_profiling_notes.md` and the `[patch.'https://github.com/zkonduit/halo2']` entry in `Cargo.toml`. Without this pin, none of the benchmark data in `results/` would have been reachable (`prove` failed outright). |
| `time -v` | this container has no `/usr/bin/time`; extracted from the Debian `.deb` into `.tools/time-pkg/usr/bin/time` (kept inside `/data`, not registered with `dpkg` globally) -- see `bench_stages.py`'s docstring. |

## What this environment means for interpreting the numbers in this report

- All wall-clock/memory numbers are single-container measurements, not
  averaged across hardware -- `bench_stages.py` runs 3 timed reps (+1
  discarded warm-up) for `setup`/`prove` specifically to smooth out
  within-container noise, but there's no cross-machine variance data here.
- 64 cores means default-thread-count `prove` runs have a *lot* of rayon
  worker threads relative to these small/medium circuits -- see
  `PPROF_README.md`'s caveat about idle-thread sampling noise inflating the
  `other` bucket in the flamegraph breakdown at default thread count.
- Peak-RSS numbers reported throughout `results/summary.csv` reflect this
  251 GiB-RAM container's ability to let large `pk.key` files (tens of GB)
  materialize without swapping; a smaller-memory host would OOM at a lower
  `logrows` than the ceiling observed here (`seq32_d256_l1`, see
  `models/README.md`).
