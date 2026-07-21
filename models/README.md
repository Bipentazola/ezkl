# models/ — index

Several generations of generated models live here from different rounds of
work. Nothing is deleted, so read this before assuming a directory is part of
the current active sweep.

## Active family (current `scripts/gen_models.py`)

`seq4_d16_l1`, `seq8_d16_l1` (base), `seq16_d16_l1`, `seq32_d16_l1`
(seq_len scan), `seq8_d32_l1`, `seq8_d64_l1` (width scan), `seq8_d16_l2`,
`seq8_d16_l4` (depth scan) — each has `model.onnx` + `input.json`, generated
and documented by **`gen_models.py`**'s module docstring (also see
**`MANIFEST.csv`** for params/FLOPs/op-counts per config). This is the family
`results/summary.csv` benchmarks seq_len/width/depth against.

## Pre-rescale leftovers (kept, not part of the active sweep)

`gen_models.py` originally used a larger base (`seq_len=32, d_model=64`)
before being rescaled 4x down after `seq32_d64_l1` measured 66-90GB peak RSS
and `seq32_d256_l1` OOM'd outright (see the comment above `BASE = dict(...)`
in `gen_models.py`). Two of those original-scale configs are deliberately
still benchmarked (present in `results/summary.csv`) as reference points:

- `seq32_d64_l1` — the original ("real") base config, all pipeline stages OK
- `seq32_d256_l1` — the width-scan extreme that OOM'd during `setup`, kept as
  the documented OOM ceiling for this container

The rest of that old-scale family (`seq128_d64_l1`, `seq16_d64_l1`,
`seq32_d128_l1`, `seq32_d64_l2`, `seq32_d64_l4`, `seq64_d64_l1`) were
generated during that same round but never benchmarked after the rescale —
not part of `results/summary.csv`, not actively used. Not deleted only
because deleting wasn't asked for; treat as stale unless a future task
explicitly revives the old-scale sweep.

## Other, unrelated model dirs

- `BertForSequenceClassification/` — real BERT, used for the separate
  BERT-profiling investigation; OOMs before `mock` even completes at this
  container's memory ceiling. See `.claude/ezkl_bert_profiling_notes.md`.
- `ScaleBaseline/`, `ScaleLargeD/`, `ScaleLargeN/`, `TinyTransformer1L/` —
  from the earlier `create_proof` flamegraph scaling experiment (control-
  variable sweep on tiny synthetic models, not the current seq/width/depth
  matrix). Fully documented in
  `.claude/ezkl_create_proof_flamegraph_scaling_notes.md`.

## Convention going forward

New generated model families should get their own manifest/docstring (like
`gen_models.py` + `MANIFEST.csv`) and a one-line entry added to this file.
