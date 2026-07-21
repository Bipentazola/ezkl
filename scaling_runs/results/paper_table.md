# create_proof phase breakdown (mean % of sampled CPU time, n=3 reps, ± std)

| Config | MSM | FFT/NTT | Lookup | Perm. Z | Quotient | Synth. | Other |
|---|---|---|---|---|---|---|---|
| baseline (all params at minimum) | 19.9 ± 0.1 | 31.1 ± 0.5 | 2.5 ± 0.1 | 1.5 ± 0.1 | 26.5 ± 0.5 | 1.4 ± 0.3 | 17.0 ± 0.4 |
| large-n (seq_len 4→64, arch@min) | 15.0 ± 0.1 | 40.5 ± 0.2 | 3.6 ± 0.1 | 1.5 ± 0.0 | 24.6 ± 0.2 | 1.8 ± 0.1 | 12.9 ± 0.4 |
| large-d (hidden 8→128, arch@min) | 15.4 ± 0.7 | 36.9 ± 1.7 | 3.1 ± 0.1 | 1.4 ± 0.0 | 27.1 ± 1.7 | 2.2 ± 0.1 | 13.9 ± 0.7 |
| large-scale (param_scale 12→13, arch@min) | 19.0 ± 0.4 | 34.4 ± 0.7 | 3.1 ± 0.2 | 1.5 ± 0.1 | 28.5 ± 1.4 | 0.7 ± 0.2 | 12.7 ± 0.6 |

## Config / calibration metadata

| Config | Varied param | logrows | input_scale | param_scale | prove time (s, rep 1) |
|---|---|---|---|---|---|
| baseline | num_layers=1, hidden_size=8, num_heads=1, intermediate_size=8, vocab_size=8, seq_len=4 | 15 | 0 | 12 | 13.03 |
| large_n | num_layers=1, hidden_size=8, num_heads=1, intermediate_size=8, vocab_size=8, seq_len=64 | 20 | 0 | 12 | 307.7 |
| large_d | num_layers=1, hidden_size=128, num_heads=1, intermediate_size=8, vocab_size=8, seq_len=4 | 18 | 0 | 13 | 89.29 |
| large_scale | num_layers=1, hidden_size=8, num_heads=1, intermediate_size=8, vocab_size=8, seq_len=4 | 16 | 0 | 13 | 23.58 |
