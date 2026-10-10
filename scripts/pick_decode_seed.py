#!/usr/bin/env python3

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
GEN = REPO / "scripts" / "gen_decode_step_onnx.py"


def simulate(model_dir, vocab, t_max, steps, start_tok, shapes):
    import numpy as np
    import onnxruntime as ort

    sess = ort.InferenceSession(
        str(Path(model_dir) / "model.onnx"), providers=["CPUExecutionProvider"]
    )
    names = [i.name for i in sess.get_inputs()]
    kc = np.zeros(shapes[1], dtype=np.float32)
    vc = np.zeros(shapes[2], dtype=np.float32)
    tok, seq = start_tok, [start_tok]
    for t in range(steps):
        oh = np.zeros(shapes[0], dtype=np.float32)
        oh[0, 0, tok % vocab] = 1.0
        mw = np.zeros(shapes[3], dtype=np.float32)
        mw[0, 0, t] = 1.0
        ma = np.zeros(shapes[4], dtype=np.float32)
        ma[0, 0, 0, : t + 1] = 1.0
        out = sess.run(None, dict(zip(names, [oh, kc, vc, mw, ma])))
        _, kc, vc, logits = out
        tok = int(np.argmax(logits.reshape(-1)))
        seq.append(tok)
    return seq


def main():
    ap = argparse.ArgumentParser(
        description=None, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--d-model", type=int, default=128)
    ap.add_argument("--t-max", type=int, default=16)
    ap.add_argument("--vocab", type=int, default=256)
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--start-token", type=int, default=7)
    ap.add_argument("--max-seed", type=int, default=32)
    ap.add_argument(
        "--min-distinct",
        type=int,
        default=3,
        help="序列里至少要有这么多个不同的 token 才算不退化",
    )
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()

    tried = []
    with tempfile.TemporaryDirectory(prefix="seedpick_") as tmp:
        for seed in range(args.max_seed):
            d = Path(tmp) / f"s{seed}"
            r = subprocess.run(
                [
                    sys.executable,
                    str(GEN),
                    "--variant",
                    "where",
                    "--d-model",
                    str(args.d_model),
                    "--t-max",
                    str(args.t_max),
                    "--vocab",
                    str(args.vocab),
                    "--seed",
                    str(seed),
                    "--token",
                    str(args.start_token),
                    "--out-dir",
                    str(d),
                ],
                capture_output=True,
                text=True,
            )
            if r.returncode != 0:
                tried.append({"seed": seed, "error": r.stderr[-200:]})
                continue
            shapes = json.loads((d / "input.json").read_text())["input_shapes"]
            seq = simulate(
                d, args.vocab, args.t_max, args.steps, args.start_token, shapes
            )
            n = len(set(seq))
            tried.append({"seed": seed, "sequence": seq, "distinct": n})
            print(f"  seed {seed:3d}  distinct={n:2d}  {seq}")
            if n >= args.min_distinct:
                res = {
                    "chosen_seed": seed,
                    "float_sequence": seq,
                    "distinct": n,
                    "tried": tried,
                }
                if args.json_out:
                    Path(args.json_out).write_text(json.dumps(res, indent=2) + "\n")
                print(f"\n[chosen] seed={seed} with {n} distinct tokens")
                return 0

    print(
        f"\n[no seed found] tried {args.max_seed}; "
        f"greedy decode is a fixed point for all of them",
        file=sys.stderr,
    )
    if args.json_out:
        Path(args.json_out).write_text(
            json.dumps({"chosen_seed": None, "tried": tried}, indent=2) + "\n"
        )
    return 1


if __name__ == "__main__":
    sys.exit(main())
