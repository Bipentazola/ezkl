#!/usr/bin/env python3
"""
gen_models.py -- generate a family of minimal pre-LN transformer models (ONNX)
for ezkl performance profiling, plus matching ezkl-format input.json files and
a parameter-count/FLOPs manifest.

Model design closely follows ezkl's own known-good
`examples/onnx/self_attention/gen.py` reference (manual multi-head attention
via one fused QKV Linear + Split, plain nn.LayerNorm, tanh-approx GELU,
opset 10) since that exact op combination is already exercised by ezkl's own
test suite. Two differences from that reference:
  - attention here is bidirectional (no causal mask) -- these are
    encoder-style blocks, not GPT-style decoder blocks;
  - depth is configurable: N stacked pre-LN blocks instead of a single one.

Usage:
    python gen_models.py                  # generate the full config matrix + MANIFEST.csv
    python gen_models.py --skip-verify     # skip the `ezkl gen-settings` sanity check on the base config
"""
import argparse
import csv
import json
import math
import subprocess
import sys
from pathlib import Path

import onnx
import torch
import torch.nn as nn
import torch.nn.functional as F

REPO_DIR = Path(__file__).resolve().parent
MODELS_DIR = REPO_DIR / "models"
EZKL_BIN = REPO_DIR / "target" / "release" / "ezkl"

# base config + the three single-axis scans (each scan holds every other axis
# at the base value); duplicates of `base` across scans collapse to one dir
#
# Scaled down 4x from the original base (seq_len=32, d_model=64) after
# seq32_d64_l1 alone measured 66-90GB peak RSS for setup/prove, and the
# width-scan extreme (seq32_d256_l1) hit this container's ~156GB cgroup
# memory ceiling and OOM'd during `setup` (see results/summary.csv). This
# family keeps the same relative scan shape (0.5x/1x/2x/4x for seq_len,
# 1x/2x/4x for d_model and n_layers) on a smaller anchor, so every axis's
# largest point stays below what the old (safely-completed) base already
# proved feasible -- the goal is the shape of the scaling trend, not
# absolute size, per explicit instruction.
BASE = dict(seq_len=8, d_model=16, n_head=2, n_layers=1)
SEQ_SCAN = (4, 8, 16, 32)
WIDTH_SCAN = (16, 32, 64)
DEPTH_SCAN = (1, 2, 4)


def cfg_name(cfg: dict) -> str:
    return f"seq{cfg['seq_len']}_d{cfg['d_model']}_l{cfg['n_layers']}"


def build_config_matrix() -> dict:
    configs = {}

    def add(cfg):
        configs[cfg_name(cfg)] = dict(cfg)

    add(BASE)
    for s in SEQ_SCAN:
        add({**BASE, "seq_len": s})
    for d in WIDTH_SCAN:
        add({**BASE, "d_model": d})
    for l in DEPTH_SCAN:
        add({**BASE, "n_layers": l})
    return configs


def new_gelu(x):
    """tanh-approximation GELU -- matches examples/onnx/self_attention/gen.py."""
    return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x * x * x)))


class SelfAttention(nn.Module):
    """Bidirectional (non-causal) multi-head self-attention.

    Uses a single fused QKV Linear + tensor.split (not nn.MultiheadAttention)
    -- this is the exact op pattern ezkl's own self_attention example uses.
    """

    def __init__(self, d_model: int, n_head: int):
        super().__init__()
        assert d_model % n_head == 0, f"d_model ({d_model}) must be divisible by n_head ({n_head})"
        self.n_head = n_head
        self.d_model = d_model
        self.c_attn = nn.Linear(d_model, 3 * d_model)
        self.c_proj = nn.Linear(d_model, d_model)

    def forward(self, x):
        b, t, c = x.size()
        q, k, v = self.c_attn(x).split(self.d_model, dim=2)
        hs = c // self.n_head
        k = k.view(b, t, self.n_head, hs).transpose(1, 2)
        q = q.view(b, t, self.n_head, hs).transpose(1, 2)
        v = v.view(b, t, self.n_head, hs).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(hs))
        att = F.softmax(att, dim=-1)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(b, t, c)
        return self.c_proj(y)


class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        return self.fc2(new_gelu(self.fc1(x)))


class PreLNBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, d_ff: int):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = SelfAttention(d_model, n_head)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model, d_ff)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


class TransformerStack(nn.Module):
    def __init__(self, d_model: int, n_head: int, n_layers: int, d_ff: int):
        super().__init__()
        self.blocks = nn.ModuleList([PreLNBlock(d_model, n_head, d_ff) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(d_model)

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return self.ln_f(x)


def flops_estimate(seq_len: int, d_model: int, n_layers: int, d_ff: int) -> int:
    """Analytic forward-pass FLOPs estimate (multiply-add = 2 FLOPs), summing
    only the matmuls (QKV proj, output proj, attention scores/context, FFN
    fc1/fc2). LayerNorm, softmax normalization, and bias adds are lower-order
    (linear in s*d) and are not included.

    Per layer:
      qkv + out proj:  2 * s*d*(3d) + 2*s*d*d = 8*s*d^2
      attn scores+ctx:  2*(2*s*s*d) = 4*s^2*d   (n_head cancels: sum over heads of s*s*hs = s*s*d)
      ffn fc1+fc2:      2*s*d*d_ff + 2*s*d_ff*d = 4*s*d*d_ff
    """
    s, d = seq_len, d_model
    per_layer = 8 * s * d * d + 4 * s * s * d + 4 * s * d * d_ff
    return n_layers * per_layer


def export_one(name: str, cfg: dict, opset: int, seed: int) -> dict:
    d_model, n_head, seq_len, n_layers = cfg["d_model"], cfg["n_head"], cfg["seq_len"], cfg["n_layers"]
    d_ff = 4 * d_model

    torch.manual_seed(seed)
    model = TransformerStack(d_model, n_head, n_layers, d_ff)
    model.eval()

    x = torch.randn(1, seq_len, d_model)
    with torch.no_grad():
        torch_out = model(x)

    out_dir = MODELS_DIR / name
    out_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = out_dir / "model.onnx"

    torch.onnx.export(
        model,
        x,
        str(onnx_path),
        export_params=True,
        opset_version=opset,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamo=False,  # tract/ezkl expects the classic TorchScript-traced exporter output
    )

    onnx_model = onnx.load(str(onnx_path))
    onnx.checker.check_model(onnx_model)

    import onnxruntime as ort

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    ort_out = sess.run(None, {"input": x.numpy()})[0]
    max_diff = float(abs(torch_out.numpy() - ort_out).max())
    assert max_diff < 1e-3, f"{name}: onnx export diverges from torch reference (max abs diff {max_diff:.3e})"

    input_flat = x.detach().numpy().reshape(-1).tolist()
    output_flat = torch_out.detach().numpy().reshape(-1).tolist()
    input_json = {
        "input_data": [input_flat],
        "input_shapes": [[1, seq_len, d_model]],
        "output_data": [output_flat],
    }
    (out_dir / "input.json").write_text(json.dumps(input_json))

    params = sum(p.numel() for p in model.parameters())
    flops = flops_estimate(seq_len, d_model, n_layers, d_ff)

    op_counts = {}
    for node in onnx_model.graph.node:
        op_counts[node.op_type] = op_counts.get(node.op_type, 0) + 1

    return {
        "name": name,
        "seq_len": seq_len,
        "d_model": d_model,
        "n_head": n_head,
        "n_layers": n_layers,
        "d_ff": d_ff,
        "params": params,
        "flops": flops,
        "onnx_path": str(onnx_path.relative_to(REPO_DIR)),
        "input_json_path": str((out_dir / "input.json").relative_to(REPO_DIR)),
        "onnx_max_abs_diff": max_diff,
        "op_counts": json.dumps(op_counts, sort_keys=True),
    }


def verify_with_ezkl(name: str) -> None:
    if not EZKL_BIN.exists():
        print(f"warning: ezkl binary not found at {EZKL_BIN}, skipping gen-settings verification", file=sys.stderr)
        return

    onnx_path = MODELS_DIR / name / "model.onnx"
    settings_path = MODELS_DIR / name / "settings.json"
    cmd = [str(EZKL_BIN), "gen-settings", "-M", str(onnx_path), "-O", str(settings_path)]
    print(f"running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    print(result.stdout)
    if result.returncode != 0:
        print(result.stderr, file=sys.stderr)
        raise SystemExit(f"ezkl gen-settings FAILED for {name} (exit {result.returncode}) -- adjust the model and rerun")
    print(f"ezkl gen-settings OK for {name} -> {settings_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--opset", type=int, default=10, help="ONNX opset (10 matches the known-good self_attention example)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-verify", action="store_true", help="skip running `ezkl gen-settings` on the base config")
    args = parser.parse_args()

    configs = build_config_matrix()
    print(f"generating {len(configs)} configs: {list(configs.keys())}")

    rows = []
    for name, cfg in configs.items():
        print(f"\n== {name}: {cfg} ==")
        row = export_one(name, cfg, args.opset, args.seed)
        rows.append(row)
        print(f"  params={row['params']:,}  flops~={row['flops']:,}  onnx_max_abs_diff={row['onnx_max_abs_diff']:.2e}")

    manifest_path = MODELS_DIR / "MANIFEST.csv"
    fieldnames = [
        "name", "seq_len", "d_model", "n_head", "n_layers", "d_ff",
        "params", "flops", "onnx_path", "input_json_path", "onnx_max_abs_diff", "op_counts",
    ]
    with open(manifest_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row[k] for k in fieldnames})
    print(f"\nwrote {manifest_path}")

    if not args.skip_verify:
        base_name = cfg_name(BASE)
        print(f"\nverifying base config ({base_name}) compiles with ezkl gen-settings...")
        verify_with_ezkl(base_name)


if __name__ == "__main__":
    main()
