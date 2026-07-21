"""
Generate a configurable, minimal transformer-encoder ONNX model for ezkl profiling.

Unlike the HF BERT export in models/BertForSequenceClassification, this model:
  - takes a single `input_ids` input (no attention_mask / token_type_ids), so
    there's no huge `-3.4028235e+38` float32-min mask constant to patch around.
  - uses only decomposed ops already known to work with ezkl/tract (manual
    LayerNorm via Sub/Pow/ReduceMean/Sqrt/Div, erf-based GELU, Softmax,
    MatMul/Gemm) -- the same op set the working BERT export uses.
  - has fully configurable depth/width, so you can generate anything from a
    1-layer toy model up to something BERT-sized.

Usage:
    python scripts/gen_transformer_onnx.py --help

    # single-layer transformer, tiny dims, good for a first ezkl profiling pass
    python scripts/gen_transformer_onnx.py \
        --output-dir models/TinyTransformer1L \
        --num-layers 1 --hidden-size 64 --num-heads 4 \
        --intermediate-size 256 --vocab-size 128 --max-seq-len 32

Output layout (mirrors models/BertForSequenceClassification so
scripts/profile_ezkl.sh's config.json auto-discovery works unmodified):
    <output-dir>/onnx/model.onnx
    <output-dir>/config.json

Requires: torch, onnx (onnxruntime optional, used only for --verify).
"""
import argparse
import json
import math
from pathlib import Path

import torch
import torch.nn as nn


class LayerNorm(nn.Module):
    """Manually decomposed LayerNorm (Sub/Pow/ReduceMean/Sqrt/Div/Mul/Add) --
    avoids the fused `LayerNormalization` op, matching the op set the known
    -good BERT ONNX export decomposes to."""

    def __init__(self, hidden_size: int, eps: float = 1e-12):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        var = (x - mean).pow(2).mean(-1, keepdim=True)
        x_norm = (x - mean) / torch.sqrt(var + self.eps)
        return x_norm * self.weight + self.bias


def gelu_erf(x):
    return 0.5 * x * (1.0 + torch.erf(x / math.sqrt(2.0)))


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int):
        super().__init__()
        if hidden_size % num_heads != 0:
            raise ValueError(f"hidden_size ({hidden_size}) must be divisible by num_heads ({num_heads})")
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.hidden_size = hidden_size
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.o_proj = nn.Linear(hidden_size, hidden_size)

    def forward(self, x):
        bsz, seq_len, _ = x.shape
        q = self.q_proj(x).view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = torch.softmax(scores, dim=-1)
        ctx = torch.matmul(attn, v)

        ctx = ctx.transpose(1, 2).contiguous().view(bsz, seq_len, self.hidden_size)
        return self.o_proj(ctx)


class FeedForward(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int, activation: str):
        super().__init__()
        self.fc1 = nn.Linear(hidden_size, intermediate_size)
        self.fc2 = nn.Linear(intermediate_size, hidden_size)
        self.activation = activation

    def forward(self, x):
        x = self.fc1(x)
        x = gelu_erf(x) if self.activation == "gelu" else torch.relu(x)
        return self.fc2(x)


class EncoderLayer(nn.Module):
    """Standard post-LN transformer encoder block."""

    def __init__(self, hidden_size: int, num_heads: int, intermediate_size: int, activation: str):
        super().__init__()
        self.attn = MultiHeadSelfAttention(hidden_size, num_heads)
        self.ln1 = LayerNorm(hidden_size)
        self.ff = FeedForward(hidden_size, intermediate_size, activation)
        self.ln2 = LayerNorm(hidden_size)

    def forward(self, x):
        x = self.ln1(x + self.attn(x))
        x = self.ln2(x + self.ff(x))
        return x


class TinyTransformer(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        num_layers: int,
        num_heads: int,
        intermediate_size: int,
        max_seq_len: int,
        num_classes: int,
        activation: str,
    ):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, hidden_size)
        self.position_embedding = nn.Embedding(max_seq_len, hidden_size)
        self.register_buffer("position_ids", torch.arange(max_seq_len).unsqueeze(0), persistent=False)

        self.layers = nn.ModuleList(
            [EncoderLayer(hidden_size, num_heads, intermediate_size, activation) for _ in range(num_layers)]
        )
        self.final_ln = LayerNorm(hidden_size)
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, input_ids):
        seq_len = input_ids.size(1)
        position_ids = self.position_ids[:, :seq_len]
        x = self.token_embedding(input_ids) + self.position_embedding(position_ids)
        for layer in self.layers:
            x = layer(x)
        x = self.final_ln(x)
        pooled = x.mean(dim=1)  # mean pooling over sequence_length
        return self.classifier(pooled)


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--output-dir", type=Path, default=Path("models/TinyTransformer"),
                    help="output dir; writes <output-dir>/onnx/model.onnx + <output-dir>/config.json")
    p.add_argument("--num-layers", type=int, default=1)
    p.add_argument("--hidden-size", type=int, default=64)
    p.add_argument("--num-heads", type=int, default=4)
    p.add_argument("--intermediate-size", type=int, default=256)
    p.add_argument("--vocab-size", type=int, default=128)
    p.add_argument("--max-seq-len", type=int, default=32, help="size of the learned position-embedding table")
    p.add_argument("--seq-len", type=int, default=None,
                    help="sequence length used for the traced example input (default: --max-seq-len)")
    p.add_argument("--batch-size", type=int, default=1, help="batch size used for the traced example input")
    p.add_argument("--num-classes", type=int, default=2)
    p.add_argument("--activation", choices=["gelu", "relu"], default="gelu")
    p.add_argument("--opset", type=int, default=11, help="ONNX opset (11 matches the known-good BERT export)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-verify", action="store_true", help="skip onnxruntime-vs-torch numerical cross-check")
    return p


def main():
    args = build_arg_parser().parse_args()
    seq_len = args.seq_len or args.max_seq_len
    if seq_len > args.max_seq_len:
        raise SystemExit(f"--seq-len ({seq_len}) cannot exceed --max-seq-len ({args.max_seq_len})")

    torch.manual_seed(args.seed)

    model = TinyTransformer(
        vocab_size=args.vocab_size,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        intermediate_size=args.intermediate_size,
        max_seq_len=args.max_seq_len,
        num_classes=args.num_classes,
        activation=args.activation,
    )
    model.eval()

    total_params = sum(p.numel() for p in model.parameters())
    print(f"model: {args.num_layers}-layer transformer, hidden={args.hidden_size}, heads={args.num_heads}, "
          f"ffn={args.intermediate_size}, vocab={args.vocab_size}, max_seq_len={args.max_seq_len}")
    print(f"total parameters: {total_params:,} (~{total_params * 4 / 1024:.1f} KB fp32)")

    onnx_dir = args.output_dir / "onnx"
    onnx_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = onnx_dir / "model.onnx"

    dummy_input = torch.randint(0, args.vocab_size, (args.batch_size, seq_len), dtype=torch.long)

    with torch.no_grad():
        torch_out = model(dummy_input)

    torch.onnx.export(
        model,
        dummy_input,
        str(onnx_path),
        input_names=["input_ids"],
        output_names=["logits"],
        dynamic_axes={"input_ids": {0: "batch_size", 1: "sequence_length"}, "logits": {0: "batch_size"}},
        opset_version=args.opset,
        do_constant_folding=True,
        dynamo=False,  # tract/ezkl expects the classic TorchScript-traced exporter output
    )
    print(f"wrote {onnx_path}")

    import onnx
    onnx_model = onnx.load(str(onnx_path))
    onnx.checker.check_model(onnx_model)
    op_counts = {}
    for node in onnx_model.graph.node:
        op_counts[node.op_type] = op_counts.get(node.op_type, 0) + 1
    print(f"onnx graph ops: {dict(sorted(op_counts.items()))}")

    if not args.no_verify:
        import onnxruntime as ort
        sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        ort_out = sess.run(None, {"input_ids": dummy_input.numpy()})[0]
        max_diff = float((torch_out.numpy() - ort_out).__abs__().max())
        print(f"torch-vs-onnxruntime max abs diff: {max_diff:.3e}")
        assert max_diff < 1e-4, "onnx export diverges from torch reference"

    config = {
        "model_type": "tiny_transformer",
        "vocab_size": args.vocab_size,
        "hidden_size": args.hidden_size,
        "num_hidden_layers": args.num_layers,
        "num_attention_heads": args.num_heads,
        "intermediate_size": args.intermediate_size,
        "max_position_embeddings": args.max_seq_len,
        "num_classes": args.num_classes,
        "activation": args.activation,
        "type_vocab_size": 1,
        "example_batch_size": args.batch_size,
        "example_sequence_length": seq_len,
    }
    config_path = args.output_dir / "config.json"
    config_path.write_text(json.dumps(config, indent=2) + "\n")
    print(f"wrote {config_path}")


if __name__ == "__main__":
    main()
