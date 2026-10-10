#!/usr/bin/env python3

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
import gen_models  # noqa: E402

N_HEAD_BY_D = {16: 2, 32: 2, 64: 2, 128: 2, 256: 4, 384: 6, 768: 12}
MASK_BIAS = 30.0


class MaskWrite(nn.Module):
    def forward(self, cache, kv, m_write):
        return cache * (1.0 - m_write) + kv * m_write


class WhereWrite(nn.Module):
    def forward(self, cache, kv, m_write):
        return torch.where(m_write > 0.5, kv.expand_as(cache), cache)


class ScatterWrite(nn.Module):
    def forward(self, cache, kv, index):
        return cache.scatter(2, index, kv)


def make_write(variant):
    return {"mask": MaskWrite, "where": WhereWrite, "scatter": ScatterWrite}[variant]()


class DecodeStep(nn.Module):
    def __init__(
        self,
        d_model,
        n_head,
        d_ff,
        t_max,
        variant,
        vocab=0,
        tie=False,
        embed_outside=False,
    ):
        super().__init__()
        assert d_model % n_head == 0
        self.d_model, self.n_head, self.hs = d_model, n_head, d_model // n_head
        self.t_max, self.variant, self.vocab, self.tie = t_max, variant, vocab, tie
        assert not (embed_outside and tie), "--embed-outside 与 --tie-embeddings 互斥"
        self.embed_outside = embed_outside
        if vocab:
            wte = torch.randn(vocab, d_model)
            if embed_outside:
                self.wte_external = wte
            else:
                self.wte = nn.Parameter(wte)
            if not tie:
                self.lm_head = nn.Linear(d_model, vocab, bias=False)
        self.ln1 = nn.LayerNorm(d_model)
        self.c_attn = nn.Linear(d_model, 3 * d_model)
        self.c_proj = nn.Linear(d_model, d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.ln_f = nn.LayerNorm(d_model)
        self.write = make_write(variant)

    def forward(self, x, k_cache, v_cache, m_write, m_attn):
        b = 1
        if self.vocab and not self.embed_outside:
            x = x @ self.wte
        h = self.ln1(x)
        q, k, v = self.c_attn(h).split(self.d_model, dim=2)
        shape = (b, 1, self.n_head, self.hs)
        q = q.view(*shape).transpose(1, 2)
        k = k.view(*shape).transpose(1, 2)
        v = v.view(*shape).transpose(1, 2)

        if self.variant == "scatter":
            idx = m_write.to(torch.int64).expand(b, self.n_head, 1, self.hs)
            k_out, v_out = self.write(k_cache, k, idx), self.write(v_cache, v, idx)
        else:
            k_out, v_out = self.write(k_cache, k, m_write), self.write(
                v_cache, v, m_write
            )

        att = (q @ k_out.transpose(-2, -1)) * (1.0 / math.sqrt(self.hs))
        att = att + (m_attn - 1.0) * MASK_BIAS
        att = F.softmax(att, dim=-1)
        y = (att @ v_out).transpose(1, 2).reshape(b, 1, self.d_model)

        x = x + self.c_proj(y)
        x = x + self.fc2(gen_models.new_gelu(self.fc1(self.ln2(x))))
        x_out = self.ln_f(x)
        if self.vocab:
            logits = x_out @ self.wte.t() if self.tie else self.lm_head(x_out)
            return x_out, k_out, v_out, logits
        return x_out, k_out, v_out


class WriteOnly(nn.Module):
    def __init__(self, variant):
        super().__init__()
        self.variant = variant
        self.write = make_write(variant)

    def forward(self, k_cache, v_cache, kv_k, kv_v, m_write):
        if self.variant != "scatter":
            return self.write(k_cache, kv_k, m_write), self.write(
                v_cache, kv_v, m_write
            )
        nh, hs = k_cache.shape[1], k_cache.shape[3]
        idx = m_write.to(torch.int64).expand(1, nh, 1, hs)
        return self.write(k_cache, kv_k, idx), self.write(v_cache, kv_v, idx)


def export(model, sample_inputs, input_names, output_names, out_dir, opset=17):
    import onnx
    import onnxruntime as ort

    out_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = out_dir / "model.onnx"
    model.eval()
    with torch.no_grad():
        ref = model(*sample_inputs)
    torch.onnx.export(
        model,
        tuple(sample_inputs),
        str(onnx_path),
        export_params=True,
        opset_version=opset,
        do_constant_folding=True,
        input_names=input_names,
        output_names=output_names,
        dynamo=False,
    )

    m = onnx.load(str(onnx_path))
    onnx.checker.check_model(m)
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    feed = {n: t.numpy() for n, t in zip(input_names, sample_inputs)}
    got = sess.run(None, feed)
    ref = ref if isinstance(ref, tuple) else (ref,)
    max_diff = max(float(np.abs(r.numpy() - g).max()) for r, g in zip(ref, got))
    assert (
        max_diff < 1e-3
    ), f"onnx export diverges from torch (max abs diff {max_diff:.3e})"

    (out_dir / "input.json").write_text(
        json.dumps(
            {
                "input_data": [t.numpy().reshape(-1).tolist() for t in sample_inputs],
                "input_shapes": [list(t.shape) for t in sample_inputs],
                "output_data": [r.numpy().reshape(-1).tolist() for r in ref],
            }
        )
    )
    op_counts = {}
    for node in m.graph.node:
        op_counts[node.op_type] = op_counts.get(node.op_type, 0) + 1
    return dict(onnx_path=str(onnx_path), max_abs_diff=max_diff, op_counts=op_counts)


def build(args):
    torch.manual_seed(args.seed)
    extra = {}
    d, t_max = args.d_model, args.t_max
    nh = args.n_head or N_HEAD_BY_D[d]
    hs = d // nh
    t = args.position
    assert 0 <= t < t_max

    k_cache = torch.randn(1, nh, t_max, hs) * 0.5
    v_cache = torch.randn(1, nh, t_max, hs) * 0.5
    k_cache[:, :, t:, :] = 0.0
    v_cache[:, :, t:, :] = 0.0
    m_write = torch.zeros(1, 1, t_max, 1)
    m_write[:, :, t, :] = 1.0

    if args.write_only:
        model = WriteOnly(args.variant)
        if args.variant == "scatter":
            m_write = torch.full((1, 1, 1, 1), float(t))
        inputs = [
            k_cache,
            v_cache,
            torch.randn(1, nh, 1, hs),
            torch.randn(1, nh, 1, hs),
            m_write,
        ]
        names = ["k_cache", "v_cache", "k_t", "v_t", "m_write"]
        outs = ["k_cache_out", "v_cache_out"]
    else:
        v = args.vocab
        model = DecodeStep(
            d,
            nh,
            4 * d,
            t_max,
            args.variant,
            vocab=v,
            tie=args.tie_embeddings,
            embed_outside=args.embed_outside,
        )
        if args.variant == "scatter":
            m_write = torch.full((1, 1, 1, 1), float(t))
        m_attn = torch.zeros(1, 1, 1, t_max)
        m_attn[:, :, :, : t + 1] = 1.0
        if v and args.embed_outside:
            extra["wte"] = model.wte_external
            first = model.wte_external[args.token % v].reshape(1, 1, d)
            names0, outs = "x_t", ["x_out", "k_cache_out", "v_cache_out", "logits"]
        elif v:
            first = torch.zeros(1, 1, v)
            first[0, 0, args.token % v] = 1.0
            names0, outs = "tok_onehot", [
                "x_out",
                "k_cache_out",
                "v_cache_out",
                "logits",
            ]
        else:
            first = torch.randn(1, 1, d)
            names0, outs = "x_t", ["x_out", "k_cache_out", "v_cache_out"]
        inputs = [first, k_cache, v_cache, m_write, m_attn]
        names = [names0, "k_cache", "v_cache", "m_write", "m_attn"]

    return model, inputs, names, outs, nh, extra


def main():
    ap = argparse.ArgumentParser(
        description=None, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--d-model", type=int, default=384)
    ap.add_argument("--n-head", type=int, default=None, help="默认按 N_HEAD_BY_D 取")
    ap.add_argument("--t-max", type=int, default=128, help="KV cache 的固定槽位数")
    ap.add_argument(
        "--position", type=int, default=None, help="本步写入的位置 t，默认 t_max//2"
    )
    ap.add_argument("--variant", choices=["mask", "where", "scatter"], default="where")
    ap.add_argument(
        "--write-only", action="store_true", help="只导出缓存写入段（R1 对照）"
    )
    ap.add_argument(
        "--vocab",
        type=int,
        default=0,
        help="词表大小；>0 时挂 LM head（输入换成 one-hot token，多一个 logits 输出）",
    )
    ap.add_argument("--token", type=int, default=0, help="--vocab>0 时的初始 token id")
    ap.add_argument(
        "--tie-embeddings",
        action="store_true",
        help="LM head 复用 wte^T（GPT-2 式）；默认使用独立的 LM head 权重",
    )
    ap.add_argument(
        "--embed-outside",
        action="store_true",
        help="把嵌入查表移出电路：输入直接是 x_t=wte[tok] 而非 one-hot。"
        "验证方自算 wte[tok] 并用 Com(x_t) 钉死（权重公开，tok 也是它算的）。"
        "与 --tie-embeddings 互斥",
    )
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    if args.position is None:
        args.position = args.t_max // 2

    model, inputs, names, outs, nh, extra = build(args)
    kind = "write" if args.write_only else "step"
    name = f"{kind}_{args.variant}_d{args.d_model}_T{args.t_max}"
    if args.vocab:
        name += f"_V{args.vocab}" + ("_tied" if args.tie_embeddings else "")
        if args.embed_outside:
            name += "_embedout"
    out_dir = Path(args.out_dir) if args.out_dir else REPO / "models_incremental" / name
    info = export(model, inputs, names, outs, out_dir)
    if "wte" in extra:
        wte = extra["wte"]
        (out_dir / "wte.json").write_text(
            json.dumps({"shape": list(wte.shape), "data": wte.reshape(-1).tolist()})
        )
        info["wte_path"] = str(out_dir / "wte.json")
    info.update(
        name=name,
        d_model=args.d_model,
        n_head=nh,
        t_max=args.t_max,
        variant=args.variant,
        write_only=args.write_only,
        vocab=args.vocab,
        cache_elems=2 * nh * args.t_max * (args.d_model // nh),
    )
    print(json.dumps(info, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
