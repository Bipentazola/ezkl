#!/usr/bin/env python3

import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import onnx
import torch
import torch.nn as nn
import torch.nn.functional as F

REPO = Path(__file__).resolve().parent.parent
os.environ.setdefault("HF_HOME", "/data/hf_cache")

MASK_BIAS = 100.0


def new_gelu(x):
    return (
        0.5
        * x
        * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x * x * x)))
    )


class Attention(nn.Module):
    def __init__(self, d_model: int, n_head: int, seq_len: int, scale_qk: bool):
        super().__init__()
        assert d_model % n_head == 0
        self.n_head, self.d_model = n_head, d_model
        self.hs = d_model // n_head
        self.scale_qk = scale_qk
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=True)
        causal = torch.tril(torch.ones(seq_len, seq_len))
        self.register_buffer("attn_bias", (causal - 1.0) * MASK_BIAS, persistent=False)

    def forward(self, x):
        b, t, c = x.size()
        q = self.q_proj(x).view(b, t, self.n_head, self.hs).transpose(1, 2)
        k = self.k_proj(x).view(b, t, self.n_head, self.hs).transpose(1, 2)
        v = self.v_proj(x).view(b, t, self.n_head, self.hs).transpose(1, 2)
        att = q @ k.transpose(-2, -1)
        if self.scale_qk:
            att = att * (1.0 / math.sqrt(self.hs))
        att = att + self.attn_bias
        att = F.softmax(att, dim=-1)
        y = (att @ v).transpose(1, 2).contiguous().view(b, t, c)
        return self.out_proj(y)


class MLP(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.c_fc = nn.Linear(d_model, d_ff)
        self.c_proj = nn.Linear(d_ff, d_model)

    def forward(self, x):
        return self.c_proj(new_gelu(self.c_fc(x)))


class Block(nn.Module):
    def __init__(self, d_model, n_head, d_ff, seq_len, scale_qk, eps):
        super().__init__()
        self.ln_1 = nn.LayerNorm(d_model, eps=eps)
        self.attn = Attention(d_model, n_head, seq_len, scale_qk)
        self.ln_2 = nn.LayerNorm(d_model, eps=eps)
        self.mlp = MLP(d_model, d_ff)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class PretrainedLM(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                Block(
                    cfg["d_model"],
                    cfg["n_head"],
                    cfg["d_ff"],
                    cfg["seq_len"],
                    cfg["scale_qk"],
                    cfg["eps"],
                )
                for _ in range(cfg["n_layers"])
            ]
        )
        self.ln_f = nn.LayerNorm(cfg["d_model"], eps=cfg["eps"])
        self.lm_head = nn.Linear(cfg["d_model"], cfg["vocab_size"], bias=False)

    def forward(self, h):
        for blk in self.blocks:
            h = blk(h)
        return self.lm_head(self.ln_f(h))


def load_gpt_neo(hf_model, cfg):
    sd = hf_model.state_dict()
    m = PretrainedLM(cfg)
    with torch.no_grad():
        for i, blk in enumerate(m.blocks):
            p = f"transformer.h.{i}."
            blk.ln_1.weight.copy_(sd[p + "ln_1.weight"])
            blk.ln_1.bias.copy_(sd[p + "ln_1.bias"])
            a = p + "attn.attention."
            blk.attn.q_proj.weight.copy_(sd[a + "q_proj.weight"])
            blk.attn.k_proj.weight.copy_(sd[a + "k_proj.weight"])
            blk.attn.v_proj.weight.copy_(sd[a + "v_proj.weight"])
            blk.attn.out_proj.weight.copy_(sd[a + "out_proj.weight"])
            blk.attn.out_proj.bias.copy_(sd[a + "out_proj.bias"])
            blk.ln_2.weight.copy_(sd[p + "ln_2.weight"])
            blk.ln_2.bias.copy_(sd[p + "ln_2.bias"])
            blk.mlp.c_fc.weight.copy_(sd[p + "mlp.c_fc.weight"])
            blk.mlp.c_fc.bias.copy_(sd[p + "mlp.c_fc.bias"])
            blk.mlp.c_proj.weight.copy_(sd[p + "mlp.c_proj.weight"])
            blk.mlp.c_proj.bias.copy_(sd[p + "mlp.c_proj.bias"])
        m.ln_f.weight.copy_(sd["transformer.ln_f.weight"])
        m.ln_f.bias.copy_(sd["transformer.ln_f.bias"])
        m.lm_head.weight.copy_(sd["transformer.wte.weight"])
    return m, sd["transformer.wte.weight"].clone(), sd["transformer.wpe.weight"].clone()


def build(model_id: str, seq_len: int):
    from transformers import AutoConfig, AutoModelForCausalLM

    hf_cfg = AutoConfig.from_pretrained(model_id)
    hf = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.float32).eval()

    if hf_cfg.model_type == "gpt_neo":
        cfg = dict(
            d_model=hf_cfg.hidden_size,
            n_head=hf_cfg.num_heads,
            n_layers=hf_cfg.num_layers,
            d_ff=hf_cfg.intermediate_size or 4 * hf_cfg.hidden_size,
            vocab_size=hf_cfg.vocab_size,
            eps=hf_cfg.layer_norm_epsilon,
            scale_qk=False,
            qkv_bias=False,
            seq_len=seq_len,
        )
        if seq_len > hf_cfg.window_size:
            sys.exit(
                f"seq_len {seq_len} > window_size {hf_cfg.window_size}: "
                "local layers would stop being plain causal attention"
            )
        m, wte, wpe = load_gpt_neo(hf, cfg)
    elif hf_cfg.model_type == "gpt2":
        cfg = dict(
            d_model=hf_cfg.n_embd,
            n_head=hf_cfg.n_head,
            n_layers=hf_cfg.n_layer,
            d_ff=4 * hf_cfg.n_embd,
            vocab_size=hf_cfg.vocab_size,
            eps=hf_cfg.layer_norm_epsilon,
            scale_qk=True,
            qkv_bias=True,
            seq_len=seq_len,
        )
        m, wte, wpe = load_gpt2(hf, cfg)
    else:
        sys.exit(f"unsupported model_type {hf_cfg.model_type}")
    return hf, m, cfg, wte, wpe


def load_gpt2(hf, cfg):
    sd = hf.state_dict()
    d = cfg["d_model"]
    m = PretrainedLM(cfg)
    for blk in m.blocks:
        blk.attn.q_proj = nn.Linear(d, d, bias=True)
        blk.attn.k_proj = nn.Linear(d, d, bias=True)
        blk.attn.v_proj = nn.Linear(d, d, bias=True)
    with torch.no_grad():
        for i, blk in enumerate(m.blocks):
            p = f"transformer.h.{i}."
            blk.ln_1.weight.copy_(sd[p + "ln_1.weight"])
            blk.ln_1.bias.copy_(sd[p + "ln_1.bias"])
            w, b = sd[p + "attn.c_attn.weight"], sd[p + "attn.c_attn.bias"]
            wq, wk, wv = w.split(d, dim=1)
            bq, bk, bv = b.split(d, dim=0)
            blk.attn.q_proj.weight.copy_(wq.t())
            blk.attn.q_proj.bias.copy_(bq)
            blk.attn.k_proj.weight.copy_(wk.t())
            blk.attn.k_proj.bias.copy_(bk)
            blk.attn.v_proj.weight.copy_(wv.t())
            blk.attn.v_proj.bias.copy_(bv)
            blk.attn.out_proj.weight.copy_(sd[p + "attn.c_proj.weight"].t())
            blk.attn.out_proj.bias.copy_(sd[p + "attn.c_proj.bias"])
            blk.ln_2.weight.copy_(sd[p + "ln_2.weight"])
            blk.ln_2.bias.copy_(sd[p + "ln_2.bias"])
            blk.mlp.c_fc.weight.copy_(sd[p + "mlp.c_fc.weight"].t())
            blk.mlp.c_fc.bias.copy_(sd[p + "mlp.c_fc.bias"])
            blk.mlp.c_proj.weight.copy_(sd[p + "mlp.c_proj.weight"].t())
            blk.mlp.c_proj.bias.copy_(sd[p + "mlp.c_proj.bias"])
        m.ln_f.weight.copy_(sd["transformer.ln_f.weight"])
        m.ln_f.bias.copy_(sd["transformer.ln_f.bias"])
        m.lm_head.weight.copy_(sd["transformer.wte.weight"])
    return m, sd["transformer.wte.weight"].clone(), sd["transformer.wpe.weight"].clone()


def embed(wte, wpe, ids):
    pos = torch.arange(ids.shape[1])
    return wte[ids] + wpe[pos][None, :, :]


def main():
    global MASK_BIAS
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="roneneldan/TinyStories-1M")
    ap.add_argument("--seq-len", type=int, default=8)
    ap.add_argument("--out", required=True)
    ap.add_argument("--opset", type=int, default=10)
    ap.add_argument("--tol", type=float, default=1e-3)
    ap.add_argument("--mask-bias", type=float, default=MASK_BIAS)
    ap.add_argument(
        "--corpus",
        default="models_pretrained/corpus/tinystories_valid_head.txt",
        help="核对用的真实语料；随机 token id 会低估掩码泄漏，必须用真实文本核",
    )
    ap.add_argument("--verify-windows", type=int, default=32)
    args = ap.parse_args()
    MASK_BIAS = args.mask_bias

    out = Path(args.out)
    if not out.is_absolute():
        out = REPO / out
    out.mkdir(parents=True, exist_ok=True)

    hf, m, cfg, wte, wpe = build(args.model, args.seq_len)
    m.eval()

    corpus = Path(args.corpus)
    if not corpus.is_absolute():
        corpus = REPO / corpus
    from transformers import AutoTokenizer

    tokzr = AutoTokenizer.from_pretrained(args.model)
    text = corpus.read_text(errors="ignore").split("\n", 1)[-1]
    ids_flat = tokzr(text)["input_ids"][: args.seq_len * args.verify_windows]
    ids = torch.tensor(np.array(ids_flat).reshape(-1, args.seq_len))
    with torch.no_grad():
        ref = hf(ids).logits
        h0_all = embed(wte, wpe, ids)
        got = m(h0_all)
    diff = float((ref - got).abs().max())
    n_arg = int((ref.argmax(-1) != got.argmax(-1)).sum())
    print(
        f"[verify] {ids.numel()} 真实语料 token: max |HF - reimpl| = {diff:.3e}, "
        f"argmax 不一致 {n_arg}/{ids.numel()}  (mask_bias={MASK_BIAS:g})"
    )
    if diff > args.tol:
        sys.exit(
            f"reimplementation diverges from {args.model} (>{args.tol})；"
            "多半是 --mask-bias 太小导致因果掩码泄漏"
        )
    h0 = h0_all[:1]
    got = got[:1]

    onnx_path = out / "model.onnx"
    torch.onnx.export(
        m,
        h0,
        str(onnx_path),
        export_params=True,
        opset_version=args.opset,
        do_constant_folding=True,
        input_names=["h0"],
        output_names=["logits"],
        dynamo=False,
    )
    onnx.checker.check_model(onnx.load(str(onnx_path)))

    import onnxruntime as ort

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    ort_out = sess.run(None, {"h0": h0.numpy()})[0]
    ort_diff = float(np.abs(ort_out - got.numpy()).max())
    print(f"[verify] max |torch - onnxruntime| = {ort_diff:.3e}")
    assert ort_diff < args.tol, "onnx export diverges from torch"

    (out / "input.json").write_text(
        json.dumps(
            {
                "input_data": [h0.numpy().reshape(-1).tolist()],
                "input_shapes": [list(h0.shape)],
                "output_data": [got.numpy().reshape(-1).tolist()],
            }
        )
    )

    np.savez(out / "embed.npz", wte=wte.numpy(), wpe=wpe.numpy())
    meta = dict(
        cfg,
        model=args.model,
        opset=args.opset,
        verify_max_abs_diff=diff,
        verify_argmax_mismatches=n_arg,
        verify_tokens=int(ids.numel()),
        verify_corpus=str(corpus.relative_to(REPO)),
        onnx_max_abs_diff=ort_diff,
        mask_bias=MASK_BIAS,
        embed_outside=True,
    )
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    print(
        f"[ok] {onnx_path}  (d={cfg['d_model']}, L={cfg['n_layers']}, V={cfg['vocab_size']}, S={args.seq_len})"
    )


if __name__ == "__main__":
    main()
