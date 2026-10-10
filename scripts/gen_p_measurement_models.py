#!/usr/bin/env python3

import csv
import sys
from pathlib import Path

REPO_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_DIR))
import gen_models  # noqa: E402

gen_models.MODELS_DIR = REPO_DIR / "models_p"
gen_models.MODELS_DIR.mkdir(parents=True, exist_ok=True)

N_HEAD_BY_D = {16: 2, 32: 2, 64: 2, 128: 2, 256: 4, 384: 6, 768: 12}
WIDTH_SCAN_D = (16, 32, 64, 128, 256, 384, 768)
WIDTH_SCAN_SEQ = 16
SEQ_SCAN = (8, 16, 32, 64, 128, 256, 512, 1024)
SEQ_SCAN_D = 128


def build_configs() -> dict:
    configs = {}

    def add(seq_len, d_model, n_head, group):
        cfg = dict(
            seq_len=seq_len, d_model=d_model, n_head=n_head, n_layers=1, group=group
        )
        name = gen_models.cfg_name(cfg)
        if name in configs:
            configs[name]["group"] += f"+{group}"
        else:
            configs[name] = cfg

    for d in WIDTH_SCAN_D:
        add(WIDTH_SCAN_SEQ, d, N_HEAD_BY_D[d], "width_scan")
    for s in SEQ_SCAN:
        add(s, SEQ_SCAN_D, N_HEAD_BY_D[SEQ_SCAN_D], "seq_scan")
    add(64, 384, 6, "anchor_nanoGPT")
    add(64, 768, 12, "anchor_gpt2small")

    return configs


def main():
    configs = build_configs()
    print(f"generating {len(configs)} configs: {list(configs.keys())}")

    rows = []
    for name, cfg in configs.items():
        print(f"\n== {name}: {cfg} ==")
        export_cfg = {k: v for k, v in cfg.items() if k != "group"}
        row = gen_models.export_one(name, export_cfg, opset=10, seed=42)
        row["group"] = cfg["group"]
        row["n_head"] = cfg["n_head"]
        rows.append(row)
        print(
            f"  params={row['params']:,}  flops~={row['flops']:,}  onnx_max_abs_diff={row['onnx_max_abs_diff']:.2e}"
        )

    manifest_path = gen_models.MODELS_DIR / "MANIFEST.csv"
    fieldnames = [
        "name",
        "group",
        "seq_len",
        "d_model",
        "n_head",
        "n_layers",
        "d_ff",
        "params",
        "flops",
        "onnx_path",
        "input_json_path",
        "onnx_max_abs_diff",
        "op_counts",
    ]
    with open(manifest_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row[k] for k in fieldnames})
    print(f"\nwrote {manifest_path}")


if __name__ == "__main__":
    main()
