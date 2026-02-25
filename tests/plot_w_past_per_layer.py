#!/usr/bin/env python3
"""
Plot W_past Frobenius norm per transformer block (encoder and decoder separate).

W_past = lora_B @ lora_A (the full accumulated adapter up to that task).
For each task folder we load the whole lora_A and lora_B, compute W_past per module,
then sum ||W_past||_F per block. One curve per task checkpoint.
"""

import argparse
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch


def parse_block_from_key(key: str) -> Optional[Tuple[str, int]]:
    """Return ('encoder', block_i) or ('decoder', block_j) from state_dict key, else None."""
    enc = re.search(r"\.encoder\.block\.(\d+)\.", key)
    dec = re.search(r"\.decoder\.block\.(\d+)\.", key)
    if enc:
        return ("encoder", int(enc.group(1)))
    if dec:
        return ("decoder", int(dec.group(1)))
    return None


def get_block_order(keys: List[str]) -> Tuple[List[Tuple[str, int]], List[Tuple[str, int]]]:
    """Return (encoder_blocks, decoder_blocks) each sorted by block index."""
    blocks = set()
    for k in keys:
        b = parse_block_from_key(k)
        if b:
            blocks.add(b)
    enc_blocks = sorted([b for b in blocks if b[0] == "encoder"], key=lambda x: x[1])
    dec_blocks = sorted([b for b in blocks if b[0] == "decoder"], key=lambda x: x[1])
    return enc_blocks, dec_blocks


def load_w_past_norms_per_block(adapter_dir: Path) -> Dict[Tuple[str, int], float]:
    """
    Load adapter; compute W_past = lora_B @ lora_A (full accumulated adapter) per module.
    Return dict mapping (enc/dec, block_idx) -> sum of ||W_past||_F over modules in that block.
    """
    sd_path = adapter_dir / "adapter_model.bin"
    if not sd_path.is_file():
        return {}

    state_dict = torch.load(sd_path, map_location="cpu", weights_only=True)
    a_keys = [k for k in state_dict if ".lora_A.weight" in k]
    norms_by_block: Dict[Tuple[str, int], float] = {}

    for a_key in a_keys:
        prefix = a_key.replace(".lora_A.weight", "")
        b_key = prefix + ".lora_B.weight"
        if b_key not in state_dict:
            continue
        A = state_dict[a_key].float()  # [r_sum, in]
        B = state_dict[b_key].float()  # [out, r_sum]
        # W_past = whole adapter = B @ A  [out, in]
        w_past = B @ A
        norm = torch.norm(w_past, p="fro").item()
        block = parse_block_from_key(a_key)
        if block:
            norms_by_block[block] = norms_by_block.get(block, 0.0) + norm

    return norms_by_block


def collect_task_folders(base_dir: Path) -> List[Tuple[int, str, Path]]:
    """Return list of (task_nr, task_name, path_to_adapter_dir) sorted by task_nr."""
    tasks = []
    for d in base_dir.iterdir():
        if not d.is_dir():
            continue
        name = d.name
        if "-" in name:
            prefix = name.split("-", 1)[0]
            if prefix.isdigit():
                task_nr = int(prefix)
                task_name = name.split("-", 1)[1]
                adapter_dir = d / "adapter"
                if (adapter_dir / "adapter_model.bin").is_file() and (adapter_dir / "adapter_config.json").is_file():
                    tasks.append((task_nr, task_name, adapter_dir))
    tasks.sort(key=lambda x: x[0])
    return tasks


def main():
    parser = argparse.ArgumentParser(description="Plot W_past (full lora_B@lora_A) Frobenius norm per block")
    parser.add_argument(
        "--base_dir",
        type=str,
        default="logs_and_outputs/ella/long_1000train/order_4/outputs2",
        help="Directory containing 1-taskname, 2-taskname, ... adapter folders",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="w_past_per_layer",
        help="Output plot path prefix (<out>_encoder.png, <out>_decoder.png)",
    )
    parser.add_argument("--tasks", type=int, nargs="+", default=None, help="Restrict to these task numbers (default: all)")
    args = parser.parse_args()

    base_dir = Path(args.base_dir)
    if not base_dir.is_dir():
        raise SystemExit(f"Not a directory: {base_dir}")

    tasks = collect_task_folders(base_dir)
    if not tasks:
        raise SystemExit(f"No task adapter folders found under {base_dir}")

    if args.tasks is not None:
        tasks = [t for t in tasks if t[0] in args.tasks]
    if not tasks:
        raise SystemExit("No tasks left after filter.")

    # Get block order from last task
    last_adapter = tasks[-1][2]
    sd = torch.load(last_adapter / "adapter_model.bin", map_location="cpu", weights_only=True)
    keys = [k for k in sd if "lora_A" in k]
    enc_blocks, dec_blocks = get_block_order(keys)

    data_enc = []
    data_dec = []
    for task_nr, task_name, adapter_dir in tasks:
        norms_map = load_w_past_norms_per_block(adapter_dir)
        data_enc.append((task_nr, task_name, [norms_map.get(b, 0.0) for b in enc_blocks]))
        data_dec.append((task_nr, task_name, [norms_map.get(b, 0.0) for b in dec_blocks]))

    out_prefix = args.out.rstrip(".png")
    if out_prefix == args.out and not out_prefix.endswith("_"):
        out_prefix = args.out

    def do_plot(block_order: List[Tuple[str, int]], data: List, title: str, out_path: str) -> None:
        block_labels = [str(b[1]) for b in block_order]
        x = np.arange(len(block_order))
        fig, ax = plt.subplots(figsize=(12, 6))
        for task_nr, task_name, norms_list in data:
            line, = ax.plot(x, norms_list, marker="o", markersize=3, label=f"{task_nr}-{task_name}", alpha=0.8)
            # Label task number at end of line so colors don't confuse
            ax.annotate(
                str(task_nr),
                xy=(x[-1], norms_list[-1]),
                xytext=(4, 0),
                textcoords="offset points",
                fontsize=8,
                color=line.get_color(),
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor=line.get_color(), alpha=0.9),
            )
        ax.set_xticks(x)
        ax.set_xticklabels(block_labels, rotation=0)
        ax.set_xlabel("Block index")
        ax.set_ylabel("Frobenius norm (sum of ||W_past||_F per block)")
        ax.set_title(title)
        ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=7)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved {out_path}")

    do_plot(
        enc_blocks,
        data_enc,
        "W_past Frobenius norm per block (encoder) — full lora_B @ lora_A",
        f"{out_prefix}_encoder.png",
    )
    do_plot(
        dec_blocks,
        data_dec,
        "W_past Frobenius norm per block (decoder) — full lora_B @ lora_A",
        f"{out_prefix}_decoder.png",
    )


if __name__ == "__main__":
    main()
