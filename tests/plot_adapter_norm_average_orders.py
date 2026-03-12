#!/usr/bin/env python3
"""
Plot average Frobenius norm of each task's adapter per transformer block, aggregated
across different orders (and optionally runs) with confidence intervals.

Loads adapters from multiple order directories (e.g. order_4, order_5, order_6) and
optionally multiple runs (outputs1, outputs2, outputs3). For each task POSITION (1, 2, 3...),
computes the mean and 95% CI of the per-block norms across all (order, run) combinations.

Usage:
  python tests/plot_adapter_norm_average_orders.py --base_dir logs_and_outputs/ella/long_1000train --orders 4 5 6
  python tests/plot_adapter_norm_average_orders.py --base_dir logs_and_outputs/ella/long_1000train --orders 4 5 6 --runs 1 2 3
"""

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy import stats


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


def load_task_slice_norms_per_block(
    adapter_dir: Path,
    task_nr: int,
    r: int,
    r_sum: int,
) -> Dict[Tuple[str, int], float]:
    """
    Load adapter from adapter_dir; extract the slice for task_nr (last r rows of A, last r cols of B).
    Return dict mapping (enc/dec, block_idx) -> Frobenius norm (sum of ||B@A||_F over modules in that block).
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
        A = state_dict[a_key]
        B = state_dict[b_key]
        if A.shape[0] != r_sum or B.shape[1] != r_sum:
            continue
        A_t = A[-r:, :].float()
        B_t = B[:, -r:].float()
        update = B_t @ A_t
        norm = torch.norm(update, p="fro").item()
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


def compute_mean_and_ci(
    values: np.ndarray,
    confidence: float = 0.95,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute mean and confidence interval for each column (block).
    values: shape (n_samples, n_blocks)
    Returns: (mean, lower, upper) each shape (n_blocks,)
    """
    n = values.shape[0]
    mean = np.nanmean(values, axis=0)
    if n <= 1:
        return mean, mean, mean
    std = np.nanstd(values, axis=0, ddof=1)
    # Use t-distribution for small samples
    t_val = stats.t.ppf((1 + confidence) / 2, df=n - 1)
    margin = t_val * std / np.sqrt(n)
    lower = mean - margin
    upper = mean + margin
    return mean, lower, upper


def main():
    parser = argparse.ArgumentParser(
        description="Plot adapter Frobenius norm per block, averaged across orders with CI"
    )
    parser.add_argument(
        "--base_dir",
        type=str,
        default="logs_and_outputs/ella/short_32_batch",
        help="Base directory containing order_4, order_5, ... subdirs",
    )
    parser.add_argument(
        "--orders",
        type=int,
        nargs="+",
        default=[1,2,3],
        help="Order numbers (e.g. 4 5 6)",
    )
    parser.add_argument(
        "--runs",
        type=int,
        nargs="+",
        default=[2],
        help="Run numbers -> outputs1, outputs2, ... (e.g. 1 2 3). Default: [2] for outputs2 only.",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="adapter_norm_avg_orders",
        help="Output plot path prefix",
    )
    parser.add_argument(
        "--tasks",
        type=int,
        nargs="+",
        default=None,
        help="Restrict to these task numbers (default: all)",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.95,
        help="Confidence level for interval (default: 0.95)",
    )
    args = parser.parse_args()

    base_dir = Path(args.base_dir)
    if not base_dir.is_dir():
        raise SystemExit(f"Not a directory: {base_dir}")

    # Collect all (order, run) -> base path
    order_run_paths: List[Tuple[int, int, Path]] = []
    for order in args.orders:
        for run_id in args.runs:
            path = base_dir / f"order_{order}" / f"outputs{run_id}"
            if path.is_dir():
                order_run_paths.append((order, run_id, path))
            else:
                print(f"Warning: skipping {path} (not found)")

    if not order_run_paths:
        raise SystemExit(
            f"No valid (order, run) paths found. Check --base_dir {base_dir} --orders {args.orders} --runs {args.runs}"
        )

    # Use first path to get block structure and max task count
    first_path = order_run_paths[0][2]
    first_tasks = collect_task_folders(first_path)
    if not first_tasks:
        raise SystemExit(f"No task adapter folders found under {first_path}")

    if args.tasks is not None:
        first_tasks = [t for t in first_tasks if t[0] in args.tasks]
    if not first_tasks:
        raise SystemExit("No tasks left after filter.")

    # Get block order from last task of first path
    last_adapter = first_tasks[-1][2]
    sd = torch.load(last_adapter / "adapter_model.bin", map_location="cpu", weights_only=True)
    keys = [k for k in sd if "lora_A" in k]
    enc_blocks, dec_blocks = get_block_order(keys)
    task_positions = [t[0] for t in first_tasks]

    # Build data: for each task position -> list of (norms_enc, norms_dec) from each order/run
    # task_position -> list of (enc_norms, dec_norms)
    enc_by_task: Dict[int, List[np.ndarray]] = {tp: [] for tp in task_positions}
    dec_by_task: Dict[int, List[np.ndarray]] = {tp: [] for tp in task_positions}

    for order, run_id, path in order_run_paths:
        tasks = collect_task_folders(path)
        if args.tasks is not None:
            tasks = [t for t in tasks if t[0] in args.tasks]
        for task_nr, task_name, adapter_dir in tasks:
            if task_nr not in enc_by_task:
                continue
            with open(adapter_dir / "adapter_config.json") as f:
                cfg = json.load(f)
            r_task = cfg["r"]
            r_sum_task = cfg["r_sum"]
            norms_map = load_task_slice_norms_per_block(adapter_dir, task_nr, r_task, r_sum_task)
            enc_norms = np.array([norms_map.get(b, 0.0) for b in enc_blocks])
            dec_norms = np.array([norms_map.get(b, 0.0) for b in dec_blocks])
            enc_by_task[task_nr].append(enc_norms)
            dec_by_task[task_nr].append(dec_norms)

    n_samples = len(order_run_paths)
    print(f"Aggregating over {n_samples} (order, run) combinations: {[(o, r) for o, r, _ in order_run_paths]}")

    out_prefix = args.out.rstrip(".png")
    if out_prefix == args.out and not out_prefix.endswith("_"):
        out_prefix = args.out

    def do_plot(
        block_order: List[Tuple[str, int]],
        by_task: Dict[int, List[np.ndarray]],
        title: str,
        out_path: str,
    ) -> None:
        block_labels = [f"{b[1]}" for b in block_order]
        x = np.arange(len(block_order))
        fig, ax = plt.subplots(figsize=(12, 6))

        colors = plt.cm.tab10(np.linspace(0, 1, max(len(by_task), 10)))
        for i, task_nr in enumerate(sorted(by_task.keys())):
            samples = by_task[task_nr]
            if not samples:
                continue
            arr = np.array(samples)
            mean, lower, upper = compute_mean_and_ci(arr, confidence=args.confidence)
            color = colors[i % len(colors)]
            ax.plot(x, mean, marker="o", markersize=4, label=f"Task {task_nr}", color=color, alpha=0.9)
            ax.fill_between(x, lower, upper, color=color, alpha=0.25)
            # Label task number at end of line so colors don't confuse
            ax.annotate(
                str(task_nr),
                xy=(x[-1], mean[-1]),
                xytext=(4, 0),
                textcoords="offset points",
                fontsize=8,
                color=color,
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor=color, alpha=0.9),
            )

        ax.set_xticks(x)
        ax.set_xticklabels(block_labels, rotation=0)
        ax.set_xlabel("Block index")
        ax.set_ylabel("Frobenius norm (mean ± 95% CI across orders)")
        ax.set_title(title)
        ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved {out_path}")

    do_plot(
        enc_blocks,
        enc_by_task,
        f"Adapter Frobenius norm per block (encoder) — mean across {n_samples} orders/runs, {int(args.confidence*100)}% CI",
        f"{out_prefix}_encoder.png",
    )
    do_plot(
        dec_blocks,
        dec_by_task,
        f"Adapter Frobenius norm per block (decoder) — mean across {n_samples} orders/runs, {int(args.confidence*100)}% CI",
        f"{out_prefix}_decoder.png",
    )


if __name__ == "__main__":
    main()
