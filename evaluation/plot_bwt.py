#!/usr/bin/env python3
"""
Plot per-task BWT (Backward Transfer) as a line chart.

X-axis: task number and name (e.g. 1-dbpedia, 2-amazon, ...)
Y-axis: per-task BWT (a_{T,t} - a_{t,t})
One line per config; each config can have multiple runs (averaged).

Config format: CONFIGS = [{"base_dir": ..., "label": "...", "run_ids": [1,2,3]}, ...]
All configs use the same order (single value in ORDERS).

Usage:
  python evaluation/plot_bwt.py --config evaluation/bwt_plot_config.py
  python evaluation/plot_bwt.py --base_dir logs_and_outputs/ella/short_32_batch --order 1 --run_ids 1 2 3 --label ELLA --output bwt_plot.png
"""

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    import matplotlib.pyplot as plt
    import numpy as np
except ImportError:
    print("Error: matplotlib and numpy required. Install with: pip install matplotlib numpy", file=sys.stderr)
    sys.exit(1)

# Import from compute_bwt_fwt (same directory)
_here = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("compute_bwt_fwt", _here / "compute_bwt_fwt.py")
_cb = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_cb)
TASK_ORDERS = _cb.TASK_ORDERS
load_a_T = _cb.load_a_T
load_a_tt = _cb.load_a_tt
compute_bwt_overall = _cb.compute_bwt


def compute_per_task_bwt(
    a_T: List[Optional[float]], a_tt: List[Optional[float]]
) -> List[Optional[float]]:
    """Per-task BWT: bwt_t = a_{T,t} - a_{t,t} for t=1..T-1 (last task excluded)."""
    T = len(a_tt)
    result = []
    for t in range(T - 1):
        r_T_t, r_t_t = a_T[t], a_tt[t]
        result.append((r_T_t - r_t_t) if (r_T_t is not None and r_t_t is not None) else None)
    return result


def get_bwt_for_config(
    base_dir: Path,
    order: int,
    run_ids: List[int],
    cwd: Path,
) -> Tuple[Optional[float], Optional[float], List[Optional[float]], List[str]]:
    """
    Returns: (mean_bwt, mean_oa, per_task_bwt_list, task_names).
    per_task_bwt_list has length T-1 (tasks 1..T-1).
    """
    base_dir = cwd / base_dir if not isinstance(base_dir, Path) else base_dir
    if order not in TASK_ORDERS:
        return None, None, [], []
    tasks = TASK_ORDERS[order]
    T = len(tasks)
    task_names = tasks[:-1]  # Exclude last task for per-task BWT

    bwts, oas, per_task_sums, per_task_count = [], [], [], []
    for run_id in run_ids:
        a_tt = load_a_tt(base_dir, order, run_id, tasks)
        a_T = load_a_T(base_dir, order, run_id, tasks)
        if all(x is None for x in a_tt):
            continue
        bwt = compute_bwt_overall(a_T, a_tt)
        if bwt is not None:
            bwts.append(bwt)
        per = compute_per_task_bwt(a_T, a_tt)
        if not per_task_sums:
            per_task_sums = [0.0] * len(per)
            per_task_count = [0] * len(per)
        for i, v in enumerate(per):
            if v is not None:
                per_task_sums[i] += v
                per_task_count[i] += 1
        # OA from last folder
        out_dir = base_dir / f"order_{order}" / f"outputs{run_id}"
        last_task = tasks[-1]
        rp = out_dir / f"{T}-{last_task}" / "all_results.json"
        if rp.is_file():
            with open(rp) as f:
                oa = json.load(f).get("predict_exact_match")
                if oa is not None:
                    oas.append(float(oa))

    n = len(bwts)
    if n == 0:
        return None, None, [], task_names
    mean_bwt = sum(bwts) / n
    mean_oa = sum(oas) / len(oas) if oas else None
    per_task_list = [
        per_task_sums[i] / per_task_count[i] if per_task_count[i] > 0 else None
        for i in range(len(per_task_sums))
    ]
    return mean_bwt, mean_oa, per_task_list, task_names


def load_config_from_file(path: Path) -> Dict[str, Any]:
    """Load CONFIGS from a Python file."""
    spec = importlib.util.spec_from_file_location("bwt_config", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    configs = getattr(mod, "CONFIGS", [])
    # Order: from first config's "order", else ORDER, else ORDERS[0], else 1
    order = None
    if configs and "order" in configs[0]:
        order = configs[0]["order"]
    if order is None:
        order = _single_order(getattr(mod, "ORDER", None), getattr(mod, "ORDERS", None))
    return {
        "configs": configs,
        "order": order,
        "run_ids": getattr(mod, "RUN_IDS", [1, 2, 3]),
        "output": getattr(mod, "OUTPUT", "bwt_plot.png"),
        "title": getattr(mod, "TITLE", "Per-task Backward Transfer (BWT)"),
    }


def _single_order(order: Optional[int], orders: Optional[List[int]]) -> int:
    """Return single order: ORDER if set, else first of ORDERS, else 1."""
    if order is not None:
        return order
    if orders and len(orders) > 0:
        return orders[0]
    return 1


def plot_bwt_line(
    configs: List[Dict[str, Any]],
    order: int,
    cwd: Path,
    output: str = "bwt_plot.png",
    title: str = "Per-task Backward Transfer (BWT)",
    run_ids: Optional[List[int]] = None,
) -> None:
    """
    Plot per-task BWT as line chart. One line per config.
    X-axis: task number and name (1-dbpedia, 2-amazon, ...)
    Y-axis: per-task BWT (a_{T,t} - a_{t,t})
    """
    run_ids = run_ids or [1, 2, 3]
    if order not in TASK_ORDERS:
        print(f"Error: unknown order {order}", file=sys.stderr)
        return
    tasks = TASK_ORDERS[order]
    task_labels = [f"{t}-{name}" for t, name in enumerate(tasks[:-1], start=1)]  # 1..T-1
    x = np.arange(len(task_labels))

    n_lines = sum(len(cfg.get("run_ids", run_ids)) for cfg in configs)
    fig, ax = plt.subplots(figsize=(max(10, len(task_labels) * 0.5), 6))
    colors = plt.cm.tab20(np.linspace(0, 1, max(n_lines, 20)))

    line_idx = 0
    for i, cfg in enumerate(configs):
        base_dir = cfg.get("base_dir")
        cfg_label = cfg.get("label", str(base_dir))
        cfg_run_ids = cfg.get("run_ids", run_ids)
        for run_id in cfg_run_ids:
            _, _, per_task, _ = get_bwt_for_config(base_dir, order, [run_id], cwd)
            if not per_task:
                print(f"Warning: no data for {cfg_label} run {run_id}", file=sys.stderr)
                continue
            vals = np.array([v if v is not None else np.nan for v in per_task])
            label = f"{cfg_label} run {run_id}" if len(configs) > 1 or len(cfg_run_ids) > 1 else cfg_label
            ax.plot(x, vals, "o-", label=label, color=colors[line_idx % len(colors)], linewidth=2, markersize=6)
            line_idx += 1

    ax.set_ylabel(r"$a_{T,t} - a_{t,t}$ (BWT per task)", fontsize=12)
    ax.set_xlabel("Task", fontsize=12)
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(task_labels, rotation=45, ha="right")
    ax.axhline(y=0, color="gray", linestyle="-", linewidth=0.5)
    ax.legend(loc="best", fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {output}")


def plot_per_task_bwt(
    base_dir: Path,
    order: int,
    run_ids: List[int],
    cwd: Path,
    output: str = "bwt_per_task.png",
    title: Optional[str] = None,
) -> None:
    """Plot per-task BWT (a_{T,t} - a_{t,t}) for each task t=1..T-1."""
    bwt_overall, oa, per_task, task_names = get_bwt_for_config(base_dir, order, run_ids, cwd)
    if not per_task or not task_names:
        print("No data for per-task BWT plot", file=sys.stderr)
        return

    fig, ax = plt.subplots(figsize=(max(8, len(task_names) * 0.6), 5))
    colors = ["#2ecc71" if (v is not None and v >= 0) else "#e74c3c" for v in per_task]
    vals = [v if v is not None else 0 for v in per_task]
    bars = ax.bar(range(len(task_names)), vals, color=colors, edgecolor="black", linewidth=0.5)
    ax.set_xticks(range(len(task_names)))
    ax.set_xticklabels(task_names, rotation=45, ha="right")
    ax.set_ylabel(r"$a_{T,t} - a_{t,t}$ (BWT per task)")
    ax.set_xlabel("Task")
    ax.set_title(title or f"Per-task Backward Transfer (Order {order})")
    ax.axhline(y=0, color="gray", linestyle="-", linewidth=0.5)
    if bwt_overall is not None:
        ax.axhline(y=bwt_overall, color="blue", linestyle="--", linewidth=1, label=f"Mean BWT={bwt_overall:.2f}")
        ax.legend()
    fig.tight_layout()
    fig.savefig(output, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {output}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Plot per-task BWT as line chart")
    parser.add_argument("--config", type=str, help="Path to config Python file defining CONFIGS, ORDER, RUN_IDS, OUTPUT")
    parser.add_argument("--base_dir", type=str, help="Base output dir (single config)")
    parser.add_argument("--order", type=int, default=1, help="Single task order (all configs use this)")
    parser.add_argument("--run_ids", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--label", type=str, default="ELLA")
    parser.add_argument("--output", type=str, default="bwt_plot.png")
    parser.add_argument("--cwd", type=str, default=None)
    args = parser.parse_args()

    cwd = Path(args.cwd) if args.cwd else Path(__file__).resolve().parent.parent

    if args.config:
        cfg = load_config_from_file(cwd / args.config)
        configs = cfg["configs"]
        order = cfg.get("order", args.order)
        run_ids = cfg.get("run_ids", args.run_ids)
        output = cfg.get("output", args.output)
        title = cfg.get("title", "Per-task Backward Transfer (BWT)")
        if not configs:
            print("No CONFIGS in config file", file=sys.stderr)
            return 1
    elif args.base_dir:
        configs = [{"base_dir": args.base_dir, "label": args.label, "run_ids": args.run_ids}]
        order = args.order
        run_ids = args.run_ids
        output = args.output
        title = "Per-task Backward Transfer (BWT)"
    else:
        print("Provide --config or --base_dir", file=sys.stderr)
        return 1

    plot_bwt_line(configs, order, cwd, output=output, title=title, run_ids=run_ids)
    return 0


if __name__ == "__main__":
    sys.exit(main())
