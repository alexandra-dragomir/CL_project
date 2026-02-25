#!/usr/bin/env python3
"""
Compute mean and std of a metric over multiple runs. (e.g. outputs1, outputs2, outputs3).

For each order, reads the metric from the last task folder in each run dir
(e.g. order_4/outputs1/<last_folder>/all_results.json) and reports mean and std.

Usage:
    python evaluation/return_results_from_runs.py
    python evaluation/return_results_from_runs.py --base_dir logs_and_outputs/ella/long_1000train --orders 4 5 6 --runs 1 2 3
"""

import argparse
import json
from pathlib import Path
from typing import Optional


# Default config
DEFAULT_BASE_DIR = "logs_and_outputs/long_1000train/"
DEFAULT_ORDERS = [1, 2, 3]
DEFAULT_RUNS = [1, 2, 3]  # outputs1, outputs2, outputs3
DEFAULT_RESULTS_FILE = "all_results.json"
DEFAULT_METRIC = "predict_exact_match"


def last_task_folder(output_dir: Path) -> Optional[Path]:
    """Return path to the last task folder in output_dir (by numeric prefix, e.g. 15-yahoo)."""
    if not output_dir.is_dir():
        return None
    dirs = []
    for p in output_dir.iterdir():
        if not p.is_dir():
            continue
        name = p.name
        if "-" in name:
            prefix = name.split("-", 1)[0]
            if prefix.isdigit():
                dirs.append((int(prefix), p))
    if not dirs:
        return None
    dirs.sort(key=lambda x: x[0])
    return dirs[-1][1]


def get_metric_from_run(
    base_dir: Path,
    order: int,
    run_id: int,
    results_file: str = DEFAULT_RESULTS_FILE,
    metric: str = DEFAULT_METRIC,
) -> Optional[float]:
    """
    Read metric from base_dir/order_N/outputs{run_id}/<last_folder>/{results_file}.
    Returns the value or None if missing.
    """
    output_dir = base_dir / f"order_{order}" / f"outputs{run_id}"
    last = last_task_folder(output_dir)
    if last is None:
        return None
    results_path = last / results_file
    if not results_path.is_file():
        return None
    with open(results_path) as f:
        data = json.load(f)
    return data.get(metric)


def main():
    parser = argparse.ArgumentParser(description="Mean ± std of metric over runs")
    parser.add_argument(
        "--base_dir",
        type=str,
        default=DEFAULT_BASE_DIR,
        help="Base dir (e.g. logs_and_outputs/ella/long_1000train)",
    )
    parser.add_argument(
        "--orders",
        type=int,
        nargs="+",
        default=DEFAULT_ORDERS,
        help="Order numbers (e.g. 4 5 6)",
    )
    parser.add_argument(
        "--runs",
        type=int,
        nargs="+",
        default=DEFAULT_RUNS,
        help="Run numbers for outputs1, outputs2, ... (e.g. 1 2 3)",
    )
    parser.add_argument(
        "--results_file",
        type=str,
        default=DEFAULT_RESULTS_FILE,
        help="Results JSON filename in each task folder",
    )
    parser.add_argument(
        "--metric",
        type=str,
        default=DEFAULT_METRIC,
        help="Metric key to aggregate (e.g. predict_exact_match)",
    )
    parser.add_argument(
        "--cwd",
        type=str,
        default=None,
        help="Working directory (default: current). Base dir is relative to this.",
    )
    args = parser.parse_args()

    cwd = Path(args.cwd) if args.cwd else Path.cwd()
    base_dir = cwd / args.base_dir
    if not base_dir.is_dir():
        print(f"Error: base_dir not found: {base_dir}")
        return 1

    print(f"Base dir: {base_dir}")
    print(f"Orders: {args.orders}, Runs: {args.runs}")
    print(f"Metric: {args.metric} from {args.results_file}")
    print()

    for order in args.orders:
        values = []
        for run_id in args.runs:
            v = get_metric_from_run(
                base_dir, order, run_id,
                results_file=args.results_file,
                metric=args.metric,
            )
            if v is not None:
                values.append(v)
            else:
                path = base_dir / f"order_{order}" / f"outputs{run_id}"
                print(f"  Warning: no value for order_{order} outputs{run_id}")
        if not values:
            print(f"Order {order}: no data")
            continue
        n = len(values)
        mean = sum(values) / n
        if n > 1:
            variance = sum((x - mean) ** 2 for x in values) / (n - 1)
            std = variance ** 0.5
        else:
            std = 0.0
        print(f"Order {order}: {args.metric} = {mean:.4f} +/- {std:.4f}  (n={n})")

    return 0


if __name__ == "__main__":
    exit(main())
