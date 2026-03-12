#!/usr/bin/env python3
"""
Print per-task accuracy: after each adapter (1..15), accuracies for each task, tab-separated.
Configuration is in this file.

Usage:
  python evaluation/print_per_task_acc.py
"""

import json
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration (edit here)
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
# OUTPUTS_DIR = ROOT / "logs_and_outputs/ella/long_ella_drop_0.1_first_base_W/order_4/outputs1"
OUTPUTS_DIR = ROOT / "logs_and_outputs/tasks_results/"
ORDER = 4

TASK_ORDERS = {
    1: ["dbpedia", "amazon", "yahoo", "agnews"],
    2: ["dbpedia", "amazon", "agnews", "yahoo"],
    3: ["yahoo", "amazon", "agnews", "dbpedia"],
    4: ["MNLI", "CB", "WiC", "COPA", "QQP", "BoolQA", "RTE", "IMDB", "yelp", "amazon", "SST-2", "dbpedia", "agnews", "MultiRC", "yahoo"],
    5: ["MultiRC", "BoolQA", "WiC", "MNLI", "CB", "COPA", "QQP", "RTE", "IMDB", "SST-2", "dbpedia", "agnews", "yelp", "amazon", "yahoo"],
    6: ["yelp", "amazon", "MNLI", "CB", "COPA", "QQP", "RTE", "IMDB", "SST-2", "dbpedia", "agnews", "yahoo", "MultiRC", "BoolQA", "WiC"],
}


def get_acc_for_task(data: dict, task: str):
    """Get accuracy for a task from all_results.json."""
    for key in [
        f"predict_exact_match_for_{task}",
        f"predict_exact_match_for_{task.lower()}",
        "predict_exact_match_for_MNLI",
        "predict_exact_match_for_NLI",
        "predict_exact_match",
    ]:
        if key in data and data[key] is not None:
            return float(data[key])
    return None


def main():
    outputs_dir = Path(OUTPUTS_DIR)
    if not outputs_dir.is_dir():
        print(f"Error: not a directory: {outputs_dir}", file=sys.stderr)
        return 1
    if ORDER not in TASK_ORDERS:
        print(f"Error: unknown order {ORDER}", file=sys.stderr)
        return 1
    tasks = TASK_ORDERS[ORDER]
    T = len(tasks)

    print("\t".join(["exact_match"] + tasks))
    for t in range(1, T + 1):
        task_name = tasks[t - 1]
        results_path = outputs_dir / f"{t}-{task_name}" / "all_results.json"
        exact_match = "-"
        row = []
        if results_path.is_file():
            with open(results_path) as f:
                data = json.load(f)
            em = data.get("predict_exact_match")
            if em is not None:
                exact_match = f"{float(em):.2f}"
            for j in range(T):
                if j < t:
                    acc = get_acc_for_task(data, tasks[j])
                    row.append(f"{acc:.2f}" if acc is not None else "-")
                else:
                    row.append("-")
        else:
            row = ["-"] * T
        print("\t".join([exact_match] + row))
    return 0


if __name__ == "__main__":
    sys.exit(main())
