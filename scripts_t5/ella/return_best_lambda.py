#!/usr/bin/env python3
"""
Return the list of best lambda_1 values chosen by the grid-search runner.

Example:
    python scripts_t5/ella/return_best_lambda.py \
        --run_dir logs_and_outputs/ella_new_splits/long_16_2_grid/order_4
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Tuple


ROOT = Path(__file__).resolve().parent.parent.parent


def _load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _collect_task_summaries(run_dir: Path) -> List[Tuple[int, str, float]]:
    summaries = []
    for summary_path in run_dir.glob("outputs*/**/lambda_search_summary.json"):
        payload = _load_json(summary_path)
        task_num = int(payload["task_num"])
        task = str(payload["task"])
        best_lambda = float(payload["best_lambda_1"])
        summaries.append((task_num, task, best_lambda))
    summaries.sort(key=lambda item: item[0])
    return summaries


def _format_lambda(value: float) -> str:
    if value == 0:
        return "0"
    s = f"{value:.0e}"
    base, exp = s.split("e")
    exp = str(int(exp))
    return f"{base}e{exp}"


def main(run_dir: Path) -> int:
    summaries = _collect_task_summaries(run_dir)
    if not summaries:
        print(f"Error: no lambda_search_summary.json files found under {run_dir}", file=sys.stderr)
        return 1

    lambda_list = [best_lambda for _, _, best_lambda in summaries]
    return [_format_lambda(v) for v in lambda_list]


if __name__ == "__main__":
    paths = [
        "logs_and_outputs/ella_new_splits/long_16_2_grid/order_4",
    ]
    for path in paths:
        run_dir = (ROOT / path).resolve()
        lambda_list = main(run_dir)
        print(f"Run dir: {run_dir}")
        print("Chosen lambda_1 values:")
        print(lambda_list)
        print()
    sys.exit(0)
