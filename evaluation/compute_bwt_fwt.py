#!/usr/bin/env python3
"""
Compute BWT (Backward Transfer) and FWT (Forward Transfer vs from-scratch) as in the ELLA paper.

BWT_T = (1/(T-1)) * sum_{t=1}^{T-1} (a_{T,t} - a_{t,t})
  - a_{T,t} = performance on task t after learning all T tasks (from last folder)
  - a_{t,t} = performance on task t right after learning task t (from each folder)

FWT = (1/T) * sum_{t=1}^{T} (a_{T,t} - a_scratch_t)
  - a_scratch_t = performance on task t when trained from scratch only on t (from tasks_results_*)

Reads:
  - CL run: base_dir/order_*/outputs{run_id}/...
  - From-scratch: scratch_dir/<task>/all_results.json (default logs_and_outputs/tasks_results_1)

Usage:
  python evaluation/compute_bwt_fwt.py --base_dir logs_and_outputs/ella/short_32_batch --order 1 --run_id 1
  python evaluation/compute_bwt_fwt.py ... --scratch_dir logs_and_outputs/tasks_results_1
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple

ROOT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_SCRATCH_DIR = ROOT_DIR / "logs_and_outputs" / "tasks_results_1"

TASK_ORDERS = {
    1: ["dbpedia", "amazon", "yahoo", "agnews"],
    2: ["dbpedia", "amazon", "agnews", "yahoo"],
    3: ["yahoo", "amazon", "agnews", "dbpedia"],
    4: ["MNLI", "CB", "WiC", "COPA", "QQP", "BoolQA", "RTE", "IMDB", "yelp", "amazon", "SST-2", "dbpedia", "agnews", "MultiRC", "yahoo"],
    5: ["MultiRC", "BoolQA", "WiC", "MNLI", "CB", "COPA", "QQP", "RTE", "IMDB", "SST-2", "dbpedia", "agnews", "yelp", "amazon", "yahoo"],
    6: ["yelp", "amazon", "MNLI", "CB", "COPA", "QQP", "RTE", "IMDB", "SST-2", "dbpedia", "agnews", "yahoo", "MultiRC", "BoolQA", "WiC"],
}


def get_metric_for_task(data: Dict[str, Any], task: str) -> Optional[float]:
    """Extract accuracy for a specific task from all_results.json."""
    candidates = [
        f"predict_exact_match_for_{task}",
        f"predict_exact_match_for_{task.lower()}",
        "predict_exact_match_for_MNLI",
        "predict_exact_match_for_NLI",
        "predict_exact_match",
    ]
    for key in candidates:
        if key in data and data[key] is not None:
            return float(data[key])
    return None


def load_a_tt(base_dir: Path, order: int, run_id: int, tasks: List[str]) -> List[Optional[float]]:
    """a_{t,t}: from folder {t}-{task}/all_results.json, metric for that task."""
    out_dir = base_dir / f"order_{order}" / f"outputs{run_id}"
    a_tt = []
    for t, task in enumerate(tasks, start=1):
        results_path = out_dir / f"{t}-{task}" / "all_results.json"
        if not results_path.is_file():
            a_tt.append(None)
            continue
        with open(results_path) as f:
            data = json.load(f)
        a_tt.append(get_metric_for_task(data, task))
    return a_tt


def load_overall_accuracy(base_dir: Path, order: int, run_id: int, tasks: List[str]) -> Optional[float]:
    """Overall accuracy = predict_exact_match from last task folder."""
    out_dir = base_dir / f"order_{order}" / f"outputs{run_id}"
    last_task = tasks[-1]
    results_path = out_dir / f"{len(tasks)}-{last_task}" / "all_results.json"
    if not results_path.is_file():
        return None
    with open(results_path) as f:
        data = json.load(f)
    return data.get("predict_exact_match")


def load_a_T(base_dir: Path, order: int, run_id: int, tasks: List[str]) -> List[Optional[float]]:
    """a_{T,t}: from last folder's all_results.json (has eval on all tasks)."""
    out_dir = base_dir / f"order_{order}" / f"outputs{run_id}"
    last_task = tasks[-1]
    results_path = out_dir / f"{len(tasks)}-{last_task}" / "all_results.json"
    if not results_path.is_file():
        return [None] * len(tasks)
    with open(results_path) as f:
        data = json.load(f)
    return [get_metric_for_task(data, task) for task in tasks]


def load_a_scratch(scratch_dir: Path, tasks: List[str]) -> List[Optional[float]]:
    """a_scratch_t: from scratch_dir/<task>/all_results.json for each task."""
    a_scratch = []
    for task in tasks:
        results_path = scratch_dir / task / "all_results.json"
        if not results_path.is_file():
            a_scratch.append(None)
            continue
        with open(results_path) as f:
            data = json.load(f)
        a_scratch.append(get_metric_for_task(data, task))
    return a_scratch


def compute_bwt(a_T: List[Optional[float]], a_tt: List[Optional[float]]) -> Optional[float]:
    """BWT_T = (1/(T-1)) * sum_{t=1}^{T-1} (a_{T,t} - a_{t,t})"""
    T = len(a_tt)
    if T < 2:
        return None
    terms = []
    for t in range(T - 1):
        r_T_t, r_t_t = a_T[t], a_tt[t]
        if r_T_t is not None and r_t_t is not None:
            terms.append(r_T_t - r_t_t)
    if not terms:
        return None
    return sum(terms) / (T - 1)


def compute_fwt(
    a_T: List[Optional[float]], a_scratch: List[Optional[float]]
) -> Optional[float]:
    """FWT = (1/T) * sum_t (a_{T,t} - a_scratch_t). CL vs from-scratch per task."""
    if not a_T or not a_scratch or len(a_T) != len(a_scratch):
        return None
    terms = []
    for r_T_t, r_scratch_t in zip(a_T, a_scratch):
        if r_T_t is not None and r_scratch_t is not None:
            terms.append(r_T_t - r_scratch_t)
    if not terms:
        return None
    return sum(terms) / len(terms)


def main(config: Dict[str, Any]) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    base_dir = Path(config["base_dir"])
    order = config["order"]
    scratch_dir = Path(config.get("scratch_dir", DEFAULT_SCRATCH_DIR))
    run_id = config["run_id"]

    if order not in TASK_ORDERS:
        print(f"Error: unknown order {order}", file=sys.stderr)
        return None, None, None
    tasks = TASK_ORDERS[order]
    T = len(tasks)

    if not base_dir.is_dir():
        print(f"Error: base_dir not found: {base_dir}", file=sys.stderr)
        return None, None, None

    a_tt = load_a_tt(base_dir, order, run_id, tasks)
    a_T = load_a_T(base_dir, order, run_id, tasks)

    if all(x is None for x in a_tt):
        print(f"Error: no results in {base_dir / f'order_{order}' / f'outputs{run_id}'}", file=sys.stderr)
        return None, None, None

    bwt = compute_bwt(a_T, a_tt)
    if bwt is None:
        print("Error: cannot compute BWT (missing a_{T,t} or a_{t,t})", file=sys.stderr)
        return None, None, None

    a_scratch = load_a_scratch(scratch_dir, tasks)
    fwt = compute_fwt(a_T, a_scratch)

    oa = load_overall_accuracy(base_dir, order, run_id, tasks)
    if oa is not None:
        oa = float(oa)
    return oa, bwt, fwt


def fmt(x: Optional[float]) -> str:
    return f"{x:.4f}" if x is not None else "N/A"


if __name__ == "__main__":
    base_dir = ROOT_DIR / "logs_and_outputs/ella_new_splits/long_16_2"
    order = 4  
    CONFIGS = [
        {
            "base_dir": base_dir,
            "order": 4,
            "run_id": 1,
            "scratch_dir": ROOT_DIR / "logs_and_outputs/tasks_results_42",
        },
        
        
       
    ]

    print("-" * 40)
    print("OA\tBWT\tFWT")
    for config in CONFIGS:
        oa, bwt, fwt = main(config)
        print(f"{fmt(oa)}\t{fmt(bwt)}\t{fmt(fwt)}")
    print("-" * 40)
