#!/usr/bin/env python3
"""
Run final-model evaluation on all tasks to compute a_{T,t} for BWT and OA.

Evaluates the adapter after learning all T tasks on each task's test set,
then writes final_eval_matrix.json with a_T = [a_{T,1}, ..., a_{T,T}].

Usage:
  python evaluation/run_final_eval.py --base_dir logs_and_outputs/ella/long_ella_first_base_W --order 4 --run_id 1

Run from CL_project directory.
"""

import argparse
import json
import os
import random
import subprocess
import sys
from pathlib import Path
from typing import Optional

# Task orders (must match run_long_sequence.py and compute_bwt_fwt.py)
TASK_ORDERS = {
    1: ["dbpedia", "amazon", "yahoo", "agnews"],
    2: ["dbpedia", "amazon", "agnews", "yahoo"],
    3: ["yahoo", "amazon", "agnews", "dbpedia"],
    4: ["MNLI", "CB", "WiC", "COPA", "QQP", "BoolQA", "RTE", "IMDB", "yelp", "amazon", "SST-2", "dbpedia", "agnews", "MultiRC", "yahoo"],
    5: ["MultiRC", "BoolQA", "WiC", "MNLI", "CB", "COPA", "QQP", "RTE", "IMDB", "SST-2", "dbpedia", "agnews", "yelp", "amazon", "yahoo"],
    6: ["yelp", "amazon", "MNLI", "CB", "COPA", "QQP", "RTE", "IMDB", "SST-2", "dbpedia", "agnews", "yahoo", "MultiRC", "BoolQA", "WiC"],
}

METRIC_KEYS = ["predict_exact_match"]


def get_project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def get_metric_from_results(data: dict) -> Optional[float]:
    for k in METRIC_KEYS:
        if k in data and data[k] is not None:
            return float(data[k])
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description="Run final model eval on all tasks for BWT/OA")
    parser.add_argument("--base_dir", type=str, required=True, help="Base output dir")
    parser.add_argument("--order", type=int, required=True, choices=[1, 2, 3, 4, 5, 6])
    parser.add_argument("--run_id", type=int, default=1)
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--cwd", type=str, default=None)
    args = parser.parse_args()

    cwd = Path(args.cwd) if args.cwd else get_project_root()
    base_dir = cwd / args.base_dir
    if args.order not in TASK_ORDERS:
        print(f"Error: unknown order {args.order}", file=sys.stderr)
        return 1
    tasks = TASK_ORDERS[args.order]
    T = len(tasks)

    out_base = base_dir / f"order_{args.order}" / f"outputs{args.run_id}"
    last_task = tasks[-1]
    final_adapter = out_base / f"{T}-{last_task}" / "adapter"
    if not final_adapter.is_dir():
        print(f"Error: final adapter not found: {final_adapter}", file=sys.stderr)
        return 1

    # Config prefix: short for orders 1-3, long for 4-6
    config_prefix = "short" if args.order <= 3 else "long"
    run_script = cwd / "src" / "run_uie_lora.py"
    ds_config = cwd / "configs" / "ds_configs" / "stage2.config"
    if not run_script.is_file():
        print(f"Error: {run_script} not found", file=sys.stderr)
        return 1

    env = os.environ.copy()
    env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    env["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    env.setdefault("TRANSFORMERS_CACHE", os.path.expanduser("~/.cache/huggingface"))

    eval_dir = out_base / "final_eval"
    eval_dir.mkdir(parents=True, exist_ok=True)

    a_T = []
    port = random.randint(25000, 30000)

    for t, task in enumerate(tasks):
        task_config = cwd / "configs" / config_prefix / f"order{args.order}_configs" / task
        if not task_config.is_dir():
            print(f"Warning: config not found for {task}, skipping", file=sys.stderr)
            a_T.append(None)
            continue
        task_out = eval_dir / f"task{t+1}-{task}"
        task_out.mkdir(parents=True, exist_ok=True)

        cmd = [
            "deepspeed",
            "--master_port", str(port),
            str(run_script),
            "--cl_method", "ella",
            "--do_predict",
            "--predict_with_generate",
            "--model_name_or_path", str(final_adapter),
            "--data_dir", "CL_Benchmark",
            "--task_config_dir", str(task_config.relative_to(cwd)),
            "--instruction_file", "configs/instruction_config.json",
            "--instruction_strategy", "single",
            "--output_dir", str(task_out.relative_to(cwd)),
            "--per_device_eval_batch_size", "128",
            "--deepspeed", str(ds_config.relative_to(cwd)),
            "--run_name", f"final_eval_{t+1}",
            "--max_source_length", "512",
            "--max_target_length", "50",
            "--generation_max_length", "50",
            "--add_task_name", "True",
            "--add_dataset_name", "True",
            "--overwrite_output_dir",
            "--overwrite_cache",
            "--seed", "73",
            "--report_to", "none",
        ]
        print(f"[{t+1}/{T}] Evaluating final model on task {t+1} ({task})...")
        r = subprocess.run(cmd, env=env, cwd=cwd)
        if r.returncode != 0:
            print(f"  Failed (exit {r.returncode})", file=sys.stderr)
            a_T.append(None)
            continue

        results_path = task_out / "all_results.json"
        if results_path.is_file():
            with open(results_path) as f:
                data = json.load(f)
            val = get_metric_from_results(data)
            a_T.append(val)
            print(f"  -> {val:.4f}" if val is not None else "  -> N/A")
        else:
            a_T.append(None)
            print("  -> no all_results.json")

    matrix_path = out_base / "final_eval_matrix.json"
    with open(matrix_path, "w") as f:
        json.dump({"a_T": a_T, "tasks": tasks}, f, indent=2)
    print(f"\nWrote {matrix_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
