#!/usr/bin/env python3
"""
Train each task from scratch (no continual learning order).

Uses configs/single_tasks_configs only: train/dev/test on the current task only
(no other tasks at eval time). Each task is trained from the base model with
the same default settings as short/long sequence scripts. Results and adapters
are saved under logs_and_outputs/tasks_results/<task_name>/.

Usage:
    # Long benchmark (15 tasks), 1000 instances per task (default)
    python scripts_t5/ella/run_tasks_from_scratch.py --config_type long

    # Short benchmark (4 tasks), custom max instances
    python scripts_t5/ella/run_tasks_from_scratch.py --config_type short --max_num_instances_per_task 500

    # Single task, dry run
    python scripts_t5/ella/run_tasks_from_scratch.py --config_type long --tasks MNLI --dry_run
"""

import os
import sys
import subprocess
import argparse
import random
import time
from pathlib import Path
from typing import List, Dict, Optional

def log_print(*args, **kwargs):
    print(*args, **kwargs)
    sys.stdout.flush()

# ---------------------------------------------------------------------------
# Task sets and config path: single-task configs only (train/dev/test on current task only)
# ---------------------------------------------------------------------------
CONFIG_DIR_BASE = "configs/single_tasks_configs"

# Long: 15 tasks
LONG_TASKS = [
    "MNLI", "CB", "WiC", "COPA", "QQP", "BoolQA", "RTE", "IMDB",
    "yelp", "amazon", "SST-2", "dbpedia", "agnews", "MultiRC", "yahoo",
]
# Short: 4 tasks (subset)
SHORT_TASKS = ["dbpedia", "amazon", "yahoo", "agnews"]

OUTPUT_BASE = "logs_and_outputs/tasks_results"


def get_common_params(
    seed: int,
    max_num_instances_per_task: int,
    cl_method: str = "ella",
    generation_max_length: str = "50",
) -> Dict[str, str]:
    """Same defaults as run_long_sequence / run_short_sequence."""
    params = {
        "--do_train": "",
        "--do_predict": "",
        "--predict_with_generate": "",
        "--data_dir": "CL_Benchmark",
        "--instruction_file": "configs/instruction_config.json",
        "--instruction_strategy": "single",
        "--per_device_train_batch_size": "32",
        "--per_device_eval_batch_size": "128",
        "--gradient_accumulation_steps": "1",
        "--learning_rate": "1e-03",
        "--deepspeed": "configs/ds_configs/stage2.config",
        "--max_source_length": "512",
        "--max_target_length": "50",
        "--generation_max_length": generation_max_length,
        "--add_task_name": "True",
        "--add_dataset_name": "True",
        "--overwrite_output_dir": "",
        "--overwrite_cache": "",
        "--lr_scheduler_type": "constant",
        "--warmup_steps": "0",
        "--logging_strategy": "steps",
        "--logging_steps": "10",
        "--eval_strategy": "no",
        "--save_strategy": "no",
        "--save_steps": "1500",
        "--cl_method": cl_method,
        "--seed": str(seed),
        "--max_num_instances_per_task": str(max_num_instances_per_task),
        "--report_to": "wandb",
        "--log_cl_metrics": "True",
    }
    if cl_method == "ella":
        params["--ella_variant"] = "ella"
    return params


def build_command(
    port: int,
    model_path: str,
    task_config_dir: str,
    output_dir: str,
    run_name: str,
    num_epochs: int,
    common_params: Dict[str, str],
    gpu_id: int = 0,
) -> List[str]:
    """Build deepspeed command for one task (from scratch: lamda_1=0, lamda_2=0)."""
    cmd = [
        "deepspeed",
        "--master_port", str(port),
        "src/run_uie_lora.py",
    ]
    for key, value in common_params.items():
        cmd.append(key)
        if value:
            cmd.append(value)
    cmd.extend([
        "--model_name_or_path", model_path,
        "--task_config_dir", task_config_dir,
        "--output_dir", output_dir,
        "--run_name", run_name,
        "--num_train_epochs", str(num_epochs),
        "--lamda_1", "0",
        "--lamda_2", "0",
    ])
    return cmd


def run_task(
    cmd: List[str],
    task_name: str,
    task_idx: int,
    gpu_id: int = 0,
    log_file: Optional[str] = None,
    wandb_project: Optional[str] = None,
) -> bool:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    env["TRANSFORMERS_CACHE"] = os.path.expanduser("~/.cache/huggingface")
    if wandb_project is not None:
        env["WANDB_PROJECT"] = wandb_project

    log_print(f"\n{'='*60}")
    log_print(f"Task {task_idx + 1}: {task_name}")
    log_print(f"{'='*60}")
    log_print(f"Command: {' '.join(cmd)}")
    if log_file:
        log_print(f"Log file: {log_file}")
    log_print()

    try:
        if log_file:
            with open(log_file, "a") as f:
                f.write(f"\n{'='*60}\n")
                f.write(f"Task {task_idx + 1}: {task_name}\n")
                f.write(f"{'='*60}\n")
                f.write(f"Command: {' '.join(cmd)}\n\n")
                f.flush()
                subprocess.run(cmd, env=env, check=True, stdout=f, stderr=subprocess.STDOUT)
        else:
            subprocess.run(cmd, env=env, check=True)
        log_print(f"✓ Task {task_idx + 1} ({task_name}) completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        log_print(f"✗ Task {task_idx + 1} ({task_name}) failed with return code {e.returncode}")
        if log_file:
            log_print(f"  Check log file: {log_file}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Train each task from scratch; save results and adapters under logs_and_outputs/tasks_results/<task>/"
    )
    parser.add_argument(
        "--config_type",
        type=str,
        default="long",
        choices=["long", "short"],
        help="Task set: 'long' (15 tasks) or 'short' (4 tasks). Configs: configs/single_tasks_configs (test only on current task).",
    )
    parser.add_argument(
        "--max_num_instances_per_task",
        type=int,
        default=1000,
        help="Max training instances per task (default: 1000, same as long sequence).",
    )
    parser.add_argument("--seed", type=int, default=3, help="Random seed.")
    parser.add_argument("--base_model", type=str, default="initial_model/t5-large", help="Base model path.")
    parser.add_argument("--gpu_id", type=int, default=0, help="GPU ID.")
    parser.add_argument("--num_train_epochs", type=int, default=1, help="Epochs per task.")
    parser.add_argument(
        "--tasks",
        type=str,
        default=None,
        help="Comma-separated task names to run (default: all for chosen config_type). "
             "E.g. --tasks MNLI,CB,RTE",
    )
    parser.add_argument("--dry_run", action="store_true", help="Print commands only.")
    parser.add_argument("--run_name", type=str, default="from_scratch", help="Run name for wandb/logs.")
    parser.add_argument("--wandb_project", type=str, default="CL", help="Wandb project name.")
    parser.add_argument("--cl_method", type=str, default="ella", choices=["ella", "olora"])

    args = parser.parse_args()

    all_tasks = LONG_TASKS if args.config_type == "long" else SHORT_TASKS

    if args.tasks:
        requested = [t.strip() for t in args.tasks.split(",") if t.strip()]
        invalid = [t for t in requested if t not in all_tasks]
        if invalid:
            parser.error(f"Unknown tasks for config_type={args.config_type}: {invalid}. Valid: {all_tasks}")
        tasks = requested
    else:
        tasks = all_tasks

    common_params = get_common_params(
        seed=args.seed,
        max_num_instances_per_task=args.max_num_instances_per_task,
        cl_method=args.cl_method,
    )

    log_dir = os.path.join(OUTPUT_BASE, "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"{args.run_name}_{args.config_type}.log")
    port = random.randint(25000, 30000)

    log_print(f"\n{'#'*60}")
    log_print(f"# Tasks from scratch (no order)")
    log_print(f"# config_type: {args.config_type} | config_dir: {CONFIG_DIR_BASE}")
    log_print(f"# max_num_instances_per_task: {args.max_num_instances_per_task}")
    log_print(f"# Output base: {OUTPUT_BASE}")
    log_print(f"# Log file: {log_file}")
    log_print(f"# Tasks: {tasks}")
    log_print(f"{'#'*60}\n")

    if not args.dry_run:
        with open(log_file, "w") as f:
            f.write(f"# Tasks from scratch - config_type={args.config_type} - configs: {CONFIG_DIR_BASE}\n")
            f.write(f"# max_num_instances_per_task={args.max_num_instances_per_task}\n")
            f.write(f"# Tasks: {tasks}\n\n")

    for i, task in enumerate(tasks):
        task_output_dir = os.path.join(OUTPUT_BASE, task)
        os.makedirs(task_output_dir, exist_ok=True)
        task_config_dir = os.path.join(CONFIG_DIR_BASE, task)
        if not os.path.isdir(task_config_dir):
            log_print(f"Skipping {task}: config dir not found: {task_config_dir}")
            continue

        cmd = build_command(
            port=port,
            model_path=args.base_model,
            task_config_dir=task_config_dir,
            output_dir=task_output_dir,
            run_name=f"{args.run_name}_{task}",
            num_epochs=args.num_train_epochs,
            common_params=common_params,
            gpu_id=args.gpu_id,
        )

        if args.dry_run:
            log_print(f"[DRY RUN] {task}: {' '.join(cmd)}")
            continue

        success = run_task(
            cmd, task, i,
            gpu_id=args.gpu_id,
            log_file=log_file,
            wandb_project=args.wandb_project,
        )
        if not success:
            log_print(f"Training failed at task {task}. Exiting.")
            sys.exit(1)
        if i < len(tasks) - 1:
            log_print("Sleeping 5s before next task...")
            time.sleep(5)

    log_print(f"\n{'='*60}")
    log_print("All tasks completed successfully.")
    log_print(f"Results and adapters: {OUTPUT_BASE}/<task>/")
    if not args.dry_run:
        log_print(f"Log: {log_file}")
    log_print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
