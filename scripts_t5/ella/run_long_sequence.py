#!/usr/bin/env python3
"""
Long Sequence Benchmark Training Script for ELLA/O-LoRA

This script runs sequential training on 15 tasks for continual learning experiments.
It replaces the bash scripts with a cleaner, more maintainable Python implementation.

Usage:
    python scripts_t5/ella/long/run_long_sequence.py --order 5 --run_number 1 --cl_method ella
    python scripts_t5/ella/long/run_long_sequence.py --order 6 --run_number 1 --cl_method olora
"""

import json
import os
import sys
import subprocess
import argparse
import importlib.util
import random
import time
from pathlib import Path
from typing import List, Dict, Optional, Any


def log_print(*args, **kwargs):
    """Print and flush immediately for nohup compatibility."""
    print(*args, **kwargs)
    sys.stdout.flush()

# ============================================================================
# Task Orders Configuration
# ============================================================================
TASK_ORDERS = {
    # Order 4: mnli -> cb -> wic -> copa -> qqp -> boolqa -> rte -> imdb -> yelp -> amazon -> sst-2 -> dbpedia -> ag -> multirc -> yahoo
    4: ["MNLI", "CB", "WiC", "COPA", "QQP", "BoolQA", "RTE", "IMDB", "yelp", "amazon", "SST-2", "dbpedia", "agnews", "MultiRC", "yahoo"],
    
    # Order 5: multirc -> boolqa -> wic -> mnli -> cb -> copa -> qqp -> rte -> imdb -> sst-2 -> dbpedia -> agnews -> yelp -> amazon -> yahoo
    5: ["MultiRC", "BoolQA", "WiC", "MNLI", "CB", "COPA", "QQP", "RTE", "IMDB", "SST-2", "dbpedia", "agnews", "yelp", "amazon", "yahoo"],
    
    # Order 6: yelp -> amazon -> mnli -> cb -> copa -> qqp -> rte -> imdb -> sst-2 -> dbpedia -> agnews -> yahoo -> multirc -> boolqa -> wic
    6: ["yelp", "amazon", "MNLI", "CB", "COPA", "QQP", "RTE", "IMDB", "SST-2", "dbpedia", "agnews", "yahoo", "MultiRC", "BoolQA", "WiC"],
}

# Lambda configurations per order (lamda_1, lamda_2) for each task
LAMBDA_CONFIGS_ELLA = {
    4: {
        "lamda_1": [0, 5e5, 5e5, 5e5, 5e5, 5e5, 5e5, 5e5, 5e5, 5e5, 5e5, 5e5, 5e5, 5e5, 5e7],
        "lamda_2": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    },
    5: {
        "lamda_1": [0, 5e6, 5e6, 5e6, 5e6, 5e6, 5e6, 5e6, 5e6, 5e6, 5e6, 5e6, 5e7, 5e7, 5e7],
        "lamda_2": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    },
    6: {
        "lamda_1": [0, 5e5, 5e5, 5e5, 5e5, 5e5, 5e5, 5e5, 5e5, 5e5, 5e5, 5e5, 5e5, 5e5, 5e5],
        "lamda_2": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    },
}

LAMBDA_CONFIGS_OLORA = {
    4: {
        "lamda_1": [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 5, 5, 5, 5],
        "lamda_2": [0, 0, 0.1, 0, 0, 0, 0.3, 0.1, 0.05, 0, 0.1, 0.1, 0.1, 0, 0.1],
    },
    5: {
        "lamda_1": [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5],
        "lamda_2": [0, 0.1, 0, 0.1, 0.1, 0, 0.1, 0.3, 0.1, 0.5, 0, 0.1, 0, 0.1, 0.1],
    },
    6: {
        "lamda_1": [0.5, 0.5, 0.02, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
        "lamda_2": [0, 0, 0, 0.1, 0, 0, 0.3, 0, 0.1, 0.1, 0, 0.1, 0, 0.1, 0.3],
    },
}

# Number of epochs per task (if different from default)
EPOCHS_CONFIG = {
    4: [1] * 15,
    5: [1] * 15,
    6: [1] * 15,
}


def get_common_params(
    cl_method: str,
    seed: int,
    ella_variant: str = "ella",
    train_batch_size: Optional[int] = None,
    gradient_accumulation_steps: Optional[int] = None,
) -> Dict[str, str]:
    """Return common parameters for all training runs."""
    if train_batch_size is None:
        train_batch_size = 32 if cl_method == "ella" else 8
    if gradient_accumulation_steps is None:
        gradient_accumulation_steps = 1 if cl_method == "ella" else 8
    params = {
        "--do_train": "",
        "--do_predict": "",
        "--predict_with_generate": "",
        "--data_dir": "CL_Benchmark_dev",
        "--instruction_file": "configs/instruction_config.json",
        "--instruction_strategy": "single",
        "--per_device_train_batch_size": str(train_batch_size),
        "--per_device_eval_batch_size": "128",
        "--gradient_accumulation_steps": str(gradient_accumulation_steps),
        "--learning_rate": "1e-03",
        "--deepspeed": "configs/ds_configs/stage2.config",
        "--max_source_length": "512",
        "--max_target_length": "50",
        "--generation_max_length": "50",
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
        # "--max_num_instances_per_task": "1000",
        # "--max_num_instances_per_eval_task": "500",  # test on full instead!!!
        "--report_to": "wandb",
        "--log_cl_metrics": "True",
    }
    if cl_method == "ella":
        params["--ella_variant"] = ella_variant
    return params


def build_command(
    port: int,
    model_path: str,
    task_config_dir: str,
    output_dir: str,
    run_name: str,
    num_epochs: int,
    lamda_1: float,
    lamda_2: float,
    common_params: Dict[str, str],
    gpu_id: int = 0,
    lamda_1_list: Optional[str] = None,
    ella_drop: bool = False,
    drop_lowest: float = 0.1,
) -> List[str]:
    """Build the deepspeed command for a single task."""
    cmd = [
        "deepspeed",
        "--master_port", str(port),
        "src/run_uie_lora.py",
    ]
    
    # Add common parameters
    for key, value in common_params.items():
        cmd.append(key)
        if value:  # Only add value if it's not empty (for flags)
            cmd.append(value)
    
    # Add task-specific parameters
    cmd.extend([
        "--model_name_or_path", model_path,
        "--task_config_dir", task_config_dir,
        "--output_dir", output_dir,
        "--run_name", run_name,
        "--num_train_epochs", str(num_epochs),
        "--lamda_1", str(lamda_1),
        "--lamda_2", str(lamda_2),
    ])
    if lamda_1_list:
        cmd.extend(["--lamda_1_list", lamda_1_list])
    if ella_drop:
        cmd.append("--ella_drop")
        cmd.extend(["--drop_lowest", str(drop_lowest)])
    
    return cmd


def run_task(cmd: List[str], task_name: str, task_idx: int, gpu_id: int = 0,
             log_file: Optional[str] = None, wandb_project: Optional[str] = None) -> bool:
    """Run a single task and return success status.

    Args:
        cmd: Command to run
        task_name: Name of the task
        task_idx: Index of the task (0-based)
        gpu_id: GPU ID to use
        log_file: Path to log file for redirecting output
        wandb_project: Wandb project name (sets WANDB_PROJECT env)
    """
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
            # Redirect output to log file
            with open(log_file, "a") as f:
                f.write(f"\n{'='*60}\n")
                f.write(f"Task {task_idx + 1}: {task_name}\n")
                f.write(f"{'='*60}\n")
                f.write(f"Command: {' '.join(cmd)}\n\n")
                f.flush()
                result = subprocess.run(cmd, env=env, check=True, stdout=f, stderr=subprocess.STDOUT)
        else:
            result = subprocess.run(cmd, env=env, check=True)
        log_print(f"✓ Task {task_idx + 1} ({task_name}) completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        log_print(f"✗ Task {task_idx + 1} ({task_name}) failed with return code {e.returncode}")
        if log_file:
            log_print(f"  Check log file for details: {log_file}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Run Long Sequence Benchmark for ELLA/O-LoRA")
    parser.add_argument("--order", type=int, required=True, choices=list(TASK_ORDERS.keys()),
                        help="Task order number (5 or 6)")
    parser.add_argument("--run_number", type=int, default=1,
                        help="Run number for output directory naming")
    parser.add_argument("--seed", type=int, default=73,
                        help="Random seed")
    parser.add_argument("--cl_method", type=str, default="ella", choices=["ella", "olora"],
                        help="Continual learning method")
    parser.add_argument("--ella_variant", type=str, default="ella",
                        choices=["ella", "ella_with_base_w", "ella_first_base_w"],
                        help="ELLA loss variant: 'ella' (W_past), 'ella_with_base_w' (base model W), "
                             "'ella_first_base_w' (base W for task 1 only, then ella)")
    parser.add_argument("--base_model", type=str, default="initial_model/t5-large",
                        help="Path to base model")
    parser.add_argument("--gpu_id", type=int, default=0,
                        help="GPU ID to use")
    parser.add_argument("--start_task", type=int, default=1,
                        help="Task number to start from (1-indexed, useful for resuming)")
    parser.add_argument("--dry_run", action="store_true",
                        help="Print commands without executing")
    parser.add_argument("--run_name", type=str, default="long_round1",
                        help="Run name")
    parser.add_argument("--output_root", type=str, default=None,
                        help="Base directory for outputs/logs. Default: logs_and_outputs/{cl_method}")
    parser.add_argument("--wandb_project", type=str, default="CL",
                        help="Wandb project name. If not set, uses env WANDB_PROJECT or wandb default.")
    parser.add_argument("--lamda_1", type=str, default=None,
                        help="Comma-separated lamda_1 per task (e.g. '0,5e5,5e5,...'). "
                             "If not set, uses LAMBDA_CONFIGS for this order.")
    parser.add_argument("--lamda_2", type=str, default=None,
                        help="Comma-separated lamda_2 per task (e.g. '0,0,0,...'). "
                             "If not set, uses LAMBDA_CONFIGS for this order.")
    parser.add_argument("--lamda_1_list", type=str, default=None,
                        help="Incremental ELLA (string fallback): per-partition lambda_1, comma/semicolon-separated. "
                             "Prefer --lamda_1_list_json or --config_file with list of lists.")
    parser.add_argument("--lamda_1_list_json", type=str, default=None,
                        help="Incremental ELLA: per-task partition lambdas as JSON list of lists. "
                             "E.g. '[[0,0],[0.1,10.0],[0.1,10.0],...]'. Each inner list has 2 or 3 values.")
    parser.add_argument("--ella_drop", action="store_true",
                        help="Use ELLA trainer with drop_lowest (drop smallest fraction of elements in ELLA loss).")
    parser.add_argument("--drop_lowest", type=float, default=0.1,
                        help="Fraction of elements to drop per module when --ella_drop (e.g. 0.9 = keep top 10%%).")
    parser.add_argument("--config_file", type=str, default=None,
                        help="Python config file that may define lamda_1_list (list of lists). "
                             "E.g. lamda_1_list = [[0,0], [0.1,10.0], ...]")
    parser.add_argument("--train_batch_size", type=int, default=None,
                        help="Per-device train batch size. If not set: 32 for ella, 8 for olora.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=None,
                        help="Gradient accumulation steps. If not set: 1 for ella, 8 for olora.")

    args = parser.parse_args()

    def _list_of_lists_to_per_task_strings(ll: List[List[Any]], num_tasks: int) -> List[str]:
        """Convert list of lists to per-task comma-separated strings. Validates length."""
        if len(ll) != num_tasks:
            raise ValueError(f"lamda_1_list has {len(ll)} tasks but order has {num_tasks}")
        return [",".join(str(float(x)) for x in row) for row in ll]

    def _parse_lamda_1_list_per_task(
        num_tasks: int,
        str_arg: Optional[str] = None,
        json_arg: Optional[str] = None,
        config_file_arg: Optional[str] = None,
    ) -> Optional[List[str]]:
        """Resolve lamda_1_list: config file (list of lists) < JSON < string. Returns list of per-task strings or None."""
        lamda_1_list_per_task = None

        if config_file_arg:
            path = Path(config_file_arg).resolve()
            if not path.is_file():
                path = Path.cwd() / config_file_arg
            if path.is_file():
                spec = importlib.util.spec_from_file_location("config_module", path)
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)
                if hasattr(mod, "lamda_1_list"):
                    ll = getattr(mod, "lamda_1_list")
                    if isinstance(ll, list) and ll and isinstance(ll[0], (list, tuple)):
                        lamda_1_list_per_task = _list_of_lists_to_per_task_strings(
                            [[float(x) for x in row] for row in ll], num_tasks
                        )
            else:
                parser.error(f"--config_file not found: {config_file_arg}")

        if json_arg and json_arg.strip():
            try:
                ll = json.loads(json_arg)
                if isinstance(ll, list) and ll and isinstance(ll[0], (list, tuple)):
                    lamda_1_list_per_task = _list_of_lists_to_per_task_strings(
                        [[float(x) for x in row] for row in ll], num_tasks
                    )
            except (json.JSONDecodeError, ValueError) as e:
                parser.error(f"--lamda_1_list_json invalid: {e}")

        if str_arg and str_arg.strip():
            raw = str_arg.strip()
            if ";" in raw:
                groups = [g.strip() for g in raw.split(";") if g.strip()]
                if len(groups) != num_tasks:
                    parser.error(
                        f"--lamda_1_list has {len(groups)} semicolon-separated groups but order has {num_tasks} tasks"
                    )
                lamda_1_list_per_task = groups
            else:
                lamda_1_list_per_task = [raw] * num_tasks

        return lamda_1_list_per_task

    def _parse_lamda_list(s: Optional[str], name: str, expected_len: int) -> List[float]:
        if s is None:
            return None
        try:
            vals = [float(x.strip()) for x in s.split(",")]
            if len(vals) != expected_len:
                parser.error(
                    f"--{name} must have {expected_len} comma-separated values (one per task), got {len(vals)}"
                )
            return vals
        except ValueError as e:
            parser.error(f"--{name} must be comma-separated floats: {e}")

    # Setup
    tasks = TASK_ORDERS[args.order]
    default_lambdas = LAMBDA_CONFIGS_ELLA[args.order] if args.cl_method == "ella" else LAMBDA_CONFIGS_OLORA[args.order]
    lamda_1_list = _parse_lamda_list(args.lamda_1, "lamda_1", len(tasks))
    lamda_2_list = _parse_lamda_list(args.lamda_2, "lamda_2", len(tasks))
    lambdas = {
        "lamda_1": lamda_1_list if lamda_1_list is not None else default_lambdas["lamda_1"],
        "lamda_2": lamda_2_list if lamda_2_list is not None else default_lambdas["lamda_2"],
    }
    # Per-task lamda_1_list for incremental ELLA: None, or list of len(tasks) strings (one per task).
    # Prefer list of lists via --config_file or --lamda_1_list_json; fallback to --lamda_1_list string.
    lamda_1_list_per_task = _parse_lamda_1_list_per_task(
        len(tasks),
        str_arg=args.lamda_1_list,
        json_arg=args.lamda_1_list_json,
        config_file_arg=args.config_file,
    )
    epochs = EPOCHS_CONFIG[args.order]
    output_root = args.output_root or f"logs_and_outputs/{args.cl_method}"
    output_base = f"{output_root}/{args.run_name}/order_{args.order}/outputs{args.run_number}"
    log_dir = f"{output_root}/{args.run_name}/order_{args.order}/logs"
    log_file = f"{log_dir}/train_and_infer{args.run_number}.log"
    port = random.randint(25000, 30000)
    
    # Create log directory
    os.makedirs(log_dir, exist_ok=True)
    
    log_print(f"\n{'#'*60}")
    log_print(f"# Long Sequence Benchmark - Order {args.order}")
    log_print(f"# CL Method: {args.cl_method}")
    log_print(f"# Run Number: {args.run_number}")
    log_print(f"# Seed: {args.seed}")
    log_print(f"# Output Base: {output_base}")
    log_print(f"# Log File: {log_file}")
    log_print(f"# ELLA variant: {args.ella_variant}")
    log_print(f"# lamda_1: {lambdas['lamda_1']}")
    log_print(f"# lamda_2: {lambdas['lamda_2']}")
    if lamda_1_list_per_task is not None:
        log_print(f"# lamda_1_list (incremental, per task): {lamda_1_list_per_task}")
    log_print(f"# Tasks: {' -> '.join(tasks)}")
    log_print(f"{'#'*60}\n")

    # Write header to log file
    if not args.dry_run:
        with open(log_file, "w") as f:
            f.write(f"{'#'*60}\n")
            f.write(f"# Long Sequence Benchmark - Order {args.order}\n")
            f.write(f"# CL Method: {args.cl_method}\n")
            f.write(f"# Run Number: {args.run_number}\n")
            f.write(f"# Seed: {args.seed}\n")
            f.write(f"# ELLA variant: {args.ella_variant}\n")
            f.write(f"# lamda_1: {lambdas['lamda_1']}\n")
            f.write(f"# lamda_2: {lambdas['lamda_2']}\n")
            if getattr(args, "ella_drop", False):
                f.write(f"# ella_drop: True, drop_lowest: {getattr(args, 'drop_lowest', 0.1)}\n")
            if lamda_1_list_per_task is not None:
                f.write(f"# lamda_1_list (incremental, per task): {lamda_1_list_per_task}\n")
            f.write(f"# Tasks: {' -> '.join(tasks)}\n")
            f.write(f"{'#'*60}\n\n")
    
    # Run tasks sequentially
    for i, task in enumerate(tasks):
        task_num = i + 1
        
        # Skip tasks before start_task
        if task_num < args.start_task:
            log_print(f"Skipping Task {task_num}: {task}")
            continue
        
        # Determine model path
        if i == 0:
            model_path = args.base_model
        else:
            prev_task = tasks[i - 1]
            model_path = f"{output_base}/{i}-{prev_task}/adapter"
        
        # Determine ELLA variant for this task (ella_first_base_w = base W for task 1 only)
        if args.ella_variant == "ella_first_base_w":
            ella_variant_this_task = "ella_with_base_w" if i == 0 else "ella"
        else:
            ella_variant_this_task = args.ella_variant
        common_params = get_common_params(
            args.cl_method, args.seed, ella_variant_this_task,
            train_batch_size=args.train_batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
        )
        
        # Use incremental lamda_1_list when: cl_method=ella, lamda_1!=0, and lamda_1_list_per_task set (from config/JSON/string)
        lamda_1_list_arg = (lamda_1_list_per_task[i] if lamda_1_list_per_task is not None else None) if (
            args.cl_method == "ella" and lambdas["lamda_1"][i] != 0
        ) else None

        # Build command
        cmd = build_command(
            port=port,
            model_path=model_path,
            task_config_dir=f"configs/long/order{args.order}_configs/{task}",
            output_dir=f"{output_base}/{task_num}-{task}",
            run_name=f"long_round{task_num}_run{args.run_number}",
            num_epochs=epochs[i],
            lamda_1=lambdas["lamda_1"][i],
            lamda_2=lambdas["lamda_2"][i],
            common_params=common_params,
            gpu_id=args.gpu_id,
            lamda_1_list=lamda_1_list_arg,
            ella_drop=getattr(args, "ella_drop", False),
            drop_lowest=getattr(args, "drop_lowest", 0.1),
        )
        
        if args.dry_run:
            log_print(f"\n[DRY RUN] Task {task_num}: {task}")
            log_print(f"Command: {' '.join(cmd)}")
            continue
        
        # Run task
        success = run_task(cmd, task, i, args.gpu_id, log_file=log_file, wandb_project=args.wandb_project)
        
        if not success:
            log_print(f"\nTraining failed at Task {task_num}. Exiting.")
            log_print(f"Check log file for details: {log_file}")
            exit(1)
        
        # Sleep between tasks
        if i < len(tasks) - 1:
            log_print("Sleeping for 5 seconds before next task...")
            time.sleep(5)
    
    log_print(f"\n{'='*60}")
    log_print(f"All tasks completed successfully!")
    if not args.dry_run:
        log_print(f"Full log: {log_file}")
    log_print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
