#!/usr/bin/env python3
"""
Short Sequence Training Script for ELLA / O-LoRA (Orders 1, 2, 3)

Runs sequential training on 4 tasks per order (dbpedia, amazon, yahoo, agnews)
in the same task orders as scripts_t5/order_1.sh, order_2.sh, order_3.sh.
Supports --cl_method ella or olora, and for ELLA: variants ella, ella_with_base_w, ella_first_base_w.

Usage:
    python scripts_t5/ella/run_short_sequence.py --order 1 --run_number 1 --cl_method ella
    python scripts_t5/ella/run_short_sequence.py --order 2 --run_number 1 --cl_method olora
    python scripts_t5/ella/run_short_sequence.py --order 1 --ella_variant ella_first_base_w
"""

import os
import sys
import subprocess
import argparse
import random
import time
from typing import List, Dict, Optional


def log_print(*args, **kwargs):
    """Print and flush immediately for nohup compatibility."""
    print(*args, **kwargs)
    sys.stdout.flush()

# ============================================================================
# Task Orders Configuration (short: 4 tasks each, matching order_1/2/3.sh)
# ============================================================================
TASK_ORDERS = {
    # Order 1: dbpedia -> amazon -> yahoo -> agnews
    1: ["dbpedia", "amazon", "yahoo", "agnews"],
    # Order 2: dbpedia -> amazon -> agnews -> yahoo
    2: ["dbpedia", "amazon", "agnews", "yahoo"],
    # Order 3: yahoo -> amazon -> agnews -> dbpedia
    3: ["yahoo", "amazon", "agnews", "dbpedia"],
}

# Lambda and epochs: same for all short orders (matching order_1.sh)
LAMBDA_CONFIGS_ELLA = {
    1: {"lamda_1": [0, 3e4, 3e4, 3e4], "lamda_2": [0, 0, 0, 0]},
    2: {"lamda_1": [0, 3e4, 3e4, 3e4], "lamda_2": [0, 0, 0, 0]},
    3: {"lamda_1": [0, 3e4, 3e4, 3e4], "lamda_2": [0, 0, 0, 0]},
}

LAMBDA_CONFIGS_OLORA = {
    1: {"lamda_1": [0.5, 0.5, 0.5, 0.5], "lamda_2": [0, 0, 0, 0]},
    2: {"lamda_1": [0.5, 0.5, 0.5, 0.5], "lamda_2": [0, 0, 0, 0]},
    3: {"lamda_1": [0.5, 0.5, 0.5, 0.5], "lamda_2": [0, 0, 0, 0]},
}

EPOCHS_CONFIG = {
    1: [1] * 4,
    2: [1] * 4,
    3: [1] * 4,
}


def get_common_params(
    seed: int,
    cl_method: str = "ella",
    ella_variant: str = "ella",
    train_batch_size: Optional[int] = None,
    gradient_accumulation_steps: Optional[int] = None,
) -> Dict[str, str]:
    """Return common parameters for short sequence. Supports ella, olora, and ELLA variants."""
    if train_batch_size is None:
        train_batch_size = 32 if cl_method == "ella" else 8
    if gradient_accumulation_steps is None:
        gradient_accumulation_steps = 1 if cl_method == "ella" else 8
    return {
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
        "--ella_variant": ella_variant,
        "--seed": str(seed),
        "--report_to": "wandb",
        "--log_cl_metrics": "True",
    }


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
) -> List[str]:
    """Build the deepspeed command for a single task."""
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
        "--lamda_1", str(lamda_1),
        "--lamda_2", str(lamda_2),
    ])

    return cmd


def run_task(cmd: List[str], task_name: str, task_idx: int, gpu_id: int = 0,
             log_file: Optional[str] = None, wandb_project: Optional[str] = None) -> bool:
    """Run a single task and return success status."""
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
    parser = argparse.ArgumentParser(
        description="Run Short Sequence (Orders 1, 2, 3) with ELLA, O-LoRA, or ELLA variants (ella_with_base_w, ella_first_base_w)."
    )
    parser.add_argument("--order", type=int, default=1, choices=[1, 2, 3],
                        help="Task order number (1, 2, or 3)")
    parser.add_argument("--run_number", type=int, default=1,
                        help="Run number for output directory naming")
    parser.add_argument("--seed", type=int, default=73,
                        help="Random seed")
    parser.add_argument("--ella_variant", type=str, default="ella",
                        choices=["ella", "ella_with_base_w", "ella_first_base_w"],
                        help="ELLA loss variant: 'ella' (W_past), 'ella_with_base_w' (base model W), "
                             "'ella_first_base_w' (base W for task 1 only, then ella)")
    parser.add_argument("--base_model", type=str, default="initial_model/t5-large",
                        help="Path to base model")
    parser.add_argument("--gpu_id", type=int, default=0,
                        help="GPU ID to use")
    parser.add_argument("--start_task", type=int, default=1,
                        help="Task number to start from (1-indexed, for resuming)")
    parser.add_argument("--dry_run", action="store_true",
                        help="Print commands without executing")
    parser.add_argument("--run_name", type=str, default="short_round1",
                        help="Run name (e.g. short_round1)")
    parser.add_argument("--output_root", type=str, default=None,
                        help="Base directory for outputs/logs. Default: logs_and_outputs/{cl_method}")
    parser.add_argument("--wandb_project", type=str, default="CL",
                        help="Wandb project name (e.g. O-LoRA). If not set, uses env WANDB_PROJECT or wandb default.")
    parser.add_argument("--lamda_1", type=str, default=None,
                        help="Comma-separated lamda_1 per task (e.g. '0,3e4,3e4,3e4'). "
                             "If not set, uses LAMBDA_CONFIGS for this order.")
    parser.add_argument("--lamda_2", type=str, default=None,
                        help="Comma-separated lamda_2 per task (e.g. '0,0,0,0'). "
                             "If not set, uses LAMBDA_CONFIGS for this order.")
    parser.add_argument("--cl_method", type=str, default="ella",
                        choices=["ella", "olora"],
                        help="CL method: 'ella' or 'olora'")
    parser.add_argument("--train_batch_size", type=int, default=None,
                        help="Per-device train batch size. If not set: 32 for ella, 8 for olora.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=None,
                        help="Gradient accumulation steps. If not set: 1 for ella, 8 for olora.")
    args = parser.parse_args()

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

    tasks = TASK_ORDERS[args.order]
    default_lambdas = LAMBDA_CONFIGS_ELLA[args.order] if args.cl_method == "ella" else LAMBDA_CONFIGS_OLORA[args.order]
    lamda_1_list = _parse_lamda_list(args.lamda_1, "lamda_1", len(tasks))
    lamda_2_list = _parse_lamda_list(args.lamda_2, "lamda_2", len(tasks))
    lambdas = {
        "lamda_1": lamda_1_list if lamda_1_list is not None else default_lambdas["lamda_1"],
        "lamda_2": lamda_2_list if lamda_2_list is not None else default_lambdas["lamda_2"],
    }
    epochs = EPOCHS_CONFIG[args.order]
    output_root = args.output_root or f"logs_and_outputs/{args.cl_method}"
    output_base = f"{output_root}/{args.run_name}/order_{args.order}/outputs{args.run_number}"
    log_dir = f"{output_root}/{args.run_name}/order_{args.order}/logs"
    log_file = f"{log_dir}/train_and_infer{args.run_number}.log"
    port = random.randint(25000, 30000)

    os.makedirs(log_dir, exist_ok=True)

    log_print(f"\n{'#'*60}")
    log_print(f"# Short Sequence ({args.cl_method}) - Order {args.order}")
    log_print(f"# Run Number: {args.run_number}")
    log_print(f"# Seed: {args.seed}")
    log_print(f"# Output Base: {output_base}")
    log_print(f"# Log File: {log_file}")
    log_print(f"# ELLA variant: {args.ella_variant}")
    log_print(f"# lamda_1: {lambdas['lamda_1']}")
    log_print(f"# lamda_2: {lambdas['lamda_2']}")
    log_print(f"# Tasks: {' -> '.join(tasks)}")
    log_print(f"{'#'*60}\n")

    if not args.dry_run:
        with open(log_file, "w") as f:
            f.write(f"{'#'*60}\n")
            f.write(f"# Short Sequence ({args.cl_method}) - Order {args.order}\n")
            f.write(f"# Run Number: {args.run_number}\n")
            f.write(f"# Seed: {args.seed}\n")
            f.write(f"# ELLA variant: {args.ella_variant}\n")
            f.write(f"# lamda_1: {lambdas['lamda_1']}\n")
            f.write(f"# lamda_2: {lambdas['lamda_2']}\n")
            f.write(f"# Tasks: {' -> '.join(tasks)}\n")
            f.write(f"{'#'*60}\n\n")

    for i, task in enumerate(tasks):
        task_num = i + 1

        if task_num < args.start_task:
            log_print(f"Skipping Task {task_num}: {task}")
            continue

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
            args.seed, args.cl_method, ella_variant_this_task,
            train_batch_size=args.train_batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
        )

        task_config_dir = f"configs/short/order{args.order}_configs/{task}"
        run_name = f"order{args.order}_round{task_num}_run{args.run_number}"

        cmd = build_command(
            port=port,
            model_path=model_path,
            task_config_dir=task_config_dir,
            output_dir=f"{output_base}/{task_num}-{task}",
            run_name=run_name,
            num_epochs=epochs[i],
            lamda_1=lambdas["lamda_1"][i],
            lamda_2=lambdas["lamda_2"][i],
            common_params=common_params,
            gpu_id=args.gpu_id,
        )

        if args.dry_run:
            log_print(f"\n[DRY RUN] Task {task_num}: {task}")
            log_print(f"Command: {' '.join(cmd)}")
            continue

        success = run_task(cmd, task, i, args.gpu_id, log_file=log_file, wandb_project=args.wandb_project)

        if not success:
            log_print(f"\nTraining failed at Task {task_num}. Exiting.")
            log_print(f"Check log file for details: {log_file}")
            exit(1)

        if i < len(tasks) - 1:
            log_print("Sleeping for 5 seconds before next task...")
            time.sleep(5)

    log_print(f"\n{'='*60}")
    log_print("All tasks completed successfully!")
    if not args.dry_run:
        log_print(f"Full log: {log_file}")
    log_print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
