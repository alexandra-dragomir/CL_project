#!/usr/bin/env python3
"""
Run test prediction for the best-lambda adapter of each task in a grid-search run.

By default this targets:
`logs_and_outputs/ella_new_splits/long_16_2_grid/order_4/outputs1`
and predicts on `CL_Benchmark_dev`, writing results into each task root
such as `.../outputs1/1-MNLI`.

Example:
    python scripts_t5/ella/predict_only.py
"""

import argparse
import json
import os
import random
import re
import shutil
import subprocess
import sys
from pathlib import Path


DEFAULT_RUN_DIR = "logs_and_outputs/ella_new_splits/long_16_2_grid/order_4/outputs1"
DEFAULT_DATA_DIR = "CL_Benchmark_dev"
DEFAULT_SEQUENCE_TYPE = "long"
DEFAULT_CL_METHOD = "ella"
DEFAULT_ELLA_VARIANT = "ella"
DEFAULT_SEED = 73
DEFAULT_GPU_ID = 0


def get_project_root() -> Path:
    return Path(__file__).resolve().parent.parent.parent


def get_deepspeed(root: Path) -> str:
    """Prefer venv_olora/bin/deepspeed so it works without activating venv."""
    venv_bin = root / "venv_olora" / "bin" / "deepspeed"
    if venv_bin.is_file():
        return str(venv_bin)
    exe = shutil.which("deepspeed")
    return exe or "deepspeed"


def get_hf_modules_cache() -> Path:
    return Path.home() / ".cache" / "huggingface" / "modules"


def prepend_pythonpath(env: dict, path: Path) -> None:
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{path}{os.pathsep}{existing}" if existing else str(path)


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def format_lambda_label(value) -> str:
    value = float(value)
    if value == 0:
        return "0"
    base, exp = f"{value:.0e}".split("e")
    return f"{base}e{int(exp)}"


def infer_order(run_dir: Path) -> int:
    for part in run_dir.parts:
        match = re.fullmatch(r"order_(\d+)", part)
        if match:
            return int(match.group(1))
    raise ValueError(f"Could not infer order from run directory: {run_dir}")


def parse_task_dir_name(task_dir: Path):
    match = re.fullmatch(r"(\d+)-(.+)", task_dir.name)
    if not match:
        raise ValueError(f"Unexpected task directory name: {task_dir.name}")
    return int(match.group(1)), match.group(2)


def resolve_selected_adapter(task_root: Path, summary: dict) -> Path:
    best_label = summary.get("best_lambda_label")
    if best_label is None and summary.get("best_lambda_1") is not None:
        best_label = format_lambda_label(summary["best_lambda_1"])
    if best_label is None:
        raise ValueError(f"Missing best lambda in {task_root / 'lambda_search_summary.json'}")

    adapter_path = task_root / f"adapter_lambda_{best_label}"
    if adapter_path.is_dir():
        return adapter_path

    selected_from_summary = summary.get("selected_adapter_path")
    if selected_from_summary:
        selected_path = Path(selected_from_summary)
        if selected_path.is_dir():
            return selected_path

    raise FileNotFoundError(f"Could not find selected adapter for {task_root.name}")


def collect_jobs(root: Path, run_dir: Path, sequence_type: str, order: int):
    jobs = []
    task_dirs = []
    for child in run_dir.iterdir():
        if child.is_dir() and re.fullmatch(r"\d+-.+", child.name):
            task_dirs.append(child)
    task_dirs.sort(key=lambda path: parse_task_dir_name(path)[0])

    for task_root in task_dirs:
        task_num, fallback_task_name = parse_task_dir_name(task_root)
        summary_path = task_root / "lambda_search_summary.json"
        if not summary_path.is_file():
            print(f"Skip (missing summary): {summary_path}")
            continue

        summary = load_json(summary_path)
        task_name = str(summary.get("task", fallback_task_name))
        adapter_path = resolve_selected_adapter(task_root, summary)
        task_config_dir = root / "configs" / sequence_type / f"order{order}_configs" / task_name
        if not task_config_dir.is_dir():
            raise FileNotFoundError(f"Missing task config dir: {task_config_dir}")

        jobs.append(
            {
                "task_num": task_num,
                "task_name": task_name,
                "task_root": task_root,
                "adapter_path": adapter_path,
                "task_config_dir": task_config_dir,
                "best_lambda_label": summary.get("best_lambda_label", "unknown"),
            }
        )
    return jobs


def build_predict_command(
    deepspeed_exe: str,
    run_script: Path,
    ds_config: Path,
    model_path: Path,
    data_dir: str,
    task_config_dir: Path,
    output_dir: Path,
    run_name: str,
    seed: int,
    cl_method: str,
    ella_variant: str,
    port: int,
):
    cmd = [
        deepspeed_exe,
        "--master_port",
        str(port),
        str(run_script),
        "--do_predict",
        "--predict_with_generate",
        "--data_dir",
        data_dir,
        "--instruction_file",
        "configs/instruction_config.json",
        "--instruction_strategy",
        "single",
        "--per_device_eval_batch_size",
        "128",
        "--deepspeed",
        str(ds_config),
        "--max_source_length",
        "512",
        "--max_target_length",
        "50",
        "--generation_max_length",
        "50",
        "--add_task_name",
        "True",
        "--add_dataset_name",
        "True",
        "--overwrite_output_dir",
        "--overwrite_cache",
        "--cl_method",
        cl_method,
        "--seed",
        str(seed),
        "--report_to",
        "none",
        "--model_name_or_path",
        str(model_path),
        "--task_config_dir",
        str(task_config_dir),
        "--output_dir",
        str(output_dir),
        "--run_name",
        run_name,
    ]
    if cl_method == "ella":
        cmd.extend(["--ella_variant", ella_variant])
    return cmd


def parse_args():
    parser = argparse.ArgumentParser(
        description="Predict on CL_Benchmark_dev using best-lambda adapters from a grid-search run."
    )
    parser.add_argument("--run_dir", type=str, default=DEFAULT_RUN_DIR, help="Grid-search outputs directory.")
    parser.add_argument("--data_dir", type=str, default=DEFAULT_DATA_DIR, help="Dataset directory to predict on.")
    parser.add_argument("--sequence_type", type=str, default=DEFAULT_SEQUENCE_TYPE, help="Config sequence type.")
    parser.add_argument("--order", type=int, default=None, help="Order number. If omitted, infer from run_dir.")
    parser.add_argument("--start_task", type=int, default=1, help="Start predicting from this 1-indexed task number.")
    parser.add_argument("--gpu_id", type=int, default=DEFAULT_GPU_ID, help="GPU ID to use.")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Seed for prediction.")
    parser.add_argument(
        "--cl_method",
        type=str,
        default=DEFAULT_CL_METHOD,
        choices=["ella", "olora"],
        help="Continual learning method used by the adapter.",
    )
    parser.add_argument(
        "--ella_variant",
        type=str,
        default=DEFAULT_ELLA_VARIANT,
        choices=["ella", "ella_with_base_w", "ella_first_base_w"],
        help="ELLA variant to pass during prediction when cl_method=ella.",
    )
    parser.add_argument("--dry_run", action="store_true", help="Print commands without executing them.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = get_project_root()
    os.chdir(root)

    run_dir = (root / args.run_dir).resolve() if not os.path.isabs(args.run_dir) else Path(args.run_dir).resolve()
    if not run_dir.is_dir():
        print(f"Error: run directory not found: {run_dir}", file=sys.stderr)
        return 1

    order = args.order if args.order is not None else infer_order(run_dir)
    jobs = collect_jobs(root=root, run_dir=run_dir, sequence_type=args.sequence_type, order=order)
    jobs = [job for job in jobs if job["task_num"] >= args.start_task]
    if not jobs:
        print(f"Error: no task directories found under {run_dir} for start_task={args.start_task}", file=sys.stderr)
        return 1

    deepspeed_exe = get_deepspeed(root)
    run_script = root / "src" / "run_uie_lora.py"
    ds_config = root / "configs" / "ds_configs" / "stage2.config"
    if not run_script.is_file():
        print(f"Error: {run_script} not found. Run from project root.", file=sys.stderr)
        return 1

    env = os.environ.copy()
    env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    env["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    env.setdefault("TRANSFORMERS_CACHE", os.path.expanduser("~/.cache/huggingface"))
    prepend_pythonpath(env, get_hf_modules_cache())
    base_port = random.randint(25000, 29000)

    print(f"Run dir: {run_dir}")
    print(f"Data dir: {args.data_dir}")
    print(f"Order: {order}")
    print(f"Start task: {args.start_task}")
    print(f"Found {len(jobs)} tasks")

    for index, job in enumerate(jobs, start=1):
        cmd = build_predict_command(
            deepspeed_exe=deepspeed_exe,
            run_script=run_script,
            ds_config=ds_config,
            model_path=job["adapter_path"],
            data_dir=args.data_dir,
            task_config_dir=job["task_config_dir"],
            output_dir=job["task_root"],
            run_name=f"predict_best_lambda_task{job['task_num']}",
            seed=args.seed,
            cl_method=args.cl_method,
            ella_variant=args.ella_variant,
            port=base_port + job["task_num"],
        )
        print(
            f"\n[{index}/{len(jobs)}] "
            f"{job['task_root'].name} | best lambda={job['best_lambda_label']} | output={job['task_root']}"
        )
        if args.dry_run:
            print(" ".join(cmd))
            continue

        result = subprocess.run(cmd, env=env, cwd=root)
        if result.returncode != 0:
            print(f"Failed with exit code {result.returncode}", file=sys.stderr)
            return result.returncode

    print("\nAll predictions done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
