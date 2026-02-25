#!/usr/bin/env python3
"""
Run prediction only (no training) for one or more trained adapters.

Set PRED_JOBS below: each entry is (model_path, task_config_dir, output_dir, seed).
Run from project root: python scripts_t5/ella/predict_only.py
"""

import os
import sys
import random
import subprocess
from pathlib import Path

# -----------------------------------------------------------------------------
# Configure here: list of (model_path, task_config_dir, output_dir, seed)
# Paths are relative to project root (directory containing scripts_t5).
# Add or remove rows to run prediction on the tasks you want.
# -----------------------------------------------------------------------------
DEFAULT_SEED = 73
PRED_JOBS = [
    # (model adapter path, task config dir, output dir, seed)
    (
        "logs_and_outputs/ella/long_1000train/order_5/outputs1/15-yahoo/adapter",
        "configs/long/order5_configs/yahoo",
        "logs_and_outputs/ella/long_1000train/order_5/outputs1/",
        1,
    ),
    (
        "logs_and_outputs/ella/long_1000train/order_5/outputs2/15-yahoo/adapter",
        "configs/long/order5_configs/yahoo",
        "logs_and_outputs/ella/long_1000train/order_5/outputs2/",
        2,
    ),
    (
        "logs_and_outputs/ella/long_1000train/order_5/outputs3/15-yahoo/adapter",
        "configs/long/order5_configs/yahoo",
        "logs_and_outputs/ella/long_1000train/order_5/outputs3/",
        3,
    ),
    (
        "logs_and_outputs/ella/long_1000train/order_6/outputs1/15-WiC/adapter",
        "configs/long/order6_configs/WiC",
        "logs_and_outputs/ella/long_1000train/order_6/outputs1/",
        1,
    ),
    (
        "logs_and_outputs/ella/long_1000train/order_6/outputs2/15-WiC/adapter",
        "configs/long/order6_configs/WiC",
        "logs_and_outputs/ella/long_1000train/order_6/outputs2/",
        2,
    ),
    (
        "logs_and_outputs/ella/long_1000train/order_6/outputs3/15-WiC/adapter",
        "configs/long/order6_configs/WiC",
        "logs_and_outputs/ella/long_1000train/order_6/outputs3/",
        3,
    ),
   
]

GPU_ID = 0


def get_project_root():
    return Path(__file__).resolve().parent.parent.parent


def get_deepspeed(root: Path):
    """Prefer venv_olora/bin/deepspeed so it works without activating venv."""
    venv_bin = root / "venv_olora" / "bin" / "deepspeed"
    if venv_bin.is_file():
        return str(venv_bin)
    import shutil
    exe = shutil.which("deepspeed")
    return exe or "deepspeed"


def main():
    root = get_project_root()
    os.chdir(root)

    deepspeed_exe = get_deepspeed(root)
    env = os.environ.copy()
    env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    env["CUDA_VISIBLE_DEVICES"] = str(GPU_ID)
    env.setdefault("TRANSFORMERS_CACHE", os.path.expanduser("~/.cache/huggingface"))

    port = random.randint(25000, 30000)
    run_script = root / "src" / "run_uie_lora.py"
    ds_config = root / "configs" / "ds_configs" / "stage2.config"

    if not run_script.is_file():
        print(f"Error: {run_script} not found. Run from project root.", file=sys.stderr)
        return 1

    for i, job in enumerate(PRED_JOBS):
        if len(job) == 4:
            model_path, task_config_dir, output_dir, seed = job
        else:
            model_path, task_config_dir, output_dir = job
            seed = DEFAULT_SEED
        run_name = f"predict_only_{i+1}"
        model_path_abs = root / model_path
        if not model_path_abs.is_dir():
            print(f"Skip (model not found): {model_path}")
            continue
        cmd = [
            deepspeed_exe,
            "--master_port", str(port),
            str(run_script),
            "--cl_method", "ella",
            "--do_predict",
            "--predict_with_generate",
            "--model_name_or_path", str(model_path_abs),
            "--data_dir", "CL_Benchmark",
            "--task_config_dir", task_config_dir,
            "--instruction_file", "configs/instruction_config.json",
            "--instruction_strategy", "single",
            "--output_dir", output_dir,
            "--per_device_eval_batch_size", "128",
            "--deepspeed", str(ds_config),
            "--run_name", run_name,
            "--max_source_length", "512",
            "--max_target_length", "50",
            "--generation_max_length", "50",
            "--add_task_name", "True",
            "--add_dataset_name", "True",
            "--overwrite_output_dir",
            "--overwrite_cache",
            "--seed", str(seed),
            "--report_to", "none",
        ]
        print(f"\n[{i+1}/{len(PRED_JOBS)}] {task_config_dir} -> {output_dir}")
        print(" ".join(cmd[:12]), "...")
        r = subprocess.run(cmd, env=env, cwd=root)
        if r.returncode != 0:
            print(f"Failed with exit code {r.returncode}", file=sys.stderr)
            return r.returncode

    print("\nAll predictions done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
