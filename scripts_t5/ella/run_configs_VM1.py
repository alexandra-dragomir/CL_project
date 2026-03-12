#!/usr/bin/env python3
"""
Run multiple training configs from a config file or built-in CONFIGS.

Each config is merged with DEFAULT_CONFIG and passed to run_long_sequence or
run_short_sequence. Use --config_file to load CONFIGS from an external Python file.

Usage:
    python scripts_t5/ella/run_configs.py --config_file scripts_t5/ella/example_configs.py
    python scripts_t5/ella/run_configs.py  # uses built-in CONFIGS
"""

import argparse
import importlib.util
import json
import os
import random
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# Order-specific lambdas (used when lamda_1/lamda_2 not in config)
LAMBDA_CONFIGS_LONG = {
    4: {"lamda_1": [0, 5e5, 5e5, 5e5, 5e5, 5e5, 5e5, 5e5, 5e5, 5e5, 5e5, 5e5, 5e5, 5e5, 5e7],
        "lamda_2": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]},
    5: {"lamda_1": [0, 5e6, 5e6, 5e6, 5e6, 5e6, 5e6, 5e6, 5e6, 5e6, 5e6, 5e6, 5e7, 5e7, 5e7],
        "lamda_2": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]},
    6: {"lamda_1": [0, 5e5, 5e5, 5e5, 5e5, 5e5, 5e5, 5e5, 5e5, 5e5, 5e5, 5e5, 5e5, 5e5, 5e5],
        "lamda_2": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]},
}

LAMBDA_CONFIGS_SHORT = {
    1: {"lamda_1": [0, 3e4, 3e4, 3e4], "lamda_2": [0, 0, 0, 0]},
    2: {"lamda_1": [0, 3e4, 3e4, 3e4], "lamda_2": [0, 0, 0, 0]},
    3: {"lamda_1": [0, 3e4, 3e4, 3e4], "lamda_2": [0, 0, 0, 0]},
}


def _lamda_list_to_str(lst: List[float]) -> str:
    return ", ".join(str(x) for x in lst)


def _resolve_config(cfg: Dict[str, Any], defaults: Dict[str, Any]) -> Dict[str, Any]:
    """Merge cfg into defaults. Handle lamda_1/lamda_2 from config or LAMBDA_CONFIGS."""
    out = dict(defaults)
    for k, v in cfg.items():
        if v is not None:
            out[k] = v

    # Resolve lamda_1/lamda_2 (support lambda_1/lambda_2 as aliases; prefer user value when given)
    if "lambda_1" in out:
        out["lamda_1"] = out.pop("lambda_1", None)
    if "lambda_2" in out:
        out["lamda_2"] = out.pop("lambda_2", None)

    seq = out.get("sequence_type", "long")
    order = out.get("order", 5 if seq == "long" else 1)
    lambdas = LAMBDA_CONFIGS_LONG.get(order, LAMBDA_CONFIGS_LONG[4]) if seq == "long" else LAMBDA_CONFIGS_SHORT.get(order, LAMBDA_CONFIGS_SHORT[1])

    if out["lamda_1"] is None:
        out["lamda_1"] = _lamda_list_to_str(lambdas["lamda_1"])
    elif isinstance(out["lamda_1"], list):
        out["lamda_1"] = _lamda_list_to_str(out["lamda_1"])

    if out["lamda_2"] is None:
        out["lamda_2"] = _lamda_list_to_str(lambdas["lamda_2"])
    elif isinstance(out["lamda_2"], list):
        out["lamda_2"] = _lamda_list_to_str(out["lamda_2"])

    return out


def run_config(
    cfg: Dict[str, Any],
    root: Path,
    dry_run: bool = False,
    gpu_id: int = 0,
) -> bool:
    """Run a single config via subprocess."""
    seq = cfg.get("sequence_type", "long")
    script = "run_long_sequence.py" if seq == "long" else "run_short_sequence.py"
    script_path = root / "scripts_t5" / "ella" / script

    cmd = [
        "python", str(script_path),
        "--order", str(cfg["order"]),
        "--run_number", str(cfg["run_number"]),
        "--seed", str(cfg["seed"]),
        "--run_name", str(cfg["run_name"]),
        "--gpu_id", str(gpu_id),
        "--lamda_1", cfg["lamda_1"],
        "--lamda_2", cfg["lamda_2"],
    ]
    if seq == "long":
        cmd.extend(["--cl_method", cfg.get("cl_method", "ella")])
        cmd.extend(["--ella_variant", cfg.get("ella_variant", "ella")])
        cmd.extend(["--base_model", cfg.get("base_model", "initial_model/t5-large")])
        # lamda_1_list as list of lists (incremental ELLA per-task partition lambdas)
        if cfg.get("lamda_1_list") is not None and isinstance(cfg["lamda_1_list"], list) and cfg["lamda_1_list"] and isinstance(cfg["lamda_1_list"][0], (list, tuple)):
            cmd.extend(["--lamda_1_list_json", json.dumps(cfg["lamda_1_list"])])
        if cfg.get("ella_drop"):
            cmd.append("--ella_drop")
            cmd.extend(["--drop_lowest", str(cfg.get("drop_lowest", 0.1))])
    else:
        cmd.extend(["--ella_variant", cfg.get("ella_variant", "ella")])
    if cfg.get("wandb_project"):
        cmd.extend(["--wandb_project", cfg["wandb_project"]])
    if cfg.get("output_root"):
        cmd.extend(["--output_root", cfg["output_root"]])
    if cfg.get("train_batch_size") is not None:
        cmd.extend(["--train_batch_size", str(cfg["train_batch_size"])])
    if cfg.get("gradient_accumulation_steps") is not None:
        cmd.extend(["--gradient_accumulation_steps", str(cfg["gradient_accumulation_steps"])])

    print(f"\n{'='*60}")
    print(f"Running: {cfg.get('run_name', 'config')} order={cfg['order']} run={cfg['run_number']} seed={cfg['seed']}")
    print(f"{' '.join(cmd[:8])} ...")
    if dry_run:
        print("[DRY RUN]")
        return True

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    r = subprocess.run(cmd, cwd=root, env=env)
    return r.returncode == 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Run multiple configs from file or built-in")
    parser.add_argument("--config_file", type=str, default=None,
                        help="Python file with CONFIGS and optional DEFAULT_CONFIG")
    parser.add_argument("--dry_run", action="store_true", help="Print commands without running")
    parser.add_argument("--gpu_id", type=int, default=0)
    args = parser.parse_args()

    root = Path(__file__).resolve().parent.parent.parent

    DEFAULT_CONFIG = {
        "sequence_type": "long",
        # "order": 5,
        # "run_name": "long_ella_first_base_W",
        "run_number": 1,
        "seed": 42,
        # "ella_variant": "ella_first_base_w",
        "cl_method": "ella",
        "base_model": "initial_model/t5-large",
        "gpu_id": 0,
        "wandb_project": "CL",
        "output_root": "logs_and_outputs/ella_new_splits",  # e.g. "logs_and_outputs/ella_new_splits"; default uses logs_and_outputs/{cl_method}
        "lamda_1": None,
        "lamda_2": None,
        "lamda_1_list": None,  # optional: list of lists for incremental ELLA, one per task
        "train_batch_size": None,  # per-device batch size; if set, passed to run_long/short_sequence
        "gradient_accumulation_steps": None,  # if set, passed to run_long/short_sequence
    }

    if args.config_file:
        path = root / args.config_file
        if not path.is_file():
            print(f"Error: config file not found: {path}", file=sys.stderr)
            return 1
        spec = importlib.util.spec_from_file_location("config_module", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        DEFAULT_CONFIG = {**DEFAULT_CONFIG, **getattr(mod, "DEFAULT_CONFIG", {})}
        CONFIGS = getattr(mod, "CONFIGS", [])
    else:
       CONFIGS = [
            {
                "order": 4,
                "run_name": "long_16_2", 
                "run_number": 1, "seed": 42,
                "ella_variant": "ella",
                "train_batch_size": 16,
                "gradient_accumulation_steps": 2,
                
            },
            # {
            #     "order": 4,
            #     "run_name": "long_16_2_drop_energies_0.4", 
            #     "run_number": 42, "seed": 42,
            #     "ella_drop": True,
            #     "drop_lowest": 0.4, 
            #     "ella_variant": "ella",
            #     "train_batch_size": 16,
            #     "gradient_accumulation_steps": 2,
            # },
            # {  # run again this <-
            #     "order": 4,
            #     "run_name": "long_16_2_drop_energies_0.9", 
            #     "run_number": 42, "seed": 42,
            #     "ella_drop": True,
            #     "drop_lowest": 0.9, 
            #     "ella_variant": "ella",
            #     "train_batch_size": 16,
            #     "gradient_accumulation_steps": 2,
            # },
            # {
            #     "order": 4,
            #     "run_name": "long_16_2_test", 
            #     "run_number": 42, "seed": 42,
            #     "ella_variant": "ella",
            #     "train_batch_size": 16,
            #     "gradient_accumulation_steps": 2,
            # },
            
        ]

    if not CONFIGS:
        print("No CONFIGS to run.", file=sys.stderr)
        return 1

    success_count = 0
    for i, c in enumerate(CONFIGS):
        resolved = _resolve_config(c, dict(DEFAULT_CONFIG))
        ok = run_config(resolved, root, dry_run=args.dry_run, gpu_id=args.gpu_id)
        if ok:
            success_count += 1

    print(f"\nCompleted {success_count}/{len(CONFIGS)} configs")
    return 0 if success_count == len(CONFIGS) else 1


if __name__ == "__main__":
    sys.exit(main())
