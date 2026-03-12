#!/usr/bin/env python3
"""
Run continual-learning configs with per-task lambda_1 grid search.

For each task in a sequence:
1. Train candidate models with lambda_1 in {default, default*10, default/10}
   (or just {0} when default lambda_1 is 0).
2. Evaluate each candidate on the aggregate validation set of all previously
   seen tasks using dev.json (implemented by mirroring dev.json as test.json in
   a temporary data directory).
3. Pick the candidate with the best aggregate exact match.
4. Use that candidate's adapter as the base model for the next task.

The script is structured so short-sequence support can be enabled later via the
same `sequence_type` config field.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional


ROOT = Path(__file__).resolve().parent.parent.parent


# Long sequence defaults (same values as VM2 / run_long_sequence)
LONG_TASK_ORDERS = {
    4: ["MNLI", "CB", "WiC", "COPA", "QQP", "BoolQA", "RTE", "IMDB", "yelp", "amazon", "SST-2", "dbpedia", "agnews", "MultiRC", "yahoo"],
    5: ["MultiRC", "BoolQA", "WiC", "MNLI", "CB", "COPA", "QQP", "RTE", "IMDB", "SST-2", "dbpedia", "agnews", "yelp", "amazon", "yahoo"],
    6: ["yelp", "amazon", "MNLI", "CB", "COPA", "QQP", "RTE", "IMDB", "SST-2", "dbpedia", "agnews", "yahoo", "MultiRC", "BoolQA", "WiC"],
}

LONG_LAMBDA_CONFIGS_ELLA = {
    4: {"lamda_1": [0, 5e5, 5e5, 5e5, 5e5, 5e5, 5e5, 5e5, 5e5, 5e5, 5e5, 5e5, 5e5, 5e5, 5e7],
        "lamda_2": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]},
    5: {"lamda_1": [0, 5e6, 5e6, 5e6, 5e6, 5e6, 5e6, 5e6, 5e6, 5e6, 5e6, 5e6, 5e7, 5e7, 5e7],
        "lamda_2": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]},
    6: {"lamda_1": [0, 5e5, 5e5, 5e5, 5e5, 5e5, 5e5, 5e5, 5e5, 5e5, 5e5, 5e5, 5e5, 5e5, 5e5],
        "lamda_2": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]},
}

LONG_LAMBDA_CONFIGS_OLORA = {
    4: {"lamda_1": [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 5, 5, 5, 5],
        "lamda_2": [0, 0, 0.1, 0, 0, 0, 0.3, 0.1, 0.05, 0, 0.1, 0.1, 0.1, 0, 0.1]},
    5: {"lamda_1": [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5],
        "lamda_2": [0, 0.1, 0, 0.1, 0.1, 0, 0.1, 0.3, 0.1, 0.5, 0, 0.1, 0, 0.1, 0.1]},
    6: {"lamda_1": [0.5, 0.5, 0.02, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
        "lamda_2": [0, 0, 0, 0.1, 0, 0, 0.3, 0, 0.1, 0.1, 0, 0.1, 0, 0.1, 0.3]},
}

SHORT_TASK_ORDERS = {
    1: ["dbpedia", "amazon", "yahoo", "agnews"],
    2: ["dbpedia", "amazon", "agnews", "yahoo"],
    3: ["yahoo", "amazon", "agnews", "dbpedia"],
}

SHORT_LAMBDA_CONFIGS_ELLA = {
    1: {"lamda_1": [0, 3e4, 3e4, 3e4], "lamda_2": [0, 0, 0, 0]},
    2: {"lamda_1": [0, 3e4, 3e4, 3e4], "lamda_2": [0, 0, 0, 0]},
    3: {"lamda_1": [0, 3e4, 3e4, 3e4], "lamda_2": [0, 0, 0, 0]},
}

SHORT_LAMBDA_CONFIGS_OLORA = {
    1: {"lamda_1": [0.5, 0.5, 0.5, 0.5], "lamda_2": [0, 0, 0, 0]},
    2: {"lamda_1": [0.5, 0.5, 0.5, 0.5], "lamda_2": [0, 0, 0, 0]},
    3: {"lamda_1": [0.5, 0.5, 0.5, 0.5], "lamda_2": [0, 0, 0, 0]},
}

EPOCHS_CONFIG_LONG = {4: [1] * 15, 5: [1] * 15, 6: [1] * 15}
EPOCHS_CONFIG_SHORT = {1: [1] * 4, 2: [1] * 4, 3: [1] * 4}


def log_print(*args, **kwargs) -> None:
    print(*args, **kwargs)
    sys.stdout.flush()


def _lamda_list_to_str(values: List[float]) -> str:
    return ", ".join(str(x) for x in values)


def _parse_lamda_list(value: Any, expected_len: int, name: str) -> Optional[List[float]]:
    if value is None:
        return None
    if isinstance(value, list):
        vals = [float(x) for x in value]
    elif isinstance(value, str):
        vals = [float(x.strip()) for x in value.split(",")]
    else:
        raise ValueError(f"{name} must be a list, string, or None")
    if len(vals) != expected_len:
        raise ValueError(f"{name} must have {expected_len} values, got {len(vals)}")
    return vals


def _format_lambda_label(value: float) -> str:
    if value == 0:
        return "0"
    if abs(value) >= 1e3 or abs(value) < 1e-2:
        s = f"{value:.0e}"
        base, exp = s.split("e")
        exp = str(int(exp))
        return f"{base}e{exp}"
    return f"{value:g}"


def _candidate_lambdas(default_value: float) -> List[float]:
    if default_value == 0:
        return [0.0]
    values = [default_value, default_value * 10.0, default_value / 10.0]
    deduped: List[float] = []
    for v in values:
        if not any(abs(v - existing) <= max(1e-12, abs(v) * 1e-12) for existing in deduped):
            deduped.append(v)
    return deduped


def _sequence_spec(sequence_type: str) -> Dict[str, Any]:
    if sequence_type == "long":
        return {
            "task_orders": LONG_TASK_ORDERS,
            "lambda_ella": LONG_LAMBDA_CONFIGS_ELLA,
            "lambda_olora": LONG_LAMBDA_CONFIGS_OLORA,
            "epochs": EPOCHS_CONFIG_LONG,
            "config_dir_prefix": "configs/long",
            "default_data_dir": "CL_Benchmark_dev",
            "default_train_batch_size": lambda cl_method: 32 if cl_method == "ella" else 8,
            "default_grad_accum": lambda cl_method: 1 if cl_method == "ella" else 8,
        }
    if sequence_type == "short":
        return {
            "task_orders": SHORT_TASK_ORDERS,
            "lambda_ella": SHORT_LAMBDA_CONFIGS_ELLA,
            "lambda_olora": SHORT_LAMBDA_CONFIGS_OLORA,
            "epochs": EPOCHS_CONFIG_SHORT,
            "config_dir_prefix": "configs/short",
            "default_data_dir": "CL_Benchmark_dev",
            "default_train_batch_size": lambda cl_method: 32 if cl_method == "ella" else 64,
            "default_grad_accum": lambda cl_method: 1 if cl_method == "ella" else 8,
        }
    raise ValueError(f"Unsupported sequence_type: {sequence_type}")


def _resolve_config(cfg: Dict[str, Any], defaults: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(defaults)
    for key, value in cfg.items():
        if value is not None:
            out[key] = value

    if "lambda_1" in out:
        out["lamda_1"] = out.pop("lambda_1")
    if "lambda_2" in out:
        out["lamda_2"] = out.pop("lambda_2")

    spec = _sequence_spec(out["sequence_type"])
    order = out["order"]
    lambda_defaults = spec["lambda_ella"][order] if out["cl_method"] == "ella" else spec["lambda_olora"][order]
    task_count = len(spec["task_orders"][order])

    lamda_1 = _parse_lamda_list(out.get("lamda_1"), task_count, "lamda_1")
    lamda_2 = _parse_lamda_list(out.get("lamda_2"), task_count, "lamda_2")
    out["lamda_1_values"] = lamda_1 if lamda_1 is not None else lambda_defaults["lamda_1"]
    out["lamda_2_values"] = lamda_2 if lamda_2 is not None else lambda_defaults["lamda_2"]
    out["tasks"] = spec["task_orders"][order]
    out["epochs"] = spec["epochs"][order]

    if out.get("train_batch_size") is None:
        out["train_batch_size"] = spec["default_train_batch_size"](out["cl_method"])
    if out.get("gradient_accumulation_steps") is None:
        out["gradient_accumulation_steps"] = spec["default_grad_accum"](out["cl_method"])
    if out.get("data_dir") is None:
        out["data_dir"] = spec["default_data_dir"]

    return out


def _create_dev_as_test_mirror(source_root: Path, target_root: Path, dry_run: bool) -> None:
    if dry_run:
        return
    if target_root.exists():
        shutil.rmtree(target_root)
    target_root.mkdir(parents=True, exist_ok=True)

    for dirpath, dirnames, filenames in os.walk(source_root):
        rel_dir = Path(dirpath).relative_to(source_root)
        out_dir = target_root / rel_dir
        out_dir.mkdir(parents=True, exist_ok=True)

        for filename in filenames:
            src_file = Path(dirpath) / filename
            dst_file = out_dir / filename
            if filename == "test.json":
                dev_file = Path(dirpath) / "dev.json"
                link_src = dev_file if dev_file.exists() else src_file
            else:
                link_src = src_file
            os.symlink(link_src.resolve(), dst_file)


def _build_common_params(cfg: Dict[str, Any], data_dir: str, ella_variant: str) -> Dict[str, str]:
    params = {
        "--do_train": "",
        "--do_predict": "",
        "--predict_with_generate": "",
        "--data_dir": data_dir,
        "--instruction_file": "configs/instruction_config.json",
        "--instruction_strategy": "single",
        "--per_device_train_batch_size": str(cfg["train_batch_size"]),
        "--per_device_eval_batch_size": "128",
        "--gradient_accumulation_steps": str(cfg["gradient_accumulation_steps"]),
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
        "--cl_method": cfg["cl_method"],
        "--seed": str(cfg["seed"]),
        "--report_to": "wandb",
        "--log_cl_metrics": "True",
    }
    if cfg["cl_method"] == "ella":
        params["--ella_variant"] = ella_variant
    if cfg.get("lamda_1_list") is not None and isinstance(cfg["lamda_1_list"], list):
        params["--lamda_1_list_json"] = json.dumps(cfg["lamda_1_list"])
    if cfg.get("ella_drop"):
        params["--ella_drop"] = ""
        params["--drop_lowest"] = str(cfg.get("drop_lowest", 0.1))
    return params


def _build_command(
    cfg: Dict[str, Any],
    data_dir: str,
    task_config_dir: str,
    model_path: str,
    output_dir: str,
    run_name: str,
    num_epochs: int,
    lamda_1: float,
    lamda_2: float,
    ella_variant: str,
    port: int,
) -> List[str]:
    params = _build_common_params(cfg, data_dir=data_dir, ella_variant=ella_variant)
    cmd = ["deepspeed", "--master_port", str(port), "src/run_uie_lora.py"]
    for key, value in params.items():
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


def _build_predict_command(
    cfg: Dict[str, Any],
    data_dir: str,
    task_config_dir: str,
    model_path: str,
    output_dir: str,
    run_name: str,
    ella_variant: str,
    port: int,
) -> List[str]:
    params = {
        "--do_predict": "",
        "--predict_with_generate": "",
        "--data_dir": data_dir,
        "--instruction_file": "configs/instruction_config.json",
        "--instruction_strategy": "single",
        "--per_device_eval_batch_size": "128",
        "--deepspeed": "configs/ds_configs/stage2.config",
        "--max_source_length": "512",
        "--max_target_length": "50",
        "--generation_max_length": "50",
        "--add_task_name": "True",
        "--add_dataset_name": "True",
        "--overwrite_output_dir": "",
        "--overwrite_cache": "",
        "--cl_method": cfg["cl_method"],
        "--seed": str(cfg["seed"]),
        "--report_to": "none",
    }
    if cfg["cl_method"] == "ella":
        params["--ella_variant"] = ella_variant
    if cfg.get("ella_drop"):
        params["--ella_drop"] = ""
        params["--drop_lowest"] = str(cfg.get("drop_lowest", 0.1))

    cmd = ["deepspeed", "--master_port", str(port), "src/run_uie_lora.py"]
    for key, value in params.items():
        cmd.append(key)
        if value:
            cmd.append(value)
    cmd.extend([
        "--model_name_or_path", model_path,
        "--task_config_dir", task_config_dir,
        "--output_dir", output_dir,
        "--run_name", run_name,
    ])
    return cmd


def _run_command(cmd: List[str], gpu_id: int, log_file: Path, wandb_project: str, dry_run: bool) -> bool:
    log_print(f"Command: {' '.join(cmd)}")
    if dry_run:
        return True

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    env["TRANSFORMERS_CACHE"] = os.path.expanduser("~/.cache/huggingface")
    env["WANDB_PROJECT"] = wandb_project

    with open(log_file, "a") as f:
        f.write(f"Command: {' '.join(cmd)}\n")
        f.flush()
        result = subprocess.run(cmd, cwd=ROOT, env=env, stdout=f, stderr=subprocess.STDOUT)
    return result.returncode == 0


def _load_metrics(metrics_path: Path) -> Dict[str, Any]:
    with open(metrics_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _copy_adapter(candidate_output_dir: Path, destination: Path, dry_run: bool) -> None:
    if dry_run:
        return
    source = candidate_output_dir / "adapter"
    if destination.exists():
        shutil.rmtree(destination)
    shutil.copytree(source, destination)


def _prepare_run_dirs(cfg: Dict[str, Any]) -> Dict[str, Path]:
    output_root = cfg.get("output_root")
    if output_root:
        base_root = ROOT / output_root
    else:
        base_root = ROOT / "logs_and_outputs" / cfg["cl_method"]
    run_root = base_root / cfg["run_name"] / f"order_{cfg['order']}"
    outputs_root = run_root / f"outputs{cfg['run_number']}"
    logs_root = run_root / "logs"
    dev_as_test_root = run_root / "_dev_as_test_data"
    logs_root.mkdir(parents=True, exist_ok=True)
    outputs_root.mkdir(parents=True, exist_ok=True)
    return {
        "run_root": run_root,
        "outputs_root": outputs_root,
        "logs_root": logs_root,
        "log_file": logs_root / f"train_and_select{cfg['run_number']}.log",
        "dev_as_test_root": dev_as_test_root,
    }


def _task_config_dir(sequence_type: str, order: int, task: str) -> Path:
    return ROOT / "configs" / sequence_type / f"order{order}_configs" / task


def _candidate_ella_variant(base_variant: str, task_index: int) -> str:
    if base_variant == "ella_first_base_w":
        return "ella_with_base_w" if task_index == 0 else "ella"
    return base_variant


def _write_json(path: Path, payload: Dict[str, Any], dry_run: bool) -> None:
    if dry_run:
        return
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def run_grid_search_config(cfg: Dict[str, Any], dry_run: bool, gpu_id: int) -> bool:
    dirs = _prepare_run_dirs(cfg)
    source_data_dir = (ROOT / cfg["data_dir"]).resolve()
    _create_dev_as_test_mirror(source_data_dir, dirs["dev_as_test_root"], dry_run=dry_run)

    log_print(f"\n{'=' * 70}")
    log_print(f"Grid search run: {cfg['run_name']} order={cfg['order']} run={cfg['run_number']} seed={cfg['seed']}")
    log_print(f"Sequence type: {cfg['sequence_type']}")
    log_print(f"CL method: {cfg['cl_method']}")
    log_print(f"Tasks: {' -> '.join(cfg['tasks'])}")
    log_print(f"Lambda_1 defaults: {cfg['lamda_1_values']}")
    log_print(f"Lambda_2 defaults: {cfg['lamda_2_values']}")
    log_print(f"{'=' * 70}\n")

    if not dry_run:
        with open(dirs["log_file"], "w", encoding="utf-8") as f:
            f.write(f"Grid search run: {cfg['run_name']}\n")
            f.write(f"order={cfg['order']} run={cfg['run_number']} seed={cfg['seed']}\n")
            f.write(f"tasks={' -> '.join(cfg['tasks'])}\n")
            f.write(f"lamda_1_defaults={cfg['lamda_1_values']}\n")
            f.write(f"lamda_2_defaults={cfg['lamda_2_values']}\n\n")

    base_model_path = cfg["base_model"]
    selection_summary: List[Dict[str, Any]] = []

    for task_index, task in enumerate(cfg["tasks"]):
        task_num = task_index + 1
        task_root = dirs["outputs_root"] / f"{task_num}-{task}"
        task_root.mkdir(parents=True, exist_ok=True)
        source_task_config_dir = _task_config_dir(cfg["sequence_type"], cfg["order"], task)

        default_lamda_1 = float(cfg["lamda_1_values"][task_index])
        lamda_2 = float(cfg["lamda_2_values"][task_index])
        candidates = _candidate_lambdas(default_lamda_1)
        ella_variant = _candidate_ella_variant(cfg["ella_variant"], task_index)

        log_print(f"\n{'-' * 60}")
        log_print(f"Task {task_num}: {task}")
        log_print(f"Base model: {base_model_path}")
        log_print(f"Lambda candidates: {candidates}")
        log_print(f"{'-' * 60}")

        candidate_results: List[Dict[str, Any]] = []
        for lamda_1 in candidates:
            lamda_label = _format_lambda_label(lamda_1)
            candidate_output_dir = task_root / f"lambda_{lamda_label}"
            candidate_output_dir.mkdir(parents=True, exist_ok=True)
            candidate_run_name = f"{cfg['run_name']}_task{task_num}_lambda_{lamda_label}"
            cmd = _build_command(
                cfg=cfg,
                data_dir=str(dirs["dev_as_test_root"]),
                task_config_dir=str(source_task_config_dir),
                model_path=base_model_path,
                output_dir=str(candidate_output_dir),
                run_name=candidate_run_name,
                num_epochs=cfg["epochs"][task_index],
                lamda_1=lamda_1,
                lamda_2=lamda_2,
                ella_variant=ella_variant,
                port=25000 + (cfg["run_number"] % 1000) * 10 + task_num,
            )
            ok = _run_command(
                cmd=cmd,
                gpu_id=gpu_id,
                log_file=dirs["log_file"],
                wandb_project=cfg["wandb_project"],
                dry_run=dry_run,
            )
            if not ok:
                candidate_results.append({
                    "lambda_1": lamda_1,
                    "lambda_label": lamda_label,
                    "success": False,
                    "exact_match": None,
                    "output_dir": str(candidate_output_dir),
                })
                continue

            metrics_path = candidate_output_dir / "all_results.json"
            metrics = _load_metrics(metrics_path) if not dry_run else {"predict_exact_match": 0.0}
            exact_match = metrics.get("predict_exact_match")

            adapter_path = task_root / f"adapter_lambda_{lamda_label}"
            _copy_adapter(candidate_output_dir, adapter_path, dry_run=dry_run)

            result_payload = {
                "task": task,
                "task_num": task_num,
                "lambda_1": lamda_1,
                "lambda_label": lamda_label,
                "exact_match": exact_match,
                "success": True,
                "adapter_path": str(adapter_path),
                "candidate_output_dir": str(candidate_output_dir),
                "metrics": metrics,
            }
            _write_json(task_root / f"lambda_{lamda_label}.json", result_payload, dry_run=dry_run)
            candidate_results.append(result_payload)

        successful = [r for r in candidate_results if r.get("success") and r.get("exact_match") is not None]
        if not successful:
            log_print(f"No successful candidates for task {task_num}: {task}")
            _write_json(task_root / "lambda_search_summary.json", {"task": task, "candidates": candidate_results}, dry_run=dry_run)
            return False

        best = max(successful, key=lambda item: item["exact_match"])
        final_test_run_name = f"{cfg['run_name']}_task{task_num}_best_test"
        test_cmd = _build_predict_command(
            cfg=cfg,
            data_dir=str(source_data_dir),
            task_config_dir=str(source_task_config_dir),
            model_path=best["adapter_path"],
            output_dir=str(task_root),
            run_name=final_test_run_name,
            ella_variant=ella_variant,
            port=26000 + (cfg["run_number"] % 1000) * 10 + task_num,
        )
        ok = _run_command(
            cmd=test_cmd,
            gpu_id=gpu_id,
            log_file=dirs["log_file"],
            wandb_project=cfg["wandb_project"],
            dry_run=dry_run,
        )
        if not ok:
            log_print(f"Final test prediction failed for task {task_num}: {task}")
            return False

        final_test_metrics = _load_metrics(task_root / "all_results.json") if not dry_run else {"predict_exact_match": 0.0}
        best_summary = {
            "task": task,
            "task_num": task_num,
            "best_lambda_1": best["lambda_1"],
            "best_lambda_label": best["lambda_label"],
            "best_exact_match": best["exact_match"],
            "selection_exact_match_dev": best["exact_match"],
            "final_test_exact_match": final_test_metrics.get("predict_exact_match"),
            "final_test_metrics": final_test_metrics,
            "selected_adapter_path": best["adapter_path"],
            "candidates": candidate_results,
        }
        _write_json(task_root / "lambda_search_summary.json", best_summary, dry_run=dry_run)
        selection_summary.append(best_summary)
        base_model_path = best["adapter_path"]
        log_print(
            f"Best for task {task_num} ({task}): lambda_1={best['lambda_1']} "
            f"(label={best['lambda_label']}) dev_exact_match={best['exact_match']} "
            f"test_exact_match={final_test_metrics.get('predict_exact_match')}"
        )

    _write_json(
        dirs["run_root"] / f"grid_search_summary_run{cfg['run_number']}.json",
        {
            "run_name": cfg["run_name"],
            "sequence_type": cfg["sequence_type"],
            "order": cfg["order"],
            "run_number": cfg["run_number"],
            "seed": cfg["seed"],
            "cl_method": cfg["cl_method"],
            "ella_variant": cfg["ella_variant"],
            "tasks": selection_summary,
        },
        dry_run=dry_run,
    )
    return True


def main(configs: List[Dict[str, Any]], dry_run: bool, gpu_id: int) -> int:
    default_config = {
        "sequence_type": "long",
        "order": 5,
        "run_name": "long_16_2_grid",
        "run_number": 1,
        "seed": 42,
        "cl_method": "ella",
        "ella_variant": "ella",
        "base_model": "initial_model/t5-large",
        "output_root": "logs_and_outputs/ella_new_splits",
        "wandb_project": "CL",
        "data_dir": None,
        "lamda_1": None,
        "lamda_2": None,
        "lamda_1_list": None,
        # "ella_drop": False,
        # "drop_lowest": 0.1,
        "train_batch_size": 16,
        "gradient_accumulation_steps": 2,
    }

    if not configs:
        print("No CONFIGS to run.", file=sys.stderr)
        return 1

    success_count = 0
    for cfg in configs:
        resolved = _resolve_config(cfg, default_config)
        ok = run_grid_search_config(resolved, dry_run=dry_run, gpu_id=gpu_id)
        if ok:
            success_count += 1

    log_print(f"\nCompleted {success_count}/{len(configs)} configs")
    return 0 if success_count == len(configs) else 1


if __name__ == "__main__":
    configs = [
        {
            "order": 5,
            "run_name": "long_16_2_grid",
            "run_number": 1,
            "seed": 42,
            "ella_variant": "ella",
        },
        {
            "order": 6,
            "run_name": "long_16_2_grid",
            "run_number": 1,
            "seed": 42,
            "ella_variant": "ella",
        },
    ]
    sys.exit(main(configs, dry_run=False, gpu_id=0))
