"""
Example config for plot_bwt.py (line chart: per-task BWT, one line per config).

Usage:
  python evaluation/plot_bwt.py --config evaluation/bwt_plot_config_example.py

All configs use the same ORDER. Each config has base_dir, label, run_ids.
Multiple runs per config are averaged for the line.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Single order for all configs (use order from first config if set)
ORDER = 6

# Configs to compare: each gets one line. run_ids are averaged.
CONFIGS = [
    {
        "base_dir": ROOT / "logs_and_outputs/ella/long_32_batch_train_1000_test_full",
        "label": "ELLA",
        "order": 6,
        "run_ids": [1],
    },
    # {
    #     "base_dir": ROOT / "logs_and_outputs/ella/long_32_batch_train_1000_test_full",
    #     "label": "ELLA",
    #     "order": 5,
    #     "run_ids": [1],
    # },
    # {
    #     "base_dir": ROOT / "logs_and_outputs/ella/long_32_batch_train_1000_test_full",
    #     "label": "ELLA",
    #     "order": 6,
    #     "run_ids": [1],
    # },
    # {
    #     "base_dir": ROOT / "logs_and_outputs/ella/order_5_ablations_lambda_first_base_W/long_ella_first_base_W_lamda_1_5e6",
    #     "label": "ELLA_lamda_1_5e6",
    #     "order": 5,
    #     "run_ids": [1],
    # },
    # {
    #     "base_dir": ROOT / "logs_and_outputs/ella/order_5_ablations_lambda_first_base_W/long_ella_first_base_W_lamda_1_5e5",
    #     "label": "ELLA_lamda_1_5e5",
    #     "order": 5,
    #     "run_ids": [1],
    # },
    # {
    #     "base_dir": ROOT / "logs_and_outputs/ella/order_5_ablations_lambda_first_base_W/long_ella_first_base_W_lamda_1_5e4",
    #     "label": "ELLA_lamda_1_5e4",
    #     "order": 5,
    #     "run_ids": [1],
    # },
    # {
    #     "base_dir": ROOT / "logs_and_outputs/ella/order_5_ablations_lambda_first_base_W/long_ella_first_base_W_lamda_1_5e3",
    #     "label": "ELLA_lamda_1_5e3",
    #     "order": 5,
    #     "run_ids": [1],
    # },
    # {
    #     "base_dir": ROOT / "logs_and_outputs/ella/order_5_ablations_lambda_first_base_W/long_ella_first_base_W_lamda_1_5e2",
    #     "label": "ELLA_lamda_1_5e2",
    #     "order": 5,
    #     "run_ids": [1],
    # },
    # {
    #     "base_dir": ROOT / "logs_and_outputs/ella/order_5_ablations_lambda_first_base_W/long_ella_first_base_W_lamda_1_5e1",
    #     "label": "ELLA_lamda_1_5e1",
    #     "order": 5,
    #     "run_ids": [1],
    # },
    # {
    #     "base_dir": ROOT / "logs_and_outputs/ella/order_5_ablations_lambda_first_base_W/long_ella_first_base_W_lamda_1_0",
    #     "label": "ELLA_lamda_1_0",
    #     "order": 5,
    #     "run_ids": [1],
    # },
    # Add more configs, e.g. different methods or lambda values:
    # {"base_dir": ROOT / "logs_and_outputs/olora/short", "label": "O-LoRA", "run_ids": [1, 2, 3]},
]

OUTPUT = "evaluation/results/bwt_plot_long_o6.png"
TITLE = "Per-task Backward Transfer (BWT)"
