"""
Example config file for run_configs.py.

Use: python scripts_t5/ella/run_configs.py --config_file scripts_t5/ella/example_configs.py

Override DEFAULT_CONFIG to change base values for all configs.
CONFIGS: list of dicts; each overrides DEFAULT_CONFIG. Only specify what differs.
"""

# Optional: override defaults for all configs
DEFAULT_CONFIG = {
    "sequence_type": "long",
    "order": 5,
    "run_name": "long_ella_first_base_W",
    "run_number": 1,
    "seed": 73,
    "ella_variant": "ella_first_base_w",
    "cl_method": "ella",
    "base_model": "initial_model/t5-large",
    "gpu_id": 0,
    "wandb_project": "CL",
}

# Configs to run. Each is merged with DEFAULT_CONFIG.
# Between seeds: keep same order, run_name, etc.; only seed and run_number change.
CONFIGS = [
    # 3 seeds for order 5
    {"order": 4, "run_number": 1, "seed": 1, "lambda_1": "5e5, 5e5, 5e5, 5e5, 5e5, 5e5, 5e5, 5e5, 5e5, 5e5, 5e5, 5e5, 5e5, 5e5, 5e7"},
    {"order": 5, "run_number": 1, "seed": 1, "lambda_1": "5e5, 5e5, 5e5, 5e5, 5e5, 5e5, 5e5, 5e5, 5e5, 5e5, 5e5, 5e5, 5e5, 5e5, 5e7"},
    {"order": 6, "run_number": 1, "seed": 1, "lambda_1": "5e5, 5e5, 5e5, 5e5, 5e5, 5e5, 5e5, 5e5, 5e5, 5e5, 5e5, 5e5, 5e5, 5e5, 5e7"},
    
    
    # Short sequence
    # {"sequence_type": "short", "order": 1, "run_name": "exp_short", "run_number": 1, "seed": 73},
]
