#!/bin/bash
# Run prediction only (no training) for a trained adapter.
# Usage (from project root, e.g. CL_project):
#   bash scripts_t5/ella/predict_only.sh
# Or set env vars to override:
#   MODEL_PATH=path/to/adapter TASK_CONFIG=configs/short/order1_configs/dbpedia OUTPUT_DIR=out/predict bash scripts_t5/ella/predict_only.sh
set -x

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
VENV_BIN="$PROJECT_ROOT/venv_olora/bin"
if [ -x "$VENV_BIN/deepspeed" ]; then
  export PATH="$VENV_BIN:$PATH"
fi

export CUDA_DEVICE_ORDER="PCI_BUS_ID"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-/root/.cache/huggingface}"
export SEED="${SEED:-73}"

port=$(shuf -i25000-30000 -n1)

# Configurable via env (defaults: one task for order 1)
MODEL_PATH="${MODEL_PATH:-logs_and_outputs/ella/long_1000train/order_4/outputs1/15-yahoo/adapter}"
TASK_CONFIG="${TASK_CONFIG:-configs/long/order4_configs/yahoo}"
OUTPUT_DIR="${OUTPUT_DIR:-logs_and_outputs/ella/long_1000train/order_4/outputs1/}"
RUN_NAME="${RUN_NAME:-predict_only}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" deepspeed --master_port "$port" src/run_uie_lora.py \
   --cl_method ella \
   --do_predict \
   --predict_with_generate \
   --model_name_or_path "$MODEL_PATH" \
   --data_dir CL_Benchmark \
   --task_config_dir "$TASK_CONFIG" \
   --instruction_file configs/instruction_config.json \
   --instruction_strategy single \
   --output_dir "$OUTPUT_DIR" \
   --per_device_eval_batch_size 128 \
   --deepspeed configs/ds_configs/stage2.config \
   --run_name "$RUN_NAME" \
   --max_source_length 512 \
   --max_target_length 50 \
   --generation_max_length 50 \
   --add_task_name True \
   --add_dataset_name True \
   --overwrite_output_dir \
   --overwrite_cache \
   --seed "$SEED" \
   --report_to none
