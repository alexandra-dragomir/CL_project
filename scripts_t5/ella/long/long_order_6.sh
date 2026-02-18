#!/bin/bash
set -x

export CUDA_DEVICE_ORDER="PCI_BUS_ID"
export TRANSFORMERS_CACHE=/root/.cache/huggingface

#############################################
# Configuration - Edit these values
#############################################
ORDER=1                          # Task order number (1, 2, 3, ...)
RUN_NUMBER=1                     # Run number (1, 2, 3, ...)
SEED=73                          # Random seed
BASE_MODEL="initial_model/t5-large"
OUTPUT_BASE=f"logs_and_outputs/long_olora/order_{ORDER}/outputs{RUN_NUMBER}"
CL_METHOD="olora"                # 'olora' or 'ella'

COMMON_PARAMS="
   --max_source_length 512
   --max_target_length 50
   --generation_max_length 50
   --add_task_name True
   --add_dataset_name True
   --overwrite_output_dir
   --overwrite_cache
   --lr_scheduler_type constant
   --warmup_steps 0
   --logging_strategy steps
   --logging_steps 10
   --eval_strategy no
   --save_strategy no
   --save_steps 1500
   --cl_method ${CL_METHOD}
   --seed ${SEED}
   --num_train_epochs 1
"

port=$(shuf -i25000-30000 -n1)

# Task sequence for Order 6 (Long Sequence Benchmark):
# yelp -> amazon -> MNLI -> CB -> COPA -> QQP -> RTE -> IMDB -> SST-2 -> dbpedia -> agnews -> yahoo -> MultiRC -> BoolQA -> WiC

#############################################
# Task 1: yelp (first task, lamda_1=0)
#############################################
CUDA_VISIBLE_DEVICES=0 deepspeed --master_port $port src/run_uie_lora.py \
   ${COMMON_PARAMS} \
   --model_name_or_path ${BASE_MODEL} \
   --task_config_dir configs/long_configs/yelp \
   --output_dir ${OUTPUT_BASE}/1-yelp \
   --run_name long_round1 \
   --lamda_1 0 \
   --lamda_2 0

sleep 5

#############################################
# Task 2: amazon
#############################################
CUDA_VISIBLE_DEVICES=0 deepspeed --master_port $port src/run_uie_lora.py \
   ${COMMON_PARAMS} \
   --model_name_or_path ${OUTPUT_BASE}/1-yelp/adapter \
   --task_config_dir configs/long_configs/amazon \
   --output_dir ${OUTPUT_BASE}/2-amazon \
   --run_name long_round2 \
   --num_train_epochs 1 \
   --lamda_1 0.5 \
   --lamda_2 0

sleep 5

#############################################
# Task 3: MNLI
#############################################
CUDA_VISIBLE_DEVICES=0 deepspeed --master_port $port src/run_uie_lora.py \
   ${COMMON_PARAMS} \
   --model_name_or_path ${OUTPUT_BASE}/2-amazon/adapter \
   --task_config_dir configs/long_configs/MNLI \
   --output_dir ${OUTPUT_BASE}/3-MNLI \
   --run_name long_round3 \
   --num_train_epochs 2 \
   --lamda_1 0.5 \
   --lamda_2 0

sleep 5

#############################################
# Task 4: CB
#############################################
CUDA_VISIBLE_DEVICES=0 deepspeed --master_port $port src/run_uie_lora.py \
   ${COMMON_PARAMS} \
   --model_name_or_path ${OUTPUT_BASE}/3-MNLI/adapter \
   --task_config_dir configs/long_configs/CB \
   --output_dir ${OUTPUT_BASE}/4-CB \
   --run_name long_round4 \
   --num_train_epochs 1 \
   --lamda_1 0.5 \
   --lamda_2 0

sleep 5

#############################################
# Task 5: COPA
#############################################
CUDA_VISIBLE_DEVICES=0 deepspeed --master_port $port src/run_uie_lora.py \
   ${COMMON_PARAMS} \
   --model_name_or_path ${OUTPUT_BASE}/4-CB/adapter \
   --task_config_dir configs/long_configs/COPA \
   --output_dir ${OUTPUT_BASE}/5-COPA \
   --run_name long_round5 \
   --num_train_epochs 1 \
   --lamda_1 0.5 \
   --lamda_2 0.1

sleep 5

#############################################
# Task 6: QQP
#############################################
CUDA_VISIBLE_DEVICES=0 deepspeed --master_port $port src/run_uie_lora.py \
   ${COMMON_PARAMS} \
   --model_name_or_path ${OUTPUT_BASE}/5-COPA/adapter \
   --task_config_dir configs/long_configs/QQP \
   --output_dir ${OUTPUT_BASE}/6-QQP \
   --run_name long_round6 \
   --num_train_epochs 1 \
   --lamda_1 0.5 \
   --lamda_2 0.1

sleep 5

#############################################
# Task 7: RTE
#############################################
CUDA_VISIBLE_DEVICES=0 deepspeed --master_port $port src/run_uie_lora.py \
   ${COMMON_PARAMS} \
   --model_name_or_path ${OUTPUT_BASE}/6-QQP/adapter \
   --task_config_dir configs/long_configs/RTE \
   --output_dir ${OUTPUT_BASE}/7-RTE \
   --run_name long_round7 \
   --num_train_epochs 1 \
   --lamda_1 0.5 \
   --lamda_2 0.3

sleep 5

#############################################
# Task 8: IMDB
#############################################
CUDA_VISIBLE_DEVICES=0 deepspeed --master_port $port src/run_uie_lora.py \
   ${COMMON_PARAMS} \
   --model_name_or_path ${OUTPUT_BASE}/7-RTE/adapter \
   --task_config_dir configs/long_configs/IMDB \
   --output_dir ${OUTPUT_BASE}/8-IMDB \
   --run_name long_round8 \
   --num_train_epochs 1 \
   --lamda_1 0.5 \
   --lamda_2 0

sleep 5

#############################################
# Task 9: SST-2
#############################################
CUDA_VISIBLE_DEVICES=0 deepspeed --master_port $port src/run_uie_lora.py \
   ${COMMON_PARAMS} \
   --model_name_or_path ${OUTPUT_BASE}/8-IMDB/adapter \
   --task_config_dir configs/long_configs/SST-2 \
   --output_dir ${OUTPUT_BASE}/9-SST-2 \
   --run_name long_round9 \
   --num_train_epochs 1 \
   --lamda_1 0.5 \
   --lamda_2 0.1

sleep 5

#############################################
# Task 10: dbpedia
#############################################
CUDA_VISIBLE_DEVICES=0 deepspeed --master_port $port src/run_uie_lora.py \
   ${COMMON_PARAMS} \
   --model_name_or_path ${OUTPUT_BASE}/9-SST-2/adapter \
   --task_config_dir configs/long_configs/dbpedia \
   --output_dir ${OUTPUT_BASE}/10-dbpedia \
   --run_name long_round10 \
   --num_train_epochs 1 \
   --lamda_1 5 \
   --lamda_2 0

sleep 5

#############################################
# Task 11: agnews
#############################################
CUDA_VISIBLE_DEVICES=0 deepspeed --master_port $port src/run_uie_lora.py \
   ${COMMON_PARAMS} \
   --model_name_or_path ${OUTPUT_BASE}/10-dbpedia/adapter \
   --task_config_dir configs/long_configs/agnews \
   --output_dir ${OUTPUT_BASE}/11-agnews \
   --run_name long_round11 \
   --num_train_epochs 1 \
   --lamda_1 5 \
   --lamda_2 0

sleep 5

#############################################
# Task 12: yahoo
#############################################
CUDA_VISIBLE_DEVICES=0 deepspeed --master_port $port src/run_uie_lora.py \
   ${COMMON_PARAMS} \
   --model_name_or_path ${OUTPUT_BASE}/11-agnews/adapter \
   --task_config_dir configs/long_configs/yahoo \
   --output_dir ${OUTPUT_BASE}/12-yahoo \
   --run_name long_round12 \
   --num_train_epochs 1 \
   --lamda_1 5 \
   --lamda_2 0.1

sleep 5

#############################################
# Task 13: MultiRC
#############################################
CUDA_VISIBLE_DEVICES=0 deepspeed --master_port $port src/run_uie_lora.py \
   ${COMMON_PARAMS} \
   --model_name_or_path ${OUTPUT_BASE}/12-yahoo/adapter \
   --task_config_dir configs/long_configs/MultiRC \
   --output_dir ${OUTPUT_BASE}/13-MultiRC \
   --run_name long_round13 \
   --num_train_epochs 1 \
   --lamda_1 5 \
   --lamda_2 0

sleep 5

#############################################
# Task 14: BoolQA
#############################################
CUDA_VISIBLE_DEVICES=0 deepspeed --master_port $port src/run_uie_lora.py \
   ${COMMON_PARAMS} \
   --model_name_or_path ${OUTPUT_BASE}/13-MultiRC/adapter \
   --task_config_dir configs/long_configs/BoolQA \
   --output_dir ${OUTPUT_BASE}/14-BoolQA \
   --run_name long_round14 \
   --num_train_epochs 1 \
   --lamda_1 5 \
   --lamda_2 0.1

sleep 5

#############################################
# Task 15: WiC (final task)
#############################################
CUDA_VISIBLE_DEVICES=0 deepspeed --master_port $port src/run_uie_lora.py \
   ${COMMON_PARAMS} \
   --model_name_or_path ${OUTPUT_BASE}/14-BoolQA/adapter \
   --task_config_dir configs/long_configs/WiC \
   --output_dir ${OUTPUT_BASE}/15-WiC \
   --run_name long_round15 \
   --num_train_epochs 1 \
   --lamda_1 5 \
   --lamda_2 0.3
