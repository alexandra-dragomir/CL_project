#!/bin/bash
set -x

export CUDA_DEVICE_ORDER="PCI_BUS_ID"
export TRANSFORMERS_CACHE=/root/.cache/huggingface

#############################################
# Configuration - Edit these values
#############################################
ORDER=5                          # Task order number (1, 2, 3, ...)
RUN_NUMBER=1                     # Run number (1, 2, 3, ...)
SEED=73                          # Random seed
BASE_MODEL="initial_model/t5-large"
OUTPUT_BASE="logs_and_outputs/ella/long/order_${ORDER}/outputs${RUN_NUMBER}"
CL_METHOD="ella"                 # 'olora' or 'ella'

# Lambda values for each task (bash arrays use parentheses, no spaces around =)
LAMBDAS1=(0 5e6 5e6 5e6 5e6 5e6 5e6 5e6 5e6 5e6 5e6 5e6 5e7 5e7 5e7)
LAMBDAS2=(0 0 0 0 0 0 0 0 0 0 0 0 0 0 0)

# Task sequence for Order 5 (Long Sequence Benchmark):
# multirc -> boolqa -> wic -> mnli -> cb -> copa -> qqp -> rte -> imdb -> sst-2 -> dbpedia -> agnews -> yelp -> amazon -> yahoo
TASK_ORDER=(MultiRC BoolQA WiC MNLI CB COPA QQP RTE IMDB SST-2 dbpedia agnews yelp amazon yahoo)

#############################################
# Common parameters
#############################################
COMMON_PARAMS="
   --do_train
   --do_predict
   --predict_with_generate
   --data_dir CL_Benchmark
   --instruction_file configs/instruction_config.json
   --instruction_strategy single
   --per_device_train_batch_size 8
   --per_device_eval_batch_size 128
   --gradient_accumulation_steps 8
   --learning_rate 1e-03
   --deepspeed configs/ds_configs/stage2.config
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

#############################################
# Task 1
#############################################
CUDA_VISIBLE_DEVICES=0 deepspeed --master_port $port src/run_uie_lora.py \
   ${COMMON_PARAMS} \
   --model_name_or_path ${BASE_MODEL} \
   --task_config_dir configs/long_configs/${TASK_ORDER[0]} \
   --output_dir ${OUTPUT_BASE}/1-${TASK_ORDER[0]} \
   --run_name long_round1 \
   --lamda_1 ${LAMBDAS1[0]} \
   --lamda_2 ${LAMBDAS2[0]}

sleep 5

#############################################
# Task 2
#############################################
CUDA_VISIBLE_DEVICES=0 deepspeed --master_port $port src/run_uie_lora.py \
   ${COMMON_PARAMS} \
   --model_name_or_path ${OUTPUT_BASE}/1-${TASK_ORDER[0]}/adapter \
   --task_config_dir configs/long_configs/${TASK_ORDER[1]} \
   --output_dir ${OUTPUT_BASE}/2-${TASK_ORDER[1]} \
   --run_name long_round2 \
   --lamda_1 ${LAMBDAS1[1]} \
   --lamda_2 ${LAMBDAS2[1]}

sleep 5

#############################################
# Task 3
#############################################
CUDA_VISIBLE_DEVICES=0 deepspeed --master_port $port src/run_uie_lora.py \
   ${COMMON_PARAMS} \
   --model_name_or_path ${OUTPUT_BASE}/2-${TASK_ORDER[1]}/adapter \
   --task_config_dir configs/long_configs/${TASK_ORDER[2]} \
   --output_dir ${OUTPUT_BASE}/3-${TASK_ORDER[2]} \
   --run_name long_round3 \
   --lamda_1 ${LAMBDAS1[2]} \
   --lamda_2 ${LAMBDAS2[2]}

sleep 5

#############################################
# Task 4
#############################################
CUDA_VISIBLE_DEVICES=0 deepspeed --master_port $port src/run_uie_lora.py \
   ${COMMON_PARAMS} \
   --model_name_or_path ${OUTPUT_BASE}/3-${TASK_ORDER[2]}/adapter \
   --task_config_dir configs/long_configs/${TASK_ORDER[3]} \
   --output_dir ${OUTPUT_BASE}/4-${TASK_ORDER[3]} \
   --run_name long_round4 \
   --lamda_1 ${LAMBDAS1[3]} \
   --lamda_2 ${LAMBDAS2[3]}

sleep 5

#############################################
# Task 5
#############################################
CUDA_VISIBLE_DEVICES=0 deepspeed --master_port $port src/run_uie_lora.py \
   ${COMMON_PARAMS} \
   --model_name_or_path ${OUTPUT_BASE}/4-${TASK_ORDER[3]}/adapter \
   --task_config_dir configs/long_configs/${TASK_ORDER[4]} \
   --output_dir ${OUTPUT_BASE}/5-${TASK_ORDER[4]} \
   --run_name long_round5 \
   --lamda_1 ${LAMBDAS1[4]} \
   --lamda_2 ${LAMBDAS2[4]}

sleep 5

#############################################
# Task 6
#############################################
CUDA_VISIBLE_DEVICES=0 deepspeed --master_port $port src/run_uie_lora.py \
   ${COMMON_PARAMS} \
   --model_name_or_path ${OUTPUT_BASE}/5-${TASK_ORDER[4]}/adapter \
   --task_config_dir configs/long_configs/${TASK_ORDER[5]} \
   --output_dir ${OUTPUT_BASE}/6-${TASK_ORDER[5]} \
   --run_name long_round6 \
   --lamda_1 ${LAMBDAS1[5]} \
   --lamda_2 ${LAMBDAS2[5]}

sleep 5

#############################################
# Task 7
#############################################
CUDA_VISIBLE_DEVICES=0 deepspeed --master_port $port src/run_uie_lora.py \
   ${COMMON_PARAMS} \
   --model_name_or_path ${OUTPUT_BASE}/6-${TASK_ORDER[5]}/adapter \
   --task_config_dir configs/long_configs/${TASK_ORDER[6]} \
   --output_dir ${OUTPUT_BASE}/7-${TASK_ORDER[6]} \
   --run_name long_round7 \
   --lamda_1 ${LAMBDAS1[6]} \
   --lamda_2 ${LAMBDAS2[6]}

sleep 5

#############################################
# Task 8
#############################################
CUDA_VISIBLE_DEVICES=0 deepspeed --master_port $port src/run_uie_lora.py \
   ${COMMON_PARAMS} \
   --model_name_or_path ${OUTPUT_BASE}/7-${TASK_ORDER[6]}/adapter \
   --task_config_dir configs/long_configs/${TASK_ORDER[7]} \
   --output_dir ${OUTPUT_BASE}/8-${TASK_ORDER[7]} \
   --run_name long_round8 \
   --lamda_1 ${LAMBDAS1[7]} \
   --lamda_2 ${LAMBDAS2[7]}

sleep 5

#############################################
# Task 9
#############################################
CUDA_VISIBLE_DEVICES=0 deepspeed --master_port $port src/run_uie_lora.py \
   ${COMMON_PARAMS} \
   --model_name_or_path ${OUTPUT_BASE}/8-${TASK_ORDER[7]}/adapter \
   --task_config_dir configs/long_configs/${TASK_ORDER[8]} \
   --output_dir ${OUTPUT_BASE}/9-${TASK_ORDER[8]} \
   --run_name long_round9 \
   --lamda_1 ${LAMBDAS1[8]} \
   --lamda_2 ${LAMBDAS2[8]}

sleep 5

#############################################
# Task 10
#############################################
CUDA_VISIBLE_DEVICES=0 deepspeed --master_port $port src/run_uie_lora.py \
   ${COMMON_PARAMS} \
   --model_name_or_path ${OUTPUT_BASE}/9-${TASK_ORDER[8]}/adapter \
   --task_config_dir configs/long_configs/${TASK_ORDER[9]} \
   --output_dir ${OUTPUT_BASE}/10-${TASK_ORDER[9]} \
   --run_name long_round10 \
   --lamda_1 ${LAMBDAS1[9]} \
   --lamda_2 ${LAMBDAS2[9]}

sleep 5

#############################################
# Task 11
#############################################
CUDA_VISIBLE_DEVICES=0 deepspeed --master_port $port src/run_uie_lora.py \
   ${COMMON_PARAMS} \
   --model_name_or_path ${OUTPUT_BASE}/10-${TASK_ORDER[9]}/adapter \
   --task_config_dir configs/long_configs/${TASK_ORDER[9]} \
   --output_dir ${OUTPUT_BASE}/11-${TASK_ORDER[9]} \
   --run_name long_round11 \
   --lamda_1 ${LAMBDAS1[9]} \
   --lamda_2 ${LAMBDAS2[9]}

sleep 5

#############################################
# Task 12
#############################################
CUDA_VISIBLE_DEVICES=0 deepspeed --master_port $port src/run_uie_lora.py \
   ${COMMON_PARAMS} \
   --model_name_or_path ${OUTPUT_BASE}/11-${TASK_ORDER[9]}/adapter \
   --task_config_dir configs/long_configs/${TASK_ORDER[10]} \
   --output_dir ${OUTPUT_BASE}/12-${TASK_ORDER[10]} \
   --run_name long_round12 \
   --lamda_1 ${LAMBDAS1[10]} \
   --lamda_2 ${LAMBDAS2[10]}

sleep 5

#############################################
# Task 13
#############################################
CUDA_VISIBLE_DEVICES=0 deepspeed --master_port $port src/run_uie_lora.py \
   ${COMMON_PARAMS} \
   --model_name_or_path ${OUTPUT_BASE}/12-${TASK_ORDER[10]}/adapter \
   --task_config_dir configs/long_configs/${TASK_ORDER[11]} \
   --output_dir ${OUTPUT_BASE}/13-${TASK_ORDER[11]} \
   --run_name long_round13 \
   --lamda_1 ${LAMBDAS1[11]} \
   --lamda_2 ${LAMBDAS2[11]}

sleep 5

#############################################
# Task 14
#############################################
CUDA_VISIBLE_DEVICES=0 deepspeed --master_port $port src/run_uie_lora.py \
   ${COMMON_PARAMS} \
   --model_name_or_path ${OUTPUT_BASE}/13-${TASK_ORDER[11]}/adapter \
   --task_config_dir configs/long_configs/${TASK_ORDER[12]} \
   --output_dir ${OUTPUT_BASE}/14-${TASK_ORDER[12]} \
   --run_name long_round14 \
   --lamda_1 ${LAMBDAS1[12]} \
   --lamda_2 ${LAMBDAS2[12]}

sleep 5

#############################################
# Task 15
#############################################
CUDA_VISIBLE_DEVICES=0 deepspeed --master_port $port src/run_uie_lora.py \
   ${COMMON_PARAMS} \
   --model_name_or_path ${OUTPUT_BASE}/14-${TASK_ORDER[12]}/adapter \
   --task_config_dir configs/long_configs/${TASK_ORDER[13]} \
   --output_dir ${OUTPUT_BASE}/15-${TASK_ORDER[13]} \
   --run_name long_round15 \
   --lamda_1 ${LAMBDAS1[13]} \
   --lamda_2 ${LAMBDAS2[13]}
