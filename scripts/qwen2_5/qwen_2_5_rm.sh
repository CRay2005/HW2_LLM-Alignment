#!/usr/bin/env bash
#
# Copyright 2025 PKU-Alignment Team. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================


#MODEL_NAME_OR_PATH="Qwen/Qwen2.5-0.5B-Instruct" # model path
#MODEL_NAME_OR_PATH="../Qwen2.5-0.5B-Instruct" # 指向本地已下载的模型目录
MODEL_NAME_OR_PATH="../outputs/qwen_2_5_rm/slice_end"

#TRAIN_DATASETS="../assets/text_to_text/preference" # rm dataset path
TRAIN_DATASETS="../align_anything_t2t" # 指向本地已下载的数据目录
EVAL_DATASETS="../align_anything_t2t"

#TRAIN_TEMPLATE="PKUSafeRLHF" # dataset template
TRAIN_TEMPLATE="HOMEWORK" # 改为 HOMEWORK template
EVAL_TEMPLATE="HOMEWORK"


TRAIN_SPLIT="train" # split the dataset
EVAL_SPLIT="validation"

OUTPUT_ROOT_DIR=$OUTPUT_ROOT_DIR

if [ -z "$OUTPUT_ROOT_DIR" ]; then
    echo "OUTPUT_ROOT_DIR is not set"
    OUTPUT_ROOT_DIR="../outputs"
fi

OUTPUT_DIR="${OUTPUT_ROOT_DIR}/qwen_2_5_rm" # output dir

# For wandb online logging
#export WANDB_API_KEY=""
export WANDB_API_KEY="0be0052ff2a2ad64c060a4df794e9ac51be87af5" # 添加 WANDB_API_KEY

# Source the setup script
source ./setup.sh

# Execute deepspeed command
# train

# deepspeed \
#      --master_port ${MASTER_PORT} \
#      --module align_anything.trainers.text_to_text.rm \
#      --model_name_or_path ${MODEL_NAME_OR_PATH} \
#      --train_template ${TRAIN_TEMPLATE} \
#      --train_datasets ${TRAIN_DATASETS} \
#      --train_split ${TRAIN_SPLIT} \
#      --output_dir ${OUTPUT_DIR} \
#      --epochs 1 


# eval
deepspeed \
     --master_port ${MASTER_PORT} \
     --module align_anything.trainers.text_to_text.rm \
     --model_name_or_path ${MODEL_NAME_OR_PATH} \
     --eval_template ${EVAL_TEMPLATE} \
     --eval_datasets ${EVAL_DATASETS} \
     --eval_split ${EVAL_SPLIT} \
     --output_dir ${OUTPUT_DIR} \
     --save_total_limit 1 \
     --epochs 1 

# train & eval
# deepspeed \
#      --master_port ${MASTER_PORT} \
#      --module align_anything.trainers.text_to_text.rm \
#      --model_name_or_path ${MODEL_NAME_OR_PATH} \
#      --train_template ${TRAIN_TEMPLATE} \
#      --eval_template ${EVAL_TEMPLATE} \
#      --train_datasets ${TRAIN_DATASETS} \
#      --eval_datasets ${EVAL_DATASETS} \
#      --train_split ${TRAIN_SPLIT} \
#      --eval_split ${EVAL_SPLIT} \
#      --output_dir ${OUTPUT_DIR} \
#      --epochs 1 
