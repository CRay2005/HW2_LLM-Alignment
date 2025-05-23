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


MODEL_NAME_OR_PATH="meta-llama/Llama-3.1-8B" # model path

TRAIN_DATASETS="PKU-Alignment/DollyTails-12K" # sft dataset path
TRAIN_TEMPLATE="O1_T2T" # sft dataset template
TRAIN_SPLIT="train" # split the sft dataset

OUTPUT_DIR="../outputs/llama_sft_o1" # output dir

# For wandb online logging
export WANDB_API_KEY=""
export O1_SPECIAL_TOKENS="['<|reserved_special_token_0|>', '<|reserved_special_token_1|>', '<|reserved_special_token_2|>']"

# Source the setup script
source ./setup.sh

# Execute deepspeed command
deepspeed \
     --master_port ${MASTER_PORT} \
     --module align_anything.trainers.text_to_text.sft \
     --model_name_or_path ${MODEL_NAME_OR_PATH} \
     --train_template ${TRAIN_TEMPLATE} \
     --train_datasets ${TRAIN_DATASETS} \
     --train_split ${TRAIN_SPLIT} \
     --output_dir ${OUTPUT_DIR}
