# Copyright 2024 PKU-Alignment Team. All Rights Reserved.
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
"""Trainer for DPO training."""


import argparse
import os
import sys
from typing import Any

import deepspeed
import torch
import torch.distributed as dist
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModelForCausalLM
from transformers.integrations.deepspeed import HfDeepSpeedConfig

from align_anything.datasets.text_to_text.preference import PreferenceBatch, PreferenceDataset
from align_anything.models.pretrained_model import load_pretrained_models
from align_anything.trainers.base import SupervisedTrainerBase
from align_anything.utils.device_utils import torch_gc, torch_set_device
from align_anything.utils.multi_process import (
    get_all_reduce_mean,
    get_current_device,
    is_main_process,
)
from align_anything.utils.tools import (
    custom_cfgs_to_dict,
    dict_to_namedtuple,
    gather_log_probabilities,
    prepare_ds_eval_cfgs,
    prepare_ds_train_cfgs,
    read_cfgs,
    seed_everything,
    update_dict,
)

from transformers import AutoTokenizer
import json
from pathlib import Path
from torch.nn.utils.rnn import pad_sequence

def strip_pad(seq: torch.Tensor, pad_token_id: int):
    # remove the pad token in the tensor
    return seq[seq != pad_token_id]


class DPOTrainer(SupervisedTrainerBase):

    def __init__(self, cfgs, ds_cfgs) -> None:
        """Initialize trainer."""
        self.cfgs = cfgs
        self.ds_train_cfgs = prepare_ds_train_cfgs(custom_cfgs=cfgs.train_cfgs, raw_ds_cfgs=ds_cfgs)
        self.ds_eval_cfgs = prepare_ds_eval_cfgs(custom_cfgs=cfgs.train_cfgs, raw_ds_cfgs=ds_cfgs)
        self.global_step = 0
        self.infer_batch = lambda batch: {k: v for k, v in batch.items() if k != 'meta_info'}

        self.init_check()
        dist.barrier()
        self.init_models()
        if hasattr(self.model, 'infer_batch'):
            self.infer_batch = self.model.infer_batch
        dist.barrier()
        self.init_datasets()
        dist.barrier()
        self.init_engines()
        dist.barrier()
        self.init_logger()

    def init_check(self) -> None:
        """Initial configuration checking."""
        super().init_check()

    def init_models(self) -> None:
        """Initialize model and tokenizer."""
        if self.ds_train_cfgs['zero_optimization']['stage'] == 3:
            self.dstchf_train = HfDeepSpeedConfig(self.ds_train_cfgs)
        if self.ds_eval_cfgs['zero_optimization']['stage'] == 3:
            self.dsechf_eval = HfDeepSpeedConfig(self.ds_eval_cfgs)
        self.bnb_cfgs = self.cfgs.bnb_cfgs
        self.lora_cfgs = self.cfgs.lora_cfgs
        self.model, self.tokenizer, self.processor = load_pretrained_models(
            self.cfgs.model_cfgs.model_name_or_path,
            model_max_length=self.cfgs.model_cfgs.model_max_length,
            padding_side='left',
            trust_remote_code=True,
            bnb_cfgs=self.bnb_cfgs,
            lora_cfgs=self.lora_cfgs,
        )
        self.reference_model, _, _ = load_pretrained_models(
            self.cfgs.model_cfgs.model_name_or_path,
            model_max_length=self.cfgs.model_cfgs.model_max_length,
            padding_side='left',
            trust_remote_code=self.cfgs.model_cfgs.trust_remote_code,
            bnb_cfgs=self.bnb_cfgs,
            lora_cfgs=self.lora_cfgs,
        )

    def init_datasets(self) -> None:
        """Initialize training and evaluation datasets."""
        self.train_dataloader, self.eval_dataloader = self.get_dataloaders(
            PreferenceDataset, PreferenceDataset
        )

    def init_engines(self) -> None:
        """Initialize DeepSpeed engines."""
        self.init_deepspeed_engines()
        self.reference_model, *_ = deepspeed.initialize(
            model=self.reference_model,
            config=self.ds_eval_cfgs,
        )

    def compute_log_probs(
        self,
        model: AutoModelForCausalLM,
        batch: PreferenceBatch,
    ) -> torch.Tensor:
        """Compute log probabilities of given sequences."""
        logits = model(**self.infer_batch(batch)).logits
        device = logits.device
        input_ids = batch['input_ids']
        batch_size = len(batch['meta_info']['response_lens'])
        logprob_list = []
        for idx in range(batch_size):
            response_length = batch['meta_info']['response_lens'][idx]
            raw_input_id = strip_pad(input_ids[idx], self.tokenizer.pad_token_id)
            logit = logits[idx][-response_length:].unsqueeze(0)
            input_id = raw_input_id[-response_length:].unsqueeze(0)
            log_p = gather_log_probabilities(logit[:, :-1], input_id[:, 1:])
            logprob_list.append(log_p.squeeze(0))
        return torch.nn.utils.rnn.pad_sequence(
            logprob_list, batch_first=True, padding_value=0.0
        ).to(device)

    def loss(  # pylint: disable=too-many-locals
        self,
        batch: PreferenceBatch,
    ) -> dict[str, torch.Tensor]:
        """Loss function for the DPO algorithm."""
        sequence_log_probs = self.compute_log_probs(
            self.model.module,
            batch,
        )
        (
            better_sequence_log_probs,  # size = (B, L - 1)
            worse_sequence_log_probs,  # size = (B, L - 1)
        ) = sequence_log_probs.chunk(chunks=2, dim=0)

        with torch.no_grad():
            ref_sequence_log_probs = self.compute_log_probs(  # size = (2 * B, L - 1)
                self.reference_model.module,
                batch,
            )
            ref_better_sequence_log_probs, ref_worse_sequence_log_probs = (
                ref_sequence_log_probs.chunk(chunks=2, dim=0)
            )

        losses = []
        better_sample_rewards = []
        worse_sample_rewards = []

        batch_size = better_sequence_log_probs.size(0)
        for i in range(batch_size):
            better_log_prob = better_sequence_log_probs[i, :].sum(dim=-1)
            worse_log_prob = worse_sequence_log_probs[i, :].sum(dim=-1)
            ref_better_log_prob = ref_better_sequence_log_probs[i, :].sum(dim=-1)
            ref_worse_log_prob = ref_worse_sequence_log_probs[i, :].sum(dim=-1)
            better_log_ratio = better_log_prob - ref_better_log_prob
            worse_log_ratio = worse_log_prob - ref_worse_log_prob

            losses.append(
                -F.logsigmoid(
                    self.cfgs.train_cfgs.scale_coeff * (better_log_ratio - worse_log_ratio),
                ),
            )
            better_sample_rewards.append(
                self.cfgs.train_cfgs.scale_coeff * better_log_ratio.detach(),
            )
            worse_sample_rewards.append(self.cfgs.train_cfgs.scale_coeff * worse_log_ratio.detach())
        loss = torch.stack(losses).mean()  # size = ()
        better_sample_reward = torch.stack(better_sample_rewards)  # size = (B,)
        worse_sample_reward = torch.stack(worse_sample_rewards)  # size = (B,)
        reward = better_sample_reward + worse_sample_reward  # size = (B,)
        reward_accuracy = (better_sample_reward > worse_sample_reward).float().mean()  # size = ()
        reward_margin = better_sample_reward - worse_sample_reward  # size = (B,)

        return {
            'loss': loss,
            'reward': reward,
            'better_sample_reward': better_sample_reward,
            'worse_sample_reward': worse_sample_reward,
            'reward_accuracy': reward_accuracy,
            'reward_margin': reward_margin,
        }

    def train_step(
        self,
        batch: PreferenceBatch,
    ) -> dict[str, Any]:
        """Perform a single training step for DPO."""
        loss_dict = self.loss(batch=batch)
        loss = loss_dict['loss']
        self.model.backward(loss)
        self.model.step()

        with torch.no_grad():
            reward = loss_dict['reward'].mean()
            better_sample_reward = loss_dict['better_sample_reward'].mean()
            worse_sample_reward = loss_dict['worse_sample_reward'].mean()
            reward_accuracy = loss_dict['reward_accuracy']
            reward_margin = loss_dict['reward_margin'].mean()

            loss = get_all_reduce_mean(loss)
            reward = get_all_reduce_mean(reward)
            better_sample_reward = get_all_reduce_mean(better_sample_reward)
            worse_sample_reward = get_all_reduce_mean(worse_sample_reward)
            reward_accuracy = get_all_reduce_mean(reward_accuracy)
            reward_margin = get_all_reduce_mean(reward_margin)

        return {
            'train/loss': loss.item(),
            'train/reward': reward.item(),
            'train/better_sample_reward': better_sample_reward.item(),
            'train/worse_sample_reward': worse_sample_reward.item(),
            'train/reward_accuracy': reward_accuracy.item(),
            'train/reward_margin': reward_margin.item(),
            'train/lr': self.model.optimizer.param_groups[0]['lr'],
        }

    def train(self) -> None:
        """Train the model."""
        self.logger.print('***** Running training *****')

        progress_bar = tqdm(
            total=self.cfgs.train_cfgs.epochs * len(self.train_dataloader),
            desc=f'Training 1/{self.cfgs.train_cfgs.epochs} epoch',
            position=0,
            leave=True,
            disable=not is_main_process(),
        )
        progress_bar.update(self.global_step)

        if self.cfgs.data_cfgs.eval_datasets:
            self.logger.print('\n***** Evaluating at the beginning *****')
            self.logger.log(self.eval(), step=0)

        if len(self.train_dataloader)==0:   #新增。当eval时，没有配置train_dataloader直接退出，避免报错
            self.logger.print("train_dataloader is empty. Skipping training step.")
            return
        
        remain_epoch = self.cfgs.train_cfgs.epochs - (
            self.global_step // len(self.train_dataloader)
        )

        start_batch_idx = self.global_step % len(self.train_dataloader)

        for epoch in range(int(remain_epoch)):
            self.model.train()
            progress_bar.set_description(
                f'Resuming from checkpoint {epoch + 1}/{self.cfgs.train_cfgs.epochs} epoch '
            )

            for batch_idx, batch in enumerate(self.train_dataloader):
                if epoch == 0 and batch_idx < start_batch_idx:
                    continue

                info = self.train_step(batch)
                torch_gc()

                self.global_step += 1
                progress_bar.set_description(
                    f'Training {epoch + 1}/{self.cfgs.train_cfgs.epochs} epoch '
                    f'(loss {info["train/loss"]:.4f})',
                )
                progress_bar.update(1)

                info['train/epoch'] = self.global_step / len(self.train_dataloader)
                self.logger.log(info, step=self.global_step)

                save_interval = (
                    self.cfgs.train_cfgs.epochs
                    * len(self.train_dataloader)
                    // self.cfgs.logger_cfgs.save_total_limit
                )
                if self.global_step % save_interval == 0:
                    self.logger.print(f'Saving checkpoint at step {self.global_step} ...')
                    self.save(tag=self.global_step)
                    self.logger.print('Checkpoint saved.')

                if (
                    self.cfgs.data_cfgs.eval_datasets
                    and self.cfgs.train_cfgs.eval_strategy == 'steps'
                    and self.global_step % self.cfgs.train_cfgs.eval_interval == 0
                ):
                    self.logger.print(f'\n***** Evaluating at step {self.global_step} *****')
                    self.logger.log(self.eval(), step=self.global_step)

            if self.cfgs.data_cfgs.eval_datasets and self.cfgs.train_cfgs.eval_strategy == 'epoch':
                self.logger.print(
                    f'\n***** Evaluating at epoch {epoch + 1}/{self.cfgs.train_cfgs.epochs} *****',
                )
                self.logger.log(self.eval(), step=self.global_step)
            self.model.tput_timer.update_epoch_count()
            

    @torch.no_grad()
    def eval(self) -> dict[str, Any]:
        """Evaluate the model on the evaluation dataset."""
        self.logger.print('\n***** Evaluating Model Responses and Reward Differences *****')
        if self.eval_dataloader is None:
            return {}

        # 禁用梯度检查点以节省内存
        if self.cfgs.train_cfgs.gradient_checkpointing:
            self.model.gradient_checkpointing_disable()

        device = self.model.device if hasattr(self.model, 'device') else get_current_device()

        # 加载原始语言模型用于生成 response
        base_model_path = "../Qwen2.5-0.5B-Instruct"
        base_tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            #torch_dtype=torch.bfloat16,
            #device_map="auto",
            torch_dtype="auto",

        ).to(device)
        base_model.eval()

        # 加载奖励模型用于打分
        reward_model_path="../outputs/qwen_2_5_rm/slice_end"
        reward_model, reward_tokenizer, _ = load_pretrained_models(
            reward_model_path,  
            is_reward_model=True,
            trust_remote_code=self.cfgs.model_cfgs.trust_remote_code,
            model_max_length=self.cfgs.model_cfgs.model_max_length,
            padding_side='right',
        )
        reward_model.to(device).eval()

        all_results = []
        dpo_scores = []
        base_scores = []

        # 获取测试集 dataloader
        eval_dataloader = tqdm(
            self.eval_dataloader,
            desc='Evaluating',
            disable=not is_main_process(),
            position=1,
            leave=False,
        )

        for batch in eval_dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            # Step 1: 使用 DPO 模型生成回答
            dpo_outputs = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=512,
                do_sample=False,
                pad_token_id=base_tokenizer.eos_token_id,
            )

            # Step 2: 使用基础模型生成回答
            # base_input_ids = input_ids.clone().detach()
            # base_attention_mask = attention_mask.clone().detach()
            # base_outputs = base_model.generate(
            #     input_ids=base_input_ids,
            #     attention_mask=base_attention_mask,
            #     max_new_tokens=512,
            #     do_sample=False,
            #     pad_token_id=base_tokenizer.eos_token_id,
            # )

            # Step 3: 解码为文本
            dpo_texts = self.tokenizer.batch_decode(dpo_outputs, skip_special_tokens=True)
            # base_texts = base_tokenizer.batch_decode(base_outputs, skip_special_tokens=True)

            # Step 4: 构造输入用于奖励模型打分
            dpo_score_inputs = reward_tokenizer(
                dpo_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
                add_special_tokens=True,
            ).to(device)

            # base_score_inputs = reward_tokenizer(
            #     base_texts,
            #     return_tensors="pt",
            #     padding=True,
            #     truncation=True,
            #     max_length=512,
            #     add_special_tokens=True,
            # ).to(device)

            # Step 5: 奖励模型打分
            dpo_scores_batch = reward_model(**dpo_score_inputs).end_scores.squeeze()
            # base_scores_batch = reward_model(**base_score_inputs).end_scores.squeeze()

            # Step 6: 保存结果
            for i in range(len(dpo_texts)):
                all_results.append({
                    'prompt': self.tokenizer.decode(input_ids[i], skip_special_tokens=True),
                    'dpo_response': dpo_texts[i],
                    # 'base_response': base_texts[i],
                    'dpo_score': float(dpo_scores_batch[i].item()),
                    # 'base_score': float(base_scores_batch[i].item()),
                })
                dpo_scores.append(dpo_scores_batch[i])
                # base_scores.append(base_scores_batch[i])

        # Step 7: 统计得分差异
        dpo_scores_tensor = torch.stack(dpo_scores)
        # base_scores_tensor = torch.stack(base_scores)
        # score_diff = (dpo_scores_tensor - base_scores_tensor).mean().item()

        info = {
            'eval/dpo_score_mean': dpo_scores_tensor.mean().item(),
            # 'eval/base_score_mean': base_scores_tensor.mean().item(),
            'eval/score_diff': score_diff,
        }

        # Step 8: 打印最后一批输出对比
        if is_main_process():
            max_num_rows = 5
            last_dpo_responses = dpo_texts[:max_num_rows]
            # last_base_responses = base_texts[:max_num_rows]

            dpo_scores_str = [f'{score:.4f}' for score in dpo_scores_tensor[:max_num_rows].tolist()]
            # base_scores_str = [f'{score:.4f}' for score in base_scores_tensor[:max_num_rows].tolist()]

            title = ', '.join(f'{key.rpartition("/")[-1]}={value:.4f}' for key, value in info.items())
            self.logger.print_table(
                title=f'Evaluation: {title}',
                # columns=['Prompt', 'DPO Response', 'Base Response', 'DPO Score', 'Base Score'],
                columns=['Prompt', 'DPO Response', 'Base Response', 'DPO Score'],
                rows=tuple(zip(
                    [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in input_ids[:max_num_rows]],
                    last_dpo_responses,
                    last_base_responses,
                    dpo_scores_str,
                    # base_scores_str,
                )),
                max_num_rows=max_num_rows,
            )

        # Step 9: 保存所有结果到文件
        # if is_main_process():
        #     save_path = Path(self.cfgs.logger_cfgs.output_dir) / "dpo_eval_comparison.json"
        #     with open(save_path, "w", encoding="utf-8") as f:
        #         json.dump(all_results, f, ensure_ascii=False, indent=4)
        #     self.logger.print(f"Saved DPO vs Base comparison to {save_path}")

        return info

    def save(
        self,
        model: deepspeed.DeepSpeedEngine | None = None,
        tag: int | None = None,
    ) -> None:
        """Save model and tokenizer in Hugging Face format."""
        self.save_transformers(model=model, tag=tag)


def main():
    # setup distribution training
    deepspeed.init_distributed()
    current_device = get_current_device()
    torch_set_device(current_device)

    # read default configs from the yaml file
    task = os.path.join('text_to_text', 'dpo')
    dict_cfgs, ds_cfgs = read_cfgs(mode='train', task=task)

    # get custom configs from command line
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    _, unparsed_args = parser.parse_known_args()
    keys = [k[2:] for k in unparsed_args[1::2]]
    values = list(unparsed_args[2::2])
    unparsed_args = dict(zip(keys, values))
    for k, v in unparsed_args.items():
        dict_cfgs = update_dict(dict_cfgs, custom_cfgs_to_dict(k, v))

    # setup training
    cfgs = dict_to_namedtuple(dict_cfgs)
    seed_everything(cfgs.train_cfgs.seed)

    # finetune the model
    trainer = DPOTrainer(cfgs=cfgs, ds_cfgs=ds_cfgs)
    trainer.train()
    #trainer.save()


if __name__ == '__main__':
    sys.exit(main())
