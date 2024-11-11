#!/bin/bash
#SBATCH -J Bamboo
#SBATCH -N 1
#SBATCH -p IAI_SLURM_HGX
#SBATCH --gres=gpu:1
#SBATCH --qos=16gpu-hgx
#SBATCH --time=72:00:00
#SBATCH -o logs/%j-Bamboo-SFTSent-Baseline.out.log
#SBATCH -e logs/%j-Bamboo-SFTSent-Baseline.err.log
#SBATCH -c 1

wandb disabled
python scripts/prediction_bamboo.py \
	--model_name_or_path models/sft-sent-fixed \
	--is_peft_model True \
	--model_max_length 7900 \
	--output_file outputs/Bamboo-SFTSent-Baseline.jsonl \
	--overwrite True \
	--input_file datasets/bamboo/reportsumsort_16k.jsonl \
	--block_size 256 \
	--len_segment 8 \
	--len_offset 3 \
	--pad_to_max_length False \
	--use_lora True \
	--lora_rank 8 \
	--load_in_4bit True \
	--num_train_epochs 0 \
	--involve_qa_epochs 0 \
	--remove_unused_columns True \
	--report_to none \
	--output_dir models/temp \
	--overwrite_output_dir True \
	--per_device_train_batch_size 1 \
	--learning_rate 1e-6 \
	--weight_decay 1e-4 \
	--adam_beta1 0.9 \
	--adam_beta2 0.98 \
	--adam_epsilon 1e-8 \
	--max_grad_norm 1.0 \
	--log_level info \
	--logging_strategy steps \
	--logging_steps 1 \
	--save_strategy no \
	--bf16 True \
	--tf32 False \
	--gradient_checkpointing True \

