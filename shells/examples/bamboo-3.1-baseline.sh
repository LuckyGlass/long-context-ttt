#!/bin/bash
#SBATCH -J bamboo
#SBATCH -N 1
#SBATCH -p IAI_SLURM_HGX
#SBATCH --gres=gpu:4
#SBATCH --qos=16gpu-hgx
#SBATCH --time=72:00:00
#SBATCH -o logs/%j-out.log
#SBATCH -e logs/%j-err.log
#SBATCH -c 1

wandb disabled
python scripts/prediction_bamboo.py \
    --model_name_or_path models/Meta-Llama-3.1-8B-Instruct \
    --output_dir models/temp \
    --overwrite_output_dir True \
    --per_device_train_batch_size 1 \
    --learning_rate 7e-6 \
    --weight_decay 1e-4 \
    --adam_beta1 0.9 \
    --adam_beta2 0.98 \
    --adam_epsilon 1e-8 \
    --max_grad_norm 1.0 \
    --num_train_epochs 0 \
    --log_level info \
    --logging_strategy steps \
    --logging_steps 1 \
    --save_strategy no \
    --bf16 True \
    --tf32 False \
    --gradient_checkpointing True \
    --full_ft True \
    --gather_batches True \
    --block_size 256 \
    --len_segment 8 \
    --len_offset 3 \
    --model_max_length 14000 \
    --pad_to_max_length False \
    --output_file outputs/ttt-reportsumsort-baseline-3.1.json \
    --input_file datasets/bamboo/reportsumsort_16k.jsonl \
    --prompt_name reportsumsort \
    --prompt_path scripts/prompt_bamboo.json \
