#!/bin/bash
#SBATCH -J lbaseline
#SBATCH -N 1
#SBATCH -p IAI_SLURM_HGX
#SBATCH --gres=gpu:4
#SBATCH --qos=16gpu-hgx
#SBATCH --time=72:00:00
#SBATCH -o logs/%j-baseline-quality-dev-out.log
#SBATCH -e logs/%j-baseline-quality-dev-err.log
#SBATCH -c 1

wandb disabled
TOKENIZERS_PARALLELISM=0 python scripts/prediction_quality.py \
    --remove_unused_columns False \
    --report_to none \
    --overwrite True \
    --model_name_or_path models/Meta-Llama-3-8B-Instruct \
    --model_max_length 7900 \
    --block_size 256 \
    --len_segment 8 \
    --len_offset 3 \
    --full_ft True \
    --input_file datasets/QuALITY/timeline-dev-summary.json \
    --enable_ICL True \
    --recite_first False \
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
    --involve_qa_epochs 0 \
    --log_level info \
    --logging_strategy steps \
    --logging_steps 1 \
    --save_strategy no \
    --bf16 True \
    --tf32 False \
    --gradient_checkpointing True \
    --output_file outputs/quality-dev-baseline-no-ttt.json \
