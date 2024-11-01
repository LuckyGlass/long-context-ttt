#!/bin/bash
#SBATCH -J quality
#SBATCH -N 1
#SBATCH -p IAI_SLURM_HGX
#SBATCH --gres=gpu:4
#SBATCH --qos=16gpu-hgx
#SBATCH --time=72:00:00
#SBATCH -o logs/%j-quality-sent-test-ttt.out.log
#SBATCH -e logs/%j-quality-sent-test-ttt.err.log
#SBATCH -c 1

python scripts/prediction_quality_sent.py \
    --model_name_or_path models/Meta-Llama-3-8B-Instruct \
    --model_max_length 7900 \
    --output_file outputs/quality-sent-test-ttt.json \
    --input_file datasets/QuALITY/quality-test-sent.json \
    --block_size 256 \
    --len_segment 8 \
    --len_offset 3 \
    --pad_to_max_length False \
    --full_ft True \
    --gather_batches True \
    --num_train_epochs 5 \
    --remove_unused_columns False \
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
    --remove_unused_columns True

