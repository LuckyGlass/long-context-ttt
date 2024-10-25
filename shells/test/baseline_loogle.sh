#!/bin/bash
#SBATCH -J loogle
#SBATCH -N 1
#SBATCH -p IAI_SLURM_HGX
#SBATCH --gres=gpu:4
#SBATCH --qos=16gpu-hgx
#SBATCH --time=72:00:00
#SBATCH -o logs/%j-loogle-baseline-no-ttt-out.log
#SBATCH -e logs/%j-loogle-baseline-no-ttt-err.log
#SBATCH -c 1

python scripts/prediction_loogle.py \
  --remove_unused_columns False \
  --output_file outputs/loogle-baseline-no-ttt.json \
  --model_name_or_path models/Meta-Llama-3-8B-Instruct \
  --model_max_length 8000 \
  --block_size 256 \
  --len_segment 8 \
  --len_offset 3 \
  --full_ft True \
  --input_file datasets/loogle/loogle-timeline-reorder.json \
  --overwrite True \
  --enable_ICL True \
  --recite_first False \
  --output_dir models/temp \
  --overwrite_output_dir True \
  --per_device_train_batch_size 1 \
  --num_train_epochs 0 \
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