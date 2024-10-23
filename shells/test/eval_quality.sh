#!/bin/bash
#SBATCH -J loogle
#SBATCH -N 1
#SBATCH -p IAI_SLURM_HGX
#SBATCH --gres=gpu:6
#SBATCH --qos=16gpu-hgx
#SBATCH --time=72:00:00
#SBATCH -o logs/%j-out.log
#SBATCH -e logs/%j-err.log
#SBATCH -c 1

wandb disabled
python scripts/prediction_quality.py \
    --remove_unused_columns False \
    --input_file datasets/QuALITY/timeline-dev-summary.json \
    --model_name_or_path models/Meta-Llama-3-8B-Instruct \
    --output_dir models/temp \
    --overwrite_output_dir True \
    --per_device_train_batch_size 1 \
    --learning_rate 7e-6 \
    --weight_decay 1e-4 \
    --adam_beta1 0.9 \
    --adam_beta2 0.98 \
    --adam_epsilon 1e-8 \
    --max_grad_norm 1.0 \
    --num_train_epochs 5 \
    --involve_qa_epochs 0 \
    --log_level info \
    --logging_strategy steps \
    --logging_steps 1 \
    --save_strategy no \
    --bf16 True \
    --tf32 False \
    --gradient_checkpointing True \
    --full_ft True \
    --gather_batches True \
    --block_size 1024 \
    --len_segment 2 \
    --len_offset 1 \
    --model_max_length 7500 \
    --output_file outputs/debug.json \
    --enable_ICL True \
    --ttt_enable_ICL True \
    --recite_first False \
    --ttt_recite_first True \
    --debug_size 1 \
    --generator_name_or_path models/Meta-Llama-3-8B-Instruct \
    --num_generate_qa 0 \
    --num_timeline_reorder 0 \
    --enable_diverse_qa False \
    --append_question True