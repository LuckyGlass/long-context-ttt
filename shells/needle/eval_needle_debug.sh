#!/bin/bash
#SBATCH -J needle
#SBATCH -N 1
#SBATCH -p IAI_SLURM_HGX
#SBATCH --gres=gpu:4
#SBATCH --qos=16gpu-hgx
#SBATCH --time=72:00:00
#SBATCH -o logs/%j-needle-ttt-debug.out.log
#SBATCH -e logs/%j-needle-ttt-debug.err.log
#SBATCH -c 1

python scripts/prediction_needle.py \
    --haystack_path datasets/needle/PaulGrahamEssays \
    --output_path outputs/needle-ttt-debug.json \
    --overwrite True \
    --test_length 1000 10000 19000 28000 37000 46000 55000 64000 \
    --test_depth_min 0 \
    --test_depth_max 100 \
    --test_depth_num 10 \
    --num_samples_per_case 1 \
    --prompt "\n\nWhat is the best thing to do in ABC University? Answer:" \
    --needle "\n\nThe best thing to do in ABC University is taking photos and singing a song on a sunny day.\n\n" \
    --enable_ICL False \
    --reciting_mode False \
    --force_regenerate False \
    --num_ttt_needles 50 \
    --ttt_needle_path datasets/needle/processed_needles_ttt.json \
    --model_name_or_path models/Meta-Llama-3-8B-Instruct \
    --model_max_length 7500 \
    --block_size 256 \
    --len_segment 8 \
    --len_offset 3 \
    --pad_to_max_length False \
    --full_ft True \
    --gather_batches True \
    --num_train_epochs 3 \
    --involve_qa_epochs 5 \
    --remove_unused_columns False \
    --report_to none \
    --output_dir models/temp \
    --overwrite_output_dir True \
    --per_device_train_batch_size 1 \
    --learning_rate 2e-5 \
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

