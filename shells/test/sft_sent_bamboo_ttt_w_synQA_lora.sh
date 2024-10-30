#!/bin/bash
#SBATCH -J bamboo
#SBATCH -N 1
#SBATCH -p IAI_SLURM_HGX
#SBATCH --gres=gpu:4
#SBATCH --qos=16gpu-hgx
#SBATCH --time=72:00:00
#SBATCH -o logs/%j-bamboo-append-question-out.log
#SBATCH -e logs/%j-bamboo-append-question-err.log
#SBATCH -c 1

wandb disabled
python scripts/prediction_bamboo.py \
    --remove_unused_columns False \
    --model_name_or_path models/sft_sent \
    --is_peft_model True \
    --model_max_length 7900 \
    --block_size 256 \
    --len_segment 8 \
    --len_offset 3 \
    --num_generate_qa 0 \
    --generator_name_or_path models/Meta-Llama-3-8B-Instruct \
    --pad_to_max_length False \
    --ttt_recite_first False \
    --ttt_enable_ICL True \
    --qa_loss_weight 5.0 \
    --enable_diverse_qa False \
    --num_timeline_reorder 5 \
    --num_timeline_reorder_events 5 \
    --append_question False \
    --use_lora True \
    --lora_rank 8 \
    --load_in_4bit True \
    --gather_batches True \
    --involve_qa_epochs 3 \
    --input_file datasets/bamboo/reportsumsort_16k.jsonl \
    --overwrite True \
    --enable_ICL True \
    --recite_first False \
    --output_dir models/temp \
    --overwrite_output_dir True \
    --per_device_train_batch_size 1 \
    --learning_rate 1e-6 \
    --weight_decay 1e-4 \
    --adam_beta1 0.9 \
    --adam_beta2 0.98 \
    --adam_epsilon 1e-8 \
    --max_grad_norm 1.0 \
    --num_train_epochs 2 \
    --log_level info \
    --logging_strategy steps \
    --logging_steps 1 \
    --save_strategy no \
    --bf16 True \
    --tf32 False \
    --gradient_checkpointing True \
    --output_file outputs/bamboo-ttt-w-synQA.jsonl \
