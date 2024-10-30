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
    --model_name_or_path /scratch/nlp/long-context-ttt/models/sft_sent \   
    --is_peft_model True \   
    --output_file /scratch/nlp/lijiaqi/long-context-ttt/output/quality-sft-ttt-w-synQA.jsonl \   
    --input_file /scratch/nlp/lijiaqi/long-context-ttt/data/generated_data/quality-test-sent.jsonl \    
    --block_size 256 \   
    --len_segment 8 \   
    --len_offset 3 \   
    --pad_to_max_length False \   
    --generator_name_or_path models/Meta-Llama-3-8B-Instruct \   
    --num_timeline_reorder 5 \   
    --num_timeline_reorder_events 4 6 \   
    --use_lora True \   
    --lora_rank 8 \   
    --load_in_4bit True \   
    --gather_batches True \   
    --involve_qa_epochs 3 \   
    --num_train_epochs 2 \   
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
    --remove_unused_columns False 
