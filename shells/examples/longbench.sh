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
python scripts/prediction_longbench.py \
    --model_name_or_path /scratch2/nlp/plm/Meta-Llama-3-8B-Instruct \
    --per_device_train_batch_size 1 \
    --output_dir models/temp \
    --learning_rate 7e-6 \
    --weight_decay 1e-4 \
    --adam_beta1 0.9 \
    --adam_beta2 0.98 \
    --adam_epsilon 1e-8 \
    --max_grad_norm 1.0 \
    --num_train_epochs 5 \
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
    --model_max_length 8000 \
    --input_file datasets/longbench/ \
    --output_file outputs/longbench-ttt-with-input/ \
    --prepend_input True \
    --recite_first False \
    --debug_size 1 \
    # --compute_attention False \
    # --attention_output_dir outputs/ttt-attns-loogle \
    # --overwrite_output_dir True \
    # --dataset_name long_qa \