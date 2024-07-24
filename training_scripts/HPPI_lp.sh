#!/bin/bash
NUM_GPUS=8
# 650M: esm2_t33_650M_UR50D
# 8M: esm2_t6_8M_UR50D
accelerate launch --num_processes=$NUM_GPUS training_scripts/train_HumanPPI.py \
    --model_name_or_path "/cto_labs/AIDD/WEIGHTS/Protein/esm2_t33_650M_UR50D" \
    --data_config_path "dataset/config/HumanPPI.yaml" \
    --base_lr 1e-2 \
    --beta1 0.9 --beta2 0.98 --wdecay 0.01 \
    --freeze_backbone \
    --optim_warmup_ratio 0.06 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps 4 \
    --dataloader_num_workers 8 \
    --fp16 True \
    --num_train_epochs 5 \
    --evaluation_strategy "epoch" \
    --save_strategy "epoch" \
    --save_total_limit 3 \
    --load_best_model_at_end True \
    --logging_steps 10 \
    --report_to "none" \
    --output_dir "output/HumanPPI_lp" 