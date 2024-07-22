# Example
## HumanPPI
```bash
bash training_scripts/HPPI.py
```

See the [training_scripts](training_scripts) directory for more examples. Take the [HPPI.sh](training_scripts/HPPI.sh) script as an example.

```bash
#!/bin/bash
NUM_GPUS=8
# 650M: esm2_t33_650M_UR50D
# 8M: esm2_t6_8M_UR50D
accelerate launch --num_processes=$NUM_GPUS training_scripts/train_HumanPPI.py \
    --model_name_or_path "/cto_labs/AIDD/WEIGHTS/Protein/esm2_t33_650M_UR50D" \
    --peft_type "LORA" \
    --task_type "SEQ_CLS" \
    --lora_r 8 \
    --lora_alpha 16 \
    --lora_dropout 0.1 \
    --target_modules "query,value" \
    --data_config_path "dataset/config/HumanPPI.yaml" \ # just define the dataset class and  path etc.
    --base_lr 5e-4 \
    --classifier_lr_ratio 1 \ # classifier_lr = base_lr * classifier_lr_ratio
    --beta1 0.9 --beta2 0.98 --wdecay 0.01 \
    --optim_warmup_ratio 0.06 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps 4 \
    --dataloader_num_workers 8 \
    --fp16 True \
    --num_train_epochs 2 \
    --evaluation_strategy "epoch" \ # evaluate every epoch
    --save_strategy "epoch" \
    --save_total_limit 3 \
    --load_best_model_at_end True \
    --logging_steps 10 \
    --report_to "none" \
    --output_dir "output/HumanPPI" # change to your own output directory
```