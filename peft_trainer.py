import sys
import yaml
from easydict import EasyDict
from tqdm import tqdm
from typing import List, Dict, Any, Tuple
import logging
import math

import torch
from torch import nn
import torchmetrics
from torcheval.metrics.functional import binary_auprc, binary_accuracy
from torch.optim import AdamW
from torch.utils.data import DataLoader
from peft import (
    get_peft_config,
    get_peft_model,
    get_peft_model_state_dict,
    set_peft_model_state_dict,
    LoraConfig,
    PeftType,
    PrefixTuningConfig,
    PromptEncoderConfig,
)
from transformers import AutoModelForSequenceClassification, \
    TrainingArguments, Trainer, TrainerCallback, \
    get_linear_schedule_with_warmup, set_seed
from transformers.trainer_utils import is_main_process
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers import get_linear_schedule_with_warmup

from utils.others import setup_seed
from utils.module_loader import load_dataset
from utils.others import trainable_parameters

if __name__ == "__main__":
    # take PPI task as example
    config_path = 'config/HumanPPI/esm/8M.yaml'
    with open(config_path, 'r', encoding='utf-8') as r:
        config = EasyDict(yaml.safe_load(r))

    if config.setting.seed:
        setup_seed(config.setting.seed)
        
    # lora config
    peft_type = PeftType.LORA
    task_type = "SEQ_CLS"
    lora_r = 8
    lora_alpha = 16
    lora_dropout = 0.1
    target_modules = ["query", "value"]
    # train config
    batch_size = 2
    gradient_accumulation_steps = 4
    num_epochs = 20
    warmup_ratio = 0.06
    # warmup_steps = 200
    lr = 1e-3
    beta1 = 0.9
    beta2 = 0.98
    weight_decay = 0.01
    eval_interval = 0.1
    fp16 = True
    save_total_limit = 3
    dataloader_num_workers = 8
    evaluate_strategy = "epoch"
    # log config
    log_steps = 100

    # fix conflict in configs
    config.dataset.dataloader_kwargs.batch_size = batch_size

    training_args = TrainingArguments(
        output_dir="output/ppi", # TODO: fix
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=num_epochs,
        fp16=fp16,
        logging_steps=log_steps,
        # optim="adamw_torch",
        # learning_rate=lr,
        # adam_beta1=beta1,
        # adam_beta2=beta2,
        # weight_decay=weight_decay,
        # warmup_steps=warmup_steps,
        evaluation_strategy=evaluate_strategy,
        eval_steps=eval_interval,
        save_strategy="epoch",
        save_total_limit=save_total_limit,
        load_best_model_at_end=True,
        report_to="none",
        run_name="none",
        dataloader_num_workers=dataloader_num_workers,
        gradient_accumulation_steps=gradient_accumulation_steps,
    )

    # init dataset
    data_module = load_dataset(config.dataset)
    train_dataset = data_module.train_dataloader().dataset
    eval_dataset = data_module.val_dataloader().dataset
    test_dataset = data_module.test_dataloader().dataset

    def print_trainable_parameters(model):
        """
        Prints the number of trainable parameters in the model.
        """
        trainable_params = 0
        all_param = 0
        for _, param in model.named_parameters():
            all_param += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
        if is_main_process(training_args.local_rank):
            print(
                f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
            )

    # init tokenizer
    tokenizer = data_module.tokenizer

    def collate_fn(batch):
        seqs_1, seqs_2, label_ids = tuple(zip(*batch))
        label_ids = torch.tensor(label_ids, dtype=torch.long)
        encoder_info_1 = tokenizer.batch_encode_plus(seqs_1, return_tensors='pt', padding=True)
        encoder_info_2 = tokenizer.batch_encode_plus(seqs_2, return_tensors='pt', padding=True)
        return {"inputs_1": encoder_info_1, "inputs_2": encoder_info_2, "labels": label_ids}

    # init model
    # model_name = "/cto_labs/AIDD/WEIGHTS/Protein/esm2_t6_8M_UR50D" # 8M try try
    model_name = "/cto_labs/AIDD/WEIGHTS/Protein/esm2_t33_650M_UR50D"
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        return_dict=True,
    )
    # fix model classifier hidden_size
    hidden_size = model.config.hidden_size * 2
    classifier = nn.Sequential(
        nn.Linear(hidden_size, hidden_size),
        nn.ReLU(),
        nn.Linear(hidden_size, 2),
    )
    setattr(model, "classifier", classifier)

    # change model forward method to support dual inputs
    def ppi_forward(self, inputs_1, inputs_2, labels, **kwargs):
        hidden_1 = self.esm(**inputs_1)[0][:, 0, :]
        hidden_2 = self.esm(**inputs_2)[0][:, 0, :]
        hidden_concat = torch.cat([hidden_1, hidden_2], dim=-1)
        logits = self.classifier(hidden_concat)
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(logits, labels)
        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
        )
        
    setattr(model, "forward", ppi_forward.__get__(model))

    # init lora
    peft_config = LoraConfig(
        task_type=task_type,
        inference_mode=False,
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules,
    )
    model = get_peft_model(model, peft_config=peft_config)
    print_trainable_parameters(model)

    # init custom optimizer for different groups of parameters
    optimizer = AdamW(
        [
            {"params": model.esm.parameters(), "lr": lr},
            {"params": model.classifier.parameters(), "lr": lr * 0.05},
        ],
        lr=lr, betas=(beta1, beta2), weight_decay=weight_decay
    )
    total_batch_size = batch_size * gradient_accumulation_steps * training_args.world_size
    warmup_steps = warmup_ratio * len(train_dataset) * num_epochs
    num_training_steps = math.ceil(len(train_dataset) * num_epochs / total_batch_size)
    scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=num_training_steps,
    )

    class TestOnEvalCallback(TrainerCallback):
        def on_evaluate(self, args, state, control, logs=None, **kwargs):
            if is_main_process(training_args.local_rank):
                print("Testing on test dataset after evaluation phase...")
            trainer.predict(test_dataset)

    def compute_metrics(eval_pred):
        labels = eval_pred.label_ids
        preds = eval_pred.predictions.argmax(-1)
        # convert to tensor
        labels = torch.tensor(labels)
        preds = torch.tensor(preds)
        acc = binary_accuracy(preds, labels)
        aupr = binary_auprc(preds, labels)
        if is_main_process(training_args.local_rank):
            print(f"acc: {acc.item()}, aupr: {aupr.item()}")
        return {
            "acc": acc,
            "aupr": aupr,
        }

    # init trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=collate_fn,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        optimizers=(optimizer, scheduler),
        compute_metrics=compute_metrics,
        callbacks=[TestOnEvalCallback],
    )
    # evaluate first before train
    trainer.evaluate()
    # start train
    trainer.train()
    # testing...
    trainer.predict(test_dataset)