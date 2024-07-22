import math
import os
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch
import yaml
from easydict import EasyDict
from torch import nn
from torch.optim import AdamW
from torcheval.metrics.functional import binary_accuracy, binary_auprc
from transformers import (AutoModelForSequenceClassification, HfArgumentParser,
                          Trainer, TrainerCallback, TrainingArguments,
                          get_linear_schedule_with_warmup)
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers.trainer_utils import is_main_process

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.module_loader import load_dataset
from utils.others import setup_seed

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    Inlcude lora settings.
    """
    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    
@dataclass
class DataArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """
    data_config_path : str = field(
        metadata={"help": "Path to data config file"},
        default="dataset/config/HumanPPI.yaml",
    )

@dataclass
class OptimArguments:
    """
    Arguments pertaining to the optimizer.
    """
    base_lr : float = field(
        metadata={"help": "Base learning rate"},
        default=1e-3,
    )
    beta1 : float = field(
        metadata={"help": "Beta1 for AdamW optimizer"},
        default=0.9,
    )
    beta2 : float = field(
        metadata={"help": "Beta2 for AdamW optimizer"},
        default=0.98,
    )
    wdecay: float = field(
        metadata={"help": "Weight decay for AdamW optimizer"},
        default=0.01,
    )
    optim_warmup_ratio: float = field(
        metadata={"help": "Warmup ratio."},
        default=0.06,
    )

if __name__ == "__main__":
    # seed
    setup_seed(42)
    
    parser = HfArgumentParser((ModelArguments, DataArguments, OptimArguments, TrainingArguments))
    model_args, data_args, optim_args, training_args = parser.parse_args_into_dataclasses()
    
    # take PPI task as example
    data_config_path = data_args.data_config_path
    with open(data_config_path, "r", encoding="utf-8") as r:
        data_config = EasyDict(yaml.safe_load(r))

    # train config
    batch_size = training_args.per_device_train_batch_size
    gradient_accumulation_steps = training_args.gradient_accumulation_steps
    num_epochs = training_args.num_train_epochs

    # init dataset
    data_module = load_dataset(data_config.dataset)
    # init tokenizer
    tokenizer = data_module.tokenizer
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

    def collate_fn(batch):
        seqs_1, seqs_2, label_ids = tuple(zip(*batch))
        label_ids = torch.tensor(label_ids, dtype=torch.long)
        encoder_info_1 = tokenizer.batch_encode_plus(
            seqs_1, return_tensors="pt", padding=True
        )
        encoder_info_2 = tokenizer.batch_encode_plus(
            seqs_2, return_tensors="pt", padding=True
        )
        return {
            "inputs_1": encoder_info_1,
            "inputs_2": encoder_info_2,
            "labels": label_ids,
        }

    # init model
    model = AutoModelForSequenceClassification.from_pretrained(
        model_args.model_name_or_path,
        return_dict=True,
    )
    # fix model classifier hidden_size for ppi
    hidden_size = model.config.hidden_size * 2
    classifier = nn.Sequential(
        nn.Linear(hidden_size, hidden_size),
        nn.ReLU(),
        nn.Linear(hidden_size, 2),
    )
    setattr(model, "classifier", classifier)
    # freeze esm backbone
    for param in model.esm.parameters():
        param.requires_grad = False
    model.esm.eval()

    # change model forward method to support dual inputs
    def ppi_forward(self, inputs_1, inputs_2, labels, **kwargs):
        with torch.no_grad():
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

    print_trainable_parameters(model)

    # init custom optimizer for different groups of parameters
    base_lr = optim_args.base_lr
    beta1 = optim_args.beta1
    beta2 = optim_args.beta2
    weight_decay = optim_args.wdecay
    warmup_ratio = optim_args.optim_warmup_ratio
    optimizer = AdamW(
        [
            {"params": model.classifier.parameters(), "lr": base_lr},
        ],
        lr=base_lr,
        betas=(beta1, beta2),
        weight_decay=weight_decay,
    )
    total_batch_size = (
        batch_size * gradient_accumulation_steps * training_args.world_size
    )
    update_steps_per_epoch = math.ceil(len(train_dataset) / total_batch_size)
    warmup_steps = int(warmup_ratio * update_steps_per_epoch * num_epochs)
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
        # convert to tensor for torch metrics usage
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
    # saving final model
    trainer.save_model()
