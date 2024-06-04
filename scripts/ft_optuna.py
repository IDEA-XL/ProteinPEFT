import sys
import torch
sys.path.append('.')

import yaml
import argparse

from easydict import EasyDict
from utils.others import setup_seed
from utils.module_loader import *

import optuna

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', help="running configurations", type=str, required=True)
    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()
    with open(args.config, 'r', encoding='utf-8') as r:
        config = EasyDict(yaml.safe_load(r))
    
    if config.setting.seed:
        setup_seed(config.setting.seed)
        
    # set os environment variables
    for k, v in config.setting.os_environ.items():
        if v is not None and k not in os.environ:
            os.environ[k] = str(v)

        elif k in os.environ:
            # override the os environment variables
            config.setting.os_environ[k] = os.environ[k]
    
    # Only the root node will print the log
    if config.setting.os_environ.NODE_RANK != 0:
        config.Trainer.logger = False
    
    # init dataset
    data_module = load_dataset(config.dataset) 
    
    # trial over head_lr, lora_lr, alpha
    def objective(trial):
        head_lr = trial.suggest_float("head_lr", 5e-4, 5e-2)
        lora_lr = trial.suggest_float("lora_lr", 1e-4, 1e-1)
        alpha = trial.suggest_categorical("alpha", [8, 12]) 
        print(f"head_lr: {head_lr}, lora_lr: {lora_lr}, alpha: {alpha}")
        # in-place modify the config
        config.model.kwargs.lora_config.update({
            "lora_alpha": alpha
        })
        config.model.lr_scheduler_kwargs.update({
            "head_lr": head_lr,
            "lora_lr": lora_lr
        })
        # set up model
        model = load_model(config.model)
        # init trainer
        trainer = load_trainer(config)
        # start fitting
        trainer.fit(model, datamodule=data_module)
        return max(model.val_history)
    
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=20)
    best_trial = study.best_trial
    print("Best hyperparameters: {}".format(best_trial.params))