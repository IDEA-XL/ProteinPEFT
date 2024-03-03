import sys
import yaml

from easydict import EasyDict
from utils.others import setup_seed
from utils.module_loader import *
from utils.others import trainable_parameters
import torch

from tqdm import tqdm

config_path = 'config/pretrain/debug.yaml'
with open(config_path, 'r', encoding='utf-8') as r:
    config = EasyDict(yaml.safe_load(r))

if config.setting.seed:
    setup_seed(config.setting.seed)
    
data_module = load_dataset(config.dataset)

# check data through iteration over data loader
for batch in tqdm(data_module.train_dataloader()):
    # check if contains NaN
    input_ids = batch[0]['inputs']['input_ids']
    attention_mask = batch[0]['inputs']['attention_mask']
    labels = batch[1]['labels']
    if torch.isnan(input_ids).any():
        print('input_ids contains NaN')
    if torch.isnan(attention_mask).any():
        print('attention_mask contains NaN')
    if torch.isnan(labels).any():
        print('labels contains NaN')