import torch
import numpy as np
import pickle
from typing import Union

from ..data_interface import register_dataset
from transformers import EsmTokenizer
from ..lmdb_dataset import LMDBDataset
from ..utils import pad_sequences


@register_dataset
class FoldDataset(LMDBDataset):
    def __init__(self,
                 tokenizer: str,
                 max_length: int = 1024,
                 **kwargs):
        """
        Args:
            tokenizer: Path to tokenizer
            max_length: Max length of sequence
            **kwargs:
        """
        super().__init__(**kwargs)
        self.tokenizer = EsmTokenizer.from_pretrained(tokenizer)
        self.max_length = max_length

    def __getitem__(self, index):
        entry = self._get(index)
        seq = entry['primary']
        fold_label = entry['fold_label']
        return seq, int(fold_label)

    def __len__(self):
        return int(self._get("num_examples"))
    
    def _get(self, key: Union[str, int]):
        return pickle.loads(self.txn.get(str(key).encode()))

    def collate_fn(self, batch):
        seqs, fold_labels = tuple(zip(*batch))

        encoder_info = self.tokenizer.batch_encode_plus(seqs, return_tensors='pt', padding=True)
        inputs = {"inputs": encoder_info}

        labels = torch.tensor(fold_labels, dtype=torch.long)
        labels = {"labels": labels}

        return inputs, labels
    

@register_dataset
class SecondaryStructureDataset(LMDBDataset):
    def __init__(self,
                 tokenizer: str,
                 max_length: int = 1024,
                 **kwargs):
        """
        Args:
            tokenizer: Path to tokenizer
            max_length: Max length of sequence
            **kwargs:
        """
        super().__init__(**kwargs)
        self.tokenizer = EsmTokenizer.from_pretrained(tokenizer)
        self.max_length = max_length

    def __getitem__(self, index):
        entry = self._get(index)
        seq = entry['primary']
        valid_mask = entry['valid_mask'][:self.max_length]
        ss3 = entry['ss3'][:self.max_length]
        ss3 = np.where(valid_mask, ss3, -1)
        return seq, ss3

    def __len__(self):
        return int(self._get("num_examples"))
    
    def _get(self, key: Union[str, int]):
        return pickle.loads(self.txn.get(str(key).encode()))

    def collate_fn(self, batch):
        seqs, ss3s = tuple(zip(*batch))

        encoder_info = self.tokenizer.batch_encode_plus(seqs, return_tensors='pt', padding=True)
        inputs = {"inputs": encoder_info}
        
        ss3s = pad_sequences(ss3s, -1)
        labels = torch.tensor(ss3s, dtype=torch.long)
        labels = {"labels": labels}

        return inputs, labels