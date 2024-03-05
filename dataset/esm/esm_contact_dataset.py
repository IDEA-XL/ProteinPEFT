import torch
import json
import random
import numpy as np
import pickle

from ..data_interface import register_dataset
from transformers import EsmTokenizer
from ..lmdb_dataset import *
from ..utils import pad_sequences
from scipy.spatial.distance import pdist, squareform


@register_dataset
class EsmContactDataset(LMDBDataset):
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
        tokens = self.tokenizer.tokenize(seq)[:self.max_length]
        seq = " ".join(tokens)

        valid_mask = np.array(entry['valid_mask'])[:self.max_length]
        coords = np.array(entry['tertiary'])[:self.max_length]
        contact_map = np.less(squareform(pdist(coords)), 8.0).astype(np.int64)

        y_inds, x_inds = np.indices(contact_map.shape)
        invalid_mask = ~(valid_mask[:, None] & valid_mask[None, :])
        invalid_mask |= np.abs(y_inds - x_inds) < 6
        contact_map[invalid_mask] = -1

        return seq, contact_map, len(contact_map)

    def __len__(self):
        return int(self._get("num_examples"))
    
    def _get(self, key: Union[str, int]):
        return pickle.loads(self.txn.get(str(key).encode()))

    def collate_fn(self, batch):
        seqs, contact_maps, lengths = tuple(zip(*batch))

        encoder_info = self.tokenizer.batch_encode_plus(seqs, return_tensors='pt', padding=True)
        inputs = {"inputs": encoder_info}

        contact_maps = pad_sequences(contact_maps, -1) # ignore_index: -1
        targets = torch.tensor(contact_maps, dtype=torch.long)
        labels = {"targets": targets, "lengths": lengths}

        return inputs, labels