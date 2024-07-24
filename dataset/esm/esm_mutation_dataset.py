import pickle
import json
import torch

from transformers import EsmTokenizer
from ..data_interface import register_dataset
from ..lmdb_dataset import LMDBDataset


@register_dataset
class BetaLactamaseDataset(LMDBDataset):
    """
    Dataset that deals with mutation data.
    """

    def __init__(self, 
                 tokenizer: str,
                 max_length: int = 1024,
                 **kwargs):
        """

        Args: **kwargs: other arguments for LMDBDataset

        """
        super().__init__(**kwargs)
        self.tokenizer = EsmTokenizer.from_pretrained(tokenizer)
        self.max_length = max_length

    def __getitem__(self, index):
        data = pickle.loads(self._get(index))

        return data["primary"], float(data["scaled_effect1"])

    def __len__(self):
        return int(pickle.loads(self._get("num_examples")))

    def collate_fn(self, batch):
        seqs, target = zip(*batch)
        seq_tokens = self.tokenizer.batch_encode_plus(
            seqs, padding="max_length", max_length=self.max_length, 
            truncation=True, return_tensors="pt")
        ys = torch.tensor(target, dtype=torch.float32)
        inputs = {"input_ids": seq_tokens["input_ids"],}
        return inputs, ys


@register_dataset
class FluorescenceDataset(LMDBDataset):
    """
    Dataset that deals with mutation data.
    """

    def __init__(self, 
                 tokenizer: str,
                 max_length: int = 1024,
                 **kwargs):
        """

        Args: **kwargs: other arguments for LMDBDataset

        """
        super().__init__(**kwargs)
        self.tokenizer = EsmTokenizer.from_pretrained(tokenizer)
        self.max_length = max_length

    def __getitem__(self, index):
        data = pickle.loads(self._get(index))

        return data["primary"], float(data["log_fluorescence"])

    def __len__(self):
        return int(pickle.loads(self._get("num_examples")))

    def collate_fn(self, batch):
        seqs, target = zip(*batch)
        seq_tokens = self.tokenizer.batch_encode_plus(
            seqs, padding="max_length", max_length=self.max_length, 
            truncation=True, return_tensors="pt")
        ys = torch.tensor(target, dtype=torch.float32)
        inputs = {"input_ids": seq_tokens["input_ids"],}
        return inputs, ys
    

@register_dataset
class StabilityDataset(LMDBDataset):
    """
    Dataset that deals with mutation data.
    """

    def __init__(self, 
                 tokenizer: str,
                 max_length: int = 1024,
                 **kwargs):
        """

        Args: **kwargs: other arguments for LMDBDataset

        """
        super().__init__(**kwargs)
        self.tokenizer = EsmTokenizer.from_pretrained(tokenizer)
        self.max_length = max_length

    def __getitem__(self, index):
        data = pickle.loads(self._get(index))

        return data["primary"], data['stability_score'].item()

    def __len__(self):
        return int(pickle.loads(self._get("num_examples")))

    def collate_fn(self, batch):
        seqs, target = zip(*batch)
        seq_tokens = self.tokenizer.batch_encode_plus(
            seqs, padding="max_length", max_length=self.max_length, 
            truncation=True, return_tensors="pt")
        ys = torch.tensor(target, dtype=torch.float32)
        inputs = {"input_ids": seq_tokens["input_ids"],}
        return inputs, ys
    

@register_dataset
class AAVDataset(LMDBDataset):
    """
    Dataset that deals with mutation data.
    """

    def __init__(self, 
                 tokenizer: str,
                 max_length: int = 1024,
                 **kwargs):
        """

        Args: **kwargs: other arguments for LMDBDataset

        """
        super().__init__(**kwargs)
        self.tokenizer = EsmTokenizer.from_pretrained(tokenizer)
        self.max_length = max_length

    def __getitem__(self, index):
        try:
            data = json.loads(self._get(index))
        except:
            index = f"{index:09d}"
            data = json.loads(self._get(index))
        return data["seq"], data['target']

    def __len__(self):
        return int(self._get("length"))

    def collate_fn(self, batch):
        seqs, target = zip(*batch)
        seq_tokens = self.tokenizer.batch_encode_plus(
            seqs, padding="max_length", max_length=self.max_length, 
            truncation=True, return_tensors="pt")
        ys = torch.tensor(target, dtype=torch.float32)
        inputs = {"input_ids": seq_tokens["input_ids"],}
        return inputs, ys