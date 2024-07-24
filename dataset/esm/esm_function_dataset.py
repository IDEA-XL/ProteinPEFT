import pickle
import torch

from transformers import EsmTokenizer
from ..data_interface import register_dataset
from ..lmdb_dataset import LMDBDataset

@register_dataset
class SolubilityDataset(LMDBDataset):
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

        return data["primary"], data['solubility']

    def __len__(self):
        return int(pickle.loads(self._get("num_examples")))

    def collate_fn(self, batch):
        seqs, target = zip(*batch)
        seq_tokens = self.tokenizer.batch_encode_plus(
            seqs, padding="max_length", max_length=self.max_length, 
            truncation=True, return_tensors="pt")
        ys = torch.tensor(target, dtype=torch.int64)
        inputs = {"input_ids": seq_tokens["input_ids"],}
        return inputs, ys