from torchmetrics import Metric
import torch
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple

class ContactPredictMetric(Metric):
    def __init__(self, range: str='long', **kwargs):
        super().__init__(**kwargs)
        self.range = range
        self.add_state("pL", default=torch.tensor(0), dist_reduce_fx="mean")

    def update(self, preds: torch.Tensor, target: torch.Tensor, src_lengths:Optional[torch.Tensor]=None):
        # short, medium, long range
        if self.range == "local":
            contact_result = compute_precision_at_lx(preds, target, src_lengths, min_sep=3, max_sep=6)
        elif self.range == "short":
            contact_result = compute_precision_at_lx(preds, target, src_lengths, min_sep=6, max_sep=12)
        elif self.range == "medium":
            contact_result = compute_precision_at_lx(preds, target, src_lengths, min_sep=12, max_sep=24)
        elif self.range == "long":
            contact_result = compute_precision_at_lx(preds, target, src_lengths, min_sep=24)
        self.pL = contact_result

    def compute(self):
        return self.pL
    

def compute_precision_at_lx(
    prediction, 
    labels, 
    sequence_lengths, 
    x:int=1, 
    _ignore_index=-1, 
    min_sep:int=6, 
    max_sep:int=None
):
    with torch.no_grad():
        valid_mask = labels != _ignore_index
        seqpos = torch.arange(valid_mask.size(1), device=prediction.device)
        x_ind, y_ind = torch.meshgrid(seqpos, seqpos)
        # valid_mask &= ((y_ind - x_ind) >= 6).unsqueeze(0)
        valid_mask &= ((y_ind - x_ind) >= min_sep).unsqueeze(0)
        if max_sep is not None:
            valid_mask &= ((y_ind - x_ind) < max_sep).unsqueeze(0)
        if prediction.dim() == 4:
            probs = F.softmax(prediction, 3)[:, :, :, 1]
        elif prediction.dim() == 3:
            # probs = F.sigmoid(prediction)
            probs = prediction
        valid_mask = valid_mask.type_as(probs)
        correct = 0
        total = 0
        for length, prob, label, mask in zip(sequence_lengths, probs, labels, valid_mask):
            masked_prob = (prob * mask).view(-1)
            most_likely = masked_prob.topk(length // x, sorted=False)
            selected = label.view(-1).gather(0, most_likely.indices)
            correct += selected.sum().float()
            total += selected.numel()
        return correct / total