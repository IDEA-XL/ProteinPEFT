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
            contact_result = compute_precision_at_lx(preds, target, src_lengths)
        elif self.range == "short":
            contact_result = compute_precision_at_lx(preds, target, src_lengths)
        elif self.range == "medium":
            contact_result = compute_precision_at_lx(preds, target, src_lengths)
        elif self.range == "long":
            contact_result = compute_precision_at_lx(preds, target, src_lengths)
        self.pL = contact_result

    def compute(self):
        return self.pL
    

def compute_precision_at_lx(prediction, labels, sequence_lengths, x:int=1, _ignore_index=-1):
    with torch.no_grad():
        valid_mask = labels != _ignore_index
        seqpos = torch.arange(valid_mask.size(1), device=prediction.device)
        x_ind, y_ind = torch.meshgrid(seqpos, seqpos)
        valid_mask &= ((y_ind - x_ind) >= 6).unsqueeze(0)
        if prediction.dim() == 4:
            probs = F.softmax(prediction, 3)[:, :, :, 1]
        elif prediction.dim() == 3:
            probs = F.sigmoid(prediction)
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
    
    
# def compute_precisions(
#     predictions: torch.Tensor,
#     targets: torch.Tensor,
#     src_lengths: Optional[torch.Tensor] = None,
#     minsep: int = 6,
#     maxsep: Optional[int] = None,
#     override_length: Optional[int] = None,  # for casp
# ):
#     """
#     ref https://github.com/facebookresearch/esm/blob/main/examples/contact_prediction.ipynb
#     contact_ranges = [
#         ("local", 3, 6),
#         ("short", 6, 12),
#         ("medium", 12, 24),
#         ("long", 24, None),
#     ]
#     """
#     if isinstance(predictions, np.ndarray):
#         predictions = torch.from_numpy(predictions)
#     if isinstance(targets, np.ndarray):
#         targets = torch.from_numpy(targets)
    
#     if src_lengths is not None and isinstance(src_lengths, Tuple):
#         src_lengths = torch.tensor(src_lengths, dtype=torch.long).to(predictions.device)
#     override_length = (targets[0, 0] >= 0).sum()

#     # Check sizes
#     if predictions.size() != targets.size():
#         raise ValueError(
#             f"Size mismatch. Received predictions of size {predictions.size()}, "
#             f"targets of size {targets.size()}"
#         )
#     device = predictions.device

#     batch_size, seqlen, _ = predictions.size()
#     seqlen_range = torch.arange(seqlen, device=device)

#     sep = seqlen_range.unsqueeze(0) - seqlen_range.unsqueeze(1)
#     sep = sep.unsqueeze(0)
#     valid_mask = sep >= minsep
#     valid_mask = valid_mask & (targets >= 0)  # negative targets are invalid

#     if maxsep is not None:
#         valid_mask &= sep < maxsep

#     if src_lengths is not None:
#         valid = seqlen_range.unsqueeze(0) < src_lengths.unsqueeze(1)
#         valid_mask &= valid.unsqueeze(1) & valid.unsqueeze(2)
#     else:
#         src_lengths = torch.full([batch_size], seqlen, device=device, dtype=torch.long)

#     predictions = predictions.masked_fill(~valid_mask, float("-inf"))

#     x_ind, y_ind = np.triu_indices(seqlen, minsep)
#     predictions_upper = predictions[:, x_ind, y_ind]
#     targets_upper = targets[:, x_ind, y_ind]

#     topk = seqlen if override_length is None else max(seqlen, override_length)
#     indices = predictions_upper.argsort(dim=-1, descending=True)[:, :topk]
#     topk_targets = targets_upper[torch.arange(batch_size).unsqueeze(1), indices]
#     if topk_targets.size(1) < topk:
#         topk_targets = F.pad(topk_targets, [0, topk - topk_targets.size(1)])

#     cumulative_dist = topk_targets.type_as(predictions).cumsum(-1)

#     gather_lengths = src_lengths.unsqueeze(1)
#     if override_length is not None:
#         gather_lengths = override_length * torch.ones_like(
#             gather_lengths, device=device
#         )

#     gather_indices = (
#         torch.arange(0.1, 1.1, 0.1, device=device).unsqueeze(0) * gather_lengths
#     ).type(torch.long) - 1
#     breakpoint()
#     binned_cumulative_dist = cumulative_dist.gather(1, gather_indices)
#     binned_precisions = binned_cumulative_dist / (gather_indices + 1).type_as(
#         binned_cumulative_dist
#     )

#     pl5 = binned_precisions[:, 1]
#     pl2 = binned_precisions[:, 4]
#     pl = binned_precisions[:, 9]
#     auc = binned_precisions.mean(-1)

#     return {"AUC": auc, "P@L": pl, "P@L2": pl2, "P@L5": pl5}