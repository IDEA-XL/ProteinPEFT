import torchmetrics
import torch
import numpy as np
import math

from torch.nn import Linear, ReLU
from torch.nn.functional import cross_entropy, linear
from ..model_interface import register_model
from .base import EsmBaseModel


@register_model
class EsmContactModel(EsmBaseModel):
    def __init__(self, **kwargs):
        """
        Args:
            num_labels: number of labels
            **kwargs: other arguments for EsmBaseModel
        """
        super().__init__(task="base", **kwargs)
        if self.freeze_backbone:
            # unfreeze the contact head
            for param in self.model.esm.contact_head.parameters():
                param.requires_grad = True

    def initialize_metrics(self, stage):
        metric_dict = {}
        for length in ["P@L", "P@L/2", "P@L/5"]:
            for range in ["long_range"]:
                metric_dict[f"{stage}_{range}_{length}"] = torchmetrics.Accuracy(ignore_index=-1)

        return metric_dict

    def forward(self, inputs):
        return self.model.predict_contacts(tokens=inputs["input_ids"], attention_mask=inputs["attention_mask"])

    def loss_func(self, stage, logits, labels):
        lengths = labels["lengths"]
        targets = labels["targets"].to(logits.device)
        targets_mask = targets != -1
        loss = cross_entropy(logits.flatten(), (targets * targets_mask).flatten().float())
        loss = loss * targets_mask.sum() / targets_mask.numel()

        # Iterate through all proteins and count accuracy
        length_dict = {"P@L": 1, "P@L/2": 2, "P@L/5": 5}
        # range_dict = ["short_range", "medium_range", "long_range"]
        range_dict = ["long_range"] # we only care about long range
        for pred_map, label_map, L in zip(logits.detach(), targets, lengths):
            x_inds, y_inds = np.indices(label_map.shape)
            for r in range_dict:
                if r == "short_range":
                    mask = (np.abs(y_inds - x_inds) < 6) | (np.abs(y_inds - x_inds) > 11)
                elif r == "medium_range":
                    mask = (np.abs(y_inds - x_inds) < 12) | (np.abs(y_inds - x_inds) > 23)
                else:
                    mask = np.abs(y_inds - x_inds) < 24

                mask = torch.from_numpy(mask)
                copy_label_map = label_map.clone()
                copy_label_map[mask] = -1
                
                # Mask the lower triangle
                mask = torch.triu(torch.ones_like(copy_label_map), diagonal=1)
                copy_label_map[mask == 0] = -1

                selector = copy_label_map != -1
                preds = pred_map[selector].float()
                labels = copy_label_map[selector]

                # probs = preds.softmax(dim=-1)[:, 1]
                probs = preds
                for k, v in length_dict.items():
                    l = min(math.ceil(L / v), (labels == 1).sum().item())

                    top_inds = torch.argsort(probs, descending=True)[:l]
                    top_labels = labels[top_inds]

                    if top_labels.numel() == 0:
                        continue

                    metric = f"{stage}_{r}_{k}"
                    self.metrics[stage][metric].update(top_labels, torch.ones_like(top_labels))

        if stage == "train":
            log_dict = self.get_log_dict("train")
            log_dict["train_loss"] = loss
            self.log_info(log_dict)

            # Reset train metrics
            self.reset_metrics("train")

        return loss

    def test_epoch_end(self, outputs):
        log_dict = self.get_log_dict("test")
        log_dict["test_loss"] = torch.cat(self.all_gather(outputs), dim=-1).mean()

        print(log_dict)
        self.log_info(log_dict)

        self.reset_metrics("test")

    def validation_epoch_end(self, outputs):
        log_dict = self.get_log_dict("valid")
        log_dict["valid_loss"] = torch.cat(self.all_gather(outputs), dim=-1).mean()

        self.log_info(log_dict)
        self.reset_metrics("valid")
        self.check_save_condition(log_dict["valid_long_range_P@L/5"], mode="max")