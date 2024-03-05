import torch
from torch import nn
import torch.nn.functional as F 
from transformers.models.esm.modeling_esm import average_product_correct, symmetrize

from ..model_interface import register_model
from .base import EsmBaseModel
from ..metrics import ContactPredictMetric

class EsmContactBinaryPredictonHead(nn.Module):
    """Performs symmetrization, apc, and computes a logistic regression on the output features
       Modified from esm 
    """

    def __init__(
        self,
        in_features: int,
        bias=True,
        eos_idx: int = 2,
    ):
        super().__init__()
        self.in_features = in_features
        self.eos_idx = eos_idx
        self.regression = nn.Linear(in_features, 1, bias)
        nn.init.zeros_(self.regression.bias)
        nn.init.kaiming_normal_(self.regression.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, tokens, attentions):
        # remove eos token attentions
        eos_mask = tokens.ne(self.eos_idx).to(attentions)
        eos_mask = eos_mask.unsqueeze(1) * eos_mask.unsqueeze(2)
        attentions = attentions * eos_mask[:, None, None, :, :]
        attentions = attentions[..., :-1, :-1]
        # remove cls token attentions
        attentions = attentions[..., 1:, 1:]
        batch_size, layers, heads, seqlen, _ = attentions.size()
        attentions = attentions.view(batch_size, layers * heads, seqlen, seqlen)

        # features: batch x channels x tokens x tokens (symmetric)
        attentions = attentions.to(
            self.regression.weight.device
        )  # attentions always float32, may need to convert to float16
        attentions = average_product_correct(symmetrize(attentions))
        attentions = attentions.permute(0, 2, 3, 1)
        return self.regression(attentions).squeeze(3)


@register_model
class EsmContactModel(EsmBaseModel):
    def __init__(self, **kwargs):
        """
        Args:
            **kwargs: other arguments for EsmBaseModel
        """
        super().__init__(task="contact", **kwargs)
        if self.freeze_backbone:
            nn.init.zeros_(self.model.esm.contact_head.regression.bias)
            nn.init.kaiming_normal_(self.model.esm.contact_head.regression.weight, mode='fan_out', nonlinearity='relu')

            for param in self.model.esm.contact_head.parameters():
                param.requires_grad = True
        
    def initialize_metrics(self, stage):
        return {f"{stage}_long_pl": ContactPredictMetric(range="long")}

    def forward(self, inputs, coords=None):
        if coords is not None:
            inputs = self.add_bias_feature(inputs, coords)

        logits = self.model.predict_contacts(
            tokens=inputs["input_ids"], 
            attention_mask=inputs["attention_mask"]
        ) # (B, L, L, 1)
        return logits

    def loss_func(self, stage, logits, labels):
        targets = labels["targets"]
        lengths = labels["lengths"]
        mask = targets != -1
        targets_masked = targets * mask
        loss = F.binary_cross_entropy(logits, targets_masked.float(), reduction="none")
        loss = (loss * mask.float()).sum() / mask.sum()

        # Update metrics
        for metric in self.metrics[stage].values():
            metric.update(logits.detach(), targets, lengths)

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
        self.check_save_condition(log_dict["valid_long_pl"], mode="max")