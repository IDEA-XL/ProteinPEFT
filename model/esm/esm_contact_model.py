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


# class EsmContactBinaryPredictonHead(nn.Module):
#     """Performs symmetrization, apc, and computes a logistic regression on the output features
#        Modified from TAPE https://github.com/songlab-cal/tape/blob/master/tape/models/modeling_utils.py#L843 
#     """

#     def __init__(
#         self,
#         in_features: int,
#         bias=True,
#         eos_idx: int = 2,
#     ):
#         super().__init__()
#         self.in_features = in_features
#         self.eos_idx = eos_idx
#         self.predict = nn.Linear(in_features, 2, bias)

#     def forward(self, tokens, attentions):
#         # attentions: (B, layer*n_head, L+2, L+2)
#         # remove eos token attentions
#         eos_mask = tokens.ne(self.eos_idx).to(attentions)
#         eos_mask = eos_mask.unsqueeze(1) * eos_mask.unsqueeze(2)
#         attentions = attentions * eos_mask[:, None, None, :, :]
#         attentions = attentions[..., :-1, :-1]
#         # remove cls token attentions
#         attentions = attentions[..., 1:, 1:]
#         batch_size, layers, heads, seqlen, _ = attentions.size()
#         attentions = attentions.view(batch_size, layers * heads, seqlen, seqlen)

#         # features: batch x channels x tokens x tokens (symmetric)
#         attentions = attentions.to(
#             self.regression.weight.device
#         )  # attentions always float32, may need to convert to float16
#         attentions = average_product_correct(symmetrize(attentions))
#         attentions = attentions.permute(0, 2, 3, 1)
#         return self.regression(attentions) # (B, L, L, 2)


@register_model
class EsmContactModel(EsmBaseModel):
    def __init__(self, **kwargs):
        """
        Args:
            **kwargs: other arguments for EsmBaseModel
        """
        super().__init__(task="contact", **kwargs)
        # If backbone is frozen, we only tune the contact head
        self.model.esm.contact_head = EsmContactBinaryPredictonHead(
            in_features=self.model.esm.config.num_hidden_layers * self.model.esm.config.num_attention_heads,
            bias=True,
            eos_idx=self.tokenizer.eos_token_id,
        )
        if self.freeze_backbone:
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
        loss = F.binary_cross_entropy_with_logits(logits, targets.float(), reduction="none")
        mask = targets != -1
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