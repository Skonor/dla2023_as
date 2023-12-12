import torch.nn as nn
import torch
from torch import Tensor


class WeightedCELossWrapper(nn.Module):
    def __init__(self, spoof_coef=1., bonafide_coef=9.0):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(weight = torch.tensor([bonafide_coef, spoof_coef]))

    def forward(self, logits, is_spoofed, **batch) -> Tensor:
        return self.ce(logits, is_spoofed)
