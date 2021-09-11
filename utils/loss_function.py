import torch
import torch.nn as nn


class RMSPELoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, logit, target):

        logit = logit.float()
        target = target.float()

        loss = torch.square((target - logit) / target)

        return torch.sqrt(loss.mean())
