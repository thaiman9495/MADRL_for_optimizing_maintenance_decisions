import torch
import torch.nn as nn


class VdnMixer(nn.Module):
    def __init__(self):
        super().__init__()

    @staticmethod
    def forward(qs):
        q_tot = torch.sum(qs, dim=1, keepdim=True)
        return q_tot

