from typing import Tuple
import torch.nn as nn
from torch.nn import functional as F
from modules.commons import sequence_mask


class InterpolateRegulator(nn.Module):
    def __init__(
            self,
            channels: int,
            sampling_ratios: Tuple,
            is_discrete: bool = False,
            codebook_size: int = 1024, # for discrete only
            out_channels: int = None,
            groups: int = 1,
    ):
        super().__init__()
        self.sampling_ratios = sampling_ratios
        out_channels = out_channels or channels
        model = nn.ModuleList([])
        if len(sampling_ratios) > 0:
            for _ in sampling_ratios:
                module = nn.Conv1d(channels, channels, 3, 1, 1)
                norm = nn.GroupNorm(groups, channels)
                act = nn.Mish()
                model.extend([module, norm, act])
        model.append(
            nn.Conv1d(channels, out_channels, 1, 1)
        )
        self.model = nn.Sequential(*model)
        self.embedding = nn.Embedding(codebook_size, channels)
        self.is_discrete = is_discrete

    def forward(self, x, ylens=None):
        if self.is_discrete:
            x = self.embedding(x)
        # x in (B, T, D)
        mask = sequence_mask(ylens).unsqueeze(-1)
        x = F.interpolate(x.transpose(1, 2).contiguous(), size=ylens.max(), mode='nearest')
        out = self.model(x).transpose(1, 2).contiguous()
        olens = ylens
        return out * mask, olens
