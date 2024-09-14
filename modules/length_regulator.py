from typing import Tuple
import torch
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
            token_dropout_prob: float = 0.5,  # randomly drop out input tokens
            token_dropout_range: float = 0.5,  # randomly drop out input tokens
            n_codebooks: int = 1,  # number of codebooks
            quantizer_dropout: float = 0.0,  # dropout for quantizer
            f0_condition: bool = False,
            n_f0_bins: int = 512,
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

        self.mask_token = nn.Parameter(torch.zeros(1, channels))

        self.n_codebooks = n_codebooks
        if n_codebooks > 1:
            self.extra_codebooks = nn.ModuleList([
                nn.Embedding(codebook_size, channels) for _ in range(n_codebooks - 1)
            ])
        self.token_dropout_prob = token_dropout_prob
        self.token_dropout_range = token_dropout_range
        self.quantizer_dropout = quantizer_dropout

        if f0_condition:
            self.f0_embedding = nn.Embedding(n_f0_bins, channels)
            self.f0_condition = f0_condition
            self.n_f0_bins = n_f0_bins
            self.f0_bins = torch.arange(2, 1024, 1024 // n_f0_bins)
            self.f0_mask = nn.Parameter(torch.zeros(1, channels))
        else:
            self.f0_condition = False

    def forward(self, x, ylens=None, n_quantizers=None, f0=None):
        # apply token drop
        if self.training:
            n_quantizers = torch.ones((x.shape[0],)) * self.n_codebooks
            dropout = torch.randint(1, self.n_codebooks + 1, (x.shape[0],))
            n_dropout = int(x.shape[0] * self.quantizer_dropout)
            n_quantizers[:n_dropout] = dropout[:n_dropout]
            n_quantizers = n_quantizers.to(x.device)
            # decide whether to drop for each sample in batch
        else:
            n_quantizers = torch.ones((x.shape[0],), device=x.device) * (self.n_codebooks if n_quantizers is None else n_quantizers)
        if self.is_discrete:
            if self.n_codebooks > 1:
                assert len(x.size()) == 3
                x_emb = self.embedding(x[:, 0])
                for i, emb in enumerate(self.extra_codebooks):
                    x_emb = x_emb + (n_quantizers > i+1)[..., None, None] * emb(x[:, i+1])
                x = x_emb
            elif self.n_codebooks == 1:
                if len(x.size()) == 2:
                    x = self.embedding(x)
                else:
                    x = self.embedding(x[:, 0])
        # x in (B, T, D)
        mask = sequence_mask(ylens).unsqueeze(-1)
        x = F.interpolate(x.transpose(1, 2).contiguous(), size=ylens.max(), mode='nearest')
        if self.f0_condition:
            quantized_f0 = torch.bucketize(f0, self.f0_bins.to(f0.device))  # (N, T)
            drop_f0 = torch.rand(quantized_f0.size(0)).to(f0.device) < self.quantizer_dropout
            f0_emb = self.f0_embedding(quantized_f0)
            f0_emb[drop_f0] = self.f0_mask
            f0_emb = F.interpolate(f0_emb.transpose(1, 2).contiguous(), size=ylens.max(), mode='nearest')
            x = x + f0_emb
        out = self.model(x).transpose(1, 2).contiguous()
        olens = ylens
        return out * mask, olens
