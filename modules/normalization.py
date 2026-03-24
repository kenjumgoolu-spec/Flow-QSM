import torch
import torch.nn.functional as F
from torch import nn
from typing import Optional


class AdaGroupNorm(nn.Module):
    """
    An adaptive group normalization module
    """

    def __init__(
        self,
        num_groups: int,
        embedding_dim: int,
        out_dim: int,
        eps: float = 1e-6,
        act_fn: Optional[str] = None,
    ):
        super().__init__()
        self.num_groups = num_groups
        self.eps = eps
        if act_fn is None:
            self.act = None
        else:
            self.act = nn.ReLU()
        self.linear = nn.Linear(embedding_dim, out_dim * 2)

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        if self.act:
            emb = self.act(emb)
        emb = self.linear(emb)
        gamma, beta = emb.chunk(2, dim=1)
        gamma = gamma.view(*gamma.shape, 1, 1, 1)
        beta = beta.view(*beta.shape, 1, 1, 1)
        x = F.group_norm(x, self.num_groups, eps=self.eps)
        x = x * (1 + gamma) + beta
        return x


class SpatialNorm(nn.Module):
    """
    Spatially conditioned normalization as defined in https://arxiv.org/abs/2209.09002.

    Args:
        f_channels (`int`):
            The number of channels for input to group normalization layer, and output of the spatial norm layer.
        zq_channels (`int`):
            The number of channels for the quantized vector as described in the paper.
    """

    def __init__(
        self,
        f_channels: int,
        zq_channels: int,
    ):
        super().__init__()
        self.norm_layer = nn.GroupNorm(
            num_channels=f_channels, num_groups=32, eps=1e-6, affine=True
        )
        self.conv_y = nn.Conv2d(
            zq_channels, f_channels, kernel_size=1, stride=1, padding=0
        )
        self.conv_b = nn.Conv2d(
            zq_channels, f_channels, kernel_size=1, stride=1, padding=0
        )

    def forward(self, f: torch.Tensor, zq: torch.Tensor) -> torch.Tensor:
        f_size = f.shape[-2:]
        zq = F.interpolate(zq, size=f_size, mode="nearest")
        norm_f = self.norm_layer(f)
        new_f = norm_f * self.conv_y(zq) + self.conv_b(zq)
        return new_f
