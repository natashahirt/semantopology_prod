"""Reusable model components for neural structural optimization."""

import math
from typing import Callable, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F


class IdentityNorm(nn.Module):
    """Identity normalization layer that passes input unchanged."""
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


class AddOffset(nn.Module):
    """Per-channel learnable offset, scaled by `scale`."""
    
    def __init__(self, channels: int, scale: float = 10.0):
        super().__init__()
        self.scale = float(scale)
        self.offset = nn.Parameter(torch.zeros(channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add scaled offset to input tensor."""
        return x + self.scale * self.offset.view(1, -1, 1, 1)


def _init_dense_orthogonal(
    dense: nn.Linear, 
    latent_size: int, 
    h_base: int, 
    w_base: int, 
    channels: int
) -> None:
    """Initialize dense layer with orthogonal weights."""
    fan_out = h_base * w_base * channels
    gain = math.sqrt(max(fan_out / latent_size, 1.0))
    nn.init.orthogonal_(dense.weight, gain=gain)
    if dense.bias is not None:
        nn.init.zeros_(dense.bias)


def _upsample_nn_to(x: torch.Tensor, size: Tuple[int, int]) -> torch.Tensor:
    """Upsample tensor to target size using nearest neighbor interpolation."""
    if x.shape[-2:] == size:
        return x
    return F.interpolate(x, size=size, mode='nearest')


class ConvStage(nn.Module):
    """One stage = activation -> UpSampling (nearest) -> normalization -> Conv2d -> AddOffset."""
    
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel_size: Union[int, Tuple[int, int]] = 5,
        norm_factor: Optional[Callable[[int], nn.Module]] = None,
        offset_scale: float = 10.0,
        neg_slope: float = 0.2,
        conv_initializer: str = "kaiming_fan_in",
        stage_name: str = "",
    ):
        super().__init__()
        self.act = nn.LeakyReLU(negative_slope=neg_slope, inplace=True)
        self.norm = (norm_factor(in_ch) if norm_factor is not None else IdentityNorm())
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, padding="same", bias=True)
        self.offset = AddOffset(out_ch, scale=offset_scale)
        
        # Initialize convolution weights
        if conv_initializer == "kaiming_fan_in":
            nn.init.kaiming_normal_(
                self.conv.weight, 
                a=neg_slope, 
                mode="fan_in", 
                nonlinearity="leaky_relu"
            )
        elif conv_initializer == "kaiming_fan_out":
            nn.init.kaiming_normal_(
                self.conv.weight, 
                a=neg_slope, 
                mode="fan_out", 
                nonlinearity="leaky_relu"
            )
        else:
            raise ValueError(f"Unknown conv_initializer: {conv_initializer}")
            
        if self.conv.bias is not None:
            nn.init.zeros_(self.conv.bias)

        self.stage_name = stage_name  # used by the adaptive transfer logic

    def forward(self, x: torch.Tensor, target_hw: Tuple[int, int]) -> torch.Tensor:
        """Forward pass through the convolutional stage."""
        x = self.act(x)
        x = _upsample_nn_to(x, target_hw)
        x = self.norm(x)
        x = self.conv(x)
        x = self.offset(x)
        return x
