"""Pixel-based structural optimization model."""

from typing import Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .model_base import Model
from .config import (
    DEFAULT_MAX_ANALYSIS_DIM,
    DEFAULT_TEMPERATURE,
    DEFAULT_EPSILON,
    DEFAULT_VOID_DENSITY,
    DEFAULT_PRESERVE_MEAN,
    DEFAULT_MAX_NEWTON_ITERATIONS,
    DEFAULT_MEAN_TOLERANCE,
    DEFAULT_STEP_TOLERANCE,
    DEFAULT_MEAN_CONVERGENCE,
)


class PixelModel(Model):
    """Direct pixel-wise optimization model."""
    
    def __init__(
        self, 
        structural_params: Optional[dict] = None, 
        clip_loss: Optional[object] = None, 
        seed: Optional[int] = None,
    ):
        super().__init__(structural_params=structural_params, clip_loss=clip_loss, seed=seed)
        
        # Initialize design parameters with volume fraction
        z_init = np.broadcast_to(
            self.env.args['volfrac'] * self.env.args['mask'], 
            self.shape
        )
        self.z = nn.Parameter(
            torch.as_tensor(z_init, dtype=torch.float32, device=self.device), 
            requires_grad=True
        )

    def forward(self) -> torch.Tensor:
        """Forward pass - return the design parameters directly."""
        return self.z

    def loss(self, logits: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Compute the total loss for the current design."""
        logits = self.forward()
        return self.get_total_loss(logits)

    @torch.no_grad()
    def upsample(
        self, 
        scale: int = 2, 
        preserve_mean: bool = DEFAULT_PRESERVE_MEAN, 
        max_dim: int = DEFAULT_MAX_ANALYSIS_DIM, 
        eps: float = DEFAULT_EPSILON, 
        void_density: float = DEFAULT_VOID_DENSITY
    ) -> None:
        """Upsample the design to higher resolution, preserving LOGITS in self.z."""
        # self.z MUST be logits when we enter
        z_coarse_log = self.z.detach()                 # logits at coarse res
        _, Hc, Wc = self.shape

        # image at coarse res (for stats only)
        x_coarse = torch.sigmoid(z_coarse_log)
        mask_coarse = self._get_mask_3d(Hc, Wc)
        x_coarse = (x_coarse * mask_coarse).clamp(0.0, 1.0)
        target_mean = (x_coarse.sum() / mask_coarse.sum().clamp_min(1)).item()

        # bump resolution
        self._update_structural_params(scale=scale)
        _, Hf, Wf = self.shape
        mask_fine = self._get_mask_3d(Hf, Wf)

        # build a coarse image with void filled, then go back to logits
        x_coarse_full = x_coarse + void_density * (1.0 - mask_coarse)
        z_coarse_full_log = torch.logit(x_coarse_full.clamp(eps, 1 - eps))

        # upsample in LOGIT space
        z_fine_log = F.interpolate(
            z_coarse_full_log.unsqueeze(0), size=(Hf, Wf), mode='bilinear', align_corners=False
        ).squeeze(0)

        # set void logits explicitly
        void_logit = torch.logit(
            torch.tensor(void_density, dtype=z_fine_log.dtype, device=z_fine_log.device).clamp(eps, 1 - eps)
        )
        z_fine_log = torch.where(mask_fine > 0, z_fine_log, torch.full_like(z_fine_log, void_logit))

        # optionally preserve mean by shifting logits by a scalar b (no extra sigmoid here)
        if preserve_mean:
            b = 0.0
            denom = mask_fine.sum().clamp_min(1)
            for _ in range(DEFAULT_MAX_NEWTON_ITERATIONS):
                x = torch.sigmoid(z_fine_log + b)
                m = (x * mask_fine).sum() / denom
                dm_db = (x * (1 - x) * mask_fine).sum() / denom
                step = ((target_mean - m) / (dm_db + 1e-8)).item()
                b += step
                if abs(step) < DEFAULT_STEP_TOLERANCE or abs((m - target_mean).item()) < DEFAULT_MEAN_CONVERGENCE:
                    break
            z_fine_log = z_fine_log + b

        # DO NOT sigmoid here; keep logits in the parameter.
        # If you want "temperature", apply it when converting logits->image in forward(), not here.
        # (E.g., image = torch.sigmoid(self.z / DEFAULT_TEMPERATURE))
        self._set_analysis_factor(max_dim=max_dim)
        z_fine_log = z_fine_log.clamp_(-6, 6)          # healthy gradient range
        self.z = torch.nn.Parameter(z_fine_log, requires_grad=True)
