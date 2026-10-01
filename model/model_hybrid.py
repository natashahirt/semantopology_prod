"""Hybrid model combining CNN and Pixel-based optimization."""

from typing import Optional, Tuple
import torch
import torch.nn as nn
from .model_base import Model
from .model_cnn import CNNModel
from .model_pixel import PixelModel

class HybridModel(Model):
    """
    Hybrid model that routes structural gradients to a CNN 
    and semantic (CLIP) gradients to a Pixel grid.
    """
    
    def __init__(
        self, 
        cnn_model: CNNModel, 
        pixel_model: PixelModel
    ):
        # Sync with CNN's structural parameters
        super().__init__(
            structural_params=cnn_model.structural_params, 
            clip_loss=cnn_model.clip_loss, 
            seed=cnn_model.seed
        )
        self.cnn = cnn_model
        self.pixel = pixel_model
        
        # Cache for split-gradient routing
        self._c = None
        self._p = None

    def forward(self) -> torch.Tensor:
        """Combine CNN skeleton and Pixel motifs."""
        self._c = self.cnn()
        self._p = self.pixel()
        return self._c + self._p

    def get_structural_loss(self, logits: torch.Tensor) -> torch.Tensor:
        """Route structural gradients ONLY to the CNN."""
        # Use provided logits if we're in a calibration step (no grad_fn)
        # otherwise use cached routed versions for parameter updates
        if logits.requires_grad and logits.grad_fn is None:
            return super().get_structural_loss(logits)
            
        # We ignore the passed 'logits' and use the cached/routed version
        # logits_for_physics = CNN (live) + Pixel (static)
        combined = self._c + self._p.detach()
        return super().get_structural_loss(combined)

    def get_semantic_loss(self, logits: torch.Tensor) -> torch.Tensor:
        """Route semantic gradients ONLY to the Pixels."""
        # Use provided logits if we're in a calibration step (no grad_fn)
        # otherwise use cached routed versions for parameter updates
        if logits.requires_grad and logits.grad_fn is None:
            return super().get_semantic_loss(logits)

        # We ignore the passed 'logits' and use the cached/routed version
        # logits_for_clip = CNN (static) + Pixel (live)
        combined = self._c.detach() + self._p
        return super().get_semantic_loss(combined)

    def get_total_loss(self, *args, **kwargs) -> torch.Tensor:
        """
        Add a 'Motif Penalty' to prevent the pixels from nuking the CNN skeleton.
        """
        total = super().get_total_loss(*args, **kwargs)
        
        # L2 penalty on pixel logits forces them to be a 'decoration'
        # rather than a structural override.
        pixel_l2 = 0.01 * torch.mean(self.pixel.z**2)
        return total + pixel_l2

    @property
    def shape(self) -> Tuple[int, int, int]:
        return self.cnn.shape

    def _unfreeze_all(self):
        """Unfreeze both sub-models."""
        self.cnn._unfreeze_all()
        if hasattr(self.pixel, '_unfreeze_all'):
            self.pixel._unfreeze_all()

    @torch.no_grad()
    def upsample(self, scale: int = 2, **kwargs):
        """Upsample both the CNN and Pixel models."""
        # CNN upsampling (handles distillation)
        distill_weight = kwargs.get('distill_weight', 0.1)
        freeze_transferred = kwargs.get('freeze_transferred', True)
        self.cnn.upsample(scale=scale, freeze_transferred=freeze_transferred, distill_weight=distill_weight)
        
        # Pixel upsampling (handles grid resizing)
        self.pixel.upsample(scale=scale)
        
        # Re-sync HybridModel's structural params with the updated sub-models
        self.structural_params = self.cnn.structural_params
        self.env = self.cnn.env
        self.args = self.cnn.args
        self.mask = self.cnn.mask

