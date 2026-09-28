"""CNN-based structural optimization model."""

from typing import Callable, Optional, Sequence, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

from .model_base import Model
from .components import ConvStage, _init_dense_orthogonal
from .config import (
    DEFAULT_LATENT_SIZE,
    DEFAULT_DENSE_CHANNELS,
    DEFAULT_CONV_UPSAMPLE,
    DEFAULT_CONV_FILTERS,
    DEFAULT_OFFSET_SCALE,
    DEFAULT_KERNEL_SIZE,
    DEFAULT_NEG_SLOPE,
    DEFAULT_CONV_INITIALIZER,
    DEFAULT_FREEZE_TRANSFERRED,
    DEFAULT_DISTILL_WEIGHT,
    DEFAULT_WARM_TAU,
)
from neural_structural_optimization.physics.physics import logit


class CNNModel(Model):
    """CNN-based optimization with upsampling stages."""
    
    def __init__(
        self,
        structural_params: Optional[dict] = None,
        latent_size: int = DEFAULT_LATENT_SIZE,
        dense_channels: int = DEFAULT_DENSE_CHANNELS,
        conv_upsample: Sequence[int] = DEFAULT_CONV_UPSAMPLE,
        conv_filters: Sequence[int] = DEFAULT_CONV_FILTERS,
        offset_scale: float = DEFAULT_OFFSET_SCALE,
        kernel_size: Union[int, Tuple[int, int]] = DEFAULT_KERNEL_SIZE,
        norm: Optional[Callable[[int], nn.Module]] = lambda C: nn.InstanceNorm2d(
            num_features=int(C), affine=False, track_running_stats=False
        ),
        conv_initializer: str = DEFAULT_CONV_INITIALIZER,
        neg_slope: float = DEFAULT_NEG_SLOPE,
        seed: Optional[int] = None,
        clip_loss: Optional[object] = None,
    ):
        super().__init__(structural_params=structural_params, clip_loss=clip_loss, seed=seed)

        # Validate inputs
        if len(conv_upsample) != len(conv_filters):
            raise ValueError("conv_upsample and conv_filters must have same length")
        if conv_filters[-1] != 1:
            raise ValueError("conv_filters[-1] must be 1 to produce (1,H,W) logits")
    
        # Store configuration
        self.structural_params = structural_params
        self.latent_size = int(latent_size)
        self.dense_channels = int(dense_channels)
        self.conv_upsample = conv_upsample
        self.conv_filters = conv_filters
        self.offset_scale = offset_scale
        self.kernel_size = kernel_size
        self.norm = norm
        self.conv_initializer = conv_initializer
        self.neg_slope = neg_slope

        # Initialize latent parameters
        if seed is not None:
            g = torch.Generator(device='cpu')
            g.manual_seed(int(seed))
            z_init = torch.randn(1, self.latent_size, generator=g) 
        else:
            z_init = torch.randn(1, self.latent_size)

        self.z = nn.Parameter(z_init)

        # Build network
        self._compute_base_sizes()
        self._build()

        # Initialize distillation buffers
        self.register_buffer("_distill_target", None, persistent=False)
        self.register_buffer("_rho_hi_logit", None, False)
        self._distill_scale = 1
        self._distill_weight = 0.0
        self._warm_tau = 0.0

    @property
    def shape(self) -> Tuple[int, int, int]:
        """Get the shape of the design grid."""
        return (1, int(self.env.args['nely']), int(self.env.args['nelx']))

    def _compute_base_sizes(self) -> None:
        """Compute base sizes for the network architecture."""
        total_up = 1
        for u in self.conv_upsample:
            total_up *= int(u)

        _, H, W = self.shape

        if H % total_up != 0 or W % total_up != 0:
            raise ValueError(f"Grid {H}x{W} too small, not divisible by Π(conv_upsample)={total_up}")

        self.h_base = H // total_up
        self.w_base = W // total_up

        # Compute sizes for each stage
        sizes = []
        ch = self.dense_channels
        h, w = self.h_base, self.w_base
        for u in self.conv_upsample:
            h *= int(u)
            w *= int(u)
            sizes.append((h, w))
        self.stage_sizes = sizes

    def _build(self) -> None:
        """Build the network architecture."""
        # Dense layer
        self.dense = nn.Linear(self.latent_size, self.h_base * self.w_base * self.dense_channels)
        _init_dense_orthogonal(
            self.dense, 
            self.latent_size, 
            self.h_base, 
            self.w_base, 
            self.dense_channels
        )        

        # Convolutional stages
        stages = []
        in_ch = self.dense_channels
        for i, (out_ch, target_hw) in enumerate(zip(self.conv_filters, self.stage_sizes)):
            stage_name = f"stage_{i:02d}_C{in_ch}->{out_ch}"
            stage = ConvStage(
                in_ch, out_ch,
                kernel_size=self.kernel_size,
                norm_factor=self.norm,
                offset_scale=self.offset_scale,
                neg_slope=self.neg_slope,
                conv_initializer=self.conv_initializer,
                stage_name=stage_name,
            )
            stages.append(stage)
            in_ch = out_ch
        self.stages = nn.ModuleList(stages)

    def forward(self) -> torch.Tensor:
        """Forward pass through the CNN architecture."""
        device_type = 'cuda' if torch.cuda.is_available() else 'cpu'
        with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
            z = self.z.float()
            x = self.dense(z).view(
                1, self.dense_channels, self.h_base, self.w_base
            ).contiguous(memory_format=torch.channels_last)

            # Process through convolutional stages
            for stage, target_hw in zip(self.stages, self.stage_sizes):
                x = stage(x, target_hw)

            # Apply warm-up if enabled
            if self._warm_tau > 0.0 and (self._rho_hi_logit is not None) and (self._rho_hi_logit.shape[-2:] == x.shape[-2:]):
                x = (1.0 - self._warm_tau) * x + self._warm_tau * self._rho_hi_logit[None, None]

        # Convert to float64 for physics computation
        return x[:, 0, :, :].to(torch.float32)

    def loss(self, logits: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Compute the total loss including physics and distillation losses."""
        if logits is None:
            logits = self.forward()

        total = self.get_total_loss(logits)

        # Add distillation loss if enabled
        if self._distill_weight > 0.0 and self._distill_target is not None and self._distill_scale >= 1:
            x = logits.to(torch.float32)
            if x.ndim == 3:
                x = x[None, None]
            if self._distill_scale > 1:
                x = F.avg_pool2d(x, kernel_size=self._distill_scale, stride=self._distill_scale)
            student_ds = x[0, 0]
            distill = F.mse_loss(student_ds, self._distill_target)
            total = total + float(self._distill_weight) * distill

        return total

    @torch.no_grad()
    def upsample(
        self, 
        scale: int, 
        *, 
        freeze_transferred: bool = DEFAULT_FREEZE_TRANSFERRED, 
        distill_weight: float = DEFAULT_DISTILL_WEIGHT
    ) -> None:
        """Upsample the model to higher resolution with knowledge distillation."""
        # Generate teacher predictions at old resolution
        old_logits = self.forward()[0].to(torch.float32).contiguous()  # (H_old, W_old)
        rho_lo = torch.sigmoid(old_logits)
        rho_hi = F.interpolate(
            rho_lo[None, None], 
            scale_factor=scale, 
            mode="nearest"
        )[0, 0]
        rho_hi = rho_hi.clamp_(0.01, 0.99)
        rho_hi_logit = logit(rho_hi)

        # Update problem parameters and rebuild network
        self._update_structural_params(scale=scale)
        ups = list(map(int, self.conv_upsample))
        ups[-1] *= scale

        self._rebuild(tuple(ups), rho_hi_logit, old_logits, scale, distill_weight, freeze_transferred)

    def _rebuild(
        self, 
        ups: Tuple[int, ...], 
        rho_hi_logit: torch.Tensor, 
        distill_target_logits: torch.Tensor, 
        scale: int, 
        distill_weight: float, 
        freeze_transferred: bool
    ) -> None:
        """Rebuild the network with new upsampling factors."""
        # Save current state
        prev_state_dict = self.state_dict()
        self.conv_upsample = tuple(int(u) for u in ups)
        self._compute_base_sizes()
        old_keys = set(prev_state_dict.keys())

        # Create new model with updated architecture
        new_self = CNNModel(
            structural_params=self.structural_params,
            latent_size=self.latent_size,
            dense_channels=self.dense_channels,
            conv_upsample=self.conv_upsample,
            conv_filters=self.conv_filters,
            offset_scale=self.offset_scale,
            kernel_size=self.kernel_size,
            norm=self.norm,
            conv_initializer=self.conv_initializer,
            neg_slope=self.neg_slope,
        )

        new_state_dict = new_self.state_dict()

        # Copy compatible parameters
        copy_map = {}
        for k, v in new_state_dict.items():
            if k in old_keys and prev_state_dict[k].shape == v.shape:
                copy_map[k] = prev_state_dict[k].clone()

        new_state_dict.update(copy_map)
        self.load_state_dict(new_state_dict)
        
        # Freeze transferred layers if requested
        def should_freeze(name: str) -> bool:
            if name == "z" or name.startswith(f"stages.{len(self.stages) - 1}."):  # keep last stage plastic
                return False
            return name in copy_map  # only freeze transferred early layers

        if freeze_transferred:
            for name, p in self.named_parameters():
                p.requires_grad_(not should_freeze(name))

        # Ensure at least one parameter is trainable
        if not any(p.requires_grad for p in self.parameters()):
            self.z.requires_grad_(True)
        
        # Set up distillation
        self.register_buffer("_rho_hi_logit", rho_hi_logit, persistent=False)
        self.register_buffer(
            "_distill_target", 
            distill_target_logits.detach().to(torch.float32), 
            persistent=False
        )
        self._distill_scale = int(scale)
        self._distill_weight = float(distill_weight)
        self._warm_tau = DEFAULT_WARM_TAU

    def _unfreeze_all(self) -> None:
        """Unfreeze all parameters and disable warm-up."""
        for p in self.parameters():
            p.requires_grad_(True)
        self._warm_tau = 0.0
