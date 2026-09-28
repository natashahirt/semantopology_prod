"""Structural loss function that bridges PyTorch models with NumPy/HIPS-autograd physics."""

import math
from typing import Any, Dict, Tuple

import autograd.numpy as np
import numpy as _np
import torch
import torch.nn.functional as F

from neural_structural_optimization.model.utils import batched_topo_loss
from neural_structural_optimization.physics import physics


class StructuralLoss(torch.autograd.Function):
    """A bridge that lets PyTorch models optimize against NumPy/HIPS-autograd physics."""
    
    @staticmethod
    def forward(ctx, logits: torch.Tensor, env: Any) -> torch.Tensor:
        """Forward pass: convert logits to NumPy and compute physics loss."""
        if not isinstance(logits, torch.Tensor):
            raise TypeError("logits must be a torch.Tensor")

        # Store shape and device/dtype to rebuild grads 
        ctx.input_shape = logits.shape
        ctx.device = logits.device
        ctx.dtype = logits.dtype
        ctx.env = env

        # Save detached tensor for backward 
        logits_cpu = logits.detach().cpu()  # keep original dtype; convert later
        ctx.save_for_backward(logits_cpu)

        # Convert to double NumPy for physics computation
        x_np = logits.detach().cpu().double().numpy()

        # Compute physics losses
        losses_np = batched_topo_loss(x_np, [env])  # -> shape (batch,)

        # Return torch tensor
        return torch.as_tensor(
            _np.asarray(losses_np), 
            dtype=ctx.dtype, 
            device=ctx.device
        )

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[torch.Tensor, None]:
        """Backward pass: compute gradients using autograd and map back to PyTorch."""
        (logits_cpu,) = ctx.saved_tensors
        env = ctx.env

        # Convert to NumPy double precision (shape: batch, H, W or similar)
        x_np = logits_cpu.double().numpy()

        # Fallback: use HIPS autograd through the legacy autograd physics
        import autograd  # type: ignore
        
        def scalar_objective(x_arr: np.ndarray) -> float:
            l = batched_topo_loss(x_arr, [env])  # -> (batch,)
            go = grad_output.detach().cpu().to(torch.float64).numpy()
            return np.sum(l * go)

        g_np = autograd.grad(scalar_objective)(x_np)

        # Map back to torch, match original dtype & device
        g = torch.from_numpy(g_np).to(ctx.device).to(ctx.dtype)
        return g, None  # no grad for env


class PhysicalDensity(torch.autograd.Function):
    """Map design variables onto the canonical physical density.

    Forward is `physics.physical_density(..., volume_constraint=True,
    cone_filter=True)` -- the same field `Environment.render` returns.
    Backward is a HIPS-autograd VJP of that map, not a Jacobian.

    The `import autograd` below is the third-party HIPS package, not
    `neural_structural_optimization.physics.autograd`.
    """

    @staticmethod
    def forward(ctx, logits: torch.Tensor, env: Any) -> torch.Tensor:
        if not isinstance(logits, torch.Tensor):
            raise TypeError("logits must be a torch.Tensor")
        ctx.device = logits.device
        ctx.dtype = logits.dtype
        ctx.input_shape = tuple(logits.shape)
        ctx.nely = int(env.args["nely"])
        ctx.nelx = int(env.args["nelx"])
        ctx.env = env
        logits_cpu = logits.detach().cpu()
        ctx.save_for_backward(logits_cpu)
        x2d = logits_cpu.double().numpy().reshape(ctx.nely, ctx.nelx)
        density = physics.physical_density(
            x2d, env.args, volume_constraint=True, cone_filter=True)
        out = torch.as_tensor(
            _np.asarray(density), dtype=ctx.dtype, device=ctx.device)
        return out.reshape(ctx.input_shape)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[torch.Tensor, None]:
        (logits_cpu,) = ctx.saved_tensors
        env = ctx.env
        nely, nelx = ctx.nely, ctx.nelx
        import autograd  # HIPS autograd; do not alias over topo_autograd

        x2d = logits_cpu.double().numpy().reshape(nely, nelx)
        go = grad_output.detach().cpu().to(torch.float64).numpy().reshape(
            nely, nelx)

        def scalar_density(x_arr):
            dens = physics.physical_density(
                x_arr, env.args, volume_constraint=True, cone_filter=True)
            return np.sum(dens * go)

        g_np = autograd.grad(scalar_density)(x2d)
        g = torch.from_numpy(_np.asarray(g_np)).to(ctx.device).to(ctx.dtype)
        return g.reshape(ctx.input_shape), None
