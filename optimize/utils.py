# lint as python3
# Copyright 2019 Google LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Utility functions for optimization framework."""

from pathlib import Path
from typing import Union

import numpy as np
import torch
from absl import logging
from PIL import Image
from torchvision import transforms

from physics import physics


def load_venice_initial_image(
    image_path: Union[str, Path],
    height: int,
    width: int,
    invert_image: bool = True,
) -> torch.Tensor:
    """Load an initial design field the way the legacy Venice run does.

    Venice's chain (`models.py` lines 659-729) is RGB -> `Resize((height,
    width))` -> `ToTensor` -> `1 - t` when inverting -> `Grayscale(1)`, and it
    is reproduced here step for step:

    * The resize runs on the PIL image, not on a tensor. PIL's bilinear
      resampling is not the same operator as `F.interpolate`, so moving the
      resize after `ToTensor` would change the field.
    * The inversion happens BEFORE the grayscale reduction. Both are affine in
      the channels and a constant-alpha image commutes them, but the source is
      only near-grayscale, so the order is kept.
    * The result stays in [0, 1] PIXEL space. No `logit` is applied: the design
      parameter of `AdaptivePixelModel` is a density-like field that the
      physics backend squashes itself (see `model_ada`), so pushing the image
      through a logit would hand the solver a different field entirely.

    `ToTensor` yields float32, which is why the reference run's design
    parameter is float32 throughout -- including after the resolution
    schedule's upsamples, which preserve dtype. Matching that is a parity
    requirement, not a precision compromise.

    Args:
        image_path: path to the source image.
        height: rows of the target grid, i.e. the model's `nely`.
        width: columns of the target grid, i.e. the model's `nelx`.
        invert_image: subtract the image from 1, so that dark ink becomes
            dense material. True for the reference run.

    Returns:
        A float32 tensor of shape `(1, height, width)` with values in [0, 1].

    Raises:
        FileNotFoundError: if `image_path` does not exist.
    """
    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f'Initial image not found: {image_path}')

    image = Image.open(image_path).convert('RGB')
    image_tensor = transforms.ToTensor()(
        transforms.Resize((height, width))(image))
    if invert_image:
        image_tensor = 1 - image_tensor
    return transforms.Grayscale(num_output_channels=1)(image_tensor)


def init_weight_with_image(
    model,
    image_path: Union[str, Path],
    invert_image: bool = True,
) -> torch.Tensor:
    """Seed `model.z` with an image, at the model's CURRENT resolution.

    This is Venice's `MMSD.init_weight_with_image`, and the resolution it reads
    is load-bearing. The reference run starts its adaptive model COARSE -- at
    32x64, the bottom of a two-step schedule up to 128x256 -- so the image is
    resampled to 32x64 here and reaches the final grid only by being carried up
    through the schedule's upsamples. Loading it at the final resolution
    instead would give the run a different starting design and a different
    trajectory.

    Args:
        model: a model exposing a `z` parameter of shape `(1, height, width)`.
        image_path: path to the source image.
        invert_image: see `load_venice_initial_image`.

    Returns:
        The newly installed parameter.
    """
    height, width = model.z.shape[1:3]
    image = load_venice_initial_image(image_path, height, width, invert_image)
    model.z = torch.nn.Parameter(
        image.to(model.z.device), requires_grad=True)
    return model.z


DEFAULT_NEUTRAL_NOISE_AMP = 0.01


def init_weight_neutral(
    model,
    density: float,
    seed: int,
    *,
    noise_amp: float = DEFAULT_NEUTRAL_NOISE_AMP,
    union_load_sites: bool = True,
) -> torch.Tensor:
    """Seed ``model.z`` with uniform volume fraction plus tiny deterministic noise.

    No pre-authored frame. The field lives in ``[0, 1]`` pixel space at the
    model's *current* (possibly coarse) grid, matching the image seeder's
    contract. Optional load-application pixels are raised to 1 so a force
    never starts on void.

    Args:
        model: a model exposing ``z`` of shape ``(1, height, width)`` and
            ``env.args`` with ``mask`` / ``forces``.
        density: target volume fraction.
        seed: RNG seed for the symmetry-breaking noise.
        noise_amp: half-width of uniform noise around ``density``.
        union_load_sites: raise load-application elements to 1.

    Returns:
        The newly installed parameter.
    """
    if not 0.0 <= float(density) <= 1.0:
        raise ValueError(f'density must be in [0, 1], got {density}')
    if float(noise_amp) < 0.0:
        raise ValueError(f'noise_amp must be >= 0, got {noise_amp}')
    height, width = int(model.z.shape[-2]), int(model.z.shape[-1])
    generator = torch.Generator(device='cpu')
    generator.manual_seed(int(seed))
    noise = (
        torch.rand((1, height, width), generator=generator) * 2.0 - 1.0
    ) * float(noise_amp)
    field = torch.full((1, height, width), float(density)) + noise
    mask_np = np.asarray(model.env.args.get('mask', 1.0), dtype=np.float32)
    if mask_np.ndim == 0:
        mask_np = np.full((height, width), float(mask_np), dtype=np.float32)
    elif mask_np.ndim == 1:
        mask_np = mask_np.reshape(height, width)
    else:
        mask_np = np.squeeze(mask_np)
        if mask_np.ndim == 3 and mask_np.shape[0] == 1:
            mask_np = mask_np[0]
        if mask_np.ndim != 2:
            raise ValueError(
                f'design mask must be 2-D after squeeze, got {mask_np.shape}')
    mask = torch.as_tensor(mask_np, dtype=torch.float32)
    if tuple(mask.shape) != (height, width):
        mask = torch.nn.functional.interpolate(
            mask.view(1, 1, mask.shape[-2], mask.shape[-1]),
            size=(height, width),
            mode='nearest',
        ).view(height, width)
    field = (field * mask.view(1, height, width)).clamp(0.0, 1.0)
    if union_load_sites:
        from guidance.loss_sketch import (
            load_site_mask)
        nely = int(model.env.args['nely'])
        nelx = int(model.env.args['nelx'])
        sites = torch.as_tensor(
            load_site_mask(model.env.args['forces'], nely=nely, nelx=nelx),
            dtype=torch.float32,
        )
        if tuple(sites.shape) != (height, width):
            sites = torch.nn.functional.interpolate(
                sites.view(1, 1, sites.shape[-2], sites.shape[-1]),
                size=(height, width),
                mode='nearest',
            ).view(height, width)
        field = torch.maximum(field, sites.view(1, height, width) * mask)
    model.z = torch.nn.Parameter(
        field.to(device=model.z.device).contiguous(), requires_grad=True)
    return model.z


def get_variables(model) -> np.ndarray:
    """Get flattened array from PyTorch model parameters."""
    return np.concatenate([
        v.detach().cpu().numpy().ravel() 
        for v in model.parameters() if v.requires_grad])

def constrained_logits(init_model) -> np.ndarray:
    """Produce matching initial conditions with volume constraints applied.

    Returns the *pre-filter* density, because the pixel model this seeds treats
    its design variables as unfiltered densities and applies the cone filter
    itself. `physical_density` still solves the volume offset against the
    filtered density, so this is the same design the CNN objective saw, just
    viewed before the filter -- handing over the filtered field instead would
    filter it twice.
    """
    logits = init_model().detach().cpu().numpy().astype(np.float64).squeeze(axis=0)
    return physics.physical_density(
        logits, init_model.env.args, volume_constraint=True, cone_filter=False)

def repeat_to_shape(design: np.ndarray, height: int, width: int) -> np.ndarray:
    """Repeat a coarse design up to `height` x `width` by an integer factor.

    Progressive schedules record designs at several resolutions, and stacking
    them into one dataset needs them on a common grid. Nearest-neighbour repeat
    keeps every coarse element visible as the block it actually was, rather
    than interpolating detail the stage never had.
    """
    factor_y, factor_x = height // design.shape[0], width // design.shape[1]
    if factor_y == 1 and factor_x == 1:
        return design
    return np.repeat(np.repeat(design, factor_y, axis=0), factor_x, axis=1)

def cosine_warmup(t: int, T: int, warmup: float = 0.1, start: float = 1.0, end: float = 0.0) -> float:
    """Cosine from `start`→`end` after a linear warmup portion."""
    Tw = max(int(T * warmup), 1)
    if t < Tw:
        return start * (t + 1) / Tw
    tt = (t - Tw) / max(T - Tw, 1)
    return end + 0.5 * (start - end) * (1 + np.cos(np.pi * tt))

def ensure_array_size(x: np.ndarray, expected_size: int, name: str = "array") -> np.ndarray:
    """Ensure array has correct size with appropriate padding or truncation."""
    x = np.asarray(x).ravel()
    if x.size != expected_size:
        logging.warning(f'Reshaping {name} from {x.size} to {expected_size}')
        if x.size == 1:
            x = np.full(expected_size, float(x[0]))
        else:
            x = x[:expected_size]
            if x.size < expected_size:
                x = np.pad(x, (0, expected_size - x.size), mode='edge')
    return x

def match_mean_std_in_logit_space(z, ref_img):
    """
    Match mean and std of current image to reference image in logit space.
    
    Args:
        z: Current logits tensor to adjust
        ref_img: Reference image tensor (already in [0,1] range)
    """
    # Get reference statistics (ensure ref_img is in [0,1] range)
    ref_img = ref_img.clamp(0, 1)
    m_ref, s_ref = ref_img.mean(), ref_img.std().clamp_min(1e-6)
    
    # Get current statistics after sigmoid
    cur_img = torch.sigmoid(z)
    m_cur, s_cur = cur_img.mean(), cur_img.std().clamp_min(1e-6)
    
    # Compute affine transformation parameters
    # b: shift parameter (difference in logit means)
    b = torch.logit(m_ref.clamp(1e-6, 1-1e-6)) - torch.logit(m_cur.clamp(1e-6, 1-1e-6))
    
    # a: scale parameter (ratio of standard deviations)
    a = (s_ref / s_cur).clamp(0.25, 4.0)  # Prevent extreme scaling
    
    # Apply transformation: z_new = a * z + b
    z.mul_(a).add_(b).clamp_(-6, 6)  # Clamp to reasonable logit range

def calibrate_lambda_clip(model, logits, R=0.9, lam_bounds=(1e-3, 10.0), ortho=True):
    """
    Pick λ_clip so that ||∂L_struct/∂logits|| ≈ R · ||λ_clip ∂L_clip/∂logits||.
    R=0.9 gives semantics almost as much 'update energy' as structure (more dramatic).
    """
    if R is None:
        return None

    # Ensure we can take grads w.r.t. logits
    needs_req = not logits.requires_grad
    if needs_req: logits.requires_grad_(True)

    Ls = model.get_structural_loss(logits)
    gS, = torch.autograd.grad(Ls, logits, retain_graph=True)

    Lc = model.get_semantic_loss(logits)
    gC, = torch.autograd.grad(Lc, logits)

    if ortho:
        # Use only CLIP's component independent of structural direction
        denom = gS.norm().pow(2) + 1e-12
        gC = gC - (gC * gS).sum() / denom * gS

    nS = gS.norm().item()
    nC = gC.norm().item()
    lam = (nS / (nC + 1e-12)) * R
    lam = float(max(lam_bounds[0], min(lam_bounds[1], lam)))

    if needs_req: logits.requires_grad_(False)
    return lam
