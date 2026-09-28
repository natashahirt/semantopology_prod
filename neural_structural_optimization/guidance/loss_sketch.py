"""Sketch occupancy and the spatial mass-prior loss (Stage 6).

A sketch is not a silhouette to match and not a CLIP image. It is a map of
where the *global* volume budget is encouraged to sit: dark ink becomes
occupancy 1, paper and faint construction drop out under a threshold, and a
differentiable loss pulls the canonical physical density toward that map.
Load-application pixels are unioned in so a force never sits on punished
void (that is catastrophically compliant and snaps the run back to the
default frame). Physics still holds ``volfrac``; this term only
redistributes the budget.

The occupancy loader uses PIL only (no torchvision, no CLIP) so a config-only
import path can stay free of those packages. The loss itself is a few tensor
ops on an already-computed density.

A motif-layout scaffold is the same occupancy prior, distilled from a raw
CLIP teacher rather than a user sketch: threshold teacher ink, then take
bounded distance envelopes at physically derived scales. Extraction is
deterministic and CLIP-free. Load-site union still happens at apply time.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence, Union

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

# Repo-relative directory of the Stage 6 corpus. Copied from the Venice
# sibling so a hardfork clone does not read across repositories at runtime.
SKETCH_DIR = Path('script/resources/input_images/sketches')
SKETCH_CORPUS = ('1.jpg', '3.jpg', '6.jpg', '9.jpg', '11.jpg', '12.jpg')

# After invert, paper sits near 0 and ink near 1. Measured on the six-sketch
# corpus: median ink is 0.00-0.13, so 0.40 drops faint grids / ghost
# rectangles while keeping occupancy in roughly 0.13-0.32, around a 0.3
# volfrac. Not Otsu -- a fixed cut so two machines agree.
DEFAULT_OCCUPANCY_THRESHOLD = 0.40


def load_sketch_occupancy(
    path: Union[str, Path],
    height: int,
    width: int,
    *,
    invert: bool = True,
    threshold: float = DEFAULT_OCCUPANCY_THRESHOLD,
    blur_sigma: float = 0.0,
) -> np.ndarray:
    """Turn a sketch image into a ``(height, width)`` occupancy map in ``[0, 1]``.

    Pipeline: grayscale ? resize to the design grid ? invert so ink is 1 ?
    threshold so paper and faint construction become 0 ? optional Gaussian
    blur so the prior is not a hard stencil.

    Args:
        path: Sketch file (JPEG or PNG).
        height: Target rows (``nely``).
        width: Target columns (``nelx``).
        invert: Dark ink becomes occupancy 1. True for pencil-on-paper.
        threshold: Cut on the inverted intensities. Below this is paper.
        blur_sigma: Gaussian sigma in *target-grid* pixels. 0 keeps the
            binary map.

    Returns:
        float32 array of shape ``(height, width)``.

    Raises:
        FileNotFoundError: if ``path`` does not exist.
        ValueError: if height/width are not positive or threshold is outside
            ``[0, 1]``.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f'Sketch not found: {path}')
    if height < 1 or width < 1:
        raise ValueError(f'occupancy grid must be positive, got {height}x{width}')
    if not 0.0 <= float(threshold) <= 1.0:
        raise ValueError(f'threshold must be in [0, 1], got {threshold!r}')

    # PIL resize takes (width, height). Bilinear matches the Venice image
    # seed's resampling family; occupancy is a prior, not a parity pin, so
    # this does not have to be the same operator as F.interpolate.
    image = Image.open(path).convert('L').resize((width, height), Image.BILINEAR)
    gray = np.asarray(image, dtype=np.float64) / 255.0
    ink = (1.0 - gray) if invert else gray
    occupancy = (ink >= float(threshold)).astype(np.float64)
    if blur_sigma and float(blur_sigma) > 0.0:
        from scipy.ndimage import gaussian_filter
        occupancy = np.clip(gaussian_filter(occupancy, sigma=float(blur_sigma)), 0.0, 1.0)
    return occupancy.astype(np.float32)


def load_site_mask(
    forces: np.ndarray,
    nely: int,
    nelx: int,
) -> np.ndarray:
    """Elements that correspond to nodes with a nonzero force (Fx or Fy).

    Loads live on nodes ``(nelx+1, nely+1, 2)``; density lives on elements
    ``(nely, nelx)``. Each loaded node ``(ix, iy)`` maps to one element by
    clamping into the grid. No extra radius: a full-width floor becomes one
    row, a point load becomes one pixel. Accepts raveled forces as stored on
    ``env.args['forces']``.
    """
    nely = int(nely)
    nelx = int(nelx)
    arr = np.asarray(forces, dtype=np.float64)
    expected = (nelx + 1) * (nely + 1) * 2
    if arr.size != expected:
        raise ValueError(
            f'forces size {arr.size} does not match ({nelx}+1)*({nely}+1)*2 '
            f'= {expected}')
    arr = arr.reshape(nelx + 1, nely + 1, 2)
    loaded = np.any(np.abs(arr) > 0.0, axis=2)
    ix, iy = np.nonzero(loaded)
    mask = np.zeros((nely, nelx), dtype=np.float32)
    if ix.size == 0:
        return mask
    mask[np.clip(iy, 0, nely - 1), np.clip(ix, 0, nelx - 1)] = 1.0
    return mask


def sketch_mass_prior_loss(
    density: torch.Tensor,
    occupancy: torch.Tensor,
    load_sites: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Fraction of physical mass that sits *off* the allowed template.

    The template is occupancy, optionally unioned with load-application
    pixels: ``allowed = max(occupancy, load_sites)``. Loads on void are
    catastrophically compliant, so those pixels must not be punished or the
    optimizer snaps back to the default frame. Between load sites, only the
    sketch is allowed.

    ``(density * (1 - allowed)).sum() / density.sum()``. Volume is already
    constrained by ``PhysicalDensity``.
    """
    dens = density.reshape(-1)
    occ = occupancy.reshape(-1)
    if dens.numel() != occ.numel():
        raise ValueError(
            f'density and occupancy must have the same number of cells, '
            f'got {tuple(density.shape)} vs {tuple(occupancy.shape)}')
    allowed = occ
    if load_sites is not None:
        sites = load_sites.reshape(-1)
        if sites.numel() != dens.numel():
            raise ValueError(
                f'density and load_sites must have the same number of cells, '
                f'got {tuple(density.shape)} vs {tuple(load_sites.shape)}')
        allowed = torch.maximum(occ, sites)
    mass = dens.sum().clamp_min(1e-12)
    return (dens * (1.0 - allowed)).sum() / mass


def teacher_ink_from_raw(field: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
    """Clamp a teacher design to ``[0, 1]`` ink (material = 1).

    Venice displays raw logits by clamping after a short-side resize. The
    layout scaffold is extracted on the design grid, so this is the same
    clamp without that resize. Unbounded logits keep their decorative
    high-value ink; a physical-density field is unchanged.
    """
    arr = _as_numpy(field).astype(np.float64)
    if arr.ndim == 3:
        arr = np.squeeze(arr, axis=0)
    if arr.ndim != 2:
        raise ValueError(
            f'teacher field must be 2-D after squeeze, got {arr.shape}')
    return np.clip(arr, 0.0, 1.0).astype(np.float32)


def rank_ink_from_raw(field: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
    """Map a field to ``[0, 1]`` by rank so a cut is a percentile of pixels.

    CLIP-only dreams are unbounded logits: clamping them to ``[0, 1]`` can
    mark a third of the grid as solid ink, and a storey-scale envelope then
    fills the facade. Rank ink makes ``threshold=0.7`` mean "the top 30%",
    independent of logit scale. Ties share a rank.
    """
    arr = _as_numpy(field).astype(np.float64)
    if arr.ndim == 3:
        arr = np.squeeze(arr, axis=0)
    if arr.ndim != 2:
        raise ValueError(
            f'teacher field must be 2-D after squeeze, got {arr.shape}')
    flat = arr.reshape(-1)
    order = np.argsort(flat, kind='mergesort')
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.linspace(0.0, 1.0, flat.size, dtype=np.float64)
    return ranks.reshape(arr.shape).astype(np.float32)


def _ink_from_field(
    field: Union[np.ndarray, torch.Tensor],
    ink_mode: str,
) -> np.ndarray:
    mode = str(ink_mode).lower()
    if mode == 'clamp':
        return teacher_ink_from_raw(field)
    if mode == 'rank':
        return rank_ink_from_raw(field)
    raise ValueError(f"ink_mode must be 'clamp' or 'rank', got {ink_mode!r}")


def motif_layout_envelope_sigmas(
    height: int,
    scale_fracs: Sequence[float],
    envelope_sigma_frac: float,
) -> tuple[float, ...]:
    """Gaussian/distance-envelope sigmas in *grid* pixels.

    Each scale fraction is a share of elevation height (building / storey /
    member), matching :func:`physical_motif_scale_fracs`. ``sigma`` is
    ``envelope_sigma_frac * frac * height``.
    """
    if height < 1:
        raise ValueError(f'height must be >= 1, got {height}')
    if float(envelope_sigma_frac) < 0.0:
        raise ValueError(
            f'envelope_sigma_frac must be >= 0, got {envelope_sigma_frac}')
    if not scale_fracs:
        raise ValueError('scale_fracs must be a non-empty sequence')
    sigmas = []
    for frac in scale_fracs:
        frac = float(frac)
        if not 0.0 < frac <= 1.0:
            raise ValueError(
                f'scale_fracs must be elevation fractions in (0, 1], got '
                f'{tuple(scale_fracs)}')
        sigmas.append(float(envelope_sigma_frac) * frac * float(height))
    return tuple(sigmas)


def _soft_distance_envelope(ink: np.ndarray, sigma: float) -> np.ndarray:
    """``[0, 1]`` falloff from binary ink. ``sigma=0`` is the ink itself."""
    binary = np.asarray(ink, dtype=bool)
    if sigma <= 0.0:
        return binary.astype(np.float64)
    from scipy.ndimage import distance_transform_edt
    dist = distance_transform_edt(np.logical_not(binary))
    return np.exp(-dist / float(sigma))


def motif_layout_scaffold(
    teacher_field: Union[np.ndarray, torch.Tensor],
    *,
    scale_fracs: Sequence[float],
    threshold: float = 0.5,
    envelope_sigma_frac: float = 0.25,
    combine: str = 'max',
    ink_mode: str = 'clamp',
) -> np.ndarray:
    """Soft multiscale occupancy envelope from teacher ink.

    Threshold the teacher field, then build a bounded distance envelope at
    each physically derived scale and combine them. This is a layout prior,
    not a silhouette match. Deterministic; no CLIP. Does not union load
    sites -- that happens when the mass prior is applied.

    Args:
        teacher_field: Raw design or density, any 2-D (or ``(1, H, W)``) array.
        scale_fracs: Elevation fractions in ``(0, 1]``. Empty is rejected;
            resolve :func:`physical_motif_scale_fracs` at the caller.
        threshold: Cut on ink in ``[0, 1]``.
        envelope_sigma_frac: Envelope width as a fraction of each scale's
            pixel size (``frac * height``).
        combine: ``'max'`` (thicker union of scales) or ``'mean'``.
        ink_mode: ``'clamp'`` (default, density-like teachers) or ``'rank'``
            (unbounded CLIP dreams).

    Returns:
        float32 array of shape ``(H, W)`` in ``[0, 1]``.
    """
    if not 0.0 <= float(threshold) <= 1.0:
        raise ValueError(f'threshold must be in [0, 1], got {threshold!r}')
    combine = str(combine).lower()
    if combine not in ('max', 'mean'):
        raise ValueError(f"combine must be 'max' or 'mean', got {combine!r}")
    ink = _ink_from_field(teacher_field, ink_mode)
    height, width = ink.shape
    sigmas = motif_layout_envelope_sigmas(
        height, scale_fracs, envelope_sigma_frac)
    binary = ink >= float(threshold)
    if not np.any(binary):
        raise ValueError(
            f'teacher ink is empty after threshold={threshold}; '
            'cannot extract a layout scaffold')
    envelopes = [
        _soft_distance_envelope(binary, sigma) for sigma in sigmas]
    if combine == 'max':
        scaffold = np.maximum.reduce(envelopes)
    else:
        scaffold = np.mean(np.stack(envelopes, axis=0), axis=0)
    return np.clip(scaffold, 0.0, 1.0).astype(np.float32)


# Stage 6 allowed templates measured 0.64-0.83 mean occupancy. Target the
# middle of that band so a CLIP-dream scaffold is comparable, not sparser
# by construction and not a filled facade.
DEFAULT_SCAFFOLD_ALLOWED_MEAN = 0.75
# Stage 6 converged with spatial_mass_loss 0.31-0.60 against its template.
# The failed CLIP-teacher scaffold scored 0.0078 - tautological. A dream
# scaffold must beat this floor against a physics-layout proxy before a
# student run is worth spending.
DEFAULT_TAUTOLOGY_MIN_MASS_OFF = 0.25


def resample_field(
    field: Union[np.ndarray, torch.Tensor],
    height: int,
    width: int,
) -> np.ndarray:
    """Bilinear resample a 2-D field to ``(height, width)``. No clamp.

    Dream fields are unbounded design parameters, same contract as
    :func:`init_weight_from_teacher`. Identity when the shape already matches.
    """
    if height < 1 or width < 1:
        raise ValueError(f'target grid must be positive, got {height}x{width}')
    arr = _as_numpy(field).astype(np.float32)
    if arr.ndim == 3:
        arr = np.squeeze(arr, axis=0)
    if arr.ndim != 2:
        raise ValueError(f'field must be 2-D after squeeze, got {arr.shape}')
    if arr.shape == (height, width):
        return arr
    tensor = torch.as_tensor(arr)[None, None]
    out = F.interpolate(
        tensor, size=(height, width), mode='bilinear', align_corners=False)
    return out[0, 0].detach().cpu().numpy().astype(np.float32)


def motif_layout_threshold_for_allowed_mean(
    teacher_field: Union[np.ndarray, torch.Tensor],
    *,
    scale_fracs: Sequence[float],
    target_mean: float = DEFAULT_SCAFFOLD_ALLOWED_MEAN,
    envelope_sigma_frac: float = 0.25,
    combine: str = 'max',
    ink_mode: str = 'clamp',
    abs_tol: float = 0.03,
    max_iter: int = 24,
) -> tuple[float, np.ndarray]:
    """Binary-search the ink cut so ``scaffold.mean()`` hits ``target_mean``.

    Higher threshold -> sparser ink -> lower allowed area. Empty-ink thresholds
    are treated as too high. Raises if no cut lands within ``abs_tol`` of
    the target - that is a field that cannot become a Stage-6-like prior.
    ``envelope_sigma_frac=0`` with ``ink_mode='rank'`` is one elevation
    occupancy, no storey dilation.
    """
    if not 0.0 < float(target_mean) < 1.0:
        raise ValueError(
            f'target_mean must be in (0, 1), got {target_mean!r}')
    if float(abs_tol) <= 0.0:
        raise ValueError(f'abs_tol must be > 0, got {abs_tol!r}')
    lo, hi = 0.0, 1.0
    best: Optional[tuple[float, np.ndarray, float]] = None
    for _ in range(int(max_iter)):
        mid = 0.5 * (lo + hi)
        try:
            scaffold = motif_layout_scaffold(
                teacher_field,
                scale_fracs=scale_fracs,
                threshold=mid,
                envelope_sigma_frac=envelope_sigma_frac,
                combine=combine,
                ink_mode=ink_mode,
            )
        except ValueError:
            hi = mid
            continue
        mean = float(scaffold.mean())
        best = (mid, scaffold, mean)
        if abs(mean - float(target_mean)) <= float(abs_tol):
            return mid, scaffold
        if mean > float(target_mean):
            lo = mid
        else:
            hi = mid
    if best is None:
        raise ValueError(
            'could not extract a non-empty scaffold at any ink threshold')
    threshold, scaffold, mean = best
    if abs(mean - float(target_mean)) > float(abs_tol):
        raise ValueError(
            f'could not hit allowed_mean={float(target_mean):.3f} '
            f'(got {mean:.3f} at threshold={threshold:.3f})')
    return threshold, scaffold


def scaffold_spatial_mass_loss(
    density: Union[np.ndarray, torch.Tensor],
    scaffold: Union[np.ndarray, torch.Tensor],
    load_sites: Optional[Union[np.ndarray, torch.Tensor]] = None,
) -> float:
    """``sketch_mass_prior_loss`` as a detached scalar (the tautology gate)."""
    dens = torch.as_tensor(_as_numpy(density), dtype=torch.float32)
    occ = torch.as_tensor(_as_numpy(scaffold), dtype=torch.float32)
    sites = None
    if load_sites is not None:
        sites = torch.as_tensor(_as_numpy(load_sites), dtype=torch.float32)
    return float(sketch_mass_prior_loss(dens, occ, load_sites=sites).detach())


def init_weight_from_teacher(model, teacher_z) -> torch.Tensor:
    """Seed ``model.z`` from a teacher design at the model's current grid.

    Bilinear resample, no clamp: teacher logits are unbounded. Occupancy
    seeding clamps to ``[0, 1]`` because a sketch is ink; a teacher is a
    continuation of the same design parameter.
    """
    height, width = int(model.z.shape[-2]), int(model.z.shape[-1])
    field = _as_numpy(teacher_z).astype(np.float32)
    if field.ndim == 3:
        field = np.squeeze(field, axis=0)
    if field.ndim != 2:
        raise ValueError(
            f'teacher design must be 2-D after squeeze, got {field.shape}')
    tensor = torch.as_tensor(field, dtype=torch.float32)
    if tensor.shape[-2] != height or tensor.shape[-1] != width:
        tensor = torch.nn.functional.interpolate(
            tensor.view(1, 1, tensor.shape[-2], tensor.shape[-1]),
            size=(height, width),
            mode='bilinear',
            align_corners=False,
        ).view(height, width)
    image = tensor.to(device=model.z.device)
    if image.ndim == 2:
        image = image.unsqueeze(0)
    model.z = torch.nn.Parameter(image.contiguous(), requires_grad=True)
    return model.z


def _as_single_channel_field(field: torch.Tensor) -> torch.Tensor:
    """Return ``field`` as ``(N, 1, H, W)`` without breaking gradients."""
    if field.ndim == 2:
        return field[None, None]
    if field.ndim == 3:
        return field[:, None]
    if field.ndim == 4 and field.shape[1] == 1:
        return field
    raise ValueError(
        f'motif field must be (H,W), (N,H,W), or (N,1,H,W), got '
        f'{tuple(field.shape)}')


def _orientation_kernels(*, device, dtype) -> torch.Tensor:
    """Sobel-like horizontal, vertical, and two diagonal edge filters."""
    kernels = torch.tensor(
        [
            [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
            [[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
            [[0, 1, 2], [-1, 0, 1], [-2, -1, 0]],
            [[-2, -1, 0], [-1, 0, 1], [0, 1, 2]],
        ],
        device=device,
        dtype=dtype,
    )
    kernels = kernels / kernels.abs().sum(dim=(1, 2), keepdim=True)
    return kernels[:, None]


def _normalized_gram(features: torch.Tensor) -> torch.Tensor:
    """Translation-invariant orientation co-occurrence statistics."""
    batch, channels = features.shape[:2]
    flat = features.reshape(batch, channels, -1)
    gram = torch.bmm(flat, flat.transpose(1, 2))
    gram = gram / flat.shape[-1]
    scale = gram.diagonal(dim1=1, dim2=2).sum(dim=1, keepdim=True)
    return (gram / scale.clamp_min(1e-8).unsqueeze(-1)).reshape(batch, -1)


def _lag_correlations(features: torch.Tensor) -> torch.Tensor:
    """Per-orientation autocorrelation at local axial and diagonal lags."""
    correlations = []
    for dy, dx in ((0, 1), (1, 0), (1, 1), (1, -1), (0, 2), (2, 0)):
        if dy >= features.shape[-2] or abs(dx) >= features.shape[-1]:
            continue
        if dx >= 0:
            a = features[..., :features.shape[-2] - dy or None,
                         :features.shape[-1] - dx or None]
            b = features[..., dy:, dx:]
        else:
            shift = -dx
            a = features[..., :features.shape[-2] - dy or None, shift:]
            b = features[..., dy:, :features.shape[-1] - shift]
        numerator = (a * b).sum(dim=(-2, -1))
        denominator = (
            a.square().sum(dim=(-2, -1))
            * b.square().sum(dim=(-2, -1))
        ).sqrt().clamp_min(1e-8)
        correlations.append(numerator / denominator)
    if not correlations:
        return features.new_zeros((features.shape[0], 0))
    return torch.cat(correlations, dim=1)


def sketch_motif_descriptor(
    field: torch.Tensor,
    *,
    scales: Sequence[int] = (1, 2, 4),
) -> torch.Tensor:
    """Describe local structural language independently of its location.

    Fixed orientation filters turn the field into horizontal, vertical, and
    diagonal edge maps. Their normalized Gram matrices and short-lag
    autocorrelations capture orientation mixtures, branching/co-occurrence,
    and repeated spacing. Blank regions contribute exactly zero edge energy,
    so they cannot dominate the descriptor as empty patches would.
    """
    x = _as_single_channel_field(field)
    kernels = _orientation_kernels(device=x.device, dtype=x.dtype)
    descriptors = []
    for raw_scale in scales:
        scale = int(raw_scale)
        if scale < 1:
            raise ValueError(f'motif scales must be positive, got {raw_scale}')
        scaled = x
        if scale > 1:
            if scale > min(x.shape[-2:]):
                continue
            scaled = F.avg_pool2d(x, kernel_size=scale, stride=scale)
        features = F.conv2d(scaled, kernels, padding=1).abs()
        orientation_mass = features.mean(dim=(-2, -1))
        orientation_mass = orientation_mass / orientation_mass.sum(
            dim=1, keepdim=True).clamp_min(1e-8)
        descriptors.extend([
            orientation_mass,
            _normalized_gram(features),
            _lag_correlations(features),
        ])
    if not descriptors:
        raise ValueError(
            f'no motif scale fits field shape {tuple(x.shape[-2:])}')
    return torch.cat(descriptors, dim=1)


def sketch_motif_loss(
    density: torch.Tensor,
    occupancy: torch.Tensor,
    *,
    scales: Sequence[int] = (1, 2, 4),
) -> torch.Tensor:
    """Match the drawing's local motif statistics, not its pixel locations.

    The reference is sketch occupancy only. Load-only collector rows are
    deliberately absent: they belong to the spatial ``allowed`` template,
    not to the drawing's recurrent structural language.
    """
    dens = _as_single_channel_field(density)
    occ = _as_single_channel_field(occupancy).to(
        device=dens.device, dtype=dens.dtype)
    if dens.shape[-2:] != occ.shape[-2:]:
        raise ValueError(
            f'density and occupancy motif grids must match, got '
            f'{tuple(dens.shape[-2:])} vs {tuple(occ.shape[-2:])}')
    if occ.shape[0] == 1 and dens.shape[0] != 1:
        occ = occ.expand(dens.shape[0], -1, -1, -1)
    if dens.shape[0] != occ.shape[0]:
        raise ValueError(
            f'density and occupancy motif batches must match, got '
            f'{dens.shape[0]} vs {occ.shape[0]}')
    density_descriptor = sketch_motif_descriptor(dens, scales=scales)
    reference_descriptor = sketch_motif_descriptor(occ, scales=scales)
    return F.mse_loss(density_descriptor, reference_descriptor)


def _sample_patch_columns(
    patches: torch.Tensor,
    valid: torch.Tensor,
    limit: int,
) -> torch.Tensor:
    """Select evenly spaced valid patch columns deterministically."""
    indices = torch.nonzero(valid, as_tuple=False).flatten()
    if indices.numel() > int(limit):
        positions = torch.linspace(
            0, indices.numel() - 1, int(limit), device=indices.device)
        indices = indices[positions.long()]
    return patches[:, indices]


def _normalized_patch_vocabulary(
    field: torch.Tensor,
    *,
    patch_size: int,
    stride: int,
    limit: int,
) -> torch.Tensor:
    """Extract nonblank, contrast-normalized patches as vocabulary rows."""
    patches = F.unfold(
        field,
        kernel_size=patch_size,
        padding=patch_size // 2,
        stride=stride,
    )
    # Motif runs use one structural design at a time. Flattening batch into
    # columns keeps the helper general without introducing batch pairings.
    patches = patches.permute(1, 0, 2).reshape(patches.shape[1], -1)
    means = patches.mean(dim=0, keepdim=True)
    centered = patches - means
    contrast = centered.square().mean(dim=0).sqrt()
    mass = means[0]
    valid = (
        (mass.detach() > 0.02)
        & (mass.detach() < 0.98)
        & (contrast.detach() > 0.025)
    )
    selected = _sample_patch_columns(centered, valid, limit)
    if selected.shape[1] == 0:
        return selected.transpose(0, 1)
    normalized = selected / selected.square().sum(
        dim=0, keepdim=True).sqrt().clamp_min(1e-6)
    return normalized.transpose(0, 1)


def sketch_patch_vocabulary_loss(
    density: torch.Tensor,
    occupancy: torch.Tensor,
    load_sites: Optional[torch.Tensor] = None,
    *,
    patch_sizes: Sequence[int] = (7, 15),
    stride: int = 2,
    temperature: float = 0.01,
    max_design_patches: int = 1024,
    max_reference_patches: int = 256,
) -> torch.Tensor:
    """Softly match local design patches to the sketch's patch vocabulary.

    Unlike global Gram statistics, this asks each sampled nonblank design
    patch to resemble an actual nonblank drawing patch, while the reverse
    direction keeps the drawing's vocabulary represented. Patch positions are
    discarded, so matching remains translation-invariant. Required load-only
    pixels are removed from the design descriptor and never enter the
    reference vocabulary; the loss therefore does not fight collector rows.
    """
    dens = _as_single_channel_field(density)
    occ = _as_single_channel_field(occupancy).to(
        device=dens.device, dtype=dens.dtype)
    if dens.shape != occ.shape:
        if occ.shape[0] == 1 and dens.shape[0] != 1:
            occ = occ.expand(dens.shape[0], -1, -1, -1)
        if dens.shape != occ.shape:
            raise ValueError(
                f'density and occupancy patch grids must match, got '
                f'{tuple(dens.shape)} vs {tuple(occ.shape)}')
    design = dens
    if load_sites is not None:
        sites = _as_single_channel_field(load_sites).to(
            device=dens.device, dtype=dens.dtype)
        if sites.shape[0] == 1 and dens.shape[0] != 1:
            sites = sites.expand(dens.shape[0], -1, -1, -1)
        if sites.shape != dens.shape:
            raise ValueError(
                f'density and load_sites patch grids must match, got '
                f'{tuple(dens.shape)} vs {tuple(sites.shape)}')
        load_only = sites * (occ < 0.5).to(dens.dtype)
        design = dens * (1.0 - load_only)

    if int(stride) < 1:
        raise ValueError(f'patch stride must be positive, got {stride}')
    if float(temperature) <= 0.0:
        raise ValueError(
            f'patch temperature must be positive, got {temperature}')
    losses = []
    for raw_size in patch_sizes:
        size = int(raw_size)
        if size < 3 or size % 2 == 0:
            raise ValueError(
                f'patch sizes must be odd integers >= 3, got {raw_size}')
        if size > min(dens.shape[-2:]):
            continue
        design_patches = _normalized_patch_vocabulary(
            design,
            patch_size=size,
            stride=int(stride),
            limit=int(max_design_patches),
        )
        reference_patches = _normalized_patch_vocabulary(
            occ,
            patch_size=size,
            stride=int(stride),
            limit=int(max_reference_patches),
        )
        if design_patches.shape[0] == 0 or reference_patches.shape[0] == 0:
            continue
        similarity = design_patches @ reference_patches.transpose(0, 1)
        design_best = (
            torch.softmax(similarity / temperature, dim=1) * similarity
        ).sum(dim=1)
        reference_best = (
            torch.softmax(similarity / temperature, dim=0) * similarity
        ).sum(dim=0)
        losses.append(
            1.0 - 0.5 * (design_best.mean() + reference_best.mean()))
    if not losses:
        return dens.sum() * 0.0
    return torch.stack(losses).mean()


def mass_fraction_on_occupancy(
    density: Union[np.ndarray, torch.Tensor],
    occupancy: Union[np.ndarray, torch.Tensor],
    *,
    threshold: float = 0.5,
) -> float:
    """Share of mass on cells whose occupancy is at least ``threshold``.

    Used by the Stage 6 gate (guided run vs no-sketch control). Detaches
    tensors; this is a metric, not a loss.
    """
    dens = _as_numpy(density).reshape(-1).astype(np.float64)
    occ = _as_numpy(occupancy).reshape(-1).astype(np.float64)
    if dens.size != occ.size:
        raise ValueError(
            f'density and occupancy must have the same number of cells, '
            f'got {dens.size} vs {occ.size}')
    mask = occ >= float(threshold)
    mass = dens.sum()
    if mass <= 0.0:
        return 0.0
    return float((dens * mask).sum() / mass)


def occupancy_to_uint8(occupancy: np.ndarray) -> np.ndarray:
    """Occupancy 1 ? white (material encouragement), 0 ? black (paper)."""
    occ = np.clip(np.asarray(occupancy, dtype=np.float64), 0.0, 1.0)
    return (occ * 255.0).round().astype(np.uint8)


def save_sketch_visual(
    path: Union[str, Path],
    panels: Sequence[np.ndarray],
) -> Path:
    """Write one or more ``(H, W)`` fields as a side-by-side grayscale PNG.

    Each panel is independently min-max scaled into ``[0, 1]`` if it is not
    already in that range, then converted with :func:`occupancy_to_uint8`.
    A standing preference of this repo is that a design run always leaves a
    visual, not just a scalar.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    images = []
    for panel in panels:
        arr = np.asarray(panel, dtype=np.float64)
        if arr.ndim != 2:
            arr = np.squeeze(arr)
        if arr.ndim != 2:
            raise ValueError(f'panel must be 2-D after squeeze, got {arr.shape}')
        peak = float(np.max(arr))
        if peak > 1.0 + 1e-6 or float(np.min(arr)) < -1e-6:
            lo, hi = float(np.min(arr)), peak
            arr = (arr - lo) / (hi - lo + 1e-12)
        images.append(Image.fromarray(occupancy_to_uint8(arr), mode='L'))
    if not images:
        raise ValueError('save_sketch_visual needs at least one panel')
    if len(images) == 1:
        images[0].save(path)
        return path
    height = max(im.height for im in images)
    width = sum(im.width for im in images)
    canvas = Image.new('L', (width, height), color=0)
    x = 0
    for im in images:
        canvas.paste(im, (x, 0))
        x += im.width
    canvas.save(path)
    return path


def init_weight_with_occupancy(model, occupancy) -> torch.Tensor:
    """Seed ``model.z`` from an occupancy map at the model's current grid.

    Same contract as the Venice image seeder: pixel space, current
    resolution, carried up by AdaptivePixel bilinear upsample. Nearest
    resample so a binary occupancy stays binary on a coarse stage. Does
    not go through the JPEG loader.
    """
    height, width = int(model.z.shape[-2]), int(model.z.shape[-1])
    occ = np.asarray(occupancy, dtype=np.float32)
    if occ.ndim == 3:
        occ = np.squeeze(occ, axis=0)
    if occ.ndim != 2:
        raise ValueError(
            f'occupancy must be 2-D after squeeze, got {occ.shape}')
    field = torch.as_tensor(occ, dtype=torch.float32)
    if field.shape[-2] != height or field.shape[-1] != width:
        field = torch.nn.functional.interpolate(
            field.view(1, 1, field.shape[-2], field.shape[-1]),
            size=(height, width),
            mode='nearest',
        ).view(height, width)
    image = field.clamp(0.0, 1.0).to(device=model.z.device)
    if image.ndim == 2:
        image = image.unsqueeze(0)
    model.z = torch.nn.Parameter(image.contiguous(), requires_grad=True)
    return model.z


def apply_sketch_config(
    model,
    sketch,
    *,
    height: int,
    width: int,
    repo_root: Optional[Union[str, Path]] = None,
):
    """Load occupancy at the *full* problem grid and attach it to ``model``.

    Coarse stages resample inside ``Model.get_sketch_loss``. ``sketch.path is
    None`` is a no-op so a Venice-preset model stays bit-identical.

    ``init_from_occupancy`` overwrites ``model.z`` with occupancy ? load
    pixels at the *current* (possibly coarse) grid.
    """
    if getattr(sketch, 'path', None) is None:
        return model
    path = Path(sketch.path)
    if not path.is_absolute():
        root = Path(repo_root) if repo_root is not None else Path(__file__).resolve().parents[2]
        path = root / path
    occupancy = load_sketch_occupancy(
        path,
        height=int(height),
        width=int(width),
        invert=bool(sketch.invert),
        threshold=float(sketch.threshold),
        blur_sigma=float(sketch.blur_sigma),
    )
    weight_end = getattr(sketch, 'weight_end', None)
    model.enable_sketch_prior(
        occupancy,
        weight=float(sketch.weight),
        weight_end=weight_end,
        motif_weight=float(getattr(sketch, 'motif_weight', 0.0)),
        motif_weight_end=getattr(sketch, 'motif_weight_end', None),
        motif_scales=tuple(getattr(sketch, 'motif_scales', (1, 2, 4))),
        patch_weight=float(getattr(sketch, 'patch_weight', 0.0)),
        patch_weight_end=getattr(sketch, 'patch_weight_end', None),
        patch_sizes=tuple(getattr(sketch, 'patch_sizes', (7, 15))),
        patch_stride=int(getattr(sketch, 'patch_stride', 2)),
    )
    if getattr(sketch, 'init_from_occupancy', False):
        nely = int(model.env.args['nely'])
        nelx = int(model.env.args['nelx'])
        sites = load_site_mask(
            model.env.args['forces'], nely=nely, nelx=nelx)
        occ_t = torch.as_tensor(occupancy, dtype=torch.float32)
        if occ_t.shape[-2] != nely or occ_t.shape[-1] != nelx:
            occ_t = torch.nn.functional.interpolate(
                occ_t.view(1, 1, occ_t.shape[-2], occ_t.shape[-1]),
                size=(nely, nelx),
                mode='nearest',
            ).view(nely, nelx)
        allowed = np.maximum(occ_t.detach().cpu().numpy(), sites)
        init_weight_with_occupancy(model, allowed)
    return model


def apply_motif_layout_config(
    model,
    layout,
    teacher_field,
    *,
    scale_fracs: Sequence[float],
):
    """Attach a teacher-distilled scaffold as the spatial mass prior.

    ``layout.enabled`` False is a no-op so a Venice-preset model stays
    bit-identical. Motif/patch sketch terms stay off: this is layout
    redistribution, not Gram/patch matching. ``init_from_teacher`` overwrites
    ``model.z`` with a bilinear downsample of the unbounded teacher design.
    """
    if not bool(getattr(layout, 'enabled', False)):
        return model
    scaffold = motif_layout_scaffold(
        teacher_field,
        scale_fracs=scale_fracs,
        threshold=float(layout.threshold),
        envelope_sigma_frac=float(layout.envelope_sigma_frac),
        combine=str(layout.combine),
    )
    model.enable_sketch_prior(
        scaffold,
        weight=float(layout.weight),
        weight_end=getattr(layout, 'weight_end', None),
        motif_weight=0.0,
        patch_weight=0.0,
    )
    if bool(getattr(layout, 'init_from_teacher', False)):
        init_weight_from_teacher(model, teacher_field)
    return model


def apply_scaffold_as_occupancy_prior(
    model,
    scaffold,
    *,
    weight: float = 4000.0,
    weight_end: Optional[float] = 400.0,
    init_from_occupancy: bool = True,
):
    """Attach a CLIP-dream scaffold the way Stage 6 attaches a sketch.

    Motif/patch terms stay off. ``init_from_occupancy`` seeds ``z`` from
    occupancy union load pixels at the *current* grid - not from a physics
    teacher, which is how the layout-distillation student became tautological.
    """
    occupancy = _as_numpy(scaffold).astype(np.float32)
    if occupancy.ndim == 3:
        occupancy = np.squeeze(occupancy, axis=0)
    if occupancy.ndim != 2:
        raise ValueError(
            f'scaffold must be 2-D after squeeze, got {occupancy.shape}')
    model.enable_sketch_prior(
        occupancy,
        weight=float(weight),
        weight_end=weight_end,
        motif_weight=0.0,
        patch_weight=0.0,
    )
    if not init_from_occupancy:
        return model
    nely = int(model.env.args['nely'])
    nelx = int(model.env.args['nelx'])
    sites = load_site_mask(model.env.args['forces'], nely=nely, nelx=nelx)
    occ_t = torch.as_tensor(occupancy, dtype=torch.float32)
    if occ_t.shape[-2] != nely or occ_t.shape[-1] != nelx:
        occ_t = F.interpolate(
            occ_t.view(1, 1, occ_t.shape[-2], occ_t.shape[-1]),
            size=(nely, nelx),
            mode='nearest',
        ).view(nely, nelx)
    allowed = np.maximum(occ_t.detach().cpu().numpy(), sites)
    init_weight_with_occupancy(model, allowed)
    return model


def _as_numpy(value: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)
