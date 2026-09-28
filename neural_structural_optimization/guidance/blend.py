# lint as python3
"""Stage 7 blending diagnostics: name the mix, log it, do not change it.

Venice glued CLIP to compliance (weight = clip_alpha * C live, plus raw CLIP
in the total). Inverse-normalized and static weights are different algebras,
not settings of that formula. This module:

* names whichever algebra is actually live (``BlendMode``),
* refuses a requested mode that contradicts the Venice seam,
* snapshots raw terms, effective weights, and (opt-in) gradient conflict,
* optionally sets the CLIP weight from EMA gradient-norm matching.

A C* controller is a later slice.
"""

from __future__ import annotations

from enum import Enum
from typing import Optional

import torch

from neural_structural_optimization.model.model_base import VeniceLossTerms

_GRAD_EPS = 1e-12

# Same ceiling as the inverse-normalized CLIP weight. Grad-match is
# ||g_C|| / ||g_d||; a near-zero CLIP gradient would otherwise explode.
GRAD_MATCH_WEIGHT_MAX = 2000.0
GRAD_MATCH_EMA_DECAY = 0.9


class BlendMode(str, Enum):
    """The live CLIP/compliance algebra. Not a mixing knob.

    ``venice``
        ``clip_weight = compliance * clip_alpha`` undetached, and the raw
        CLIP loss is added on top of the weighted term.
    ``static``
        Constant weight; no leftover unweighted CLIP term.
    ``detached_scale``
        ``clip_alpha * compliance.detach()``. Venice magnitude without
        extra gradient through C. AdaptiveAdam's default-path ``clip_alpha``.
    ``inverse``
        ``clip_alpha * C0 / C``, detached and capped. Adam/LBFGS
        ``clip_alpha``.
    ``grad_match``
        ``rho * EMA||g_C|| / EMA||g_d||``, detached and capped. Equal
        gradient energy at rho=1. Occupancy is not in this ratio.
    """

    VENICE = 'venice'
    STATIC = 'static'
    DETACHED_SCALE = 'detached_scale'
    INVERSE = 'inverse'
    GRAD_MATCH = 'grad_match'


def resolve_blend_mode(
    model,
    *,
    clip_weight: Optional[float],
    clip_alpha: Optional[float],
    optimizer: str,
    blend_rho: Optional[float] = None,
    requested: Optional[str] = None,
) -> BlendMode:
    """Name the algebra this optimizer will actually run.

    Args:
        model: owns ``venice_loss_algebra`` when the gothic path is on.
        clip_weight: static CLIP weight, or None if unset.
        clip_alpha: optimizer-level alpha, or None if unset.
        optimizer: constructor name, used to tell AdaptiveAdam's
            proportional ``clip_alpha`` from Adam/LBFGS inverse ``clip_alpha``.
        blend_rho: if set, request gradient-norm matching on the default path.
        requested: optional explicit ``BlendMode`` value.

    Returns:
        The live :class:`BlendMode`.

    Raises:
        ValueError: if the caller asked for a coupling the Venice seam
            cannot honour.
    """
    venice = getattr(model, 'venice_loss_algebra', None) is not None
    inverse_alpha = optimizer in ('Adam_Optimizer', 'LBFGS_Optimizer')
    wants_grad_match = (
        blend_rho is not None or requested == BlendMode.GRAD_MATCH.value)
    if venice:
        if wants_grad_match:
            raise ValueError(
                f'{optimizer} got blend_rho={blend_rho!r} / blend_mode='
                f'{requested!r} while the model has the Venice compatibility '
                'algebra enabled; gradient-norm matching is not a setting of '
                'venice (live compliance * clip_alpha). Disable the preset '
                'to run grad_match, or drop blend_rho to run the preset.')
        if inverse_alpha and clip_alpha is not None:
            raise ValueError(
                f'{optimizer} got clip_alpha={clip_alpha!r} while the model '
                'has the Venice compatibility algebra enabled; those are '
                'contradictory couplings. Inverse-normalized clip_alpha is '
                'not a setting of venice (live compliance * clip_alpha). '
                'Drop clip_alpha to run the preset, or disable the preset.')
        if clip_weight is not None:
            raise ValueError(
                f'{optimizer} got clip_weight={clip_weight!r} while the '
                'model has the Venice compatibility algebra enabled; those '
                'are contradictory couplings. Drop the static weight to run '
                'the preset, or disable the preset to use a static weight.')
        return BlendMode.VENICE
    if wants_grad_match:
        if clip_alpha is not None or clip_weight is not None:
            raise ValueError(
                f'{optimizer} got grad-match together with clip_alpha='
                f'{clip_alpha!r} or clip_weight={clip_weight!r}; those are '
                'different couplings. Drop clip_alpha and clip_weight to '
                'run grad_match.')
        return BlendMode.GRAD_MATCH
    if inverse_alpha and clip_alpha is not None:
        return BlendMode.INVERSE
    if clip_alpha is not None:
        return BlendMode.DETACHED_SCALE
    return BlendMode.STATIC


def freeze_blend_mode(
    requested: Optional[str],
    resolved: BlendMode,
) -> BlendMode:
    """Refuse a requested mode that is not the live algebra.

    ``requested is None`` means "log whatever is live." Asking for
    ``static`` / ``inverse`` / ``detached_scale`` / ``grad_match`` while
    the Venice seam is on is the mix-by-mutation this freeze exists to stop.
    """
    if requested is None or requested == '':
        return resolved
    try:
        want = BlendMode(str(requested))
    except ValueError as exc:
        known = ', '.join(mode.value for mode in BlendMode)
        raise ValueError(
            f'unknown blend_mode={requested!r}; known modes: {known}.') from exc
    if want != resolved:
        raise ValueError(
            f'requested blend_mode={want.value!r} but the live algebra is '
            f'{resolved.value!r}. The Venice seam is frozen: disable '
            'venice_compat to run a different coupling, or drop blend_mode '
            'to log the live algebra.')
    return resolved


class GradNormEma:
    """EMA of ||g_C|| and ||g_d|| for gradient-norm matching.

    The first finite observation seeds the averages. Non-finite or near-zero
    norms are skipped so a missing CLIP path does not poison the ratio.
    ``reset`` after an AdaptivePixel upsample: the coarse-grid norms are not
    the fine-grid ones.
    """

    def __init__(self, decay: float = GRAD_MATCH_EMA_DECAY):
        if not 0.0 <= float(decay) < 1.0:
            raise ValueError(
                f'grad-match EMA decay must be in [0, 1), got {decay!r}')
        self.decay = float(decay)
        self.g_c: Optional[float] = None
        self.g_d: Optional[float] = None

    def reset(self) -> None:
        self.g_c = None
        self.g_d = None

    def update(self, n_c: float, n_d: float) -> None:
        if not _finite_positive(n_c) or not _finite_positive(n_d):
            return
        if self.g_c is None or self.g_d is None:
            self.g_c = float(n_c)
            self.g_d = float(n_d)
            return
        keep = self.decay
        self.g_c = keep * self.g_c + (1.0 - keep) * float(n_c)
        self.g_d = keep * self.g_d + (1.0 - keep) * float(n_d)

    def weight(
        self,
        rho: float,
        cap: float = GRAD_MATCH_WEIGHT_MAX,
    ) -> float:
        """Detached CLIP weight ``rho * EMA||g_C|| / EMA||g_d||``, capped."""
        if float(rho) == 0.0 or self.g_c is None or self.g_d is None:
            return 0.0
        if self.g_d < _GRAD_EPS:
            return 0.0
        raw = float(rho) * self.g_c / self.g_d
        if raw < 0.0:
            return 0.0
        return min(raw, float(cap))


def unweighted_grad_norms(
    compliance: torch.Tensor,
    clip_z: torch.Tensor,
    logits: torch.Tensor,
) -> tuple[float, float]:
    """||dC/d logits|| and ||d clip_z / d logits||. NaN if a term is dead."""
    return _norm(_grad_wrt(compliance, logits)), _norm(_grad_wrt(clip_z, logits))


def _finite_positive(value: float) -> bool:
    return value == value and value > _GRAD_EPS


def snapshot_blend(
    model,
    terms: VeniceLossTerms,
    logits: Optional[torch.Tensor] = None,
    *,
    grads: bool = False,
) -> dict[str, float]:
    """Per-step mix diagnostics. Does not change ``terms.total_loss``.

    Scalar fields always populate from the already-computed breakdown and
    the model's last occupancy / density-CLIP readings. Gradient norms and
    cosine are extra backwards through ``logits`` and default off.

    Args:
        model: the model that just composed ``terms``.
        terms: one-step Venice-shaped breakdown (both algebras pack into it).
        logits: design field, required when ``grads`` is True.
        grads: if True, also record gradient norms and cosine.

    Returns:
        A flat dict of floats suitable for the optimizer snapshot columns.
    """
    snap = {
        'compliance': _as_float(terms.compliance_loss),
        'clip_z': _as_float(terms.clip_loss_raw),
        'clip_z_weight': _as_float(terms.clip_weight),
        'clip_rho': _optional_float(getattr(model, '_last_physical_clip', None)),
        'clip_rho_weight': _optional_float(
            getattr(model, '_last_physical_clip_coefficient', None)),
        'occupancy': _optional_float(
            getattr(model, '_last_occupancy_loss', None)),
        'occupancy_weight': _optional_float(
            getattr(model, '_last_occupancy_weight', None)),
    }
    if grads:
        if logits is None:
            raise ValueError('blend_grads requires the live logits tensor.')
        snap.update(_gradient_conflict(terms, logits))
    return snap


def _as_float(value) -> float:
    if value is None:
        return float('nan')
    if torch.is_tensor(value):
        return float(value.detach())
    return float(value)


def _optional_float(value) -> float:
    if value is None:
        return float('nan')
    return _as_float(value)


def _gradient_conflict(
    terms: VeniceLossTerms,
    logits: torch.Tensor,
) -> dict[str, float]:
    """Gradient norms of C and clip_z, and their cosine, wrt ``logits``.

    Extra VJPs; the caller must still ``retain_graph`` into the parameter
    ``backward``. Empty / unused grads become NaN rather than 0 so a
    missing CLIP path does not look orthogonal.
    """
    compliance = terms.compliance_loss
    clip_z = terms.clip_loss_raw
    g_c = _grad_wrt(compliance, logits)
    g_d = _grad_wrt(clip_z, logits)
    n_c = _norm(g_c)
    n_d = _norm(g_d)
    return {
        'g_compliance': n_c,
        'g_clip_z': n_d,
        'grad_cosine': _cosine(g_c, g_d),
    }


def _grad_wrt(loss: torch.Tensor, logits: torch.Tensor) -> Optional[torch.Tensor]:
    if not torch.is_tensor(loss) or not loss.requires_grad:
        return None
    if not logits.requires_grad:
        return None
    grads = torch.autograd.grad(
        loss,
        logits,
        retain_graph=True,
        allow_unused=True,
    )
    grad = grads[0]
    if grad is None:
        return None
    return grad


def _norm(grad: Optional[torch.Tensor]) -> float:
    if grad is None:
        return float('nan')
    return float(grad.detach().norm())


def _cosine(
    left: Optional[torch.Tensor],
    right: Optional[torch.Tensor],
) -> float:
    if left is None or right is None:
        return float('nan')
    n_left = float(left.norm())
    n_right = float(right.norm())
    if n_left < _GRAD_EPS or n_right < _GRAD_EPS:
        return float('nan')
    dot = float((left.reshape(-1) * right.reshape(-1)).sum())
    return dot / (n_left * n_right)
