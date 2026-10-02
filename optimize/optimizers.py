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

"""Optimization algorithms for neural structural optimization."""

from typing import Callable, Optional
import logging
import numpy as np
import torch
import xarray
from tqdm import tqdm

import model as models
from model import AdaptivePixelModel, PixelModel, CNNModel
from model.model_ada import (
    DEFAULT_CONVERGENCE_THRESHOLD,
    DEFAULT_MAX_RESIZE_ITERATION,
    DEFAULT_RESIZE_THRESHOLD,
    INITIAL_PREV_LOSS,
)
from model.model_base import VeniceLossTerms
from guidance.blend import (
    BlendMode,
    GradNormEma,
    freeze_blend_mode,
    grad_norm,
    grad_wrt,
    resolve_blend_mode,
    snapshot_blend,
)

from .base import BaseOptimizer
from .utils import (cosine_warmup, get_variables, constrained_logits, ensure_array_size,
                    calibrate_lambda_clip, repeat_to_shape)


# Ceiling on the default inverse-normalized CLIP weight, which grows without
# bound as compliance falls and would otherwise dominate late in a run. It
# belongs to that formula alone: the Venice algebra's weight is *directly*
# proportional to compliance and uncapped (the reference run peaks near 740),
# so the preset path must never clamp -- a cap that does not bind today would
# bind at higher compliance later and silently corrupt the trajectory.
CLIP_DYNAMIC_WEIGHT_MAX = 2000.0


def _apply_sketch_schedule(model, step: int, max_iterations: int) -> None:
    """Advance the per-step schedules (sketch weight, physics projection)."""
    model._opt_step = int(step)
    model._opt_max_iterations = int(max_iterations)
    apply = getattr(model, 'apply_sketch_schedule', None)
    if apply is not None:
        apply(step=step, max_iterations=max_iterations)
    project = getattr(model, 'apply_physics_projection_schedule', None)
    if project is not None:
        project()


def _reject_clip_alpha_under_venice_compat(model, clip_alpha, optimizer_name: str) -> None:
    """Refuse a `clip_alpha` that contradicts the model's Venice algebra.

    `Adam_Optimizer` and `LBFGS_Optimizer` read `clip_alpha` as the scale of an
    INVERSE-normalized weight, `clip_alpha * baseline_compliance / compliance`,
    capped at :data:`CLIP_DYNAMIC_WEIGHT_MAX`. The Venice algebra's weight is
    `compliance * clip_alpha`: directly proportional, undetached and uncapped.
    Those are structurally different formulas rather than two settings of one,
    so a caller asking for both has asked for two different runs and gets an
    error instead of whichever branch happens to be tested first.

    Args:
        model: the model whose `venice_loss_algebra` selects the algebra.
        clip_alpha: the caller's `clip_alpha`, or None if unset.
        optimizer_name: name used in the error message.

    Raises:
        ValueError: if both the preset and `clip_alpha` are configured.
    """
    if clip_alpha is None or model.venice_loss_algebra is None:
        return
    raise ValueError(
        f'{optimizer_name} got clip_alpha={clip_alpha!r} while the model has '
        'the Venice compatibility algebra enabled; those are contradictory '
        "couplings. This optimizer's clip_alpha scales a weight inversely "
        f'proportional to compliance and capped at {CLIP_DYNAMIC_WEIGHT_MAX}, '
        'whereas the preset weight is compliance * clip_alpha, uncapped. Drop '
        'clip_alpha to run the preset -- set its alpha with '
        'enable_venice_compat_loss(VeniceLossAlgebra(clip_alpha=...)) -- or '
        'disable the preset to use the default coupling.')


def _reject_clip_weight_under_venice_compat(
        model, clip_weight, optimizer_name: str,
        arg_name: str = 'clip_weight') -> None:
    """Refuse a static CLIP weight that the Venice algebra cannot honour.

    The preset recomputes `clip_weight = compliance * clip_alpha` undetached
    every step. A caller-supplied static weight -- `clip_weight` on
    AdaptiveAdam, `clip_weight_max` on Adam/LBFGS -- is a different coupling,
    and silently dropping it is the same defect `Model.get_total_loss`
    already refuses.

    Args:
        model: the model whose `venice_loss_algebra` selects the algebra.
        clip_weight: the caller's static weight, or None if unset.
        optimizer_name: name used in the error message.
        arg_name: the parameter the caller passed, so the message names the
            argument they can drop.

    Raises:
        ValueError: if both the preset and a static weight are set.
    """
    if clip_weight is None or model.venice_loss_algebra is None:
        return
    raise ValueError(
        f'{optimizer_name} got {arg_name}={clip_weight!r} while the model '
        'has the Venice compatibility algebra enabled; those are '
        'contradictory couplings. The preset recomputes clip_weight as '
        'compliance * clip_alpha, undetached, every step, so a static '
        f'{arg_name} has no meaning there. Drop {arg_name} to run the '
        'preset -- set its alpha with enable_venice_compat_loss('
        'VeniceLossAlgebra(clip_alpha=...)) -- or disable the preset to '
        'use a static weight.')


def _coerce_blend_rho(blend_rho, name: str = 'blend_rho') -> Optional[float]:
    """None is unset; otherwise the mixer knob must live in [0, 1]."""
    if blend_rho is None:
        return None
    rho = float(blend_rho)
    if not 0.0 <= rho <= 1.0:
        raise ValueError(
            f'{name} must be in [0, 1] (0 = CLIP off, 1 = equal gradient '
            f'energy), got {blend_rho!r}')
    return rho


def _static_clip_weight_or_default(
        value: Optional[float], default: float = 1.0) -> float:
    """None means unset; the default path then uses `default`."""
    return default if value is None else float(value)


def _add_scaled(
    total: Optional[torch.Tensor],
    grad: Optional[torch.Tensor],
    scale,
) -> Optional[torch.Tensor]:
    """``total + scale * grad``, where ``None`` is a term with no gradient."""
    if grad is None:
        return total
    scaled = grad * scale
    return scaled if total is None else total + scaled


def _reject_venice_compat_under_physics_only(model, optimizer_name: str) -> None:
    """Refuse a Venice algebra an optimizer never consults.

    `MMA_Optimizer` and `OptimalityCriteria_Optimizer` do not go through
    `Model.get_total_loss` at all: they drive `env.objective` and the physics
    optimality step directly, so neither the semantic loss nor the algebra
    weighting it reaches the design. A model carrying the preset would
    therefore be optimized for pure compliance while every configured coupling
    was discarded without a word -- the same silent swallow the `clip_alpha`
    and `clip_weight` refusals exist to prevent.

    Args:
        model: the model whose `venice_loss_algebra` selects the algebra.
        optimizer_name: name used in the error message.

    Raises:
        ValueError: if the preset is enabled.
    """
    if model.venice_loss_algebra is None:
        return
    raise ValueError(
        f'{optimizer_name} cannot honour the Venice compatibility algebra: it '
        'optimizes the physics objective directly and never evaluates the '
        'semantic loss, so clip_alpha, the undetached weight and the '
        'unweighted CLIP term would all be dropped and the run would silently '
        'minimize compliance alone. Disable the preset with '
        'enable_venice_compat_loss(False) to optimize compliance on purpose, '
        'or use a gradient optimizer -- AdaptiveAdam_Optimizer runs the '
        'preset.')


def _detach_loss_terms(terms: VeniceLossTerms) -> VeniceLossTerms:
    """Drop the autograd graph from one step's terms so a run can log them."""
    return VeniceLossTerms(*(float(term.detach()) for term in terms))


def _make_grads_contiguous(model) -> None:
    """Compact any non-contiguous gradient, which L-BFGS cannot flatten.

    `torch.optim.LBFGS` calls `view(-1)` on every gradient, and that raises on
    a non-contiguous tensor, so each closure has to normalize its own grads.
    """
    for p in model.parameters():
        if p.grad is not None and not p.grad.is_contiguous():
            p.grad = p.grad.contiguous()


def _attach_loss_terms(ds: xarray.Dataset, terms) -> xarray.Dataset:
    """Add a term-by-term loss trajectory to `ds`, beside the scalar total.

    The parity harness compares the reference log field by field, so a run has
    to carry compliance, the weighted and unweighted CLIP terms and the weight
    itself, not just `loss`.

    Args:
        ds: dataset with a `step` dimension; modified in place.
        terms: one detached :class:`VeniceLossTerms` per step, or an empty
            sequence on paths where the breakdown is unavailable without a
            second physics solve, in which case `ds` is left untouched.

    Returns:
        `ds`, for chaining onto a `create_dataset` call.
    """
    if not terms:
        return ds
    ds['compliance'] = (('step',), [t.compliance_loss for t in terms])
    ds['clip_loss'] = (('step',), [t.clip_loss for t in terms])
    ds['clip_loss_raw'] = (('step',), [t.clip_loss_raw for t in terms])
    ds['clip_weight'] = (('step',), [t.clip_weight for t in terms])
    return ds


def _snapshot_motif_scale_losses(model) -> dict:
    """Copy per-scale CLIP contributions recorded on the last forward."""
    clip = getattr(model, 'clip_loss', None)
    losses = getattr(clip, 'last_motif_scale_losses', None)
    if not losses:
        return {}
    return {str(key): float(value) for key, value in losses.items()}


def _motif_scale_column(key: str) -> str:
    """Stable xarray name for a ``last_motif_scale_losses`` key."""
    if key == 'mean':
        return 'clip_motif_mean'
    sanitized = (
        str(key).replace('frac=', 'f').replace('.', 'p').replace('=', '_'))
    return f'clip_motif_{sanitized}'


def _attach_snapshot_columns(
    ds: xarray.Dataset,
    snapshots,
    column_name: Callable[[str], str],
) -> xarray.Dataset:
    """Write one `step`-indexed column per key seen across `snapshots`.

    Diagnostic subsystems each record a dict per step and only some of them
    run on any given path, so the union of keys is taken in first-seen order
    and a step that lacks a key is filled with NaN rather than dropped. That
    keeps every column the same length as `step`.

    Args:
        ds: dataset with a `step` dimension; modified in place.
        snapshots: one dict per step. All-empty leaves `ds` untouched, which
            is how a path that never ran stays out of the schema.
        column_name: maps a snapshot key to its dataset column.

    Returns:
        `ds`, for chaining.
    """
    if not snapshots or not any(snapshots):
        return ds
    keys = []
    for snap in snapshots:
        for key in snap:
            if key not in keys:
                keys.append(key)
    for key in keys:
        ds[column_name(key)] = (
            ('step',),
            [float(snap.get(key, float('nan'))) for snap in snapshots])
    return ds


def _attach_motif_scale_terms(ds: xarray.Dataset, snapshots) -> xarray.Dataset:
    """Add per-scale CLIP columns when the second path ran. No-op if empty."""
    return _attach_snapshot_columns(ds, snapshots, _motif_scale_column)


def _snapshot_semantic_prior(model) -> dict:
    """Copy live-prior diagnostics recorded on the last refresh."""
    prior = getattr(model, 'semantic_prior', None)
    if prior is None or not getattr(prior, 'last_metrics', None):
        return {}
    snap = {str(key): float(value) for key, value in prior.last_metrics.items()}
    loss = getattr(model, '_last_semantic_prior_loss', None)
    if loss is not None:
        snap['prior_loss'] = float(loss.detach())
    return snap


def _attach_semantic_prior_terms(ds: xarray.Dataset, snapshots) -> xarray.Dataset:
    """Add live-prior columns when the path ran. No-op if empty."""
    return _attach_snapshot_columns(ds, snapshots, lambda key: f'semantic_{key}')


def _snapshot_physical_clip(model) -> dict:
    """Copy the density-CLIP term recorded on the last `add_sketch_term`.

    Empty when the term is off, so a run without it keeps its old schema.
    """
    score = getattr(model, '_last_physical_clip', None)
    if score is None:
        return {}
    snap = {'physical_clip': float(score)}
    coefficient = getattr(model, '_last_physical_clip_coefficient', None)
    if coefficient is not None:
        snap['physical_clip_coefficient'] = float(coefficient)
        snap['physical_clip_weighted'] = float(coefficient) * float(score)
    beta_fn = getattr(model, '_physical_clip_projection_beta', None)
    sigma_fn = getattr(model, '_physical_clip_projection_sigma', None)
    if callable(beta_fn):
        snap['physical_clip_beta'] = float(beta_fn())
    if callable(sigma_fn):
        snap['physical_clip_sigma'] = float(sigma_fn())
    return snap


def _attach_physical_clip_terms(ds: xarray.Dataset, snapshots) -> xarray.Dataset:
    """Add density-CLIP columns when the term was on. No-op if empty.

    Unprefixed: the keys are already named for the term they come from.
    """
    return _attach_snapshot_columns(ds, snapshots, str)


def _attach_blend_terms(ds: xarray.Dataset, snapshots) -> xarray.Dataset:
    """Add mix-diagnostics columns when blend logging ran. No-op if empty."""
    return _attach_snapshot_columns(ds, snapshots, lambda key: f'blend_{key}')


class Adam_Optimizer(BaseOptimizer):
    """Adam optimization algorithm."""
    
    def __init__(self, model, max_iterations: int, lr_init: float = 1e-2, lr_final: float = 3e-3,
                 warmup_frac: float = 0.1, save_intermediate_designs: bool = True, 
                 grad_clip: Optional[float] = None,
                 clip_weight_max: Optional[float] = None,
                 clip_warmup_steps: int = 0,
                 clip_alpha: Optional[float] = None,
                 compliance_weight: Optional[float] = None):
        super().__init__(model, max_iterations, save_intermediate_designs)
        self.lr_init = lr_init
        self.lr_final = lr_final
        self.warmup_frac = warmup_frac
        self.grad_clip = grad_clip
        self.clip_weight_max = (
            None if clip_weight_max is None else float(clip_weight_max))
        self.clip_warmup_steps = int(clip_warmup_steps)
        self.clip_alpha = clip_alpha
        self.compliance_weight = compliance_weight
        # baseline structural loss for normalized inverse coupling
        self._baseline_Ls = None
        # cap for dynamic CLIP weight to avoid late domination
        self._clip_dynamic_w_max = CLIP_DYNAMIC_WEIGHT_MAX
        # One detached VeniceLossTerms per step, appended in step with the
        # tracker -- which `optimize` does not reset either, so the two stay
        # the same length however many times a run is restarted. Only filled
        # on the preset path.
        self.loss_terms = []
        _reject_clip_alpha_under_venice_compat(model, clip_alpha, 'Adam_Optimizer')
        _reject_clip_weight_under_venice_compat(
            model, self.clip_weight_max, 'Adam_Optimizer',
            arg_name='clip_weight_max')
    
    def optimize(self) -> xarray.Dataset:
        """Run Adam optimization."""
        # Re-checked here because the seam can be flipped on an already-built
        # model, after this optimizer was constructed.
        _reject_clip_alpha_under_venice_compat(
            self.model, self.clip_alpha, 'Adam_Optimizer')
        _reject_clip_weight_under_venice_compat(
            self.model, self.clip_weight_max, 'Adam_Optimizer',
            arg_name='clip_weight_max')
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr_init)
        clip_weight_max = _static_clip_weight_or_default(self.clip_weight_max)
        
        for i in tqdm(range(self.max_iterations + 1), desc="Adam Optimizer"):
            lr = cosine_warmup(i, self.max_iterations, self.warmup_frac, self.lr_init, self.lr_final)
            cw = 0.0
            if self.model.clip_loss is not None and clip_weight_max > 0:
                if self.clip_warmup_steps > 0:
                    cw = clip_weight_max * min(1.0, (i + 1) / float(self.clip_warmup_steps))
                else:
                    cw = clip_weight_max
            
            optimizer.param_groups[0]['lr'] = lr
            optimizer.zero_grad(set_to_none=True)
            _apply_sketch_schedule(self.model, i, self.max_iterations)
            logits = self.model()
            if self.model.venice_loss_algebra is not None:
                # The legacy algebra owns the coupling outright, so neither the
                # warmed-up `cw` nor the cap above applies. One call, one FEA
                # solve, and the whole breakdown for the parity log.
                terms = self.model.get_venice_compat_losses(
                    logits, compliance_weight=self.compliance_weight)
                loss = terms.total_loss
                self.loss_terms.append(_detach_loss_terms(terms))
            elif self.clip_alpha is not None:
                # Inverse-normalized, capped dynamic coupling
                Ls = self.model.get_structural_loss(logits)
                Lc = self.model.get_semantic_loss(logits) if self.model.clip_loss is not None else Ls.new_tensor(0.0)
                Ls_eff = Ls if self.compliance_weight is None else (Ls * float(self.compliance_weight))
                if self._baseline_Ls is None:
                    self._baseline_Ls = float(Ls_eff.detach())
                denom = max(self._baseline_Ls if self._baseline_Ls is not None else 1.0, 1e-8)
                # weight grows as compliance shrinks; clamp to safe cap
                raw_w = float(self.clip_alpha) * (self._baseline_Ls / (float(Ls_eff.detach()) + 1e-8))
                w_eff = min(raw_w, self._clip_dynamic_w_max)
                loss = Ls_eff + Lc * w_eff
                loss = self.model.add_sketch_term(loss, logits)
            else:
                loss = self.model.get_total_loss(
                    logits,
                    clip_weight=cw,
                    compliance_weight=self.compliance_weight,
                )
            
            loss.backward()
            if self.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            optimizer.step()
            
            self.tracker.add_step(float(loss.detach()), logits.detach().cpu().numpy())
        
        return _attach_loss_terms(self.tracker.create_dataset(self.model), self.loss_terms)


# Venice's iteration counter is initialised to 1 rather than 0 (models.py line
# 618; `reset_iteration` exists but is never called) and is incremented before
# the resize test runs, so after N gradient steps the counter reads N + 1. The
# first upsample therefore lands one step earlier than max_resize_iteration
# reads: at 50 it fires after 49 steps. The reference run's timing depends on
# this, so it is reproduced rather than quietly corrected.
VENICE_ITERATION_OFFSET = 1


class AdaptiveAdam_Optimizer(BaseOptimizer):
    """Adam driving `AdaptivePixelModel` on Venice's coarse-to-fine schedule.

    Two policies distinguish this from `Adam_Optimizer`, and both read
    COMPLIANCE rather than the total loss, so a semantic term that is still
    moving cannot hold either one open:

    * While upsamples remain, the grid advances a stage whenever the iteration
      count divides `max_resize_iteration` OR compliance moved less than
      `resize_threshold` over the last iteration.
    * Once the schedule is exhausted, the run stops as soon as compliance moves
      less than `convergence_threshold` over an iteration.

    The learning rate is held constant, as Venice holds it -- there is no
    cosine schedule here -- and a fresh Adam is built on every upsample.
    """

    def __init__(self, model, max_iterations: int, save_intermediate_designs: bool = True,
                 lr: float = 1e-2, grad_clip: Optional[float] = None,
                 clip_weight: Optional[float] = None,
                 clip_alpha: Optional[float] = None,
                 compliance_weight: Optional[float] = None,
                 resize_threshold: float = DEFAULT_RESIZE_THRESHOLD,
                 max_resize_iteration: int = DEFAULT_MAX_RESIZE_ITERATION,
                 convergence_threshold: float = DEFAULT_CONVERGENCE_THRESHOLD,
                 blend_log: bool = False,
                 blend_grads: bool = False,
                 blend_mode: Optional[str] = None,
                 blend_rho: Optional[float] = None,
                 blend_rho_z: Optional[float] = None):
        """Configure the run.

        Args:
            model: an `AdaptivePixelModel`; it owns `resize_num` and
                `resize_scale`, which set the resolutions this schedule visits.
            max_iterations: hard cap on gradient steps. The schedule normally
                stops well short of it -- the reference run stops at 124 of 200.
            save_intermediate_designs: keep every stage's design, not just the
                best one.
            lr: constant Adam learning rate. Defaults to the repository's Adam
                default rather than to Venice's, which is 0.2 for the reference
                run and belongs in that run's configuration.
            grad_clip: optional gradient-norm clip.
            clip_weight: static weight on the semantic loss of the default
                path. Ignored when `clip_alpha` is set. REFUSED, not ignored,
                when the model carries a Venice algebra -- that algebra
                recomputes the weight from compliance every step, so a static
                value describes a different run. None (the default) means
                unset; the default path then uses 1.0.
            clip_alpha: if set, weights the semantic loss by
                `clip_alpha * compliance` each iteration, which is Venice's
                dynamic coupling. Under a Venice algebra this is the same knob,
                so it overrides the algebra's own `clip_alpha` rather than
                conflicting with it.
            compliance_weight: optional scalar on the structural loss. It
                scales the value the schedule tests, so the thresholds are read
                against the weighted compliance, exactly as Venice reads them.
            resize_threshold: compliance delta below which the grid advances.
            max_resize_iteration: iteration period that forces an advance.
            convergence_threshold: compliance delta below which the run stops,
                once no upsamples remain.
            blend_log: if True, record per-step raw losses and effective
                weights on the dataset as ``blend_*`` columns. Default False
                so a Venice replay keeps its old schema.
            blend_grads: if True, also record gradient norms of compliance
                and clip_z, and their cosine. Implies ``blend_log``. Extra
                VJPs; off by default.
            blend_mode: optional name that must match the live algebra
                (``venice`` / ``static`` / ``detached_scale`` / ``inverse``
                / ``grad_match``). None logs whatever is live. Requesting a
                different mode than the Venice seam is refused rather than
                silently retuned.
            blend_rho: if set, AdaptiveAdam's default path weights CLIP by
                ``rho * EMA||g_C|| / EMA||g_d||`` (detached, capped). ``0``
                is compliance-only; ``1`` equalizes gradient energy.
                Occupancy stays on its own anneal. REFUSED under the Venice
                seam, and incompatible with ``clip_alpha`` / ``clip_weight``.
                Implies ``blend_log``.
            blend_rho_z: optional second mixer on raw-``z`` CLIP at
                ``rho_z * EMA||g_C|| / EMA||g_z||``. Requires ``blend_rho``
                and ``physical_clip_as_semantic`` so density CLIP stays the
                structural term. Unset keeps the single-term mixer.
        """
        super().__init__(model, max_iterations, save_intermediate_designs)
        self._validate_model_type(AdaptivePixelModel, "Adaptive Adam")
        if max_resize_iteration <= 0:
            raise ValueError(
                f'max_resize_iteration must be positive, got '
                f'{max_resize_iteration}; it is the period of the forced '
                'upsample, so zero has no meaning. Raise it above '
                'max_iterations to leave only the compliance-delta trigger.')
        self.lr = lr
        self.grad_clip = grad_clip
        self.clip_weight = None if clip_weight is None else float(clip_weight)
        self.clip_alpha = clip_alpha
        self.compliance_weight = compliance_weight
        self.resize_threshold = float(resize_threshold)
        self.max_resize_iteration = int(max_resize_iteration)
        self.convergence_threshold = float(convergence_threshold)
        if blend_grads and not blend_log:
            blend_log = True
        self.blend_log = bool(blend_log)
        self.blend_grads = bool(blend_grads)
        self.blend_mode_requested = (
            None if blend_mode is None else str(blend_mode))
        self.blend_rho = _coerce_blend_rho(blend_rho)
        self.blend_rho_z = _coerce_blend_rho(blend_rho_z, name='blend_rho_z')
        if self.blend_rho_z is not None and self.blend_rho is None:
            raise ValueError(
                'blend_rho_z requires blend_rho in [0, 1]; it is a second '
                'grad-matched CLIP term on raw z, not a substitute mixer.')
        _reject_clip_weight_under_venice_compat(
            model, self.clip_weight, 'AdaptiveAdam_Optimizer')
        self.blend_mode = self._freeze_live_blend_mode()
        if self.blend_mode == BlendMode.GRAD_MATCH:
            if self.blend_rho is None:
                raise ValueError(
                    'blend_mode=grad_match requires blend_rho in [0, 1] '
                    '(0 = CLIP off, 1 = equal gradient energy).')
            self.blend_log = True
            self._grad_norm_ema = GradNormEma()
            self._grad_norm_ema_z = (
                GradNormEma() if self.blend_rho_z is not None else None)
        else:
            self._grad_norm_ema = None
            self._grad_norm_ema_z = None
        if (
                self.blend_rho_z is not None
                and float(self.blend_rho_z) != 0.0
                and not getattr(model, 'physical_clip_as_semantic', False)):
            raise ValueError(
                'blend_rho_z requires physical_clip as_semantic so the mixer '
                'CLIP term is density, not the same raw-z term twice.')
        # Gradient steps at which an upsample fired, and whether the run
        # stopped on the convergence test rather than on max_iterations.
        self.resize_steps = []
        # One detached VeniceLossTerms per step: the compliance trajectory the
        # schedule ran on, plus the breakdown the parity harness compares.
        # Appended in step with the tracker; `optimize` clears BOTH, along with
        # every other per-run field, so a second call is a fresh run rather
        # than an append onto the first. See `_reset_run_state`.
        self.loss_terms = []
        self.motif_scale_terms = []
        self.semantic_prior_terms = []
        self.physical_clip_terms = []
        self.blend_terms = []
        self.converged = False
        # Assembled step gradient from `_compose_grad_matched`; None means
        # `optimize` backpropagates the total loss instead.
        self._step_grad = None

    def _freeze_live_blend_mode(self) -> BlendMode:
        return freeze_blend_mode(
            self.blend_mode_requested,
            resolve_blend_mode(
                self.model,
                clip_weight=self.clip_weight,
                clip_alpha=self.clip_alpha,
                optimizer='AdaptiveAdam_Optimizer',
                blend_rho=self.blend_rho,
                requested=self.blend_mode_requested,
            ),
        )

    def _compose_loss(self, logits) -> VeniceLossTerms:
        """Return the whole loss breakdown from ONE physics solve.

        The schedule reads the compliance term while the parity harness reads
        every term, and both have to come out of a single FEA solve -- asking
        the model for compliance separately would double the dominant cost of
        the run.

        Under a Venice algebra this delegates to
        `Model.get_venice_compat_losses`, which is itself one solve wide and
        routes through the shared `venice_compat_total_loss`; restating that
        algebra here is what would let the two paths drift. Only the default
        composition -- a detached weight and no unweighted term -- is spelled
        out, packed into the same container so one dataset schema covers both.
        """
        self._step_grad = None
        if self.model.venice_loss_algebra is not None:
            return self.model.get_venice_compat_losses(
                logits,
                clip_alpha=self.clip_alpha,
                compliance_weight=self.compliance_weight,
            )

        compliance = self.model.get_structural_loss(logits)
        if self.compliance_weight is not None:
            compliance = compliance * float(self.compliance_weight)
        rho_d = 0.0 if self.blend_rho is None else float(self.blend_rho)
        rho_z = 0.0 if self.blend_rho_z is None else float(self.blend_rho_z)
        grad_match = self.blend_mode == BlendMode.GRAD_MATCH
        if self.model.clip_loss is None or (
                grad_match and rho_d == 0.0 and rho_z == 0.0):
            zero = compliance.detach().new_tensor(0.0)
            self._clear_raw_z_clip_state()
            return VeniceLossTerms(
                total_loss=self.model.add_sketch_term(compliance, logits),
                compliance_loss=compliance,
                clip_loss=zero,
                clip_loss_raw=zero,
                clip_weight=zero,
            )
        if grad_match:
            return self._compose_grad_matched(
                logits, compliance, rho_d=rho_d, rho_z=rho_z)
        semantic = self.model.get_semantic_loss(logits)
        if self.clip_alpha is not None:
            weight = float(self.clip_alpha) * compliance.detach()
        else:
            static = _static_clip_weight_or_default(self.clip_weight)
            weight = semantic.detach().new_tensor(static)
        clip_loss = semantic * weight
        self._clear_raw_z_clip_state()
        return VeniceLossTerms(
            total_loss=self.model.add_sketch_term(
                compliance + clip_loss, logits),
            compliance_loss=compliance,
            clip_loss=clip_loss,
            clip_loss_raw=semantic,
            clip_weight=weight,
        )

    def _compose_grad_matched(
        self,
        logits: torch.Tensor,
        compliance: torch.Tensor,
        *,
        rho_d: float,
        rho_z: float,
    ) -> VeniceLossTerms:
        """Grad-match loss, plus its gradient from one backward per term.

        The weights need each term's unweighted gradient norm, so every term
        is differentiated once here. Because the weights are detached, the
        step gradient is exactly ``g_C + w_d g_d + w_z g_z + g_sketch``; it is
        assembled from those same gradients and stashed for ``optimize``
        instead of backpropagating the total a second time. The sum rounds
        differently from autograd's accumulation (~1 ulp), nothing more.
        """
        zero = compliance.detach().new_tensor(0.0)
        g_c = grad_wrt(compliance, logits)
        step_grad = g_c
        total = compliance
        semantic = weight = clip_loss = zero
        if rho_d != 0.0:
            semantic = self.model.get_semantic_loss(logits)
            g_d = grad_wrt(semantic, logits)
            weight = semantic.detach().new_tensor(self._grad_match_weight(
                g_c, g_d, rho=rho_d, ema=self._grad_norm_ema))
            clip_loss = semantic * weight
            total = total + clip_loss
            step_grad = _add_scaled(step_grad, g_d, weight)
        if rho_z != 0.0:
            raw_z = self.model.get_raw_z_clip_loss(logits)
            g_z = grad_wrt(raw_z, logits)
            w_z = raw_z.detach().new_tensor(self._grad_match_weight(
                g_c, g_z, rho=rho_z, ema=self._grad_norm_ema_z))
            total = total + raw_z * w_z
            step_grad = _add_scaled(step_grad, g_z, w_z)
            self.model._last_raw_z_clip = raw_z.detach()
            self.model._last_raw_z_clip_weight = w_z
        else:
            self._clear_raw_z_clip_state()
        # Sketch, live-prior, and density-CLIP terms are purely additive, so
        # composing them on a constant zero isolates their gradient.
        sketch = self.model.add_sketch_term(zero, logits)
        step_grad = _add_scaled(step_grad, grad_wrt(sketch, logits), 1.0)
        self._step_grad = step_grad
        return VeniceLossTerms(
            total_loss=total + sketch,
            compliance_loss=compliance,
            clip_loss=clip_loss,
            clip_loss_raw=semantic,
            clip_weight=weight,
        )

    def _clear_raw_z_clip_state(self) -> None:
        self.model._last_raw_z_clip = None
        self.model._last_raw_z_clip_weight = None

    @staticmethod
    def _grad_match_weight(
        g_c: Optional[torch.Tensor],
        g_term: Optional[torch.Tensor],
        *,
        rho: float,
        ema: GradNormEma,
    ) -> float:
        """Detached ``rho * EMA||g_C|| / EMA||g_term||``; occupancy excluded."""
        ema.update(grad_norm(g_c), grad_norm(g_term))
        return ema.weight(rho)

    def _reset_run_state(self) -> None:
        """Discard everything a previous `optimize` call left behind.

        Every per-run field has to be cleared TOGETHER or not at all. The
        tracker and `loss_terms` accumulate across calls while `stage_of_step`
        and `stage_envs` are local to one call, so clearing only some of them
        leaves `_create_dataset` zipping this call's stages against both calls'
        frames -- which raised `conflicting sizes for dimension 'step'` on the
        second call, or silently rendered a frame through the wrong stage's
        environment when the two grids happened to agree.

        `model.prev_loss` is reset for the same reason: it is the baseline both
        schedule tests threshold against, and carrying the first run's final
        compliance into a second run would make its first step measure a delta
        against a design the second run never visited. The sentinel is larger
        than any first-step compliance, so both tests stay closed until a real
        previous value exists, exactly as on a freshly built model.

        The resolution schedule itself is NOT rewound: `upsample` is one-way,
        so a restarted run continues at whatever grid the first one reached and
        `stage_envs` is rebuilt from there.
        """
        self.tracker.losses.clear()
        self.tracker.frames.clear()
        self.loss_terms = []
        self.motif_scale_terms = []
        self.semantic_prior_terms = []
        self.physical_clip_terms = []
        self.blend_terms = []
        self.resize_steps = []
        self.converged = False
        self.model.prev_loss = INITIAL_PREV_LOSS
        if self._grad_norm_ema is not None:
            self._grad_norm_ema.reset()
        if self._grad_norm_ema_z is not None:
            self._grad_norm_ema_z.reset()
        self._clear_raw_z_clip_state()

    def optimize(self, after_step: Optional[Callable] = None) -> xarray.Dataset:
        """Run Adam, advancing and stopping on the compliance schedule.

        Calling this a second time restarts the run: the recorded trajectory is
        cleared first, so the returned dataset describes this call alone.

        ``after_step(model, step, terms)`` runs after the structure Adam step
        and before the upsample/convergence tests. It must not step
        ``model.parameters()``; occupancy updates belong there. ``None`` keeps
        the Venice schedule bit-identical.
        """
        # Re-checked here because the seam can be flipped on an already-built
        # model, after this optimizer was constructed.
        _reject_clip_weight_under_venice_compat(
            self.model, self.clip_weight, 'AdaptiveAdam_Optimizer')
        self.blend_mode = self._freeze_live_blend_mode()
        self._reset_run_state()
        model = self.model
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)

        # One environment per stage, plus the stage each frame was drawn under,
        # so frames recorded at different resolutions can be rendered later.
        stage_envs = [model.env]
        stage_of_step = []

        pbar = tqdm(range(self.max_iterations), desc="Adaptive Adam")
        for step in pbar:
            optimizer.zero_grad(set_to_none=True)
            _apply_sketch_schedule(self.model, step, self.max_iterations)
            logits = model()
            terms = self._compose_loss(logits)
            loss = terms.total_loss
            blend_snap = (
                snapshot_blend(
                    model, terms, logits, grads=self.blend_grads)
                if self.blend_log else {})
            if self.blend_log and self.blend_mode == BlendMode.GRAD_MATCH:
                ema = self._grad_norm_ema
                blend_snap = {
                    **blend_snap,
                    'rho': float(self.blend_rho),
                    'ema_g_compliance': (
                        float('nan') if ema is None or ema.g_c is None
                        else ema.g_c),
                    'ema_g_clip_z': (
                        float('nan') if ema is None or ema.g_d is None
                        else ema.g_d),
                }
                if self.blend_rho_z is not None:
                    ema_z = self._grad_norm_ema_z
                    raw_z = getattr(model, '_last_raw_z_clip', None)
                    raw_w = getattr(model, '_last_raw_z_clip_weight', None)
                    blend_snap['rho_z'] = float(self.blend_rho_z)
                    blend_snap['clip_raw_z'] = (
                        float('nan') if raw_z is None
                        else float(raw_z.detach() if torch.is_tensor(raw_z)
                                   else raw_z))
                    blend_snap['clip_raw_z_weight'] = (
                        float('nan') if raw_w is None
                        else float(raw_w.detach() if torch.is_tensor(raw_w)
                                   else raw_w))
                    blend_snap['ema_g_clip_raw_z'] = (
                        float('nan') if ema_z is None or ema_z.g_d is None
                        else ema_z.g_d)
            if self._step_grad is not None:
                logits.backward(self._step_grad)
                self._step_grad = None
            else:
                loss.backward()
            if self.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), self.grad_clip)
            optimizer.step()

            # Both losses describe the design that produced the gradient, i.e.
            # the design before this step, which is what Venice records too.
            step_terms = _detach_loss_terms(terms)
            if after_step is not None:
                after_step(model, step, step_terms)
            compliance_value = step_terms.compliance_loss
            self.tracker.add_step(step_terms.total_loss, logits.detach().cpu().numpy())
            self.loss_terms.append(step_terms)
            self.motif_scale_terms.append(_snapshot_motif_scale_losses(model))
            self.semantic_prior_terms.append(_snapshot_semantic_prior(model))
            self.physical_clip_terms.append(_snapshot_physical_clip(model))
            self.blend_terms.append(blend_snap)
            stage_of_step.append(len(stage_envs) - 1)
            pbar.set_postfix({'compliance': f'{compliance_value:.4f}',
                              'grid': f'{model.shape[2]}x{model.shape[1]}'})

            iteration = step + 1 + VENICE_ITERATION_OFFSET
            if model.can_upsample and (
                    iteration % self.max_resize_iteration == 0
                    or model.threshold_crossed(compliance_value, self.resize_threshold)):
                model.upsample()
                # Adam's moments are keyed on the parameter object, which
                # upsample has just replaced with one of a different shape.
                optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
                if self._grad_norm_ema is not None:
                    self._grad_norm_ema.reset()
                if self._grad_norm_ema_z is not None:
                    self._grad_norm_ema_z.reset()
                stage_envs.append(model.env)
                self.resize_steps.append(step)

            # Deliberately not elif: the final upsample makes the schedule
            # exhausted within this same iteration, and Venice tests for
            # convergence immediately, against the same compliance delta.
            # A projection ramp must reach its final beta, so it runs the cap.
            if (not model.can_upsample
                    and not getattr(model, 'physics_projection_active', False)
                    and model.threshold_crossed(
                        compliance_value, self.convergence_threshold)):
                self.converged = True
                break

            model.prev_loss = compliance_value

        return self._create_dataset(stage_envs, stage_of_step)

    def _create_dataset(self, stage_envs, stage_of_step) -> xarray.Dataset:
        """Collect the run into a dataset spanning more than one resolution.

        `OptimizationTracker.create_dataset` renders every frame through the
        model's current environment, which is only correct when the grid never
        moved. Each frame is rendered here through the environment it was
        optimized under, then repeated up to the final grid so that the whole
        run shares one set of coordinates.

        Raises:
            RuntimeError: if the per-step series disagree on how many steps ran.
                `zip` would truncate silently, so this is checked rather than
                left to surface downstream as an xarray coordinate conflict.
        """
        recorded = len(self.tracker.losses)
        if not (len(self.tracker.frames) == len(stage_of_step)
                == len(self.loss_terms) == len(self.motif_scale_terms)
                == len(self.semantic_prior_terms)
                == len(self.physical_clip_terms)
                == len(self.blend_terms)
                == recorded):
            raise RuntimeError(
                'per-step series disagree after the run: '
                f'{recorded} losses, {len(self.tracker.frames)} frames, '
                f'{len(stage_of_step)} stage labels, '
                f'{len(self.loss_terms)} loss breakdowns, '
                f'{len(self.motif_scale_terms)} motif-scale snapshots, '
                f'{len(self.semantic_prior_terms)} semantic-prior snapshots, '
                f'{len(self.physical_clip_terms)} density-CLIP snapshots and '
                f'{len(self.blend_terms)} blend snapshots. Every '
                'one of them is appended once per gradient step, so a mismatch '
                'means state from another run leaked in; `optimize` clears all '
                'of them before it starts.')

        _, height, width = self.model.full_shape
        designs = [
            repeat_to_shape(
                np.asarray(stage_envs[stage].render(frame, volume_constraint=True)),
                height, width)
            for frame, stage in zip(self.tracker.frames, stage_of_step)
        ]

        losses = self.tracker.losses
        logging.info(f'Final loss: {losses[int(np.nanargmin(losses))]}')
        data = {'loss': (('step',), losses)}
        if self.tracker.save_intermediate_designs:
            data['design'] = (('step', 'y', 'x'), designs)
            # Raw z per step, block-repeated so each stage's native grid is
            # recoverable exactly by striding (see figures.native_raw_frames).
            raw_planes = [
                np.asarray(stage_envs[stage].reshape(frame), dtype=np.float32)
                for frame, stage in zip(self.tracker.frames, stage_of_step)
            ]
            data['design_raw'] = (
                ('step', 'y', 'x'),
                [repeat_to_shape(plane, height, width) for plane in raw_planes])
            data['design_raw_height'] = (
                ('step',), [plane.shape[0] for plane in raw_planes])
            data['design_raw_width'] = (
                ('step',), [plane.shape[1] for plane in raw_planes])
        else:
            data['design'] = (('y', 'x'), designs[int(np.nanargmin(losses))])

        ds = xarray.Dataset(data, coords={'step': np.arange(len(losses))})
        _attach_loss_terms(ds, self.loss_terms)
        _attach_motif_scale_terms(ds, self.motif_scale_terms)
        _attach_semantic_prior_terms(ds, self.semantic_prior_terms)
        _attach_physical_clip_terms(ds, self.physical_clip_terms)
        _attach_blend_terms(ds, self.blend_terms)
        ds.attrs['resize_steps'] = list(self.resize_steps)
        ds.attrs['converged'] = int(self.converged)
        if self.blend_log:
            ds.attrs['blend_mode'] = self.blend_mode.value
            if self.blend_rho is not None:
                ds.attrs['blend_rho'] = float(self.blend_rho)
            if self.blend_rho_z is not None:
                ds.attrs['blend_rho_z'] = float(self.blend_rho_z)
        return ds


class LBFGS_Optimizer(BaseOptimizer):
    """L-BFGS optimization algorithm."""
    
    def __init__(self, model, max_iterations: int, save_intermediate_designs: bool = True,
                 lr: float = 1.0, history_size: int = 100, line_search: str = 'strong_wolfe',
                 tol_rel: float = 1e-3, tol_abs: float = 1e-2, patience: int = 5, 
                 min_steps: int = 20, coarse_start: bool = True,
                 clip_weight_max: Optional[float] = None,
                 clip_warmup_steps: int = 0,
                 clip_alpha: Optional[float] = None,
                 compliance_weight: Optional[float] = None):
        super().__init__(model, max_iterations, save_intermediate_designs)
        self.lr = lr
        self.history_size = history_size
        self.line_search = line_search
        self.tol_rel = tol_rel
        self.tol_abs = tol_abs
        self.patience = patience
        self.min_steps = min_steps
        self.coarse_start = coarse_start
        self._lam_clip = None
        self.clip_weight_max = (
            None if clip_weight_max is None else float(clip_weight_max))
        self.clip_warmup_steps = int(clip_warmup_steps)
        self.clip_alpha = clip_alpha
        self.compliance_weight = compliance_weight
        self._baseline_Ls = None
        self._clip_dynamic_w_max = CLIP_DYNAMIC_WEIGHT_MAX
        # One detached VeniceLossTerms per step, appended in step with the
        # tracker (see `Adam_Optimizer`). Only filled on the preset path.
        self.loss_terms = []
        _reject_clip_alpha_under_venice_compat(model, clip_alpha, 'LBFGS_Optimizer')
        _reject_clip_weight_under_venice_compat(
            model, self.clip_weight_max, 'LBFGS_Optimizer',
            arg_name='clip_weight_max')
    
    def optimize(self) -> xarray.Dataset:
        """Run L-BFGS optimization."""
        # Re-checked here because the seam can be flipped on an already-built
        # model, after this optimizer was constructed.
        _reject_clip_alpha_under_venice_compat(
            self.model, self.clip_alpha, 'LBFGS_Optimizer')
        _reject_clip_weight_under_venice_compat(
            self.model, self.clip_weight_max, 'LBFGS_Optimizer',
            arg_name='clip_weight_max')
        venice_compat = self.model.venice_loss_algebra is not None
        clip_weight_max = _static_clip_weight_or_default(self.clip_weight_max)
        opt = torch.optim.LBFGS(
            self.model.parameters(),
            lr=self.lr,
            history_size=self.history_size,
            max_iter=1,
            line_search_fn=self.line_search
        )
        
        pbar = tqdm(range(self.max_iterations), desc="L-BFGS")
        prev_loss = None
        stall = 0
        fine_start = None
        fine_steps = 10
        
        for step in pbar:
            _apply_sketch_schedule(self.model, step, self.max_iterations)
            if not self.coarse_start:
                self.model.analysis_factor = 1
                self.model.analysis_env = self.model.env
            
            if isinstance(self.model, CNNModel) and step > 15:
                self.model._unfreeze_all()

            with torch.no_grad():
                logits_probe = self.model()
            # The calibration below costs an extra FEA solve, so it is skipped
            # wherever its lambda would go unused -- including the preset path,
            # whose weight comes from the legacy algebra instead.
            if self.model.clip_loss is None or self.clip_alpha is not None or venice_compat:
                self._lam_clip = None
            elif self._lam_clip is None or step == 0 or step % 10 == 0:
                # clip_R is used only when clip_loss exists
                if not hasattr(self.model, "clip_R"):
                    self.model.clip_R = 1.0
                self._lam_clip = calibrate_lambda_clip(self.model, logits_probe, R=self.model.clip_R, ortho=True)
            
            if self.model.clip_loss is None or clip_weight_max <= 0:
                clip_weight = 0.0
            elif self.clip_warmup_steps > 0:
                clip_weight = clip_weight_max * min(1.0, (step + 1) / float(self.clip_warmup_steps))
            else:
                clip_weight = clip_weight_max
            
            # The line search calls `closure` several times per step, but
            # `opt.step` returns the FIRST evaluation's loss, so only the first
            # step's terms are logged -- they are the ones `loss_val` describes.
            step_terms = []

            def closure():
                opt.zero_grad(set_to_none=True)
                logits = self.model()
                if venice_compat:
                    # The legacy algebra owns the coupling outright, so neither
                    # `clip_weight` nor the cap above applies. One call, one FEA
                    # solve, and the whole breakdown for the parity log.
                    terms = self.model.get_venice_compat_losses(
                        logits, compliance_weight=self.compliance_weight)
                    loss = terms.total_loss
                    loss.backward()
                    _make_grads_contiguous(self.model)
                    step_terms.append(_detach_loss_terms(terms))
                    return loss
                if self.clip_alpha is not None:
                    # Inverse-normalized, capped dynamic coupling
                    Ls = self.model.get_structural_loss(logits)
                    Lc = self.model.get_semantic_loss(logits) if self.model.clip_loss is not None else Ls.new_tensor(0.0)
                    Ls_eff = Ls if self.compliance_weight is None else (Ls * float(self.compliance_weight))
                    if self._baseline_Ls is None:
                        self._baseline_Ls = float(Ls_eff.detach())
                    denom = max(self._baseline_Ls if self._baseline_Ls is not None else 1.0, 1e-8)
                    raw_w = float(self.clip_alpha) * (self._baseline_Ls / (float(Ls_eff.detach()) + 1e-8))
                    w_eff = min(raw_w, self._clip_dynamic_w_max)
                    loss = Ls_eff + Lc * w_eff
                    loss = self.model.add_sketch_term(loss, logits)
                    loss.backward()
                    _make_grads_contiguous(self.model)
                    return loss
                if self._lam_clip is None:
                    # No semantic loss configured; fall back to structural-only if clip_loss is absent.
                    if self.model.clip_loss is None:
                        if self.compliance_weight is None:
                            loss = self.model.get_structural_loss(logits)
                        else:
                            loss = self.model.get_structural_loss(logits) * float(self.compliance_weight)
                    else:
                        loss = self.model.get_semantic_loss(logits) * clip_weight
                    loss = self.model.add_sketch_term(loss, logits)
                    loss.backward()
                    _make_grads_contiguous(self.model)
                    return loss
                loss_structural = self.model.get_structural_loss(logits)
                loss_semantic = self.model.get_semantic_loss(logits)
                if self.compliance_weight is not None:
                    loss_structural = loss_structural * float(self.compliance_weight)
                loss = loss_structural + loss_semantic * self._lam_clip * clip_weight
                loss = self.model.add_sketch_term(loss, logits)
                loss.backward()
                _make_grads_contiguous(self.model)
                return loss
            
            loss = opt.step(closure)
            loss_val = float(loss.detach())
            
            if step_terms:
                self.loss_terms.append(step_terms[0])
            self.tracker.add_step(loss_val, self.model().detach().cpu().numpy())

            if prev_loss is not None:
                d = loss_val - prev_loss
                rel = abs(d) / (abs(prev_loss) + 1e-12)
                pbar.set_postfix({'loss': f'{loss_val:.6f}', 'Δ': f'{d:.2e}', 'relΔ': f'{rel:.2e}'})
                
                if (abs(d) <= self.tol_abs) or (rel <= self.tol_rel):
                    stall += 1
                else:
                    stall = 0
                
                if fine_start is None and (step + 1) >= self.min_steps and stall >= self.patience:
                    if (self.coarse_start and isinstance(self.model, PixelModel) and 
                        getattr(self.model, 'analysis_factor', 1) != 1):
                        self.model._set_analysis_factor(reset=True)
                        fine_start = step
                    else:
                        pbar.set_postfix({'loss': f'{loss_val:.6f}', 'stopped_at': step})
                        break
                
                if fine_start is not None and (step - fine_start + 1) >= fine_steps:
                    pbar.set_postfix({'loss': f'{loss_val:.6f}', 'stopped_at': step})
                    break
            
            prev_loss = loss_val
        
        return _attach_loss_terms(self.tracker.create_dataset(self.model), self.loss_terms)


class MMA_Optimizer(BaseOptimizer):
    """Method of Moving Asymptotes optimization algorithm."""
    
    def __init__(self, model, max_iterations: int, save_intermediate_designs: bool = True, 
                 init_model=None):
        super().__init__(model, max_iterations, save_intermediate_designs)
        self.init_model = init_model
        self._validate_model_type(models.PixelModel, "MMA")
        _reject_venice_compat_under_physics_only(model, 'MMA_Optimizer')
    
    def optimize(self) -> xarray.Dataset:
        """Run MMA optimization."""
        # Re-checked here because the seam can be flipped on an already-built
        # model, after this optimizer was constructed. Ahead of the imports so
        # a contradictory configuration reports itself rather than an optional
        # dependency that was never going to help.
        _reject_venice_compat_under_physics_only(self.model, 'MMA_Optimizer')

        import nlopt  # pylint: disable=g-import-not-at-top
        import autograd  # pylint: disable=g-import-not-at-top

        env = self.model.env
        if self.init_model is None:
            x0 = get_variables(self.model).astype(np.float64)
        else:
            x0 = constrained_logits(self.init_model).ravel()
        
        pbar = tqdm(total=self.max_iterations, desc="MMA Optimization")
        
        def objective(x):
            return env.objective(x, volume_constraint=False)
        
        def constraint(x):
            return env.constraint(x)
        
        def wrap_autograd_func(func, losses=None, frames=None):
            def wrapper(x, grad):
                if grad.size > 0:
                    value, grad[:] = autograd.value_and_grad(func)(x)
                else:
                    value = func(x)
                if losses is not None:
                    losses.append(value)
                if frames is not None:
                    frames.append(env.reshape(x).copy())
                    pbar.update(1)
                return value
            return wrapper
        
        opt = nlopt.opt(nlopt.LD_MMA, x0.size)
        opt.set_min_objective(wrap_autograd_func(objective, self.tracker.losses, self.tracker.frames))
        opt.add_inequality_constraint(wrap_autograd_func(constraint))
        opt.set_lower_bounds(1e-2)
        opt.set_upper_bounds(1.0)
        opt.set_maxeval(self.max_iterations)
        
        try:
            x = opt.optimize(x0)
            logging.info('MMA optimization completed successfully')
        except Exception as e:
            logging.info(f'MMA optimization stopped: {type(e).__name__}: {e}')
        finally:
            pbar.close()
        
        # Print min/max values across all designs
        designs_array = np.array(self.tracker.frames)
        print(f"Designs min value: {designs_array.min():.4f}")
        print(f"Designs max value: {designs_array.max():.4f}")
        
        return self.tracker.create_dataset(self.model)


class OptimalityCriteria_Optimizer(BaseOptimizer):
    """Optimality Criteria optimization algorithm."""
    
    def __init__(self, model, max_iterations: int, save_intermediate_designs: bool = True, 
                 init_model=None):
        super().__init__(model, max_iterations, save_intermediate_designs)
        self.init_model = init_model
        self._validate_model_type(models.PixelModel, "Optimality criteria")
        _reject_venice_compat_under_physics_only(
            model, 'OptimalityCriteria_Optimizer')
    
    def optimize(self) -> xarray.Dataset:
        """Run Optimality Criteria optimization."""
        # Re-checked here because the seam can be flipped on an already-built
        # model, after this optimizer was constructed.
        _reject_venice_compat_under_physics_only(
            self.model, 'OptimalityCriteria_Optimizer')

        from physics import physics

        env = self.model.env
        nely, nelx = env.args['nely'], env.args['nelx']
        expected_size = nely * nelx
        
        # Initialize design
        if self.init_model is None:
            x = get_variables(self.model).astype(np.float64)
        else:
            x = constrained_logits(self.init_model).ravel()
        
        x = ensure_array_size(x, expected_size, "design array")
        
        for i in tqdm(range(self.max_iterations), desc="Optimality Criteria"):
            try:
                step_result = physics.optimality_criteria_step(x, env.ke, env.args)
                
                if isinstance(step_result, tuple):
                    x_new = step_result[1]
                else:
                    x_new = step_result
                
                x_new = ensure_array_size(x_new, expected_size, f"step {i} result")
                x = x_new
                
                loss = env.objective(x, volume_constraint=False)
                frame = x.reshape(nely, nelx)
                
                self.tracker.add_step(loss, frame.copy())
                
                if i % max(1, self.max_iterations // 10) == 0:
                    logging.info(f'step {i}, loss {loss:.6f}')
                    
            except Exception as e:
                logging.warning(f'Step {i} failed: {e}')
                break
        
        # Ensure we have results
        if not self.tracker.losses:
            self.tracker.losses = [0.0]
            self.tracker.frames = [np.zeros((nely, nelx))]
        
        return self.tracker.create_dataset(self.model)
