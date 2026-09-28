"""Venice-compatible adaptive pixel model with a progressive resolution schedule.

`AdaptivePixelModel` reproduces the legacy `Ada` model from the Venice repo
(`models.py` lines 116-158) instead of extending `PixelModel`, because the two
differ in ways that decide whether a legacy run reproduces:

* `forward` returns the design parameter RAW. The parameter is a density-like
  field that `physics.physical_density` squashes itself, so it is unbounded --
  the reference run's tensor spans -11.78 to 13.12. Nothing is applied at the
  model boundary.
* Training STARTS coarse, at the configured resolution divided by
  `resize_scale ** resize_num`. `interval` -- the row spacing between the
  loaded floors of `multistory_building` -- is divided with it, so every stage
  carries the same number of floors rather than the same spacing.
* `upsample` bilinearly resizes the parameter and rebuilds the problem. It does
  NOT transfer through logit space, preserve the mean, or clamp, all of which
  `PixelModel.upsample` does. Those steps are right for `PixelModel`, whose
  parameter is logits; this parameter is not logits, and squashing it here is
  exactly what the reference run's value range rules out.

This model owns the resolution schedule. The policy deciding *when* to fire it
lives in `train.optimizers.AdaptiveAdam_Optimizer`.
"""

from typing import Optional, Tuple, Union
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from neural_structural_optimization.physics import physics
from neural_structural_optimization.problem.problems import StructuralParams
from .model_base import Model

# Venice's legacy schedule, from source/config/training_parameters.py lines
# 65-69 and confirmed against the reference run's logged config.
DEFAULT_RESIZE_NUM = 2
DEFAULT_RESIZE_SCALE = 2
DEFAULT_RESIZE_THRESHOLD = 0.5
DEFAULT_MAX_RESIZE_ITERATION = 50
DEFAULT_CONVERGENCE_THRESHOLD = 0.05

# Venice seeds its previous-compliance sentinel with this (models.py line 123),
# which is larger than any first-iteration compliance and so keeps both delta
# tests False until a real previous value exists.
INITIAL_PREV_LOSS = 100000.0


def _stage_filter_width(
    filter_width: Union[float, str, None],
    nelx: int,
    nely: int,
) -> Union[float, str, None]:
    """Return a cone-filter radius usable on a coarse stage grid.

    `check_filter_width` rejects a radius past the grid diagonal, and a radius
    the user authored for the full grid can exceed the diagonal of a coarse
    stage -- a 40-element radius is fine on 128x256 and impossible on 32x64.
    The coarse grids come from the resolution schedule rather than from the
    user, so an unreachable radius is clamped with a warning rather than
    refusing to start the run. The final stage is the authored grid, where the
    clamp is inert and the authored radius is used exactly.
    """
    if not isinstance(filter_width, (int, float)) or isinstance(filter_width, bool):
        return filter_width
    limit = physics.max_filter_width(nelx, nely)
    if filter_width <= limit:
        return filter_width
    warnings.warn(
        f'filter_width of {filter_width} exceeds the {nelx}x{nely} diagonal of '
        f'{limit:.4g} on this stage of the resolution schedule; clamping to '
        f'{limit:.4g} for the stage. Coarse stages are derived by dividing the '
        'configured grid, so the radius is unreachable there through no fault '
        'of the configuration; the full-resolution stage uses the configured '
        'radius unchanged.',
        stacklevel=3)
    return limit


def _stage_params(
    full: StructuralParams,
    resize_scale: int,
    remaining: int,
) -> StructuralParams:
    """Build the structural parameters `remaining` upsamples below `full`.

    Every stage is derived from `full` rather than from the stage before it, so
    the last stage (`remaining == 0`) is the configured problem exactly, with
    no drift from repeated integer division and multiplication.
    """
    divisor = resize_scale ** remaining
    width = full.width // divisor
    height = full.height // divisor
    return full.copy(
        width=width,
        height=height,
        # `interval` is a row spacing, so it divides with the grid to keep the
        # floor count fixed. Clamped because a zero spacing is an empty slice.
        interval=max(1, full.interval // divisor),
        filter_width=_stage_filter_width(full.filter_width, width, height),
    )


class AdaptivePixelModel(Model):
    """Direct pixel model trained coarse-to-fine on a Venice-compatible schedule."""

    def __init__(
        self,
        structural_params: Optional[StructuralParams | dict] = None,
        clip_loss: Optional[object] = None,
        seed: Optional[int] = None,
        resize_num: int = DEFAULT_RESIZE_NUM,
        resize_scale: int = DEFAULT_RESIZE_SCALE,
    ):
        """Build the model at the coarsest stage of its resolution schedule.

        Args:
            structural_params: the problem at its FULL (final) resolution.
                Training starts at `width` and `height` divided by
                `resize_scale ** resize_num` and ends here.
            clip_loss: optional semantic loss, forwarded to `Model`.
            seed: optional random seed, forwarded to `Model`.
            resize_num: number of upsamples in the schedule. Zero trains at the
                full resolution throughout.
            resize_scale: integer factor each upsample multiplies the grid by.

        Raises:
            ValueError: if the schedule is unusable, or if the full resolution
                is not divisible by it -- an indivisible grid cannot start
                coarse and land back on the configured resolution.
        """
        if structural_params is None:
            raise ValueError(
                'AdaptivePixelModel requires structural_params: the resolution '
                'schedule is derived from the full-resolution width and height.')
        full_params = (
            structural_params if isinstance(structural_params, StructuralParams)
            else StructuralParams(**structural_params))

        if int(resize_num) != resize_num or int(resize_num) < 0:
            raise ValueError(
                f'resize_num must be a non-negative integer, got {resize_num!r}.')
        if int(resize_scale) != resize_scale or int(resize_scale) < 2:
            raise ValueError(
                f'resize_scale must be an integer of at least 2, got '
                f'{resize_scale!r}. A scale of 1 makes every upsample a no-op, '
                'so the schedule would never reach a finer grid.')
        resize_num = int(resize_num)
        resize_scale = int(resize_scale)

        divisor = resize_scale ** resize_num
        if full_params.width % divisor or full_params.height % divisor:
            raise ValueError(
                f'{full_params.width}x{full_params.height} is not divisible by '
                f'resize_scale ** resize_num = {divisor}, so the schedule '
                f'would start at '
                f'{full_params.width // divisor}x{full_params.height // divisor} '
                f'and end at '
                f'{(full_params.width // divisor) * divisor}x'
                f'{(full_params.height // divisor) * divisor} rather than at '
                'the resolution you configured. Choose a width and height '
                'divisible by the schedule, or reduce resize_num.')

        super().__init__(
            structural_params=_stage_params(full_params, resize_scale, resize_num),
            clip_loss=clip_loss,
            seed=seed,
        )

        self.full_params = full_params
        self.resize_num = resize_num
        self.resize_scale = resize_scale
        self.resizes = 0
        self.prev_loss = INITIAL_PREV_LOSS

        # `broadcast_to` returns a read-only view, which torch warns about, so
        # materialise it before handing it over.
        z_init = np.ascontiguousarray(np.broadcast_to(
            self.env.args['volfrac'] * self.env.args['mask'],
            self.shape
        ))
        self.z = nn.Parameter(
            torch.as_tensor(z_init, dtype=torch.float32, device=self.device),
            requires_grad=True
        )

    @property
    def full_shape(self) -> Tuple[int, int, int]:
        """Shape of the design grid at the final stage of the schedule."""
        return (1, self.full_params.height, self.full_params.width)

    @property
    def can_upsample(self) -> bool:
        """Whether any upsamples remain in the schedule."""
        return self.resizes < self.resize_num

    def forward(self) -> torch.Tensor:
        """Forward pass - return the design parameters raw and unbounded."""
        return self.z

    def loss(self, logits: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Compute the total loss, for the current design unless given one."""
        if logits is None:
            logits = self.forward()
        return self.get_total_loss(logits)

    def threshold_crossed(self, compliance: float, threshold: float) -> bool:
        """Whether compliance moved less than `threshold` since `prev_loss`.

        Both schedule tests -- when to upsample and when to stop -- run on
        compliance rather than on the total loss, so a semantic term that is
        still moving cannot hold either one open.
        """
        return abs(self.prev_loss - compliance) < threshold

    @torch.no_grad()
    def upsample(self) -> None:
        """Advance one stage: resize the design and rebuild the problem.

        The parameter is resized bilinearly and carried over as-is. Venice does
        no logit-space transfer, no mean preservation, and no clamping here,
        and each of those would change the field the physics backend sees.

        Raises:
            RuntimeError: if the schedule has no upsamples left.
        """
        if not self.can_upsample:
            raise RuntimeError(
                f'upsample called after all {self.resize_num} upsamples in the '
                'schedule have run; the model is already at its full resolution.')

        self.resizes += 1
        params = _stage_params(
            self.full_params, self.resize_scale, self.resize_num - self.resizes)

        # Equivalent to Venice's transforms.Resize((height, width)): that also
        # resolves to bilinear interpolation with align_corners=False, and its
        # antialiasing applies only when downsampling.
        z_fine = F.interpolate(
            self.z.detach().unsqueeze(0),
            size=(params.height, params.width),
            mode='bilinear',
            align_corners=False,
        ).squeeze(0)

        self.structural_params = params
        self._refresh_physics_environment()
        self.mask = torch.as_tensor(self.args['mask'], dtype=torch.float64)

        # Venice has no analysis grid: every stage solves the physics on the
        # grid it trains on. Keeping the factor at 1 also keeps `analysis_env`
        # from going stale against the environment just rebuilt.
        self.analysis_factor = 1
        self.analysis_env = self.env

        self.z = nn.Parameter(z_fine, requires_grad=True)
