"""The dream-layout settings behind the fern and butterfly paper figures.

The hardfork strips differ by CLIP text only. Grid, dream size, coadaptation,
and grad-match are shared. ``--clip`` passes that text through unchanged.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, replace


def prompt_slug(prompt: str) -> str:
    """Filesystem token for a CLIP prompt. Empty after stripping is rejected."""
    stripped = prompt.strip()
    if not stripped:
        raise ValueError('prompt must be non-empty')
    slug = re.sub(r'[^a-z0-9]+', '_', stripped.lower()).strip('_')
    if not slug:
        raise ValueError(f'prompt {prompt!r} has no filesystem-safe characters')
    return slug


@dataclass(frozen=True)
class DreamLayoutPreset:
    """One replicable dream-then-physics run.

    Defaults match the skeleton-recipe figures: a 32x16 soft-rank dream,
    coadaptation, and grad-match (``blend_rho`` 1, ``blend_rho_z`` 0.75)
    on ``multistory_building`` at 128x256.
    """

    problem_name: str = 'multistory_building'
    width: int = 128
    height: int = 256
    density: float = 0.3
    interval: int = 64
    filter_width: float = 2.0
    penal: float = 3.0

    resize_num: int = 2
    resize_scale: int = 2
    seed: int = 12
    init_noise_amp: float = 0.01
    union_load_sites: bool = False

    clip_model_name: str = 'ViT-B/32'
    clip_rn_model_name: str = 'RN50'
    clip_prompt: str = 'fern fronds'
    num_augs: int = 32
    clip_resize_short_side: int = 512
    clip_alpha: float = 10.0
    compliance_weight: float = 1.0
    device: str = 'cpu'

    lr: float = 0.2
    max_iterations: int = 200
    resize_threshold: float = 0.5
    max_resize_iteration: int = 50
    convergence_threshold: float = 0.05

    dream_steps: int = 64
    dream_lr: float = 0.2
    control_height: int = 32
    control_width: int = 16

    sketch_weight_start: float = 4000.0
    sketch_weight_end: float = 400.0

    coadapt: bool = True
    coadapt_interval: int = 5
    coadapt_until: float = 0.6
    coadapt_release: bool = False
    mask_lr: float = 0.05
    ema_decay: float = 0.9
    mask_clip_weight: float = 1.0
    saliency_weight: float = 1.0
    overlap_weight: float = 1.0
    area_weight: float = 10.0
    anchor_weight: float = 1.0
    deficit_weight: float = 0.0

    blend_rho: float = 1.0
    blend_rho_z: float = 0.75
    physical_clip_projection_beta_max: float = 8.0
    physical_clip_projection_sigma: float = 2.0
    physical_clip_projection_sigma_end: float = 0.5
    tiled_scales: tuple = ()
    sketch_path: str | None = None
    sketch_init: bool = True
    use_sketch_weight: bool = True
    structure_kind: str = 'building'
    beta: float | None = None
    heavyside: bool = False

    def with_clip(self, clip_prompt: str) -> 'DreamLayoutPreset':
        stripped = clip_prompt.strip()
        if not stripped:
            raise ValueError('CLIP prompt must be non-empty')
        return replace(self, clip_prompt=stripped)

    def with_problem(self, problem_name: str) -> 'DreamLayoutPreset':
        name = problem_name.strip()
        if not name:
            raise ValueError('problem name must be non-empty')
        return replace(self, problem_name=name)

    def with_device(self, device: str) -> 'DreamLayoutPreset':
        return replace(self, device=device)


PAPER = DreamLayoutPreset()
