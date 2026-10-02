"""Dream, then physics, for one paper figure.

Sequence matches the skeleton-recipe runs: a coarse CLIP dream with no FEA,
a soft-rank scaffold, an occupancy prior, physical-density CLIP as the
semantic term, a coadaptive mask, and grad-match. The tautology gate is not
run; its baseline file lives in the lab archive, and a passing gate does not
change the solve.
"""

from __future__ import annotations

import runtime  # noqa: F401  # pins OpenMP before NumPy and torch load

import json
import random
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from figures import (
    native_raw_frames,
    save_field_png,
    write_comparison,
    write_progress_gif,
)
from guidance.loss_clip import CLIPLoss, VeniceClipPreset
from guidance.loss_semantic_prior import CoadaptiveMask, report_design_metrics
from guidance.loss_semantic_prior import save_design_arrays
from guidance.loss_sketch import (
    apply_scaffold_as_occupancy_prior,
    load_on_solid_fraction,
    load_site_mask,
    rank_ink_from_raw,
    resample_field,
)
from model.model_ada import AdaptivePixelModel
from model.model_base import PHYSICAL_CLIP_INK_WRAP, VeniceLossAlgebra
from optimize.optimizers import AdaptiveAdam_Optimizer
from optimize.utils import init_weight_neutral
from problem.problems import StructuralParams
from recipe.preset import DreamLayoutPreset, prompt_slug
from runtime import configure_torch_threads

_REPO = Path(__file__).resolve().parents[1]


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy, and torch before a loop that consumes those RNGs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def git_revision() -> str:
    """Commit this run was launched from. ``-dirty`` when the tree has changes."""
    try:
        sha = subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'],
            cwd=_REPO,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return 'unknown'
    dirty = subprocess.call(
        ['git', 'diff', '--quiet', 'HEAD'],
        cwd=_REPO,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return sha if dirty == 0 else f'{sha}-dirty'


def structural_params(preset: DreamLayoutPreset) -> StructuralParams:
    kwargs = dict(
        problem_name=preset.problem_name,
        width=preset.width,
        height=preset.height,
        density=preset.density,
        interval=preset.interval,
        filter_width=preset.filter_width,
    )
    if preset.beta is not None:
        kwargs['beta'] = preset.beta
        kwargs['heavyside'] = True
    elif preset.heavyside:
        kwargs['heavyside'] = True
    return StructuralParams(**kwargs)


def build_clip_loss(preset: DreamLayoutPreset) -> CLIPLoss:
    layout = None
    scales = tuple(preset.tiled_scales)
    if scales:
        from guidance.clip_scales import layout_for_structure
        layout = layout_for_structure(
            preset.height, preset.width,
            kind=preset.structure_kind,
            interval=preset.interval,
        )
    return CLIPLoss(
        clip_model_name=preset.clip_model_name,
        clip_rn_model_name=preset.clip_rn_model_name,
        device=torch.device(preset.device),
        positive_prompts=[preset.clip_prompt],
        num_augs=preset.num_augs,
        venice_compat=VeniceClipPreset(
            resize_short_side=preset.clip_resize_short_side,
            num_augs=preset.num_augs,
        ),
        motif_scale_fracs=(),
        tiled_scales=scales,
        tile_layout=layout,
    )


def build_model(
    preset: DreamLayoutPreset,
    clip_loss,
    *,
    venice_algebra: bool = True,
    init: bool = True,
) -> AdaptivePixelModel:
    """Coarse adaptive-pixel model. Venice algebra is opt-in."""
    model = AdaptivePixelModel(
        structural_params=structural_params(preset),
        clip_loss=clip_loss,
        seed=preset.seed,
        resize_num=preset.resize_num,
        resize_scale=preset.resize_scale,
    )
    model.args['penal'] = float(preset.penal)
    model.env.args['penal'] = float(preset.penal)
    model.physics_projection_beta_max = float(preset.physics_projection_beta_max)
    if venice_algebra:
        model.enable_venice_compat_loss(VeniceLossAlgebra(
            clip_alpha=preset.clip_alpha,
            compliance_weight=preset.compliance_weight,
        ))
    if init:
        init_weight_neutral(
            model,
            density=preset.density,
            seed=preset.seed,
            noise_amp=preset.init_noise_amp,
            union_load_sites=preset.union_load_sites,
        )
    return model


def run_clip_dream(model, preset: DreamLayoutPreset):
    """Optimize a smooth low-resolution field. No finite-element solve."""
    if preset.control_height < 2 or preset.control_width < 2:
        raise ValueError('control grid dimensions must both be >= 2')
    if preset.dream_steps < 1:
        raise ValueError('dream_steps must be >= 1')
    generator = torch.Generator(device='cpu')
    generator.manual_seed(int(model.seed))
    control = torch.full(
        (1, 1, preset.control_height, preset.control_width),
        float(model.env.args['volfrac']),
        device=model.device,
    )
    noise = torch.rand(control.shape, generator=generator)
    control = torch.nn.Parameter(
        control + 0.01 * (2.0 * noise.to(model.device) - 1.0))
    optimizer = torch.optim.Adam([control], lr=preset.dream_lr)
    losses = []
    field = None
    for _step in tqdm(range(int(preset.dream_steps)), desc='CLIP dream (no FEA)'):
        optimizer.zero_grad(set_to_none=True)
        logits = F.interpolate(
            control,
            size=model.shape[-2:],
            mode='bilinear',
            align_corners=False,
        )
        logits = F.avg_pool2d(logits, kernel_size=3, stride=1, padding=1)
        logits = logits[:, 0]
        loss = model.get_semantic_loss(logits)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))
        field = logits
    return losses, field.detach(), control.detach()


def extract_soft_scaffold(dream_field: np.ndarray, preset: DreamLayoutPreset):
    """Upsample the dream and keep its percentile ranks as occupancy in [0, 1]."""
    upsampled = resample_field(dream_field, preset.height, preset.width)
    scaffold = rank_ink_from_raw(upsampled)
    return upsampled, scaffold


def _plane(field: np.ndarray) -> np.ndarray:
    arr = np.asarray(field, dtype=np.float32)
    while arr.ndim > 2:
        arr = arr[0]
    return arr


def _coadapt_hook(mask, preset: DreamLayoutPreset, freeze_after: int):
    def after_step(model, step, _terms) -> None:
        if int(step) % int(preset.coadapt_interval) != 0:
            return
        if int(step) >= int(freeze_after):
            return
        logits = model()
        density = model.get_physical_density(logits)
        sites = model._load_sites_on_density(density)
        mask.step(
            density,
            lambda field: model.score_physical_clip(field),
            load_sites=sites,
            use_saliency=True,
            progress=0.0,
        )
        mask.commit_to_model(model)

    return after_step


def _attach_final(ds, model):
    raw = model.z.detach().cpu().numpy()[0]
    ds['final_design_raw'] = (('raw_y', 'raw_x'), raw)
    density = _plane(model.get_physical_density(model.z).detach().cpu().numpy())
    if density.shape != raw.shape:
        raise ValueError(
            f'physical density shape {density.shape} does not match raw {raw.shape}')
    ds['final_physical_density'] = (('raw_y', 'raw_x'), density)
    return ds


def _last(ds, name: str):
    if name not in ds:
        return None
    values = np.asarray(ds[name].values)
    if values.size == 0:
        return None
    return float(values.reshape(-1)[-1])


def run_dream_layout(preset: DreamLayoutPreset, output_dir: Path) -> dict:
    """Run the paper recipe and write images plus ``summary.json``."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    configure_torch_threads()
    started = time.perf_counter()

    seed_everything(preset.seed)
    dream_model = build_model(preset, build_clip_loss(preset))
    dream_model.venice_loss_algebra = None
    seed_everything(preset.seed)

    dream_started = time.perf_counter()
    dream_losses, dream_logits, _control = run_clip_dream(dream_model, preset)
    dream_seconds = time.perf_counter() - dream_started
    dream_coarse = _plane(dream_logits.cpu().numpy())
    upsampled, scaffold = extract_soft_scaffold(dream_coarse, preset)
    np.save(output_dir / 'dream_coarse.npy', dream_coarse)
    np.save(output_dir / 'dream_upsampled.npy', upsampled)
    np.save(output_dir / 'scaffold.npy', scaffold)
    save_field_png(
        output_dir / 'dream_coarse.png', rank_ink_from_raw(dream_coarse))
    save_field_png(
        output_dir / 'dream_upsampled.png', rank_ink_from_raw(upsampled))
    save_field_png(output_dir / 'scaffold.png', scaffold)

    seed_everything(preset.seed)
    physics_clip = build_clip_loss(preset)
    seed_everything(preset.seed)
    physics_model = build_model(preset, physics_clip)
    apply_scaffold_as_occupancy_prior(
        physics_model,
        scaffold,
        weight=preset.sketch_weight_start,
        weight_end=preset.sketch_weight_end,
        init_from_occupancy=True,
    )
    physics_model.enable_venice_compat_loss(False)
    physics_model.enable_physical_clip(
        weight=0.0,
        match_venice=False,
        as_semantic=True,
        projection_beta_max=preset.physical_clip_projection_beta_max,
        projection_sigma=preset.physical_clip_projection_sigma,
        projection_sigma_end=preset.physical_clip_projection_sigma_end,
        prompt_wrap=PHYSICAL_CLIP_INK_WRAP,
    )

    mask = None
    after_step = None
    if preset.coadapt:
        mask = CoadaptiveMask(
            scaffold,
            control_height=preset.control_height,
            control_width=preset.control_width,
            lr=preset.mask_lr,
            ema_decay=preset.ema_decay,
            lambda_clip=preset.mask_clip_weight,
            lambda_saliency=preset.saliency_weight,
            lambda_overlap=preset.overlap_weight,
            lambda_area=preset.area_weight,
            lambda_anchor=preset.anchor_weight,
            lambda_deficit=preset.deficit_weight,
        )
        mask.commit_to_model(physics_model)
        save_field_png(output_dir / 'mask_initial.png', mask.occupancy_cpu().numpy())
        freeze_after = int(preset.coadapt_until * int(preset.max_iterations))
        after_step = _coadapt_hook(mask, preset, freeze_after)

    seed_everything(preset.seed)
    optimizer = AdaptiveAdam_Optimizer(
        physics_model,
        max_iterations=int(preset.max_iterations),
        lr=preset.lr,
        compliance_weight=preset.compliance_weight,
        resize_threshold=preset.resize_threshold,
        max_resize_iteration=preset.max_resize_iteration,
        convergence_threshold=preset.convergence_threshold,
        blend_rho=float(preset.blend_rho),
        blend_rho_z=float(preset.blend_rho_z),
    )
    physics_started = time.perf_counter()
    ds = _attach_final(optimizer.optimize(after_step=after_step), physics_model)
    physics_seconds = time.perf_counter() - physics_started

    density = _plane(ds['final_physical_density'].values)
    raw = _plane(ds['final_design_raw'].values)
    save_design_arrays(output_dir, density, raw=raw)
    save_field_png(output_dir / 'physical_density.png', density)

    panels = [
        ('Physical density', density),
        ('Semantic design (raw z)', raw),
        ('Dream scaffold', scaffold),
    ]
    saliency = None
    if mask is not None:
        mask.commit_to_model(physics_model)
        mask_final = _plane(mask.occupancy_cpu().numpy())
        np.save(output_dir / 'mask_final.npy', mask_final)
        save_field_png(output_dir / 'mask_final.png', mask_final)
        panels.append(('Co-adapt mask', mask_final))
        if mask.last_saliency is not None:
            saliency = _plane(mask.last_saliency.detach().cpu().numpy())
            np.save(output_dir / 'saliency.npy', saliency)
            save_field_png(output_dir / 'saliency.png', saliency)
            panels.append(('CLIP saliency', saliency))
    write_comparison(output_dir / 'comparison.png', panels)
    write_progress_gif(output_dir / 'progress.gif', native_raw_frames(ds))

    nely = int(physics_model.env.args['nely'])
    nelx = int(physics_model.env.args['nelx'])
    sites = load_site_mask(physics_model.env.args['forces'], nely=nely, nelx=nelx)
    report = report_design_metrics(density, sites, scaffold=scaffold, ds=ds)
    total_seconds = time.perf_counter() - started
    summary = {
        'prompt': preset.clip_prompt,
        'problem': preset.problem_name,
        'slug': prompt_slug(preset.clip_prompt),
        'steps': int(np.asarray(ds['loss'].values).reshape(-1).size),
        'converged': bool(ds.attrs.get('converged', 0)),
        'progress_gif_frames': int(np.asarray(ds['design'].values).shape[0]),
        'compliance': _last(ds, 'compliance'),
        'physics_projection_beta_max': preset.physics_projection_beta_max,
        'physics_projection_beta_final': physics_model._last_physics_projection_beta,
        'clip_loss': report['clip_loss'],
        'clip_loss_raw': report['clip_loss_raw'],
        'dream_clip_loss': float(dream_losses[-1]),
        'volume_actual': float(np.mean(raw > 0.9)),
        'mean_physical_density': report['mean_physical_density'],
        'mass_on_scaffold': report['mass_on_scaffold'],
        'validity': report['validity'],
        'load_on_solid_fraction': load_on_solid_fraction(
            density, physics_model.env.args['forces']),
        'scaffold_source': 'soft_rank',
        'scaffold_mean': float(np.mean(scaffold)),
        'control_grid': [preset.control_height, preset.control_width],
        'blend_rho': preset.blend_rho,
        'blend_rho_z': preset.blend_rho_z,
        'device': preset.device,
        'seed': preset.seed,
        'dream_seconds': dream_seconds,
        'physics_seconds': physics_seconds,
        'total_seconds': total_seconds,
        'git_commit': git_revision(),
        'gate': None,
    }
    (output_dir / 'summary.json').write_text(_to_json(summary))
    return summary


def _to_json(value) -> str:
    def convert(item):
        if isinstance(item, dict):
            return {key: convert(child) for key, child in item.items()}
        if isinstance(item, (list, tuple)):
            return [convert(child) for child in item]
        if isinstance(item, np.generic):
            return item.item()
        return item

    return json.dumps(convert(value), indent=2) + '\n'
