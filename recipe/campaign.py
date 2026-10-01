"""One campaign row: any mode, one attempt directory, the output contract."""

from __future__ import annotations

import hashlib
import json
import os
import socket
import subprocess
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from figures import save_field_png, write_comparison, write_progress_gif
from guidance.clip_scales import parse_clip_scales
from guidance.loss_semantic_prior import (
    CoadaptiveMask,
    report_design_metrics,
    save_design_arrays,
)
from guidance.loss_sketch import (
    apply_scaffold_as_occupancy_prior,
    load_site_mask,
    load_sketch_occupancy,
)
from model.model_base import PHYSICAL_CLIP_INK_WRAP
from optimize.optimizers import AdaptiveAdam_Optimizer
from physics import physics
from recipe.campaign_spec import STRUCTURES
from recipe.dream_layout import (
    _attach_final,
    _coadapt_hook,
    _last,
    _plane,
    _to_json,
    build_clip_loss,
    build_model,
    extract_soft_scaffold,
    git_revision,
    run_clip_dream,
    seed_everything,
)
from recipe.preset import PAPER, DreamLayoutPreset
from runtime import configure_torch_threads

_REPO = Path(__file__).resolve().parents[1]


def onoff(value: str) -> bool:
    token = str(value).strip().lower()
    if token in ('on', 'true', '1', 'yes'):
        return True
    if token in ('off', 'false', '0', 'no'):
        return False
    raise ValueError(f'expected on/off, got {value!r}')


def pip_freeze_hash() -> str:
    try:
        text = subprocess.check_output(
            ['pip', 'freeze'], cwd=_REPO, text=True,
            stderr=subprocess.DEVNULL)
    except (OSError, subprocess.CalledProcessError):
        return 'unknown'
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def next_attempt_dir(run_root: Path) -> Path:
    run_root = Path(run_root)
    existing = [
        int(path.name.split('_', 1)[1])
        for path in run_root.glob('attempt_*')
        if path.name.split('_', 1)[-1].isdigit()
    ]
    index = (max(existing) + 1) if existing else 1
    path = run_root / f'attempt_{index}'
    path.mkdir(parents=True, exist_ok=True)
    return path


def resize_num_for(width: int, height: int, preferred: int, scale: int = 2) -> int:
    for count in range(int(preferred), -1, -1):
        divisor = int(scale) ** count
        if width % divisor == 0 and height % divisor == 0:
            return count
    return 0


def preset_from_args(args) -> DreamLayoutPreset:
    spec = STRUCTURES[args.structure]
    scale = float(args.resolution_scale)
    width = max(1, int(round(spec.width * scale)))
    height = max(1, int(round(spec.height * scale)))
    interval = max(1, int(round(spec.interval * scale)))
    beta = None if args.beta_max is None else float(args.beta_max)
    tiled = parse_clip_scales(args.clip_scales)
    prompt = args.clip or PAPER.clip_prompt
    sketch = None
    if args.sketch:
        path = Path(args.sketch)
        sketch = str(path if path.is_absolute() else _REPO / path)
    return replace(
        PAPER,
        problem_name=spec.problem_name,
        width=width,
        height=height,
        density=spec.density,
        interval=interval,
        filter_width=float(args.filter_width),
        penal=float(args.penal),
        seed=int(args.seed),
        device=args.device,
        clip_prompt=prompt,
        blend_rho=float(args.blend_rho),
        sketch_weight_end=float(args.sketch_weight_end),
        coadapt=bool(args.coadapt),
        resize_num=resize_num_for(width, height, spec.resize_num, spec.resize_scale),
        resize_scale=spec.resize_scale,
        control_height=spec.control_height,
        control_width=spec.control_width,
        tiled_scales=tiled,
        sketch_path=sketch,
        sketch_init=bool(args.sketch_init),
        use_sketch_weight=bool(args.sketch_weight),
        structure_kind=spec.kind,
        beta=beta,
        heavyside=beta is not None,
    )


def _apply_sketch(model, preset: DreamLayoutPreset) -> np.ndarray | None:
    if not preset.sketch_path:
        return None
    occupancy = load_sketch_occupancy(
        preset.sketch_path, preset.height, preset.width)
    weight = preset.sketch_weight_start if preset.use_sketch_weight else 0.0
    weight_end = preset.sketch_weight_end if preset.use_sketch_weight else 0.0
    apply_scaffold_as_occupancy_prior(
        model,
        occupancy,
        weight=weight,
        weight_end=weight_end,
        init_from_occupancy=preset.sketch_init,
    )
    if preset.use_sketch_weight:
        return occupancy
    # Weight off: drop the prior term but keep occupancy for metrics / init.
    model.sketch_weight = 0.0
    model.sketch_weight_end = 0.0
    return occupancy


def _enable_physical_clip(model, preset: DreamLayoutPreset) -> None:
    model.enable_venice_compat_loss(False)
    model.enable_physical_clip(
        weight=0.0,
        match_venice=False,
        as_semantic=True,
        projection_beta_max=preset.physical_clip_projection_beta_max,
        projection_sigma=preset.physical_clip_projection_sigma,
        projection_sigma_end=preset.physical_clip_projection_sigma_end,
        prompt_wrap=PHYSICAL_CLIP_INK_WRAP,
    )


def _maybe_coadapt(model, occupancy, preset: DreamLayoutPreset):
    if not preset.coadapt or occupancy is None:
        return None, None
    mask = CoadaptiveMask(
        occupancy,
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
    mask.commit_to_model(model)
    freeze_after = int(preset.coadapt_until * int(preset.max_iterations))
    return mask, _coadapt_hook(mask, preset, freeze_after)


def _run_physics(model, preset: DreamLayoutPreset, after_step=None):
    blend_rho = 0.0 if model.clip_loss is None else float(preset.blend_rho)
    blend_rho_z = (
        0.0 if model.clip_loss is None else float(preset.blend_rho_z))
    optimizer = AdaptiveAdam_Optimizer(
        model,
        max_iterations=int(preset.max_iterations),
        lr=preset.lr,
        compliance_weight=preset.compliance_weight,
        resize_threshold=preset.resize_threshold,
        max_resize_iteration=preset.max_resize_iteration,
        convergence_threshold=preset.convergence_threshold,
        blend_rho=blend_rho,
        blend_rho_z=blend_rho_z,
    )
    return _attach_final(optimizer.optimize(after_step=after_step), model)


def _gray_fraction(density: np.ndarray) -> float:
    flat = np.asarray(density, dtype=np.float64).reshape(-1)
    return float(np.mean((flat > 0.1) & (flat < 0.9)))


def _thresholded_compliance(model, density: np.ndarray) -> float | None:
    binary = (np.asarray(density, dtype=np.float64) >= 0.5).astype(np.float64)
    try:
        displacement = physics.displace(
            binary, model.env.ke, model.env.args['forces'],
            model.env.args['freedofs'], model.env.args['fixdofs'],
            penal=model.env.args['penal'])
        value = physics.compliance(
            binary, displacement, model.env.ke,
            penal=model.env.args['penal'])
    except Exception:
        return None
    value = float(np.asarray(value))
    if not np.isfinite(value):
        return None
    return value


def _write_contract(
    output_dir: Path,
    *,
    args,
    preset: DreamLayoutPreset,
    density: np.ndarray | None,
    raw: np.ndarray | None,
    scaffold: np.ndarray | None,
    ds,
    summary: dict,
    started: float,
    status: str,
) -> dict:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if density is not None:
        save_design_arrays(output_dir, density, raw=raw)
        save_field_png(output_dir / 'physical_density.png', density)
        save_field_png(output_dir / 'final.png', density)
        if ds is not None and 'design' in ds:
            write_progress_gif(
                output_dir / 'progress.gif', np.asarray(ds['design'].values))
        elif scaffold is not None:
            write_progress_gif(
                output_dir / 'progress.gif', np.stack([scaffold, density]))
        else:
            write_progress_gif(
                output_dir / 'progress.gif', density[None, ...])
    elif scaffold is not None:
        save_field_png(output_dir / 'final.png', scaffold)
        write_progress_gif(output_dir / 'progress.gif', scaffold[None, ...])

    nely = preset.height
    nelx = preset.width
    sites = None
    report = {'validity': {}, 'mean_physical_density': None,
              'clip_loss': None, 'clip_loss_raw': None,
              'mass_on_scaffold': None}
    if density is not None and ds is not None:
        # Sites at the *final* grid stored on the model if it exists.
        pass
    record = {
        'run_id': args.run_id,
        'experiment': args.experiment,
        'group': args.group,
        'mode': args.mode,
        'structure': args.structure,
        'prompt': None if args.mode in ('unguided', 'sketch') and not args.prompt_sketch
        else preset.clip_prompt,
        'sketch': preset.sketch_path,
        'cli': {key: getattr(args, key) for key in sorted(vars(args))},
        'git_commit': git_revision(),
        'pip_freeze_hash': pip_freeze_hash(),
        'hostname': socket.gethostname(),
        'slurm_job_id': os.environ.get('SLURM_JOB_ID'),
        'slurm_array_task_id': os.environ.get('SLURM_ARRAY_TASK_ID'),
        'start_time': started,
        'end_time': time.time(),
        'exit_status': status,
        'wall_clock_seconds': time.perf_counter() - summary.get('_t0', 0.0),
        **{k: v for k, v in summary.items() if not str(k).startswith('_')},
    }
    if density is not None:
        record['mean_density'] = float(np.mean(density))
        record['gray_fraction'] = _gray_fraction(density)
        record['volume_fraction'] = float(np.mean(density))
    (output_dir / 'run.json').write_text(_to_json(record))
    if status == 'ok':
        (output_dir / 'DONE').write_text('')
    return record


def run_campaign(args, output_dir: Path) -> dict:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    configure_torch_threads()
    t0 = time.perf_counter()
    started = time.time()
    preset = preset_from_args(args)
    seed_everything(preset.seed)
    mode = args.mode
    if args.prompt_sketch:
        if not preset.sketch_path or mode == 'unguided':
            raise ValueError('--prompt-sketch needs --sketch and a CLIP prompt')
        mode = 'prompt_sketch'

    density = None
    raw = None
    scaffold = None
    ds = None
    summary: dict = {'_t0': t0, 'problem': preset.problem_name, 'seed': preset.seed}

    needs_clip = mode in (
        'semantic', 'hybrid', 'dream_only', 'prompt_sketch')
    clip_loss = build_clip_loss(preset) if needs_clip else None

    if mode == 'dream_only':
        model = build_model(preset, clip_loss, venice_algebra=False)
        model.venice_loss_algebra = None
        dream_losses, dream_logits, _ = run_clip_dream(model, preset)
        coarse = _plane(dream_logits.cpu().numpy())
        upsampled, scaffold = extract_soft_scaffold(coarse, preset)
        np.save(output_dir / 'dream_coarse.npy', coarse)
        np.save(output_dir / 'scaffold.npy', scaffold)
        save_field_png(output_dir / 'scaffold.png', scaffold)
        write_comparison(
            output_dir / 'comparison.png',
            [('Dream scaffold', scaffold)])
        summary.update({
            'compliance': None,
            'dream_clip_loss': float(dream_losses[-1]),
            'steps': int(preset.dream_steps),
        })
        return _write_contract(
            output_dir, args=args, preset=preset, density=None, raw=None,
            scaffold=scaffold, ds=None, summary=summary, started=started,
            status='ok')

    if mode == 'hybrid':
        from recipe.dream_layout import run_dream_layout
        hybrid_summary = run_dream_layout(preset, output_dir)
        density = np.load(output_dir / 'physical_density.npy')
        raw = np.load(output_dir / 'final_design_raw.npy') if (
            output_dir / 'final_design_raw.npy').exists() else None
        scaffold = np.load(output_dir / 'scaffold.npy') if (
            output_dir / 'scaffold.npy').exists() else None
        save_field_png(output_dir / 'final.png', density)
        summary.update(hybrid_summary)
        summary['_t0'] = t0
        # Hybrid already ran FEA. Thresholded compliance uses the live model
        # from this process only if we rebuild — skip a second CHOLMOD risk
        # by using saved density without a new model when possible.
        summary['gray_fraction'] = _gray_fraction(density)
        return _write_contract(
            output_dir, args=args, preset=preset, density=density, raw=raw,
            scaffold=scaffold, ds=None, summary=summary, started=started,
            status='ok')

    venice = False
    model = build_model(
        preset, clip_loss, venice_algebra=venice, init=not (
            preset.sketch_path and preset.sketch_init))
    occupancy = _apply_sketch(model, preset)
    if occupancy is None and (preset.sketch_path is None):
        pass
    if needs_clip:
        _enable_physical_clip(model, preset)
    mask = None
    after_step = None
    if mode == 'prompt_sketch' and occupancy is not None:
        mask, after_step = _maybe_coadapt(model, occupancy, preset)
    physics_started = time.perf_counter()
    ds = _run_physics(model, preset, after_step=after_step)
    summary['physics_seconds'] = time.perf_counter() - physics_started
    density = _plane(ds['final_physical_density'].values)
    raw = _plane(ds['final_design_raw'].values)
    nely = int(model.env.args['nely'])
    nelx = int(model.env.args['nelx'])
    sites = load_site_mask(model.env.args['forces'], nely=nely, nelx=nelx)
    report = report_design_metrics(
        density, sites, scaffold=occupancy, ds=ds)
    panels = [('Physical density', density)]
    if occupancy is not None:
        panels.append(('Occupancy', occupancy))
        scaffold = occupancy
    if mask is not None:
        mask_final = _plane(mask.occupancy_cpu().numpy())
        panels.append(('Co-adapt mask', mask_final))
    write_comparison(output_dir / 'comparison.png', panels)
    summary.update({
        'compliance': _last(ds, 'compliance'),
        'clip_loss': report['clip_loss'],
        'steps': int(np.asarray(ds['loss'].values).reshape(-1).size),
        'mean_physical_density': report['mean_physical_density'],
        'mass_on_scaffold': report['mass_on_scaffold'],
        'validity': report['validity'],
        'gray_fraction': _gray_fraction(density),
        'thresholded_compliance': _thresholded_compliance(model, density),
        'connected_components': report['validity'].get('component_count'),
        'volume_actual': float(np.mean(raw > 0.9)),
    })
    return _write_contract(
        output_dir, args=args, preset=preset, density=density, raw=raw,
        scaffold=scaffold, ds=ds, summary=summary, started=started,
        status='ok')
