"""Expand the experiment table into ``slurm/campaign.tsv``.

Do not hand-edit the TSV. Re-run this script.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from guidance.clip_scales import parse_clip_scales, scales_token
from recipe.campaign_spec import (
    BLEND_RHOS,
    COUNTER_PROMPT,
    F1_SKETCHES,
    F3_SKETCHES,
    GRAVITY_LOAD,
    H3_SKETCHES,
    PROMPTS,
    REPORTED_STRUCTURES,
    SCALE_ARMS,
    SKETCH_ROOT,
    WEIGHT_ENDS,
    prompt_token,
    rho_token,
    sketch_token,
    wend_token,
)

_REPO = Path(__file__).resolve().parents[1]

# Adopted campaign-wide after the projection gate (instructions section 0.5).
PHYSICS_BETA_MAX = 8.0


def _flag(name: str, value) -> list[str]:
    if isinstance(value, bool):
        return [name, 'on' if value else 'off']
    return [name, str(value)]


def _row(
        run_id: str,
        experiment: str,
        group: str,
        argv: list[str],
        *,
        physics_beta_max: float = PHYSICS_BETA_MAX) -> dict:
    return {
        'run_id': run_id,
        'argv': [
            '--run-id', run_id,
            '--experiment', experiment,
            '--group', group,
            *argv,
            '--physics-beta-max', f'{physics_beta_max:.6g}',
        ],
    }


def _structure_args(key: str) -> list[str]:
    return ['--structure', key]


def latin_hypercube(n: int, dim: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    cut = np.linspace(0.0, 1.0, n + 1)
    samples = cut[:n, None] + rng.random((n, dim)) * (cut[1] - cut[0])
    for axis in range(dim):
        rng.shuffle(samples[:, axis])
    return samples


def _lerp(lo: float, hi: float, u: float) -> float:
    return float(lo + (hi - lo) * u)


def experiment_rows(*, include_s1b: bool) -> list[dict]:
    rows: list[dict] = []

    for key in REPORTED_STRUCTURES:
        rows.append(_row(
            f'B/{key}/unguided', 'B', 'baseline',
            ['--mode', 'unguided', *_structure_args(key)],
        ))

    for stem in F1_SKETCHES:
        token = sketch_token(stem)
        for arm, init, weight in (
                ('init', True, False),
                ('weight', False, True),
                ('both', True, True)):
            rows.append(_row(
                f'F1/tall/{token}/{arm}', 'F1', 'formal',
                ['--mode', 'sketch', *_structure_args('tall'),
                 '--sketch', str(SKETCH_ROOT / stem),
                 *_flag('--sketch-init', init),
                 *_flag('--sketch-weight', weight)],
            ))

    for end in WEIGHT_ENDS:
        rows.append(_row(
            f'F2/tall/{sketch_token("12.jpg")}/{wend_token(end)}',
            'F2', 'formal',
            ['--mode', 'sketch', *_structure_args('tall'),
             '--sketch', str(SKETCH_ROOT / '12.jpg'),
             '--sketch-weight-end', str(end)],
        ))

    for stem in F3_SKETCHES:
        rows.append(_row(
            f'F3/tall/{sketch_token(stem)}', 'F3', 'formal',
            ['--mode', 'sketch', *_structure_args('tall'),
             '--sketch', str(SKETCH_ROOT / stem)],
        ))

    def scale_rows(experiment: str, structure: str) -> list[dict]:
        out = []
        for prompt in PROMPTS:
            for arm in SCALE_ARMS:
                token = scales_token(parse_clip_scales(arm))
                out.append(_row(
                    f'{experiment}/{structure}/{prompt_token(prompt)}/{token}',
                    experiment, 'semantic',
                    ['--mode', 'semantic', *_structure_args(structure),
                     '--clip', prompt, '--clip-scales', arm],
                ))
        return out

    rows.extend(scale_rows('S1', 'tall'))

    for prompt in PROMPTS:
        for rho in BLEND_RHOS:
            rows.append(_row(
                f'S2/tall/{prompt_token(prompt)}/{rho_token(rho)}',
                'S2', 'semantic',
                ['--mode', 'semantic', *_structure_args('tall'),
                 '--clip', prompt, '--blend-rho', str(rho)],
            ))

    for key in REPORTED_STRUCTURES:
        for prompt in PROMPTS:
            rows.append(_row(
                f'S3/{key}/{prompt_token(prompt)}', 'S3', 'semantic',
                ['--mode', 'semantic', *_structure_args(key), '--clip', prompt],
            ))
    rows.append(_row(
        f'S3/tall/{prompt_token(COUNTER_PROMPT)}', 'S3', 'semantic',
        ['--mode', 'semantic', *_structure_args('tall'), '--clip', COUNTER_PROMPT],
    ))

    for prompt in PROMPTS:
        token = prompt_token(prompt)
        rows.append(_row(
            f'H1/tall/{token}/hybrid', 'H1', 'hybrid',
            ['--mode', 'hybrid', *_structure_args('tall'), '--clip', prompt],
        ))
        rows.append(_row(
            f'H1/tall/{token}/dream_only', 'H1', 'hybrid',
            ['--mode', 'dream_only', *_structure_args('tall'), '--clip', prompt],
        ))

    for key in ('short', 'bridge'):
        for prompt in PROMPTS:
            rows.append(_row(
                f'H2/{key}/{prompt_token(prompt)}/hybrid', 'H2', 'hybrid',
                ['--mode', 'hybrid', *_structure_args(key), '--clip', prompt],
            ))

    for stem in H3_SKETCHES:
        for prompt in PROMPTS:
            rows.append(_row(
                f'H3/tall/{sketch_token(stem)}/{prompt_token(prompt)}',
                'H3', 'hybrid',
                ['--mode', 'semantic', *_structure_args('tall'),
                 '--clip', prompt,
                 '--sketch', str(SKETCH_ROOT / stem),
                 '--prompt-sketch',
                 *_flag('--coadapt', True)],
            ))

    for prompt in PROMPTS:
        rows.append(_row(
            f'H4/tall/{prompt_token(prompt)}/coadapt-off', 'H4', 'hybrid',
            ['--mode', 'hybrid', *_structure_args('tall'),
             '--clip', prompt, *_flag('--coadapt', False)],
        ))

    for key in REPORTED_STRUCTURES:
        for prompt in PROMPTS:
            rows.append(_row(
                f'H5/{key}/{prompt_token(prompt)}/hybrid', 'H5', 'hybrid',
                ['--mode', 'hybrid', *_structure_args(key),
                 '--clip', prompt, '--gravity-load', str(GRAVITY_LOAD)],
            ))

    samples = latin_hypercube(24, 5, seed=0)
    for index, unit in enumerate(samples):
        filter_width = _lerp(1.5, 4.0, unit[0])
        penal = _lerp(3.0, 4.0, unit[1])
        beta_max = _lerp(4.0, 16.0, unit[2])
        resolution_scale = 0.5 if unit[3] < 0.5 else 1.0
        seed = int(round(_lerp(0.0, 1000.0, unit[4])))
        # Submitted Phase 2: D still samples conventional --beta-max (4–16);
        # the campaign-wide physics ramp is separately capped at 8.
        rows.append(_row(
            f'D/tall/lhs-{index:02d}', 'D', 'conventional',
            ['--mode', 'unguided', *_structure_args('tall'),
             '--filter-width', f'{filter_width:.6g}',
             '--penal', f'{penal:.6g}',
             '--beta-max', f'{beta_max:.6g}',
             '--resolution-scale', str(resolution_scale),
             '--seed', str(seed)],
        ))
    if include_s1b:
        # Appended last so the original 0..117 array indices stay valid.
        rows.extend(scale_rows('S1b', 'short'))
        rows.extend(scale_rows('S1b', 'bridge'))
    return rows


def write_manifest(path: Path, *, include_s1b: bool) -> list[dict]:
    rows = experiment_rows(include_s1b=include_s1b)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ['run_id\targv_json\n']
    for row in rows:
        lines.append(f"{row['run_id']}\t{json.dumps(row['argv'], ensure_ascii=True)}\n")
    path.write_text(''.join(lines))
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--include-s1b', action='store_true',
        help='Append S1b only after the butterfly-wing-venation {m} gate passes.')
    parser.add_argument(
        '--out', default=str(_REPO / 'slurm' / 'campaign.tsv'))
    args = parser.parse_args(argv)
    rows = write_manifest(Path(args.out), include_s1b=args.include_s1b)
    print(f'wrote {len(rows)} rows to {args.out}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
