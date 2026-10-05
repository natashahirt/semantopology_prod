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
    CONTROL_PROMPTS,
    COUNTER_PROMPT,
    DIAL_STRUCTURES,
    DIVERSITY_PROMPTS,
    F1_SKETCHES,
    F3_SKETCHES,
    FERN_WORDINGS,
    GRAVITY_LOAD,
    H3_SKETCHES,
    LADDER_PROMPTS,
    NONSENSE_PROMPTS,
    NONSTRUCTURAL_PROMPTS,
    PROMPTS,
    REPLICATE_SEEDS,
    REPORTED_STRUCTURES,
    SCALE_ARMS,
    SCRAMBLED_PROMPTS,
    SKETCH_ROOT,
    STRUCTURAL_PROMPTS,
    TYPOLOGY_PROMPTS,
    VOLUME_FRACTIONS,
    VOLUME_STRUCTURES,
    WEIGHT_ENDS,
    prompt_token,
    rho_token,
    seed_token,
    sketch_token,
    vf_token,
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


def fern_wording_rows() -> list[dict]:
    """S4 plural/count panel. Official ``fern fronds`` stays on H1/S3."""
    rows: list[dict] = []
    official = PROMPTS[0]
    for prompt in FERN_WORDINGS:
        if prompt == official:
            continue
        token = prompt_token(prompt)
        for mode, group in (('hybrid', 'hybrid'), ('semantic', 'semantic')):
            rows.append(_row(
                f'S4/tall/{token}/{mode}', 'S4', group,
                ['--mode', mode, *_structure_args('tall'), '--clip', prompt],
            ))
    return rows


def control_prompt_rows() -> list[dict]:
    """C1 control prompts on every structure, matched to S3."""
    return [
        _row(
            f'C1/{key}/{prompt_token(prompt)}', 'C1', 'semantic',
            ['--mode', 'semantic', *_structure_args(key), '--clip', prompt],
        )
        for key in REPORTED_STRUCTURES
        for prompt in CONTROL_PROMPTS
    ]


def fixed_weight_rows(clip_weight: float, clip_weight_z: float) -> list[dict]:
    """C2: the S3 grid with grad-match replaced by one pair of fixed weights."""
    return [
        _row(
            f'C2/{key}/{prompt_token(prompt)}', 'C2', 'semantic',
            ['--mode', 'semantic', *_structure_args(key), '--clip', prompt,
             '--clip-weight', f'{clip_weight:.6g}',
             '--clip-weight-z', f'{clip_weight_z:.6g}'],
        )
        for key in REPORTED_STRUCTURES
        for prompt in PROMPTS
    ]


def _semantic(run_id: str, experiment: str, structure: str, prompt: str,
              *extra: str) -> dict:
    """An S3-recipe semantic row plus the flags its panel varies."""
    return _row(
        run_id, experiment, 'semantic',
        ['--mode', 'semantic', *_structure_args(structure), '--clip', prompt,
         *extra],
    )


def _unguided(run_id: str, experiment: str, structure: str, *extra: str) -> dict:
    """A B-recipe unguided row plus the flags its panel varies."""
    return _row(
        run_id, experiment, 'baseline',
        ['--mode', 'unguided', *_structure_args(structure), *extra],
    )


def extension_rows() -> list[dict]:
    """Phase 2c panels in queue order; the R seed replicates come last."""
    rows: list[dict] = []

    for key in DIAL_STRUCTURES:
        for prompt in PROMPTS:
            for rho in BLEND_RHOS:
                rows.append(_semantic(
                    f'S2b/{key}/{prompt_token(prompt)}/{rho_token(rho)}',
                    'S2b', key, prompt, '--blend-rho', str(rho)))

    for prompt in PROMPTS:
        rows.append(_semantic(
            f'A/tall/{prompt_token(prompt)}/density-only',
            'A', 'tall', prompt, '--blend-rho-z', '0'))

    for experiment, prompts in (
            ('P', STRUCTURAL_PROMPTS + NONSTRUCTURAL_PROMPTS),
            ('N', NONSENSE_PROMPTS + SCRAMBLED_PROMPTS),
            ('L', LADDER_PROMPTS)):
        for prompt in prompts:
            rows.append(_semantic(
                f'{experiment}/tall/{prompt_token(prompt)}',
                experiment, 'tall', prompt))

    for key in VOLUME_STRUCTURES:
        for volume in VOLUME_FRACTIONS:
            flag = ('--volume-fraction', f'{volume:.6g}')
            rows.append(_unguided(
                f'V/{key}/unguided/{vf_token(volume)}', 'V', key, *flag))
            for prompt in PROMPTS:
                rows.append(_semantic(
                    f'V/{key}/{prompt_token(prompt)}/{vf_token(volume)}',
                    'V', key, prompt, *flag))

    for prompt in DIVERSITY_PROMPTS:
        rows.append(_semantic(
            f'M/tall/{prompt_token(prompt)}', 'M', 'tall', prompt))

    for seed in REPLICATE_SEEDS:
        flag = ('--seed', str(seed))
        for key in REPORTED_STRUCTURES:
            rows.append(_unguided(
                f'R/{key}/unguided/{seed_token(seed)}', 'R', key, *flag))
            for prompt in PROMPTS:
                rows.append(_semantic(
                    f'R/{key}/{prompt_token(prompt)}/{seed_token(seed)}',
                    'R', key, prompt, *flag))
    return rows


def typology_rows() -> list[dict]:
    """G: the same geometric-compatibility prompts on all three structures.

    Structure varies in the outer loop so tall completes first if the queue
    is cut short, and every row is otherwise an S3 recipe. The point of the
    panel is the comparison ACROSS structures: a prompt native to a span and
    awkward on a tower separates compatibility from wording.
    """
    return [
        _semantic(f'G/{key}/{prompt_token(prompt)}', 'G', key, prompt)
        for key in REPORTED_STRUCTURES
        for prompt in TYPOLOGY_PROMPTS
    ]


# Off-table panels: flag -> (default TSV name, row builder).
OFF_TABLE_PANELS = {
    'fern_wording': ('fern_wording.tsv', fern_wording_rows),
    'controls': ('control_prompts.tsv', control_prompt_rows),
    'extensions': ('extensions.tsv', extension_rows),
    'typology': ('typology.tsv', typology_rows),
}


def write_manifest(path: Path, rows: list[dict]) -> list[dict]:
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
        '--fern-wording', action='store_true',
        help='Write only the S4 fern-plural panel, never the 118-row campaign TSV.')
    parser.add_argument(
        '--controls', action='store_true',
        help='Write only the C1 control-prompt panel, never the 118-row campaign TSV.')
    parser.add_argument(
        '--extensions', action='store_true',
        help='Write only the Phase 2c panels, never the 118-row campaign TSV.')
    parser.add_argument(
        '--typology', action='store_true',
        help='Write only the G geometric-compatibility panel, never the '
             '118-row campaign TSV.')
    parser.add_argument(
        '--fixed-weight', nargs=2, type=float, metavar=('W_DENSITY', 'W_Z'),
        help='Write only the C2 fixed-weight panel at these two CLIP weights.')
    parser.add_argument(
        '--out', default=None)
    args = parser.parse_args(argv)
    default_campaign = _REPO / 'slurm' / 'campaign.tsv'
    panels = [name for name in OFF_TABLE_PANELS if getattr(args, name)]
    if args.fixed_weight is not None:
        panels.append('fixed_weight')
    if len(panels) > 1:
        raise SystemExit('choose one off-table panel per call')
    if panels:
        if panels[0] == 'fixed_weight':
            filename = 'fixed_weight.tsv'
            build_rows = lambda: fixed_weight_rows(*args.fixed_weight)
        else:
            filename, build_rows = OFF_TABLE_PANELS[panels[0]]
        out = Path(args.out) if args.out else _REPO / 'slurm' / filename
        if out.resolve() == default_campaign.resolve():
            raise SystemExit(f'refusing to write {panels[0]} over slurm/campaign.tsv')
        rows = write_manifest(out, build_rows())
    else:
        out = Path(args.out) if args.out else default_campaign
        rows = write_manifest(out, experiment_rows(include_s1b=args.include_s1b))
    print(f'wrote {len(rows)} rows to {out}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
