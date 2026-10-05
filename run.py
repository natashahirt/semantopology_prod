"""Replicable entry point for a paper figure or one campaign row."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from recipe.campaign import next_attempt_dir, onoff, run_campaign
from recipe.campaign_spec import STRUCTURES
from recipe.preset import PAPER, prompt_slug


def resolve_prompt(clip: str | None, sentence: str | None, *, required: bool = True) -> str | None:
    """Return CLIP text. Campaign unguided/sketch rows may omit it."""
    clip_text = (clip or '').strip()
    sentence_text = (sentence or '').strip()
    if clip_text and sentence_text:
        raise ValueError('pass only one of --clip or --sentence')
    if clip_text:
        return clip_text
    if sentence_text:
        from language.interpret import interpret_motive
        return interpret_motive(sentence_text)
    if required:
        raise ValueError('pass exactly one of --clip or --sentence')
    return None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description='Run one semantopology recipe (campaign row or paper figure).')
    parser.add_argument('--run-id', default=None)
    parser.add_argument('--experiment', default='')
    parser.add_argument('--group', default='')
    parser.add_argument(
        '--mode',
        default='hybrid',
        choices=('unguided', 'sketch', 'semantic', 'hybrid', 'dream_only'),
    )
    parser.add_argument(
        '--structure', default='tall', choices=tuple(STRUCTURES))
    parser.add_argument(
        '--problem', default=None,
        help='Override the structure problem name (debug only).',
    )
    parser.add_argument('--clip', default=None)
    parser.add_argument('--sentence', default=None)
    parser.add_argument('--sketch', default=None)
    parser.add_argument('--sketch-init', default='on')
    parser.add_argument('--sketch-weight', default='on')
    parser.add_argument(
        '--sketch-weight-end', type=float, default=PAPER.sketch_weight_end)
    parser.add_argument(
        '--sketch-weight-start', type=float, default=PAPER.sketch_weight_start,
        help='Sketch prior weight on the coarsest AdaptivePixel grid, where '
             'topology is decided. The weight ramps from here to '
             '--sketch-weight-end at full resolution.')
    parser.add_argument(
        '--prompt-sketch', action='store_true',
        help='Sketch occupancy plus a CLIP prompt (no dream).')
    parser.add_argument('--blend-rho', type=float, default=PAPER.blend_rho)
    parser.add_argument(
        '--blend-rho-z', type=float, default=None,
        help=f'Grad-match mixer on raw-z CLIP (default {PAPER.blend_rho_z}; '
             '0 = density CLIP only). Not combinable with --clip-weight.')
    parser.add_argument(
        '--volume-fraction', type=float, default=None,
        help='Target volume fraction in (0, 1); default is the structure\'s.')
    parser.add_argument(
        '--clip-weight', type=float, default=None,
        help='Fixed weight on density CLIP, replacing grad-match (C2 only; '
             'semantic mode; set with --clip-weight-z).')
    parser.add_argument(
        '--clip-weight-z', type=float, default=None,
        help='Fixed weight on raw-z CLIP, paired with --clip-weight.')
    parser.add_argument('--clip-scales', default='')
    parser.add_argument(
        '--coadapt', default=None,
        help='on/off. Default on for hybrid, off otherwise.',
    )
    parser.add_argument('--filter-width', type=float, default=PAPER.filter_width)
    parser.add_argument('--penal', type=float, default=PAPER.penal)
    parser.add_argument('--beta-max', type=float, default=None)
    parser.add_argument(
        '--physics-beta-max', type=float,
        default=PAPER.physics_projection_beta_max,
        help='Ramp the physics Heaviside beta from 1 to this value over the '
             'run (0 = off). Disables early convergence stopping.')
    parser.add_argument(
        '--gravity-load', type=float, default=PAPER.gravity_load,
        help='Per-pixel self-weight as a fraction of the live-load '
             'resultant (0 = off). A solid design then carries this share '
             'of the applied load as gravity, split equally across pixels.')
    parser.add_argument('--resolution-scale', type=float, default=1.0)
    parser.add_argument('--seed', type=int, default=PAPER.seed)
    parser.add_argument('--out', default=None)
    parser.add_argument('--device', default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    needs_prompt = args.mode in ('semantic', 'hybrid', 'dream_only') or args.prompt_sketch
    try:
        prompt = resolve_prompt(args.clip, args.sentence, required=needs_prompt)
    except ValueError as exc:
        print(exc, file=sys.stderr)
        return 2
    args.clip = prompt
    args.sketch_init = onoff(args.sketch_init)
    args.sketch_weight = onoff(args.sketch_weight)
    if args.coadapt is None:
        args.coadapt = (
            'on' if args.mode == 'hybrid' or args.prompt_sketch else 'off')
    args.coadapt = onoff(args.coadapt)
    args.device = args.device or PAPER.device
    if args.device == 'cpu':
        os.environ['CUDA_VISIBLE_DEVICES'] = ''

    if args.mode == 'sketch' and not args.sketch:
        print('sketch mode needs --sketch', file=sys.stderr)
        return 2

    run_id = args.run_id
    if not run_id:
        slug = prompt_slug(prompt) if prompt else args.mode
        run_id = f'{args.mode}/{args.structure}/{slug}'
        args.run_id = run_id
    if args.problem:
        # Keep the structure grid; only the problem name is overridden later
        # if we ever need it. Unused for the campaign.
        pass

    if args.out:
        output_dir = Path(args.out)
        output_dir.mkdir(parents=True, exist_ok=True)
    else:
        output_dir = next_attempt_dir(Path('results') / run_id)

    try:
        record = run_campaign(args, output_dir)
    except Exception as exc:
        print(f'run failed: {exc}', file=sys.stderr)
        raise
    print(json.dumps({
        'out': str(output_dir),
        'run_id': record.get('run_id'),
        'compliance': record.get('compliance'),
        'exit_status': record.get('exit_status'),
        'git_commit': record.get('git_commit'),
    }, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
