"""Replicable entry point.

Paper figures pass ``--clip`` and a named ``--problem``. A free sentence goes
through ``language.interpret`` and becomes CLIP text only; it does not choose
the structure.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def resolve_prompt(clip: str | None, sentence: str | None) -> str:
    """Return CLIP text. Exactly one of ``clip`` or ``sentence`` must be set."""
    clip_text = (clip or '').strip()
    sentence_text = (sentence or '').strip()
    if bool(clip_text) == bool(sentence_text):
        raise ValueError('pass exactly one of --clip or --sentence')
    if clip_text:
        return clip_text
    from language.interpret import interpret_motive
    return interpret_motive(sentence_text)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description='Dream a layout from CLIP text, then optimize the structure.')
    parser.add_argument(
        '--problem',
        default='multistory_building',
        help='Named structural problem (default: multistory_building).',
    )
    parser.add_argument(
        '--clip',
        default=None,
        help='CLIP text, used exactly. Paper figures use this, not --sentence.',
    )
    parser.add_argument(
        '--sentence',
        default=None,
        help='Free sentence. A language model extracts the image motive only.',
    )
    parser.add_argument(
        '--out',
        default=None,
        help='Output directory. Default: results/<prompt-slug>.',
    )
    parser.add_argument(
        '--device',
        default=None,
        help='cpu or cuda. Default is cpu, which matches the paper runs.',
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        prompt = resolve_prompt(args.clip, args.sentence)
    except ValueError as exc:
        print(exc, file=sys.stderr)
        return 2

    from recipe.preset import PAPER, prompt_slug

    preset = PAPER.with_problem(args.problem).with_clip(prompt)
    if args.device:
        preset = preset.with_device(args.device)
    # The model uses CUDA whenever it is visible. Hide it unless this run
    # asked for it, so a GPU machine still reproduces the CPU paper path.
    if preset.device == 'cpu':
        os.environ['CUDA_VISIBLE_DEVICES'] = ''

    from recipe.dream_layout import run_dream_layout
    output_dir = Path(args.out) if args.out else Path('results') / prompt_slug(prompt)
    summary = run_dream_layout(preset, output_dir)
    print(json.dumps({
        'out': str(output_dir),
        'compliance': summary['compliance'],
        'clip_loss': summary['clip_loss'],
        'clip_loss_raw': summary['clip_loss_raw'],
        'steps': summary['steps'],
        'total_seconds': summary['total_seconds'],
        'git_commit': summary['git_commit'],
    }, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
