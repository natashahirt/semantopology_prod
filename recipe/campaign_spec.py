"""Campaign constants: structures, prompts, sketches, and run_id paths.

``slurm/make_manifest.py`` is the only place that expands this into jobs.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from recipe.preset import prompt_slug

PROMPTS = ('fern fronds', 'butterfly wing venation', 'skeletons')
COUNTER_PROMPT = 'human skull'
# S4: optional fern-count panel. Official campaign fern stays "fern fronds".
FERN_WORDINGS = (
    'fern frond',
    'fern fronds',
    'many fern fronds',
    'field of ferns',
    'unfurling fern fronds',
)
# C1: optional control prompts. A neutral subject and a meaningless string
# separate the effect of the prompt's meaning from generic CLIP pressure.
CONTROL_PROMPTS = ('structure', 'qzv xlrp mnek')

# Phase 2c extensions. Every prompt panel runs on tall unless noted.
# S2b: the S2 dial on the structures S2 skipped.
DIAL_STRUCTURES = ('short', 'bridge')
# P: prompts whose forms are already structural vs ones that are not.
STRUCTURAL_PROMPTS = (
    'tree branches', 'bone trabeculae', 'spider web', 'gothic tracery',
    'honeycomb',
)
NONSTRUCTURAL_PROMPTS = ('clouds', 'smoke', 'fur', 'a cat')
# N: more meaningless strings, and each campaign prompt with its letters
# scrambled within each word (same letters and word lengths, no meaning).
NONSENSE_PROMPTS = (
    'vbtq orzk plimw', 'xjeu wkqa dryv', 'gmof ztuy hcnb', 'kwyp elrj sqox',
)
SCRAMBLED_PROMPTS = (
    'rnef dsnorf', 'ytlurfebt gniw ntoeiavn', 'ntseleosk',
)
# L: a meaning ladder from fern fronds (S3) through tree branches (P).
LADDER_PROMPTS = ('bracken', 'lightning', 'brick wall')
# V: volume fractions beside the 0.3 already run by S3 and B.
VOLUME_FRACTIONS = (0.2, 0.4, 0.5)
VOLUME_STRUCTURES = ('tall', 'bridge')
# M: 24 varied prompts, matched in count to D's Latin-hypercube samples.
DIVERSITY_PROMPTS = (
    'coral reef', 'river delta', 'mangrove roots', 'lichen',
    'seashell spiral', 'pine cone', 'dragonfly wing', 'ice crystals',
    'flying buttresses', 'art nouveau ironwork', 'bamboo scaffolding',
    'chain-link fence', 'stained glass window', 'woven basket',
    'suspension bridge cables', 'lattice tower', 'circuit board', 'lace',
    'cracked mud', 'marble veins', 'sound waves', 'spiral galaxy',
    'barbed wire', 'neurons',
)
# R: seeds beside the campaign seed (12) for the S3 grid and B.
REPLICATE_SEEDS = (101, 202, 303, 404)

# Post-hoc evaluators. The first keeps the unsuffixed output names.
EVAL_MODELS = ('ViT-B/32', 'ViT-L/14')


def eval_model_suffix(model_name: str) -> str:
    """Output-file suffix for an evaluator; empty for the primary model."""
    if model_name == EVAL_MODELS[0]:
        return ''
    return '_' + re.sub(r'[^a-z0-9]+', '_', model_name.lower()).strip('_')

SKETCH_STEMS = (
    '1.jpg', '3.jpg', '6.jpg', '9.jpg', '11.jpg', '12.jpg',
    'col2.png', 'col3.png', 'col6_grid.png', 'col3_braced.png',
)
F3_SKETCHES = SKETCH_STEMS
F1_SKETCHES = ('12.jpg', '3.jpg')
H3_SKETCHES = ('12.jpg', 'col3_braced.png', 'col6_grid.png')

SCALE_ARMS = ('g', 'm', 'e', 'gme')
BLEND_RHOS = (0.0, 0.25, 0.5, 0.75, 1.0)
WEIGHT_ENDS = (200.0, 400.0, 800.0, 1200.0, 2000.0)
# H5: each solid pixel carries an equal share of this fraction of the live load.
GRAVITY_LOAD = 0.05

SKETCH_ROOT = Path('inputs/sketches')
REPORTED_STRUCTURES = ('tall', 'short', 'bridge')


@dataclass(frozen=True)
class StructureSpec:
    """One campaign structure: problem, grid, CLIP tiling, AdaptivePixel schedule."""

    key: str
    problem_name: str
    width: int
    height: int
    interval: int
    density: float
    kind: str
    resize_num: int
    control_height: int
    control_width: int
    resize_scale: int = 2


STRUCTURES = {
    'tall': StructureSpec(
        key='tall',
        problem_name='tall_building',
        width=128,
        height=256,
        interval=64,
        density=0.3,
        kind='building',
        resize_num=2,
        control_height=32,
        control_width=16,
    ),
    'short': StructureSpec(
        key='short',
        problem_name='short_cantilever_building',
        width=300,
        height=150,
        interval=50,
        density=0.3,
        kind='building',
        # 300x150 is divisible by 2, not by 4.
        resize_num=1,
        control_height=16,
        control_width=32,
    ),
    'bridge': StructureSpec(
        key='bridge',
        problem_name='double_decker_bridge',
        width=448,
        height=72,
        interval=72,
        density=0.3,
        kind='bridge',
        resize_num=2,
        control_height=8,
        control_width=32,
    ),
    # Exploratory gate only; not expanded into the main campaign manifest.
    'bridge3': StructureSpec(
        key='bridge3',
        problem_name='three_decker_bridge',
        width=448,
        height=72,
        interval=36,
        density=0.3,
        kind='bridge',
        resize_num=2,
        control_height=8,
        control_width=32,
    ),
}


def sketch_token(stem: str) -> str:
    return 'sketch-' + Path(stem).stem


def prompt_token(prompt: str) -> str:
    return prompt_slug(prompt)


def rho_token(value: float) -> str:
    return f'rho-{value:.2f}'


def wend_token(value: float) -> str:
    return f'wend-{int(value)}'


def vf_token(value: float) -> str:
    return f'vf-{value:.2f}'


def seed_token(value: int) -> str:
    return f'seed-{int(value)}'
