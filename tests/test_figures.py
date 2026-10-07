"""Analysis tables that carry a claim the mechanical scalars cannot make."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from analysis.figures import (
    MEANINGLESS_PROMPTS,
    formal_table,
    semantic_floor,
    weight_headroom,
)
from guidance.blend import GRAD_MATCH_WEIGHT_MAX
from recipe.campaign_spec import PROMPTS


def _read(path: Path) -> list[dict]:
    with path.open(newline='') as handle:
        return list(csv.DictReader(handle))


def _record(prompt, own_score):
    """One similarity row whose score for PROMPTS[0] is `own_score`."""
    return {
        'attempt': f'x/{prompt}/{own_score}',
        'run_id': f'{prompt}-{own_score}',
        'prompt': prompt,
        'similarities': {PROMPTS[0]: own_score},
    }


def test_semantic_floor_separates_meaning_from_clip_pressure(tmp_path):
    nonsense = sorted(MEANINGLESS_PROMPTS)[0]
    path = tmp_path / 'similarities.json'
    path.write_text(json.dumps([
        _record(PROMPTS[0], 0.30),
        _record(PROMPTS[0], 0.32),
        _record(nonsense, 0.20),
        _record(PROMPTS[1], 0.22),
        _record(None, 0.18),
    ]))

    semantic_floor(path, tmp_path)

    rows = {row['prompt']: row for row in _read(tmp_path / 'tables' / 'semantic_floor.csv')}
    row = rows[PROMPTS[0]]
    assert row['own_n'] == '2'
    assert float(row['own_mean']) == 0.31
    assert float(row['meaningless_mean']) == 0.20
    assert float(row['unguided_mean']) == 0.18
    assert float(row['other prompt_mean']) == 0.22
    assert float(row['own_minus_meaningless']) == pytest.approx(0.11)
    assert float(row['own_minus_unguided']) == pytest.approx(0.13)
    # A prompt nothing was scored against must not fabricate a row.
    assert rows[PROMPTS[2]]['own_mean'] == ''


def test_semantic_floor_is_silent_without_an_evaluator_pass(tmp_path):
    semantic_floor(tmp_path / 'missing.json', tmp_path)
    assert not (tmp_path / 'tables').exists()


def test_weight_headroom_flags_a_run_against_the_cap(tmp_path):
    runs = [
        {'run_id': 'unguided', 'experiment': 'B'},
        {'run_id': 'clear', 'experiment': 'S3', 'coupling': 'grad_match',
         'clip_weight_mean': 500.0, 'clip_weight_max': 900.0,
         'clip_weight_cap': GRAD_MATCH_WEIGHT_MAX},
        {'run_id': 'clipped', 'experiment': 'S3', 'coupling': 'grad_match',
         'clip_weight_mean': 1800.0, 'clip_weight_max': GRAD_MATCH_WEIGHT_MAX,
         'clip_weight_cap': GRAD_MATCH_WEIGHT_MAX},
        {'run_id': 'legacy', 'experiment': 'S3', 'coupling': 'grad_match',
         'clip_weight_mean': 1900.0},
    ]

    weight_headroom(runs, tmp_path)

    rows = {row['run_id']: row for row in _read(tmp_path / 'tables' / 'weight_headroom.csv')}
    # An unguided run has no coupling and so no weight to report.
    assert 'unguided' not in rows
    assert rows['clear']['at_cap'] == 'False'
    assert rows['clipped']['at_cap'] == 'True'
    assert float(rows['clear']['max_fraction_of_cap']) == 0.45
    # Without a logged maximum the verdict is unknown, not False: the mean
    # alone cannot say whether any step ran against the ceiling.
    assert rows['legacy']['at_cap'] == ''
    assert rows['legacy']['max_fraction_of_cap'] == ''
    assert float(rows['legacy']['mean_fraction_of_cap']) == 0.95


def test_formal_table_carries_only_runs_given_a_drawing(tmp_path):
    """Sketch adherence is meaningless where there is no sketch.

    `mass_on_scaffold` is null on unguided and prompt-only runs, which is
    what keeps them out -- so no list of formal experiment codes has to be
    maintained here as panels are added.
    """
    runs = [
        {'run_id': 'unguided', 'experiment': 'B', 'mass_on_scaffold': None},
        {'run_id': 'prompt-only', 'experiment': 'S3',
         'mass_on_scaffold': None, 'prompt': PROMPTS[0]},
        {'run_id': 'sketch', 'experiment': 'F3', 'mode': 'sketch',
         'sketch': '/cluster/home/inputs/sketches/12.jpg',
         'mass_on_scaffold': 0.72,
         'validity': {'floating_mass_fraction': 0.013,
                      'spanning_mass_fraction': 0.987,
                      'top_to_bottom_connected': 1.0,
                      'support_to_load_connected': 1.0}},
        {'run_id': 'hybrid', 'experiment': 'H3', 'mode': 'semantic',
         'sketch': '/cluster/home/inputs/sketches/12.jpg',
         'prompt': PROMPTS[0], 'mass_on_scaffold': 0.61},
    ]

    formal_table(runs, tmp_path)

    rows = {row['run_id']: row for row in _read(
        tmp_path / 'tables' / 'formal.csv')}
    assert set(rows) == {'sketch', 'hybrid'}
    # The recorded path is the cluster's, so only the filename is portable.
    assert rows['sketch']['sketch'] == '12.jpg'
    assert float(rows['sketch']['floating_mass_fraction']) == 0.013
    # A run whose validity block is absent reports blanks, not zeros: zero
    # floating mass is a strong claim and must not be invented.
    assert rows['hybrid']['floating_mass_fraction'] == ''
