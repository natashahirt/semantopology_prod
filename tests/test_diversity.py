"""Diversity representations and summaries."""

from __future__ import annotations

import numpy as np
import pytest

from analysis.diversity import (
    bootstrap_mean_distance,
    comparison_set,
    geometric_map,
    load_path_map,
    replay_args,
)
from physics import physics
from physics.api import Environment, specified_task
from problem.problems import tall_building


def _solve(field: np.ndarray) -> tuple[float, np.ndarray]:
    env = Environment(specified_task(tall_building(
        width=field.shape[1], height=field.shape[0], interval=8)))
    kwargs = {
        'penal': env.args['penal'],
        'e_min': env.args['young_min'],
        'e_0': env.args['young'],
    }
    displacement = physics.displace(
        field, env.ke, physics.calculate_forces(field, env.args),
        env.args['freedofs'], env.args['fixdofs'], **kwargs)
    energy = np.asarray(
        physics.element_strain_energy(field, displacement, env.ke, **kwargs))
    compliance = float(physics.compliance(
        field, displacement, env.ke, **kwargs))
    return compliance, energy


def _frame() -> np.ndarray:
    field = np.full((32, 16), 0.05)
    field[:, :2] = 1.0
    field[:, -2:] = 1.0
    for row in (0, 8, 16, 24, 30):
        field[row:row + 2, :] = 1.0
    return field


def test_element_energies_sum_to_compliance():
    compliance, energy = _solve(_frame())
    assert energy.sum() == pytest.approx(compliance, rel=1e-12)


def test_floating_blob_changes_geometry_more_than_load_path():
    frame = _frame()
    blob = frame.copy()
    blob[11:14, 6:9] = 1.0
    _, energy_a = _solve(frame)
    _, energy_b = _solve(blob)
    structural_change = np.linalg.norm(
        load_path_map(energy_a, (16, 8)) - load_path_map(energy_b, (16, 8)))
    geometric_change = np.linalg.norm(
        geometric_map(frame, (16, 8)) - geometric_map(blob, (16, 8)))
    assert structural_change < geometric_change
    assert geometric_change > 0.1


def test_load_path_is_resolution_and_scale_invariant():
    rng = np.random.default_rng(4)
    energy = rng.uniform(0.1, 2.0, size=(8, 4))
    upsampled = np.kron(energy, np.ones((2, 2))) / 4.0
    original = load_path_map(energy, (4, 2))
    refined = load_path_map(upsampled, (4, 2))
    np.testing.assert_allclose(refined, original)
    assert np.linalg.norm(original) == pytest.approx(1.0)


def test_identical_designs_have_zero_bootstrap_distance():
    values = np.ones((4, 12))
    assert bootstrap_mean_distance(values, samples=100) == (0.0, 0.0, 0.0)


def test_legacy_run_arguments_are_recovered_from_manifest(tmp_path):
    manifest = tmp_path / 'campaign.tsv'
    manifest.write_text(
        'D/tall/lhs-01\t'
        '["--run-id", "D/tall/lhs-01", "--experiment", "D", '
        '"--mode", "unguided", "--structure", "tall", '
        '"--filter-width", "2.75", "--penal", "3.25", '
        '"--resolution-scale", "0.5"]\n')
    args = replay_args({'run_id': 'D/tall/lhs-01'}, [manifest])
    assert args.structure == 'tall'
    assert args.filter_width == pytest.approx(2.75)
    assert args.penal == pytest.approx(3.25)
    assert args.resolution_scale == pytest.approx(0.5)


@pytest.mark.parametrize(
    ('meta', 'expected'),
    [
        ({'experiment': 'B', 'structure': 'tall', 'prompt': None},
         'unguided seeds'),
        ({'experiment': 'R', 'structure': 'tall', 'prompt': None},
         'unguided seeds'),
        ({'experiment': 'R', 'structure': 'tall', 'prompt': 'fern fronds'},
         None),
        ({'experiment': 'D', 'structure': 'tall', 'prompt': None},
         'parameter sweep'),
        ({'experiment': 'M', 'structure': 'tall', 'prompt': 'lace'}, 'ours'),
        ({'experiment': 'P', 'structure': 'tall', 'prompt': 'honeycomb'},
         'semantic prompts'),
        ({'experiment': 'M', 'structure': 'bridge', 'prompt': 'lace'}, None),
    ],
)
def test_comparison_set(meta, expected):
    assert comparison_set(meta) == expected
