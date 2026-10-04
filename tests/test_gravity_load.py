"""Per-pixel self-weight as a fraction of the live-load resultant."""

from __future__ import annotations

import numpy as np
import pytest

from physics import physics
from recipe.campaign_spec import GRAVITY_LOAD
from recipe.dream_layout import build_model
from recipe.preset import PAPER
from dataclasses import replace


def test_pixel_gravity_is_off_at_zero():
    forces = np.zeros((4, 3, 2))
    forces[:, 0, 1] = -0.5
    assert physics.pixel_gravity(forces, 3, 2, 0.0) == 0.0


def test_pixel_gravity_splits_the_live_load_across_pixels():
    forces = np.zeros((5, 4, 2))
    forces[:, 0, 1] = -0.25
    total = 5 * 0.25
    assert physics.pixel_gravity(forces, 4, 3, 0.05) == pytest.approx(0.05 * total / 12)


def test_pixel_gravity_rejects_a_negative_fraction():
    with pytest.raises(ValueError, match='gravity_load'):
        physics.pixel_gravity(np.zeros(8), 2, 2, -0.01)


def test_calculate_forces_adds_density_weighted_self_weight():
    args = physics.default_args()
    solid = np.ones((args['nely'], args['nelx']))
    live = np.asarray(physics.calculate_forces(solid, args))
    assert np.allclose(live, args['forces'])

    physics.apply_pixel_gravity(args, 0.05)
    combined = np.asarray(physics.calculate_forces(solid, args))
    extra = combined.reshape(args['nelx'] + 1, args['nely'] + 1, 2) - (
        np.asarray(args['forces']).reshape(args['nelx'] + 1, args['nely'] + 1, 2)
    )
    assert extra[..., 0].sum() == pytest.approx(0.0)
    assert extra[..., 1].sum() == pytest.approx(
        -args['g'] * args['nelx'] * args['nely'])
    live_total = float(np.sum(np.abs(args['forces'])))
    assert extra[..., 1].sum() == pytest.approx(-0.05 * live_total)


def test_build_model_applies_gravity_and_refresh_keeps_it():
    preset = replace(
        PAPER,
        problem_name='tall_building',
        width=16,
        height=16,
        interval=8,
        resize_num=0,
        gravity_load=GRAVITY_LOAD,
    )
    model = build_model(preset, clip_loss=None, venice_algebra=False, init=True)
    assert model.gravity_load == GRAVITY_LOAD
    assert model.env.args['g'] > 0.0
    expected = physics.pixel_gravity(
        model.env.args['forces'],
        model.env.args['nelx'],
        model.env.args['nely'],
        GRAVITY_LOAD,
    )
    assert model.env.args['g'] == pytest.approx(expected)
    model._refresh_physics_environment()
    assert model.env.args['g'] == pytest.approx(expected)
    model._set_analysis_factor(max_dim=8)
    if model.analysis_env is not model.env:
        assert model.analysis_env.args['g'] > 0.0
        assert model.analysis_env.args['gravity_load'] == GRAVITY_LOAD
