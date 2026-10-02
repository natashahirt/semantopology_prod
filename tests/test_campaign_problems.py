"""Paper structures: loads, supports, and a factorable stiffness system."""

from __future__ import annotations

import numpy as np
from absl.testing import absltest

from physics import physics
from physics.api import Environment, specified_task
from problem.problems import (
    X,
    Y,
    StructuralParams,
    double_decker_bridge,
    short_cantilever_building,
    tall_building,
    three_decker_bridge,
)


def _factors(problem) -> np.ndarray:
    args = specified_task(problem)
    env = Environment(args)
    density = np.full((env.nely, env.nelx), problem.density, dtype=np.float64)
    displacement = physics.displace(
        density, env.ke, env.args['forces'], env.args['freedofs'],
        env.args['fixdofs'], penal=env.args['penal'])
    if not np.isfinite(displacement).all():
        raise AssertionError('displacement is not finite')
    return np.asarray(displacement)


class TallBuildingTest(absltest.TestCase):

    def test_alias_matches_multistory_grid(self):
        problem = tall_building()
        self.assertEqual((problem.width, problem.height), (128, 256))
        self.assertTrue(np.all(problem.normals[:, -1, Y] == 1))
        self.assertTrue(np.all(problem.normals[-1, :, X] == 1))
        loaded = np.flatnonzero(np.abs(problem.forces[0, :, Y]) > 0)
        # Interval 64 also marks the supported soffit (y=256); that load is
        # inert because those DOFs are fixed, same as multistory_building.
        np.testing.assert_array_equal(loaded, [0, 64, 128, 192, 256])

    def test_params_round_trip(self):
        problem = StructuralParams(
            problem_name='tall_building', width=32, height=64,
            density=0.3, interval=16).get_problem()
        self.assertEqual(problem.name, 'tall_building_32x64')
        _factors(problem)


class ShortCantileverTest(absltest.TestCase):

    def test_loads_and_overhang(self):
        problem = short_cantilever_building()
        self.assertEqual((problem.width, problem.height), (300, 150))
        support_x = int(round(0.70 * 300))
        self.assertEqual(support_x, 210)
        self.assertTrue(np.all(problem.normals[:support_x + 1, -1, Y] == 1))
        self.assertTrue(np.all(problem.normals[support_x + 1:, -1, Y] == 0))
        self.assertTrue(np.all(problem.normals[0, :, X] == 1))
        loaded = np.flatnonzero(np.abs(problem.forces[0, :, Y]) > 0)
        np.testing.assert_array_equal(loaded, [0, 50, 100])
        np.testing.assert_allclose(problem.forces[:, 0, Y], -1.0 / 300)

    def test_factors_on_a_similar_grid(self):
        problem = short_cantilever_building(width=40, height=20, interval=10)
        support_x = int(round(0.70 * 40))
        self.assertTrue(np.all(problem.normals[:support_x + 1, -1, Y] == 1))
        loaded = np.flatnonzero(np.abs(problem.forces[0, :, Y]) > 0)
        np.testing.assert_array_equal(loaded, [0, 10])
        _factors(problem)


class DoubleDeckerBridgeTest(absltest.TestCase):

    def test_pin_roller_and_two_decks(self):
        problem = double_decker_bridge()
        self.assertEqual((problem.width, problem.height), (448, 72))
        self.assertEqual(problem.normals[0, -1, X], 1)
        self.assertEqual(problem.normals[0, -1, Y], 1)
        self.assertEqual(problem.normals[-1, -1, Y], 1)
        self.assertEqual(problem.normals[-1, -1, X], 0)
        np.testing.assert_allclose(problem.forces[:, 0, Y], -1.0 / 448)
        np.testing.assert_allclose(problem.forces[:, 71, Y], -1.0 / 448)
        self.assertTrue(np.all(problem.forces[:, 72, Y] == 0))

    def test_factors_on_a_similar_grid(self):
        problem = double_decker_bridge(width=32, height=8)
        self.assertEqual(problem.normals[0, -1, X], 1)
        np.testing.assert_allclose(problem.forces[:, 7, Y], -1.0 / 32)
        _factors(problem)


class ThreeDeckerBridgeTest(absltest.TestCase):

    def test_adds_a_middle_loaded_deck(self):
        problem = three_decker_bridge()
        loaded = np.flatnonzero(np.abs(problem.forces[0, :, Y]) > 0)
        np.testing.assert_array_equal(loaded, [0, 36, 71])
        np.testing.assert_allclose(problem.forces[:, 36, Y], -1.0 / 448)

    def test_factors_on_a_similar_grid(self):
        problem = three_decker_bridge(width=32, height=8)
        loaded = np.flatnonzero(np.abs(problem.forces[0, :, Y]) > 0)
        np.testing.assert_array_equal(loaded, [0, 4, 7])
        _factors(problem)


if __name__ == '__main__':
    absltest.main()
