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
    tee_shirt,
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

    def test_upper_decks_bear_on_end_rollers(self):
        problem = three_decker_bridge()
        for row in (0, 36):
            for end in (0, -1):
                self.assertEqual(problem.normals[end, row, Y], 1)
                self.assertEqual(problem.normals[end, row, X], 0)
        self.assertEqual(problem.normals[0, -1, X], 1)
        self.assertEqual(int(problem.normals[..., X].sum()), 1)
        self.assertTrue(np.all(problem.normals[1:-1, :-1, :] == 0))

    def test_factors_on_a_similar_grid(self):
        problem = three_decker_bridge(width=32, height=8)
        loaded = np.flatnonzero(np.abs(problem.forces[0, :, Y]) > 0)
        np.testing.assert_array_equal(loaded, [0, 4, 7])
        self.assertEqual(problem.normals[-1, 4, Y], 1)
        _factors(problem)


class TeeShirtTest(absltest.TestCase):

    def test_cloth_is_a_sleeved_front_with_a_neck_hole(self):
        problem = tee_shirt()
        self.assertEqual((problem.width, problem.height), (192, 160))
        mask = problem.mask
        self.assertEqual(mask.shape, (160, 192))
        self.assertEqual(mask[-1, 0], 0.0)
        self.assertEqual(mask[-1, -1], 0.0)
        self.assertEqual(mask[0, 96], 0.0)
        self.assertEqual(mask[100, 96], 1.0)
        self.assertEqual(mask[10, 10], 1.0)
        self.assertEqual(mask[10, 180], 1.0)

    def test_hem_footing_is_the_middle_quarter(self):
        problem = tee_shirt()
        # Body occupies elements x=48..143. Middle 25% is 24 elements, x=84..107.
        self.assertTrue(np.all(problem.normals[84:109, -1, :] == 1))
        self.assertTrue(np.all(problem.normals[48:84, -1, :] == 0))
        self.assertTrue(np.all(problem.normals[109:145, -1, :] == 0))

    def test_cuffs_are_supported_and_the_undersides_are_not(self):
        problem = tee_shirt()
        self.assertTrue(np.all(problem.normals[0, :49, :] == 1))
        self.assertTrue(np.all(problem.normals[-1, :49, :] == 1))
        self.assertEqual(problem.normals[10, 48, Y], 0)
        self.assertEqual(problem.normals[182, 48, Y], 0)

    def test_loads_ripple_out_from_the_collar(self):
        problem = tee_shirt()
        # Centerline of the second arc points at the hem.
        self.assertAlmostEqual(problem.forces[96, 80, X], 0.0)
        self.assertAlmostEqual(problem.forces[96, 80, Y], -1.0 / 192)
        # A node on the left sleeve's arc points outward, toward the cuff.
        self.assertLess(problem.forces[20, 24, X], 0.0)
        self.assertGreater(
            abs(problem.forces[20, 24, X]), abs(problem.forces[20, 24, Y]))
        # The right side mirrors it.
        self.assertGreater(problem.forces[172, 24, X], 0.0)
        # Four arcs, and nothing inside the neck or on the footing.
        loaded = np.hypot(problem.forces[..., X], problem.forces[..., Y]) > 0
        distances = np.hypot(
            np.nonzero(loaded)[0] - 96, np.nonzero(loaded)[1])
        stories = np.unique(np.round(distances / 40.0))
        np.testing.assert_array_equal(stories, [1, 2, 3, 4])
        self.assertEqual(problem.forces[96, 0, X], 0.0)
        self.assertEqual(problem.forces[96, 0, Y], 0.0)
        self.assertEqual(problem.forces[96, -1, Y], 0.0)

    def test_coarse_stage_scales_the_pattern(self):
        # AdaptivePixel's first stage is the 192x160 tee divided by 4.
        problem = tee_shirt(width=48, height=40, interval=10)
        self.assertEqual(problem.mask.shape, (40, 48))
        self.assertEqual(problem.mask[30, 0], 0.0)
        self.assertEqual(problem.mask[30, 24], 1.0)
        self.assertGreater(int(problem.normals[..., X].sum()), 0)
        self.assertGreater(int(np.count_nonzero(problem.forces)), 0)
        _factors(problem)

    def test_factors(self):
        _factors(tee_shirt())


if __name__ == '__main__':
    absltest.main()
