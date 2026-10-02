# lint as python3
"""Opt-in physics Heaviside ramp and the load-on-solid metric.

No CLIP checkpoint. Grids are small enough that a physics solve is cheap.
"""

# pylint: disable=missing-docstring
# pylint: disable=protected-access

import numpy as np
import torch
from absl.testing import absltest

from guidance.loss_sketch import load_on_solid_fraction
from model.model_ada import AdaptivePixelModel
from optimize.optimizers import AdaptiveAdam_Optimizer
from problem.problems import StructuralParams

ALWAYS_CONVERGE = 1.0e12


def _model(beta_max=0.0):
    model = AdaptivePixelModel(
        structural_params=StructuralParams(
            problem_name='multistory_building', width=16, height=32,
            density=0.3, interval=8, filter_width=1.5),
        clip_loss=None, seed=0, resize_num=0, resize_scale=2)
    model.physics_projection_beta_max = beta_max
    return model


def _run(model, max_iterations):
    optimizer = AdaptiveAdam_Optimizer(
        model, max_iterations=max_iterations, lr=0.2,
        convergence_threshold=ALWAYS_CONVERGE, max_resize_iteration=50)
    return optimizer.optimize()


class PhysicsProjectionRampTest(absltest.TestCase):

    def test_off_leaves_projection_off(self):
        model = _model()
        ds = _run(model, max_iterations=3)
        self.assertFalse(model.env.args.get('heavyside', False))
        self.assertIsNone(model._last_physics_projection_beta)
        self.assertEqual(int(ds.attrs['converged']), 1)

    def test_ramp_runs_to_the_cap_and_ends_at_beta_max(self):
        model = _model(beta_max=8.0)
        ds = _run(model, max_iterations=3)
        self.assertEqual(int(ds.attrs['converged']), 0)
        self.assertEqual(np.asarray(ds['loss'].values).reshape(-1).size, 3)
        self.assertTrue(model.env.args['heavyside'])
        self.assertAlmostEqual(model.env.args['beta'], 8.0)
        self.assertAlmostEqual(model._last_physics_projection_beta, 8.0)

    def test_ramp_is_geometric(self):
        model = _model(beta_max=16.0)
        model._opt_max_iterations = 5
        betas = []
        for step in range(5):
            model._opt_step = step
            model.apply_physics_projection_schedule()
            betas.append(model.env.args['beta'])
        np.testing.assert_allclose(betas, [1.0, 2.0, 4.0, 8.0, 16.0])

    def test_beta_max_below_one_is_rejected(self):
        model = _model(beta_max=0.5)
        with self.assertRaisesRegex(ValueError, '>= 1'):
            model.apply_physics_projection_schedule()

    def test_projection_sharpens_the_physical_density(self):
        soft, sharp = _model(), _model(beta_max=8.0)
        sharp._opt_step = sharp._opt_max_iterations = 1
        sharp.apply_physics_projection_schedule()
        z = torch.from_numpy(
            np.random.default_rng(0).normal(scale=2.0, size=soft.z.shape))
        z = z.to(soft.z.dtype)
        rho_soft = soft.get_physical_density(z).detach().numpy()
        rho_sharp = sharp.get_physical_density(z).detach().numpy()
        gray = lambda r: np.mean((r > 0.1) & (r < 0.9))
        self.assertLess(gray(rho_sharp), gray(rho_soft))


class LoadOnSolidFractionTest(absltest.TestCase):

    @staticmethod
    def _forces(nely, nelx, loaded_nodes):
        f = np.zeros((nelx + 1, nely + 1, 2))
        for ix, iy in loaded_nodes:
            f[ix, iy, 1] = -1.0
        return f.ravel()

    def test_counts_nodes_touching_any_solid_element(self):
        density = np.zeros((2, 3))
        density[0, 0] = 1.0  # touches nodes (0,0) (1,0) (0,1) (1,1)
        forces = self._forces(2, 3, [(1, 1), (3, 2)])
        self.assertAlmostEqual(load_on_solid_fraction(density, forces), 0.5)

    def test_weights_by_force_magnitude(self):
        density = np.zeros((2, 3))
        density[1, 2] = 0.6  # touches nodes (2..3, 1..2)
        f = np.zeros((4, 3, 2))
        f[3, 2, 1] = -3.0
        f[0, 0, 0] = 1.0
        self.assertAlmostEqual(load_on_solid_fraction(density, f.ravel()), 0.75)

    def test_no_load_returns_none(self):
        self.assertIsNone(
            load_on_solid_fraction(np.ones((2, 2)), np.zeros(3 * 3 * 2)))


if __name__ == '__main__':
    absltest.main()
