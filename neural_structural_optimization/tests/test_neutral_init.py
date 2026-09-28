# lint as python3
"""Neutral initialization: uniform volfrac plus tiny deterministic noise."""

import numpy as np
import torch
from absl.testing import absltest

from neural_structural_optimization.guidance.loss_sketch import load_site_mask
from neural_structural_optimization.model.model_pixel import PixelModel
from neural_structural_optimization.problem.problems import StructuralParams
from neural_structural_optimization.optimize.utils import init_weight_neutral


_WIDTH = 16
_HEIGHT = 32
_VOLFRAC = 0.3


def _params() -> StructuralParams:
  return StructuralParams(
      problem_name='multistory_building',
      width=_WIDTH,
      height=_HEIGHT,
      density=_VOLFRAC,
      interval=8,
  )


class NeutralInitTest(absltest.TestCase):

  def test_is_deterministic_and_near_volfrac(self):
    model_a = PixelModel(structural_params=_params(), seed=0)
    model_b = PixelModel(structural_params=_params(), seed=0)
    z_a = init_weight_neutral(model_a, density=_VOLFRAC, seed=12)
    z_b = init_weight_neutral(model_b, density=_VOLFRAC, seed=12)
    torch.testing.assert_close(z_a.detach(), z_b.detach())
    field = z_a.detach().cpu().numpy()
    self.assertEqual(field.shape, (1, _HEIGHT, _WIDTH))
    self.assertGreaterEqual(float(field.min()), 0.0)
    self.assertLessEqual(float(field.max()), 1.0)
    off_load = field.copy()
    sites = load_site_mask(
        model_a.env.args['forces'], nely=_HEIGHT, nelx=_WIDTH)
    off_load[0, sites > 0] = np.nan
    self.assertAlmostEqual(float(np.nanmean(off_load)), _VOLFRAC, delta=0.02)

  def test_different_seeds_differ(self):
    model_a = PixelModel(structural_params=_params(), seed=0)
    model_b = PixelModel(structural_params=_params(), seed=0)
    z_a = init_weight_neutral(model_a, density=_VOLFRAC, seed=1)
    z_b = init_weight_neutral(model_b, density=_VOLFRAC, seed=2)
    self.assertFalse(torch.allclose(z_a.detach(), z_b.detach()))

  def test_load_sites_are_material(self):
    model = PixelModel(structural_params=_params(), seed=0)
    init_weight_neutral(model, density=_VOLFRAC, seed=12, noise_amp=0.0)
    field = model.z.detach().cpu().numpy()[0]
    sites = load_site_mask(
        model.env.args['forces'], nely=_HEIGHT, nelx=_WIDTH)
    self.assertGreater(float(sites.mean()), 0.0)
    np.testing.assert_allclose(field[sites > 0], 1.0, atol=1e-6)

  def test_no_load_union_stays_near_uniform(self):
    model = PixelModel(structural_params=_params(), seed=0)
    init_weight_neutral(
        model, density=_VOLFRAC, seed=12, noise_amp=0.0,
        union_load_sites=False)
    field = model.z.detach().cpu().numpy()
    np.testing.assert_allclose(field, _VOLFRAC, atol=1e-6)



if __name__ == '__main__':
  absltest.main()
