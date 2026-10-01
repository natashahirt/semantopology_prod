# lint as python3
# Copyright 2019 Google LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Regression tests for discretization parameter propagation and density chain.

Most tests here drive physics through a BARE args dict, so `args['heavyside'] =
True` really does enable the projection. That idiom does NOT carry over to a
constructed `Model`: `Environment.__init__` normalizes `args` into its own
dict, and every physics entry point reads `env.args`, so writing to
`model.args` is a silent no-op. `ProjectionToggleOnModelTest` pins that and
shows the two ways that do work.
"""

# pylint: disable=missing-docstring
# pylint: disable=invalid-name

import warnings

import autograd
import autograd.numpy as np
import torch
from model.model_pixel import PixelModel
from guidance.loss_structural import PhysicalDensity
from physics import api as topo_api
from physics import physics
from physics import autograd as topo_autograd
from problem import problems
from problem.problems import (
    StructuralParams,
    resolve_analysis_filter_width,
)
import numpy as npo
from absl.testing import absltest

find_root = topo_autograd.find_root

# Characterization pins for default StructuralParams, heavyside=False, seed 0.
# Recomputed once on the fixed default path; any accidental reordering fails here.
_PARITY_SEED = 0
_PARITY_VOLFRAC = 0.5
_PARITY_DENSITY_SUM_VC = 1800.0582793246901
_PARITY_DENSITY_MEAN_VC = 0.5000161887013028
_PARITY_COMPLIANCE_VC = 74.20393726617235
_PARITY_DENSITY_SUM_NO_VC = 1798.9812813154135
_PARITY_COMPLIANCE_NO_VC = 81.00982672269954
_PARITY_RTOL = 1e-10
_PARITY_ATOL = 1e-12


def _default_structural_args():
  """Build physics args from default StructuralParams (heavyside=False)."""
  params = StructuralParams(density=_PARITY_VOLFRAC)
  return topo_api.specified_task(params.get_problem())


def _parity_logits_and_no_vc_input(args):
  """Reproduce the exact RandomState(0) draw order used for parity pins."""
  rng = npo.random.RandomState(_PARITY_SEED)
  logits = rng.randn(args['nely'], args['nelx']) * 0.5
  clipped = npo.clip(
      0.5 + 0.1 * rng.randn(args['nely'], args['nelx']), 0.0, 1.0)
  return logits, clipped


def _small_structural_args(**overrides):
  """Build physics args on a small grid, for tests that differentiate."""
  params = StructuralParams(width=24, height=16, density=0.4)
  args = topo_api.specified_task(params.get_problem())
  args.update(overrides)
  return args


def _shipped_structural_params(**overrides):
  """The multistory config script/run.py ships, which the parity work targets."""
  kwargs = dict(
      problem_name='multistory_building', width=50, height=100, density=0.3,
      interval=20, rmin=1.0, filter_width='linear', beta='linear')
  kwargs.update(overrides)
  return StructuralParams(**kwargs)


def _inverse_heavyside_projection(y, beta, eta):
  """Invert `physics.heavyside_projection`, which is strictly increasing."""
  offset = npo.tanh(beta * eta)
  scale = offset + npo.tanh(beta * (1.0 - eta))
  return eta + npo.arctanh(y * scale - offset) / beta


class DiscretizationParityTest(absltest.TestCase):
  """Gate 1: default heavyside=False path must stay numerically fixed."""

  def test_physical_density_volume_constraint_parity(self):
    args = _default_structural_args()
    self.assertFalse(args['heavyside'])
    logits, _ = _parity_logits_and_no_vc_input(args)
    density = physics.physical_density(logits, args, volume_constraint=True)
    npo.testing.assert_allclose(
        density.sum(), _PARITY_DENSITY_SUM_VC, rtol=_PARITY_RTOL, atol=_PARITY_ATOL)
    npo.testing.assert_allclose(
        density.mean(), _PARITY_DENSITY_MEAN_VC, rtol=_PARITY_RTOL, atol=_PARITY_ATOL)

  def test_objective_volume_constraint_parity(self):
    args = _default_structural_args()
    logits, _ = _parity_logits_and_no_vc_input(args)
    ke = physics.get_stiffness_matrix(args['young'], args['poisson'])
    compliance = physics.objective(logits, ke, args, volume_constraint=True)
    npo.testing.assert_allclose(
        compliance, _PARITY_COMPLIANCE_VC, rtol=_PARITY_RTOL, atol=_PARITY_ATOL)

  def test_physical_density_no_volume_constraint_parity(self):
    args = _default_structural_args()
    _, clipped = _parity_logits_and_no_vc_input(args)
    density = physics.physical_density(clipped, args, volume_constraint=False)
    npo.testing.assert_allclose(
        density.sum(), _PARITY_DENSITY_SUM_NO_VC, rtol=_PARITY_RTOL, atol=_PARITY_ATOL)

  def test_objective_no_volume_constraint_parity(self):
    args = _default_structural_args()
    _, clipped = _parity_logits_and_no_vc_input(args)
    ke = physics.get_stiffness_matrix(args['young'], args['poisson'])
    compliance = physics.objective(clipped, ke, args, volume_constraint=False)
    npo.testing.assert_allclose(
        compliance, _PARITY_COMPLIANCE_NO_VC, rtol=_PARITY_RTOL, atol=_PARITY_ATOL)


def _design_region_mean(density, args):
  """Mean density over the design region only.

  `density.mean()` is NOT the volume fraction on a masked problem: it averages
  the zeroed non-design elements in too. On `l_shape` at volfrac 0.7 the two
  differ by 0.25, so a test that uses the plain mean passes only because the
  default problem's mask happens to be all ones.
  """
  mask = npo.broadcast_to(args['mask'], density.shape)
  return float((density * mask).sum() / mask.sum())


class VolumeUnderProjectionTest(absltest.TestCase):
  """Gate 2: with heavyside=True, the design region's mean must equal volfrac.

  Swept OFF the volfrac == eta diagonal and over masked problems, because both
  are places the constraint could fail invisibly. At volfrac == eta the
  projection is mean-neutral, so a test pinned there cannot see a projection
  that shifts the mean; and on a masked problem the plain array mean is wrong
  by up to 0.25 regardless of whether the constraint holds.
  """

  _PROBLEMS = ('cantilever_beam_full', 'mbb_beam', 'l_shape',
               'multistory_building')
  _TOLERANCE = 1e-9

  def _args(self, problem_name, volfrac, beta, eta):
    params = StructuralParams(
        problem_name=problem_name, width=60, height=60, density=volfrac,
        heavyside=True, beta=beta, eta=eta)
    return topo_api.specified_task(params.get_problem())

  def test_mean_density_matches_volfrac_across_beta(self):
    args = _default_structural_args()
    args['heavyside'] = True
    rng = npo.random.RandomState(42)
    logits = rng.randn(args['nely'], args['nelx']) * 0.5
    for beta in (1.0, 4.0, 16.0):
      with self.subTest(beta=beta):
        trial = dict(args)
        trial['beta'] = beta
        density = physics.physical_density(
            logits, trial, volume_constraint=True)
        self.assertAlmostEqual(
            _design_region_mean(density, trial), trial['volfrac'],
            delta=self._TOLERANCE)

  def test_mean_density_matches_volfrac_off_the_eta_diagonal(self):
    for volfrac in (0.1, 0.3, 0.7):
      for eta in (0.3, 0.7):
        for beta in (4.0, 16.0):
          with self.subTest(volfrac=volfrac, eta=eta, beta=beta):
            args = self._args('cantilever_beam_full', volfrac, beta, eta)
            logits = npo.random.RandomState(0).randn(
                args['nely'], args['nelx']) * 2.0
            density = physics.physical_density(
                logits, args, volume_constraint=True)
            self.assertAlmostEqual(
                _design_region_mean(density, args), volfrac,
                delta=self._TOLERANCE)

  def test_mean_density_matches_volfrac_on_masked_problems(self):
    for problem_name in self._PROBLEMS:
      for beta in (4.0, 16.0):
        with self.subTest(problem=problem_name, beta=beta):
          args = self._args(problem_name, 0.3, beta, 0.5)
          logits = npo.random.RandomState(0).randn(
              args['nely'], args['nelx']) * 2.0
          density = physics.physical_density(
              logits, args, volume_constraint=True)
          self.assertAlmostEqual(
              _design_region_mean(density, args), 0.3, delta=self._TOLERANCE)

  def test_no_density_bleeds_outside_the_design_region(self):
    for problem_name in self._PROBLEMS:
      with self.subTest(problem=problem_name):
        args = self._args(problem_name, 0.3, 4.0, 0.5)
        mask = npo.broadcast_to(args['mask'], (args['nely'], args['nelx']))
        if mask.min() >= 1:
          continue  # trivial mask, nothing to bleed into
        logits = npo.random.RandomState(0).randn(
            args['nely'], args['nelx']) * 2.0
        density = physics.physical_density(
            logits, args, volume_constraint=True)
        self.assertEqual(float(npo.abs(density * (1 - mask)).max()), 0.0)

  def test_plain_array_mean_is_not_the_volume_fraction(self):
    """Pins WHY the helper exists, so nobody simplifies it back to .mean()."""
    args = self._args('l_shape', 0.7, 16.0, 0.5)
    logits = npo.random.RandomState(0).randn(
        args['nely'], args['nelx']) * 2.0
    density = physics.physical_density(logits, args, volume_constraint=True)
    self.assertAlmostEqual(
        _design_region_mean(density, args), 0.7, delta=self._TOLERANCE)
    self.assertGreater(abs(float(density.mean()) - 0.7), 0.2)


class MultistoryLoadPatternTest(absltest.TestCase):
  """`multistory_building` spaces floors by rows, and that choice is load-bearing.

  It previously took a floor COUNT. Switching to a SPACING changes the structure
  at a fixed call site by an order of magnitude, so these pin the semantics and
  the measured size of the difference rather than leaving it to a docstring.
  """

  _W, _H, _VOLFRAC = 128, 256, 0.3

  def _compliance(self, problem):
    args = topo_api.specified_task(problem)
    ke = physics.get_stiffness_matrix(args['young'], args['poisson'])
    x = npo.full((args['nely'], args['nelx']), self._VOLFRAC)
    return float(physics.objective(x, ke, args, volume_constraint=False))

  def _loaded_rows(self, problem):
    return int((npo.abs(problem.forces[:, :, 1]).sum(axis=0) > 0).sum())

  def test_interval_sets_the_floor_spacing(self):
    problem = problems.multistory_building(
        self._W, self._H, density=self._VOLFRAC, interval=64)
    # Rows 0, 64, 128, 192, 256 -- the last sits on the supported edge.
    self.assertEqual(self._loaded_rows(problem), 5)

  def test_floor_count_and_spacing_are_convertible(self):
    """interval == height // num_stories reproduces the old load pattern."""
    problem = problems.multistory_building(
        self._W, self._H, density=self._VOLFRAC, interval=self._H // 16)
    rows = npo.nonzero(npo.abs(problem.forces[:, :, 1]).sum(axis=0))[0]
    self.assertContainsSubset(list(range(0, self._H, 16)), rows.tolist())

  def test_the_spacing_change_is_an_order_of_magnitude(self):
    """Guards the claim that pre-change compliance figures are incomparable."""
    sparse = self._compliance(problems.multistory_building(
        self._W, self._H, density=self._VOLFRAC, interval=64))
    dense = self._compliance(problems.multistory_building(
        self._W, self._H, density=self._VOLFRAC, interval=16))
    self.assertGreater(dense / sparse, 10.0)

  # The companion measurement -- that dropping the right wall moves compliance
  # by only 0.07% (563.43 walled vs 563.80 free, 128x256, uniform 0.3) -- is
  # NOT asserted here, and deliberately so. Solving the wall-free case needs a
  # solver that tolerates the rank deficiency, and merely provoking CHOLMOD's
  # CholmodNotPositiveDefiniteError corrupts its internal state so the NEXT
  # solve in the process segfaults. A test that measures it would take the rest
  # of the suite down with it. Recompute it under SuperLU if it needs checking.

  def test_without_the_right_wall_nothing_constrains_x(self):
    """The rank deficiency CHOLMOD rejects: a free horizontal slide."""
    free = problems.multistory_building(
        self._W, self._H, density=self._VOLFRAC, fix_right_wall=False)
    walled = problems.multistory_building(
        self._W, self._H, density=self._VOLFRAC)
    self.assertEqual(int(free.normals[:, :, 0].sum()), 0)
    self.assertGreater(int(walled.normals[:, :, 0].sum()), 0)

  def test_stale_num_stories_warns_instead_of_being_dropped(self):
    params = StructuralParams(
        problem_name='multistory_building', width=50, height=100,
        density=0.3, num_stories=5)
    with self.assertWarnsRegex(UserWarning, 'num_stories.*is ignored'):
      params.get_problem()

  def test_interval_alone_does_not_warn(self):
    params = StructuralParams(
        problem_name='multistory_building', width=50, height=100,
        density=0.3, interval=20)
    with warnings.catch_warnings():
      warnings.simplefilter('error', UserWarning)
      params.get_problem()

  def test_num_stories_still_reaches_staircase(self):
    """The field is not dead -- staircase genuinely takes a count."""
    params = StructuralParams(
        problem_name='staircase', width=64, height=64, density=0.3,
        num_stories=3)
    with warnings.catch_warnings():
      warnings.simplefilter('error', UserWarning)
      params.get_problem()


class VolumeWithoutProjectionTest(absltest.TestCase):
  """The legacy path misses volfrac, and it always has -- pinned as a bound.

  With projection off, the volume offset is solved on the raw sigmoid and the
  cone filter is applied afterwards. The filter is not mean-preserving, so the
  finished density misses volfrac by a small amount. This is PRE-EXISTING
  behavior, not a regression from the density-chain reordering, and stage 3's
  Venice reproduction depends on it staying exactly as it is -- hence a bound
  rather than an equality. Enabling projection moves enforcement to the end of
  the chain and drives this to ~1e-14.
  """

  _OBSERVED_MAX_ERROR = 1.1e-4

  def test_legacy_path_misses_volfrac_by_a_bounded_amount(self):
    for problem_name in ('cantilever_beam_full', 'multistory_building'):
      for volfrac in (0.1, 0.3, 0.5, 0.7):
        with self.subTest(problem=problem_name, volfrac=volfrac):
          params = StructuralParams(
              problem_name=problem_name, width=60, height=60, density=volfrac)
          args = topo_api.specified_task(params.get_problem())
          logits = npo.random.RandomState(0).randn(
              args['nely'], args['nelx']) * 2.0
          density = physics.physical_density(
              logits, args, volume_constraint=True)
          error = abs(_design_region_mean(density, args) - volfrac)
          self.assertLess(error, self._OBSERVED_MAX_ERROR)


class ParameterPropagationTest(absltest.TestCase):
  """Gate 3: StructuralParams fields reach physics args and change behavior."""

  def test_discretization_fields_in_specified_task(self):
    params = StructuralParams(
        filter_width='linear',
        rmin=1.5,
        beta='linear',
        heavyside=True,
        eta=0.45,
        density=0.4,
    )
    args = topo_api.specified_task(params.get_problem())
    for key in ('filter_width', 'rmin', 'beta', 'eta'):
      self.assertIsInstance(args[key], float)
      self.assertNotIsInstance(args[key], str)
    self.assertIsInstance(args['heavyside'], bool)
    self.assertNotIsInstance(args['heavyside'], str)
    # Linear schedules resolve to concrete floats (2 * rmin and beta start).
    npo.testing.assert_allclose(args['filter_width'], 3.0)
    npo.testing.assert_allclose(args['beta'], 1.0)
    npo.testing.assert_allclose(args['rmin'], 1.5)
    self.assertTrue(args['heavyside'])
    npo.testing.assert_allclose(args['eta'], 0.45)

  def test_filter_width_observable_in_physical_density(self):
    rng = npo.random.RandomState(7)
    logits = rng.randn(60, 60) * 0.5
    narrow = topo_api.specified_task(
        StructuralParams(filter_width=2.0, rmin=2.0).get_problem())
    wide = topo_api.specified_task(
        StructuralParams(filter_width=8.0, rmin=8.0).get_problem())
    d_narrow = physics.physical_density(logits, narrow, volume_constraint=False)
    d_wide = physics.physical_density(logits, wide, volume_constraint=False)
    max_diff = float(npo.max(npo.abs(d_narrow - d_wide)))
    self.assertGreater(max_diff, 0.01)


class FindRootBracketTest(absltest.TestCase):
  """Optional: check_bracket guard on find_root."""

  def test_check_bracket_raises_on_bad_bracket(self):
    f = lambda x, y: y - 5.0  # always positive for any y > 0
    with self.assertRaises(ValueError):
      find_root(f, 0.0, lower_bound=1.0, upper_bound=2.0, check_bracket=True)

  def test_check_bracket_allows_valid_bracket(self):
    f = lambda x, y: y ** 2 - x
    result = find_root(
        f, 4.0, lower_bound=0.0, upper_bound=3.0, check_bracket=True)
    npo.testing.assert_allclose(result, 2.0)


class DegenerateFilterWidthTest(absltest.TestCase):
  """Gate 4: a cone filter that silently becomes the identity must not ship."""

  def test_check_filter_width_rejects_degenerate_radii(self):
    for width in (0.5, 1.0):
      with self.subTest(filter_width=width):
        with self.assertRaisesRegex(ValueError, 'degenerates the cone filter'):
          physics.check_filter_width(width)

  def test_check_filter_width_accepts_effective_radii(self):
    for width in (1.5, 2.0):
      with self.subTest(filter_width=width):
        self.assertEqual(physics.check_filter_width(width), width)

  def test_degenerate_radii_are_exactly_the_identity(self):
    """The measurement the guard exists for: <= 1.0 is a no-op filter."""
    args = _default_structural_args()
    field = npo.random.RandomState(3).rand(args['nely'], args['nelx'])
    for width in (0.5, 1.0):
      with self.subTest(filter_width=width):
        filtered = topo_autograd.cone_filter(field, width, args['mask'])
        npo.testing.assert_array_equal(filtered, field)
    for width in (1.5, 2.0):
      with self.subTest(filter_width=width):
        filtered = topo_autograd.cone_filter(field, width, args['mask'])
        self.assertGreater(float(npo.max(npo.abs(filtered - field))), 0.1)

  def test_linear_schedule_below_half_rmin_is_rejected(self):
    # rmin=0.25 resolves 'linear' to 0.5, the config that shipped unfiltered.
    with self.assertRaisesRegex(ValueError, 'degenerates the cone filter'):
      StructuralParams(rmin=0.25, filter_width='linear').get_problem()

  def test_explicit_degenerate_filter_width_is_rejected(self):
    with self.assertRaisesRegex(ValueError, 'degenerates the cone filter'):
      StructuralParams(filter_width=1.0)

  def test_physical_density_rejects_degenerate_args(self):
    args = _default_structural_args()
    args['filter_width'] = 0.5
    logits, _ = _parity_logits_and_no_vc_input(args)
    with self.assertRaisesRegex(ValueError, 'degenerates the cone filter'):
      physics.physical_density(logits, args, volume_constraint=True)

  def test_shipped_linear_schedule_actually_filters(self):
    """rmin=1.0 with 'linear' must reach the legacy radius of 2.0."""
    args = topo_api.specified_task(_shipped_structural_params().get_problem())
    npo.testing.assert_allclose(args['filter_width'], 2.0)
    field = npo.random.RandomState(3).rand(args['nely'], args['nelx'])
    filtered = topo_autograd.cone_filter(
        field, args['filter_width'], args['mask'])
    self.assertGreater(float(npo.max(npo.abs(filtered - field))), 0.1)


class FilterWidthBoundsTest(absltest.TestCase):
  """Gate 4b: the radius guard must be two-sided and finite-checked.

  A one-sided ``width <= 1.0`` test lets nan, inf and unbounded radii through,
  because every comparison against nan is False and nothing bounds a large
  radius. All three destroy the design as silently as the identity filter the
  guard was written for.
  """

  def test_rejects_non_finite_radii(self):
    for width in (float('nan'), float('inf'), float('-inf')):
      with self.subTest(filter_width=width):
        with self.assertRaisesRegex(ValueError, 'not a finite radius'):
          physics.check_filter_width(width)

  def test_rejects_radii_past_the_grid_diagonal(self):
    for width in (100.0, 200.0, 1e9):
      with self.subTest(filter_width=width):
        with self.assertRaisesRegex(
            ValueError, 'exceeds the 60x60 grid diagonal'):
          physics.check_filter_width(width, nelx=60, nely=60)

  def test_accepts_radii_inside_the_grid(self):
    for width in (1.5, 2.0, 20.0):
      with self.subTest(filter_width=width):
        self.assertEqual(
            physics.check_filter_width(width, nelx=60, nely=60), width)

  def test_radii_past_the_diagonal_are_a_global_average(self):
    """The measurement the upper bound exists for: the design is smeared out."""
    field = npo.random.RandomState(3).rand(16, 16)
    self.assertGreater(float(npo.ptp(field)), 0.9)
    # 16x16 has a diagonal of ~22.6, so every cone covers the whole domain.
    smeared = topo_autograd.cone_filter(field, 24.0, 1)
    self.assertLess(float(npo.ptp(smeared)), 0.05)

  def test_structural_params_rejects_unbounded_and_non_finite_radii(self):
    for kwargs in (dict(filter_width=float('nan')),
                   dict(filter_width=float('inf')),
                   dict(filter_width=1e9),
                   dict(rmin=float('nan')),
                   dict(rmin=float('inf'))):
      with self.subTest(**kwargs):
        with self.assertRaises(ValueError):
          StructuralParams(**kwargs)

  def test_linear_schedule_above_the_grid_diagonal_is_rejected(self):
    with self.assertRaisesRegex(ValueError, 'exceeds the 60x60 grid diagonal'):
      StructuralParams(rmin=1e9, filter_width='linear').get_problem()


class AnalysisGridFilterWidthTest(absltest.TestCase):
  """Gate 4c: a coarse ANALYSIS grid must clamp its radius, never abort a run.

  `_set_analysis_factor` divides the radius by the analysis factor, so a valid
  design-grid radius can land below the degeneracy threshold. It is called from
  `PixelModel.upsample`, i.e. after a stage has finished training, so raising
  there discards completed work over a derived value nobody authored.
  """

  def test_clamp_raises_a_degenerate_radius_and_warns(self):
    with self.assertWarnsRegex(UserWarning, 'clamping to 1.5'):
      self.assertEqual(physics.clamp_filter_width(1.0), 1.5)

  def test_clamp_leaves_usable_radii_untouched(self):
    for width in (1.5, 2.0, 8.0):
      with self.subTest(filter_width=width):
        with warnings.catch_warnings():
          warnings.simplefilter('error')
          self.assertEqual(physics.clamp_filter_width(width), width)

  def test_clamp_still_rejects_non_finite_radii(self):
    # Division never produces these, so clamping would only hide a bad input.
    for width in (float('nan'), float('inf')):
      with self.subTest(filter_width=width):
        with self.assertRaisesRegex(ValueError, 'not a finite radius'):
          physics.clamp_filter_width(width)

  def test_resolve_analysis_filter_width_clamps_the_linear_schedule(self):
    with self.assertWarns(UserWarning):
      self.assertEqual(
          resolve_analysis_filter_width('linear', rmin=0.5), 1.5)

  def test_user_authored_degenerate_radius_still_raises(self):
    """The clamp must not weaken the guard on configuration the user wrote."""
    with self.assertRaisesRegex(ValueError, 'degenerates the cone filter'):
      StructuralParams(filter_width=1.0)
    with self.assertRaisesRegex(ValueError, 'degenerates the cone filter'):
      StructuralParams(rmin=0.25, filter_width='linear').get_problem()

  def test_coarse_analysis_grid_does_not_abort_the_model(self):
    model = PixelModel(structural_params=_shipped_structural_params(), seed=0)
    # rmin=1.0 over a factor of 2 resolves 'linear' to exactly 1.0.
    with self.assertWarnsRegex(UserWarning, 'clamping to 1.5'):
      model._set_analysis_factor(max_dim=50)
    self.assertEqual(model.analysis_factor, 2)
    self.assertEqual(model.analysis_env.args['filter_width'], 1.5)
    # The design grid keeps the radius the user configured.
    self.assertEqual(model.env.args['filter_width'], 2.0)

  def test_shipped_config_upsample_chain_completes(self):
    """The chain script/run.py runs: resize_num=4 at the default max_dim."""
    model = PixelModel(structural_params=_shipped_structural_params(), seed=0)
    radii = [(model.analysis_factor, model.analysis_env.args['filter_width'])]
    for _ in range(4):
      model.upsample(scale=2)
      radii.append(
          (model.analysis_factor, model.analysis_env.args['filter_width']))
    # rmin scales up with the grid, so the analysis radius stays well clear of
    # the degeneracy threshold and nothing is clamped.
    self.assertEqual(radii, [(1, 2.0), (1, 4.0), (1, 8.0), (2, 8.0), (4, 8.0)])


class InertRminTest(absltest.TestCase):
  """Gate 4d: rmin alone does not set the filter radius, and must say so.

  `apply_discretization_params` skips None fields, so `StructuralParams(rmin=x)`
  leaves filter_width at the Problem default and discards x -- while
  `StructuralParams(rmin=x, filter_width='linear')` would raise for a small x.
  """

  def test_rmin_without_filter_width_warns(self):
    with self.assertWarnsRegex(UserWarning, 'does not set the filter radius'):
      problem = StructuralParams(rmin=0.75).get_problem()
    self.assertEqual(problem.filter_width, 2.0)
    self.assertEqual(problem.rmin, 0.75)

  def test_rmin_with_a_filter_width_does_not_warn(self):
    for filter_width in ('linear', 3.0):
      with self.subTest(filter_width=filter_width):
        with warnings.catch_warnings():
          warnings.simplefilter('error')
          StructuralParams(rmin=1.5, filter_width=filter_width).get_problem()

  def test_default_args_rmin_and_filter_width_agree(self):
    args = physics.default_args()
    self.assertEqual(args['filter_width'], 2.0 * args['rmin'])
    physics.check_filter_width(
        args['filter_width'], nelx=args['nelx'], nely=args['nely'])


class ProjectionToggleOnModelTest(absltest.TestCase):
  """`model.args['heavyside'] = True` is a silent no-op; env.args is the truth."""

  def test_writing_model_args_does_not_reach_physics(self):
    model = PixelModel(structural_params=_shipped_structural_params(), seed=0)
    model.args['heavyside'] = True
    self.assertFalse(model.env.args['heavyside'])
    self.assertIsNone(physics.projection_params(model.env.args))

  def test_refreshing_the_environment_enables_projection(self):
    model = PixelModel(structural_params=_shipped_structural_params(), seed=0)
    model.structural_params = model.structural_params.copy(
        heavyside=True, beta=4.0)
    model._refresh_physics_environment()
    self.assertTrue(model.env.args['heavyside'])
    self.assertEqual(physics.projection_params(model.env.args), (4.0, 0.5))


class ProjectionParameterValidationTest(absltest.TestCase):
  """Gate 5: beta and eta values that NaN the density field must be rejected."""

  def test_structural_params_rejects_non_positive_beta(self):
    for beta in (-4.0, 0.0):
      with self.subTest(beta=beta):
        with self.assertRaisesRegex(ValueError, 'beta must be positive'):
          StructuralParams(beta=beta)

  def test_structural_params_rejects_non_finite_beta(self):
    # nan slips past every one-sided comparison, so it needs its own check;
    # eta is already safe because its check is a range test.
    for beta in (float('nan'), float('inf'), float('-inf')):
      with self.subTest(beta=beta):
        with self.assertRaisesRegex(ValueError, 'beta must be finite'):
          StructuralParams(beta=beta)

  def test_projection_params_rejects_non_finite_beta(self):
    args = _default_structural_args()
    args['heavyside'] = True
    for beta in (float('nan'), float('inf')):
      with self.subTest(beta=beta):
        with self.assertRaisesRegex(ValueError, 'beta must be finite'):
          physics.projection_params(dict(args, beta=beta))

  def test_non_finite_beta_would_have_nan_ed_the_density_field(self):
    """The failure the check replaces: an all-NaN density and no error."""
    args = dict(_default_structural_args(), heavyside=True)
    logits, _ = _parity_logits_and_no_vc_input(args)
    nan_projected = physics.heavyside_projection(
        physics.sigmoid(logits), float('nan'), args['eta'])
    self.assertTrue(bool(npo.all(npo.isnan(nan_projected))))

  def test_structural_params_rejects_eta_outside_unit_interval(self):
    for eta in (-2.0, 1.5, 3.0, 25.0):
      with self.subTest(eta=eta):
        with self.assertRaisesRegex(ValueError, r'eta must lie in \[0, 1\]'):
          StructuralParams(eta=eta)

  def test_projection_params_rejects_negative_beta(self):
    args = _default_structural_args()
    args.update(heavyside=True, beta=-4.0)
    with self.assertRaisesRegex(ValueError, 'beta must be positive'):
      physics.projection_params(args)

  def test_projection_params_rejects_eta_outside_unit_interval(self):
    args = _default_structural_args()
    args['heavyside'] = True
    for beta, eta in ((16.0, 3.0), (16.0, -2.0), (40.0, 1.5), (2.0, 25.0),
                      (4.0, 5.0)):
      with self.subTest(beta=beta, eta=eta):
        trial = dict(args, beta=beta, eta=eta)
        with self.assertRaisesRegex(ValueError, r'eta must lie in \[0, 1\]'):
          physics.projection_params(trial)

  def test_projection_params_accepts_boundary_eta(self):
    args = _default_structural_args()
    args.update(heavyside=True, beta=4.0)
    for eta in (0.0, 0.5, 1.0):
      with self.subTest(eta=eta):
        self.assertEqual(physics.projection_params(dict(args, eta=eta)),
                         (4.0, eta))

  def test_projection_stays_off_when_heavyside_is_false(self):
    args = _default_structural_args()
    self.assertIsNone(physics.projection_params(args))


class RenderObjectiveConsistencyTest(absltest.TestCase):
  """Gate 6: the unfiltered view must be the design the objective solved for.

  `train.utils.constrained_logits` asks for the density without the cone
  filter. That has to be the same design seen before the filter, not a second
  design whose own volume offset was re-solved against the unfiltered residual
  -- otherwise the CNN-to-pixel handoff hands over something the objective
  never evaluated. `Environment.render` is the filtered field (Stage 4); this
  class pins the handoff view, not the saved image.
  """

  def test_filtering_the_render_reproduces_the_objective(self):
    # Projection off but volume enforced last, so the only difference between
    # the two views is the cone filter itself.
    args = _default_structural_args()
    args['enforce_volume_last'] = True
    logits, _ = _parity_logits_and_no_vc_input(args)
    render = physics.physical_density(
        logits, args, volume_constraint=True, cone_filter=False)
    objective = physics.physical_density(
        logits, args, volume_constraint=True, cone_filter=True)
    refiltered = topo_autograd.cone_filter(
        render, args['filter_width'], args['mask'])
    npo.testing.assert_allclose(refiltered, objective, rtol=0, atol=1e-12)

  def test_render_and_objective_share_one_offset_under_projection(self):
    # With projection on, undo it on the render, filter, and re-project: the
    # result can only match the objective if both solved the same offset.
    args = _default_structural_args()
    args['heavyside'] = True
    logits, _ = _parity_logits_and_no_vc_input(args)
    for beta in (1.0, 4.0, 16.0):
      with self.subTest(beta=beta):
        trial = dict(args, beta=beta)
        render = physics.physical_density(
            logits, trial, volume_constraint=True, cone_filter=False)
        objective = physics.physical_density(
            logits, trial, volume_constraint=True, cone_filter=True)
        unprojected = _inverse_heavyside_projection(render, beta, trial['eta'])
        rebuilt = physics.heavyside_projection(
            topo_autograd.cone_filter(
                unprojected, trial['filter_width'], trial['mask']),
            beta, trial['eta'])
        npo.testing.assert_allclose(rebuilt, objective, rtol=0, atol=1e-9)


class CanonicalRenderVolumeTest(absltest.TestCase):
  """Stage 4: the saved design is the physical density, so it holds volfrac.

  `Environment.render` used to ask for the pre-filter view. Under Heaviside
  that field's mean was 19.9% (beta=4) and 37.0% (beta=16) above volfrac on
  this split, while the objective held volfrac exactly. Render now uses
  `cone_filter=True`; the old means are the rejected near-miss so a silent
  revert is visible.

  Measured OFF the volfrac == eta diagonal deliberately. At volfrac == eta ==
  0.5 the projection is mean-neutral, which is precisely the operating point
  at which this gap was previously missed. Do not validate a projection
  change only on that diagonal.
  """

  _VOLFRAC = 0.3
  _ETA = 0.5
  # cantilever_beam_full 60x60, RandomState(0).randn(60, 60) * 2.0 logits.
  # Pre-Stage-4 Environment.render means (cone_filter=False).
  _OLD_UNFILTERED_MEAN_BY_BETA = {
      4.0: 0.35958723578062846,   # 19.9% over volfrac
      16.0: 0.4110767562774843,   # 37.0% over volfrac
  }

  def _projection_args(self, beta):
    params = StructuralParams(
        problem_name='cantilever_beam_full', width=60, height=60,
        density=self._VOLFRAC, heavyside=True, beta=beta, eta=self._ETA)
    return topo_api.specified_task(params.get_problem())

  def test_render_matches_the_objective_and_holds_volfrac(self):
    for beta, old_mean in self._OLD_UNFILTERED_MEAN_BY_BETA.items():
      with self.subTest(beta=beta):
        args = self._projection_args(beta)
        env = topo_api.Environment(args)
        logits = npo.random.RandomState(0).randn(
            args['nely'], args['nelx']) * 2.0
        objective = physics.physical_density(
            logits, args, volume_constraint=True, cone_filter=True)
        render = env.render(logits, volume_constraint=True)
        unfiltered = physics.physical_density(
            logits, args, volume_constraint=True, cone_filter=False)
        self.assertAlmostEqual(
            float(objective.mean()), self._VOLFRAC, delta=1e-9)
        self.assertAlmostEqual(
            float(render.mean()), self._VOLFRAC, delta=1e-9)
        npo.testing.assert_allclose(render, objective, rtol=0, atol=1e-12)
        npo.testing.assert_allclose(
            float(unfiltered.mean()), old_mean, rtol=1e-9, atol=0)
        self.assertGreater(
            abs(float(unfiltered.mean()) - self._VOLFRAC), 0.05,
            'the pre-filter view already holds volfrac, so the old gap '
            'is no longer a discriminating near-miss')


class FilteredProjectionGradientTest(absltest.TestCase):
  """Gate 7: gradients through filter + projection + volume enforcement.

  Verified with directional central differences rather than autograd's
  `check_grads`. `check_grads` differentiates the VJP a second time, and
  `grad_find_root` calls `autograd.grad` inside its own VJP, which leaks an
  ArrayBox; that failure is a pre-existing defect of `find_root` and reproduces
  identically on the untouched plain-sigmoid residual, so it says nothing about
  the filter/projection path. The gradients themselves are correct, as the
  finite differences below show.
  """

  # find_root resolves its offset to a bisection tolerance of 1e-12, so a step
  # much below 1e-4 amplifies that noise faster than it reduces truncation
  # error. At 1e-4 every case below agrees to better than 1e-7.
  def _assert_directional_grad(self, fn, x0, seed=0, eps=1e-4, tolerance=1e-6):
    direction = npo.random.RandomState(seed).randn(*x0.shape)
    direction /= npo.linalg.norm(direction)
    analytic = float(npo.sum(autograd.grad(fn)(x0) * direction))
    numeric = float(
        (fn(x0 + eps * direction) - fn(x0 - eps * direction)) / (2 * eps))
    scale = max(abs(analytic), abs(numeric), 1.0)
    self.assertLess(abs(analytic - numeric) / scale, tolerance,
                    msg=f'analytic={analytic!r} numeric={numeric!r}')

  def _weighted_density(self, args, cone_filter):
    weights = npo.random.RandomState(1).randn(args['nely'], args['nelx'])

    def fn(x):
      density = physics.physical_density(
          x, args, volume_constraint=True, cone_filter=cone_filter)
      return np.sum(density * weights)
    return fn

  def test_grad_through_filter_only(self):
    args = _small_structural_args(enforce_volume_last=True)
    x0 = npo.random.RandomState(0).randn(args['nely'], args['nelx']) * 0.5
    self._assert_directional_grad(self._weighted_density(args, True), x0)

  def test_grad_through_projection_only(self):
    args = _small_structural_args(heavyside=True, beta=4.0)
    x0 = npo.random.RandomState(0).randn(args['nely'], args['nelx']) * 0.5
    self._assert_directional_grad(self._weighted_density(args, False), x0)

  def test_grad_through_filter_and_projection(self):
    x0 = None
    for beta in (4.0, 16.0):
      with self.subTest(beta=beta):
        args = _small_structural_args(heavyside=True, beta=beta)
        if x0 is None:
          x0 = npo.random.RandomState(0).randn(args['nely'], args['nelx']) * 0.5
        self._assert_directional_grad(self._weighted_density(args, True), x0)

  def test_grad_through_full_objective(self):
    for beta in (4.0, 16.0):
      with self.subTest(beta=beta):
        args = _small_structural_args(heavyside=True, beta=beta)
        ke = physics.get_stiffness_matrix(args['young'], args['poisson'])
        x0 = npo.random.RandomState(0).randn(args['nely'], args['nelx']) * 0.5
        fn = lambda x: physics.objective(x, ke, args, volume_constraint=True)
        self._assert_directional_grad(fn, x0)


class CanonicalHandoffVolumeTest(absltest.TestCase):
  """KNOWN-GAP for Stage 8: the handoff field is already projected.

  `constrained_logits` still asks for `cone_filter=False`. Slice A made render
  the filtered field but left this view alone, so the pixel objective
  re-filters and re-projects a density that already went through Heaviside.
  The sign of the volume error still flips with beta.

  Crane 32x32 is the split the Stage-1 critic published. Do not "fix" these
  here.
  """

  _VOLFRAC = 0.3
  _ETA = 0.5
  # Design-region mean of the re-projected handoff, from note-7dabd6e1
  # (crane 32x32, volfrac=0.3, eta=0.5, RandomState(0)*2 logits).
  _CRITIC_AFTER_DMEAN = {
      4.0: 0.28082,    # -6.4%
      16.0: 0.33348,   # +11.2%
  }

  def test_reprojected_handoff_still_misses_volfrac_and_flips_sign(self):
    errors = {}
    for beta, published in self._CRITIC_AFTER_DMEAN.items():
      with self.subTest(beta=beta):
        params = StructuralParams(
            problem_name='crane', width=32, height=32,
            density=self._VOLFRAC, heavyside=True, beta=beta, eta=self._ETA)
        args = topo_api.specified_task(params.get_problem())
        logits = npo.random.RandomState(0).randn(
            args['nely'], args['nelx']) * 2.0
        handoff = physics.physical_density(
            logits, args, volume_constraint=True, cone_filter=False)
        after = physics.physical_density(
            handoff, args, volume_constraint=False, cone_filter=True)
        after_mean = _design_region_mean(after, args)
        err = (after_mean - self._VOLFRAC) / self._VOLFRAC
        errors[beta] = err
        self.assertGreater(
            abs(_design_region_mean(handoff, args) - self._VOLFRAC), 0.05)
        npo.testing.assert_allclose(after_mean, published, rtol=1e-4, atol=0)
        self.assertGreater(abs(err), 0.05)
    self.assertLess(errors[4.0], 0.0)
    self.assertGreater(errors[16.0], 0.0)


class CanonicalDensityBridgeTest(absltest.TestCase):
  """Default CLIP input is Environment.render, and the VJP matches HIPS."""

  def test_physical_density_bridge_matches_render(self):
    params = StructuralParams(
        problem_name='cantilever_beam_full', width=24, height=16, density=0.3,
        heavyside=True, beta=4.0, eta=0.5)
    model = PixelModel(structural_params=params, seed=0)
    logits = model()
    bridged = model.get_physical_density(logits)
    rendered = model.env.render(
        logits.detach().cpu().numpy(), volume_constraint=True)
    npo.testing.assert_allclose(
        bridged.detach().cpu().numpy().reshape(rendered.shape),
        rendered, rtol=0, atol=1e-5)

  def test_default_clip_sees_the_rendered_density(self):
    params = StructuralParams(width=24, height=16, density=0.4)
    captured = {}

    def capture(image):
      captured['image'] = image.detach().clone()
      return image.reshape(-1).sum() * 0.0

    model = PixelModel(structural_params=params, seed=0)
    model.clip_loss = capture
    logits = model()
    model.get_semantic_loss(logits)
    rendered = model.env.render(
        logits.detach().cpu().numpy(), volume_constraint=True)
    npo.testing.assert_allclose(
        captured['image'].cpu().numpy().reshape(rendered.shape),
        rendered, rtol=0, atol=1e-5)

  def test_venice_algebra_still_passes_raw_logits_to_clip(self):
    from model.model_base import VeniceLossAlgebra
    params = StructuralParams(width=24, height=16, density=0.4)
    captured = {}

    def capture(image):
      captured['image'] = image.detach().clone()
      return image.reshape(-1).sum() * 0.0

    model = PixelModel(structural_params=params, seed=0)
    model.clip_loss = capture
    model.enable_venice_compat_loss(VeniceLossAlgebra())
    logits = model()
    model.get_semantic_loss(logits)
    npo.testing.assert_allclose(
        captured['image'].cpu().numpy(),
        logits.detach().cpu().numpy(), rtol=0, atol=0)

  def test_bridge_vjp_matches_directional_finite_difference(self):
    args = _small_structural_args(heavyside=True, beta=4.0)
    env = topo_api.Environment(args)
    rng = npo.random.RandomState(0)
    x0 = rng.randn(args['nely'], args['nelx']).astype(npo.float64) * 0.5
    direction = rng.randn(*x0.shape)
    direction /= npo.linalg.norm(direction)
    eps = 1e-4

    def numpy_density(x):
      return physics.physical_density(
          x, args, volume_constraint=True, cone_filter=True)

    numeric = (
        (numpy_density(x0 + eps * direction) - numpy_density(x0 - eps * direction))
        / (2 * eps))
    # directional derivative of sum(density): sum(ddensity) along direction
    numeric_scalar = float(numeric.sum())

    x_t = torch.tensor(x0, dtype=torch.float64, requires_grad=True)
    dens = PhysicalDensity.apply(x_t, env)
    dens.sum().backward()
    analytic_scalar = float((x_t.grad.numpy() * direction).sum())
    scale = max(abs(analytic_scalar), abs(numeric_scalar), 1.0)
    self.assertLess(
        abs(analytic_scalar - numeric_scalar) / scale, 1e-5,
        msg=f'analytic={analytic_scalar!r} numeric={numeric_scalar!r}')


if __name__ == '__main__':
  absltest.main()
