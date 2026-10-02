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

"""Guards on the Venice compatibility seam: restart, refusals, and the BCs.

Three defects motivated this file, and each has the same shape -- a
configuration the seam accepted and then did not honour.

RESTARTING A RUN
----------------
`AdaptiveAdam_Optimizer.optimize` reset `resize_steps`, `converged` and its
per-call stage bookkeeping while the tracker and `loss_terms` kept accumulating.
A second call therefore zipped one call's stage labels against two calls'
frames, which raised `conflicting sizes for dimension 'step': length 3 on
'design' and length 6 on {'step': 'loss'}` -- and, when the two runs happened to
share a grid, would have silently rendered frames through the wrong stage's
environment instead. `optimize` now clears every per-run field together, and
`_create_dataset` refuses to build a dataset from series that disagree.

REFUSING WHAT CANNOT BE HONOURED
--------------------------------
`clip_alpha` was already refused where it contradicts the legacy algebra, but
`clip_weight` was silently swallowed: under the preset, `get_total_loss(z,
clip_weight=1.0)` and `get_total_loss(z, clip_weight=1000.0)` returned the same
number. `AdaptiveAdam_Optimizer(clip_weight=1000)` and
`Adam_Optimizer`/`LBFGS_Optimizer(clip_weight_max=1000)` did the same one
layer up -- they constructed, ran the algebra, and never read the static
weight. `MMA_Optimizer` and `OptimalityCriteria_Optimizer` were worse -- they
never consult the model's loss at all, so a preset-carrying model was optimized
for pure compliance with no complaint. All of those now raise. None (the
default) means unset; the default path then uses 1.0.

The asymmetry in `clip_alpha`'s handling is NOT a defect and is pinned
elsewhere (`test_training_loop_coverage.ClipAlphaUnderThePresetTest`): on
`Adam_Optimizer`/`LBFGS_Optimizer` it scales a capped INVERSE weight and
conflicts with the preset, while on `AdaptiveAdam_Optimizer` it is the same
proportional formula and is a legitimate override. Nothing here unifies them.

BOUNDARY CONDITIONS
-------------------
`fix_right_wall` existed only on `problems.multistory_building`, so no
`StructuralParams`-driven run could reach Venice's true (unconstrained) BCs.
It is exposed here, still defaulting to True, and asking for False is refused
while CHOLMOD is the active backend -- there the failure is a
`CholmodNotPositiveDefiniteError` that poisons the process so the NEXT solve
segfaults, which is not a thing to let a user discover empirically.

Everything below is dependency-free: no CLIP, no checkpoints, no network, and
no solve of the singular system. The grids are small enough that the whole file
runs in a few seconds.
"""

# pylint: disable=missing-docstring
# pylint: disable=invalid-name
# pylint: disable=protected-access

import warnings
from unittest import mock

import numpy as np
import torch
from absl.testing import absltest

from model.model_ada import (
    AdaptivePixelModel,
    INITIAL_PREV_LOSS,
)
from model.model_base import VeniceLossAlgebra
from model.model_pixel import PixelModel
from physics import autograd as topo_autograd
from problem import problems
from problem.problems import StructuralParams
from optimize.optimizers import (
    AdaptiveAdam_Optimizer,
    Adam_Optimizer,
    LBFGS_Optimizer,
    MMA_Optimizer,
    OptimalityCriteria_Optimizer,
)

# A grid small enough that a physics solve is free, and divisible by 2 so the
# resolution schedule can start one stage below it.
SMALL_WIDTH = 16
SMALL_HEIGHT = 32
SMALL_INTERVAL = 8

# A convergence threshold of zero is never crossed -- `abs(delta) < 0` is false
# for every delta -- so a run configured with it stops only at max_iterations
# and its step count is exact rather than trajectory-dependent.
NEVER_CONVERGE = 0.0
RUN_STEPS = 3

# Venice's learning rate rather than the repository default of 1e-2, so three
# steps move compliance far enough for a restart to be told from a repeat.
VENICE_LR = 0.2

# `max_resize_iteration=2` with Venice's counter offset fires an upsample at
# zero-based step 0, so the first run of the restart test changes resolution.
UPSAMPLE_AT_FIRST_STEP = 2

STUB_CLIP_VALUE = 5.0

# Two static weights three orders of magnitude apart. Under the default algebra
# they must give different totals; under the preset both must be refused, and
# the bug was that both produced the same number.
SMALL_CLIP_WEIGHT = 1.0
LARGE_CLIP_WEIGHT = 1000.0


def _small_params(**overrides) -> StructuralParams:
  """The Venice problem at a size a unit test can afford."""
  kwargs = dict(
      problem_name='multistory_building',
      width=SMALL_WIDTH,
      height=SMALL_HEIGHT,
      density=0.3,
      interval=SMALL_INTERVAL,
  )
  kwargs.update(overrides)
  return StructuralParams(**kwargs)


def _stub_clip_loss(logits):
  """Stand in for a CLIP loss with a constant, so no checkpoint is needed."""
  return logits.new_tensor(STUB_CLIP_VALUE)


def _adaptive_model(resize_num: int = 0) -> AdaptivePixelModel:
  """An adaptive model on the preset, without CLIP."""
  model = AdaptivePixelModel(
      structural_params=_small_params(), seed=0, resize_num=resize_num)
  return model.enable_venice_compat_loss(VeniceLossAlgebra())


def _adaptive_optimizer(model, **overrides) -> AdaptiveAdam_Optimizer:
  kwargs = dict(
      max_iterations=RUN_STEPS,
      lr=VENICE_LR,
      convergence_threshold=NEVER_CONVERGE,
  )
  kwargs.update(overrides)
  return AdaptiveAdam_Optimizer(model, **kwargs)


def _pixel_model(venice: bool = True) -> PixelModel:
  """A plain pixel model, on the preset or on the default algebra."""
  model = PixelModel(structural_params=_small_params(), seed=0)
  return model.enable_venice_compat_loss(VeniceLossAlgebra() if venice else False)


class AdaptiveAdamRestartTest(absltest.TestCase):
  """A second `optimize` call is a fresh run, not an append onto the first."""

  def test_restart_returns_only_its_own_trajectory(self):
    """The exact reported crash: series of different lengths reach xarray.

    With no upsamples every frame shares a grid, which is what made the old
    failure a coordinate conflict rather than a reshape error -- three stage
    labels against six recorded losses.
    """
    model = _adaptive_model()
    optimizer = _adaptive_optimizer(model)

    first = optimizer.optimize()
    self.assertEqual(int(first.sizes['step']), RUN_STEPS)

    second = optimizer.optimize()
    self.assertEqual(int(second.sizes['step']), RUN_STEPS)
    for name in ('loss', 'compliance', 'clip_loss', 'clip_loss_raw',
                 'clip_weight'):
      with self.subTest(field=name):
        self.assertEqual(second[name].shape, (RUN_STEPS,))
    self.assertEqual(second['design'].shape,
                     (RUN_STEPS, SMALL_HEIGHT, SMALL_WIDTH))

    # A restart continues from the design the first run reached, so the second
    # run's trajectory is genuinely its own rather than a copy.
    self.assertLess(float(second['compliance'].values[0]),
                    float(first['compliance'].values[0]))

  def test_raw_trajectory_keeps_each_stage_native_grid(self):
    """``design_raw`` strides back to the grid each step was optimized on."""
    from figures import native_raw_frames

    model = _adaptive_model(resize_num=1)
    optimizer = _adaptive_optimizer(
        model, max_resize_iteration=UPSAMPLE_AT_FIRST_STEP)
    ds = optimizer.optimize()
    self.assertEqual(ds['design_raw'].shape, ds['design'].shape)
    frames = native_raw_frames(ds)
    self.assertLen(frames, RUN_STEPS)
    self.assertEqual(frames[0].shape, (SMALL_HEIGHT // 2, SMALL_WIDTH // 2))
    self.assertEqual(frames[-1].shape, (SMALL_HEIGHT, SMALL_WIDTH))
    self.assertTrue(all(np.all(np.isfinite(frame)) for frame in frames))

  def test_restart_after_a_resolution_change_renders_on_one_grid(self):
    """The other half of the desync: frames drawn under a stale environment.

    The first run upsamples, so its frames are 8x16 and 16x32. Reusing them
    would pair a coarse frame with the fine environment the second run reports
    under; the second run must see only its own, full-resolution frames.
    """
    model = _adaptive_model(resize_num=1)
    optimizer = _adaptive_optimizer(
        model, max_resize_iteration=UPSAMPLE_AT_FIRST_STEP)

    first = optimizer.optimize()
    self.assertEqual(list(first.attrs['resize_steps']), [0])
    self.assertFalse(model.can_upsample)

    second = optimizer.optimize()
    self.assertEqual(int(second.sizes['step']), RUN_STEPS)
    self.assertEqual(second['design'].shape,
                     (RUN_STEPS, SMALL_HEIGHT, SMALL_WIDTH))
    self.assertTrue(np.all(np.isfinite(second['design'].values)))
    # The schedule is exhausted, so this run fires no upsample of its own and
    # must not inherit the first run's.
    self.assertEqual(list(second.attrs['resize_steps']), [])

  def test_restart_rebaselines_the_schedule_against_the_sentinel(self):
    """`prev_loss` is the baseline both schedule tests read.

    Left at the first run's final compliance, the second run's opening step
    would threshold against a design it never visited -- and, since a restart
    resumes from that very design, against a delta near zero, which is exactly
    the state that fires a spurious convergence.
    """
    model = _adaptive_model()
    optimizer = _adaptive_optimizer(model)

    first = optimizer.optimize()
    carried_over = model.prev_loss
    self.assertAlmostEqual(
        carried_over, float(first['compliance'].values[-1]), places=4)

    observed = []
    real_threshold_crossed = model.threshold_crossed

    def recording_threshold_crossed(compliance, threshold):
      observed.append(model.prev_loss)
      return real_threshold_crossed(compliance, threshold)

    model.threshold_crossed = recording_threshold_crossed
    optimizer.optimize()

    self.assertEqual(observed[0], INITIAL_PREV_LOSS)
    self.assertNotAlmostEqual(observed[0], carried_over)
    # Only the baseline is reset; the run still tracks itself after step 0.
    self.assertNotEqual(observed[1], INITIAL_PREV_LOSS)

  def test_restart_clears_the_recorded_state_on_the_optimizer(self):
    model = _adaptive_model()
    optimizer = _adaptive_optimizer(model)
    optimizer.optimize()

    optimizer._reset_run_state()
    self.assertEmpty(optimizer.tracker.losses)
    self.assertEmpty(optimizer.tracker.frames)
    self.assertEmpty(optimizer.loss_terms)
    self.assertEmpty(optimizer.motif_scale_terms)
    self.assertEmpty(optimizer.physical_clip_terms)
    self.assertEmpty(optimizer.blend_terms)
    self.assertEmpty(optimizer.resize_steps)
    self.assertFalse(optimizer.converged)
    self.assertEqual(model.prev_loss, INITIAL_PREV_LOSS)

  def test_dataset_refuses_series_that_disagree(self):
    """The backstop, in case some future path desynchronizes them again.

    `zip` truncates silently, so without this the mismatch surfaces as an
    xarray coordinate conflict several frames away from its cause -- or not at
    all, when the shorter series happens to be the one nothing else counts.
    """
    model = _adaptive_model()
    optimizer = _adaptive_optimizer(model)
    optimizer.optimize()

    with self.assertRaisesRegex(RuntimeError, 'per-step series disagree'):
      optimizer._create_dataset([model.env], [])

  def test_converged_flag_does_not_survive_a_restart(self):
    """A run that stopped early must not mark a later run as converged."""
    model = _adaptive_model()
    converging = _adaptive_optimizer(
        model, convergence_threshold=INITIAL_PREV_LOSS * 2)
    first = converging.optimize()
    self.assertTrue(bool(first.attrs['converged']))
    self.assertEqual(int(first.sizes['step']), 1)

    second = _adaptive_optimizer(model).optimize()
    self.assertFalse(bool(second.attrs['converged']))
    self.assertEqual(int(second.sizes['step']), RUN_STEPS)


class ClipWeightUnderThePresetTest(absltest.TestCase):
  """`clip_weight` is refused under the legacy algebra, not ignored.

  The preset's weight is `compliance * clip_alpha`, recomputed and undetached
  every step. A static `clip_weight` is not a setting of that formula, so a
  caller passing one has described a different run.
  """

  def test_preset_refuses_every_clip_weight(self):
    model = _pixel_model(venice=True)
    logits = model()
    for weight in (0.0, SMALL_CLIP_WEIGHT, LARGE_CLIP_WEIGHT):
      with self.subTest(clip_weight=weight):
        with self.assertRaisesRegex(ValueError, 'contradictory couplings'):
          model.get_total_loss(logits, clip_weight=weight)

  def test_the_two_weights_used_to_agree_and_now_cannot_be_compared(self):
    """The measured symptom: 1.0 and 1000.0 produced the same number.

    Both calls are refused now, so the equality is unreachable rather than
    merely unlikely; that is the point of the refusal.
    """
    model = _pixel_model(venice=True)
    model.clip_loss = _stub_clip_loss
    logits = model()
    for weight in (SMALL_CLIP_WEIGHT, LARGE_CLIP_WEIGHT):
      with self.subTest(clip_weight=weight):
        with self.assertRaises(ValueError):
          model.get_total_loss(logits, clip_weight=weight)

    # Omitting it is the supported call, and it still runs the algebra.
    unweighted = model.get_total_loss(logits)
    terms = model.get_venice_compat_losses(logits)
    self.assertAlmostEqual(
        float(unweighted) / float(terms.total_loss), 1.0, places=6)

  def test_the_preset_alone_is_accepted(self):
    model = _pixel_model(venice=True)
    logits = model()
    # None is the default and means "unset", not "zero".
    self.assertTrue(torch.isfinite(model.get_total_loss(logits)))
    self.assertTrue(
        torch.isfinite(model.get_total_loss(logits, clip_weight=None)))

  def test_the_default_algebra_still_honours_clip_weight(self):
    """The non-preset path is untouched: the weight scales the semantic term."""
    model = _pixel_model(venice=False)
    model.clip_loss = _stub_clip_loss
    logits = model()

    small = float(model.get_total_loss(logits, clip_weight=SMALL_CLIP_WEIGHT))
    large = float(model.get_total_loss(logits, clip_weight=LARGE_CLIP_WEIGHT))
    unweighted = float(model.get_total_loss(logits, clip_weight=0.0))

    self.assertNotAlmostEqual(small, large)
    # Compliance is reported to float32 resolution, so the difference of two
    # totals carries a rounding error of order 1e-4 rather than 1e-7.
    self.assertAlmostEqual(
        small - unweighted, STUB_CLIP_VALUE * SMALL_CLIP_WEIGHT, delta=1e-2)
    self.assertAlmostEqual(
        (large - unweighted) / (STUB_CLIP_VALUE * LARGE_CLIP_WEIGHT), 1.0,
        places=5)


class OptimizerStaticWeightUnderThePresetTest(absltest.TestCase):
  """The same swallow, one layer up: a static weight on the optimizer.

  `get_total_loss` already refuses `clip_weight` under the preset. The
  optimizers were still accepting it (AdaptiveAdam's `clip_weight`,
  Adam/LBFGS's `clip_weight_max`) and then never reading it, because the
  Venice algebra owns the coupling. Default None means unset; constructing
  without the argument still runs.
  """

  def test_adaptive_adam_refuses_every_static_clip_weight(self):
    model = _adaptive_model()
    for weight in (0.0, 1.0, SMALL_CLIP_WEIGHT, LARGE_CLIP_WEIGHT):
      with self.subTest(clip_weight=weight):
        with self.assertRaisesRegex(ValueError, 'contradictory couplings'):
          _adaptive_optimizer(model, clip_weight=weight)

  def test_adaptive_adam_accepts_the_preset_when_clip_weight_is_unset(self):
    model = _adaptive_model()
    optimizer = _adaptive_optimizer(model)
    self.assertIsNone(optimizer.clip_weight)

  def test_adaptive_adam_refuses_a_late_flip_onto_a_weighted_optimizer(self):
    """Constructed off the preset, then the seam is enabled: optimize refuses.

    The constructor cannot see a seam that is not on yet; the re-check at
    the start of `optimize` is what closes that hole.
    """
    model = AdaptivePixelModel(structural_params=_small_params(), seed=0)
    optimizer = AdaptiveAdam_Optimizer(
        model, max_iterations=1, lr=VENICE_LR,
        clip_weight=LARGE_CLIP_WEIGHT,
        convergence_threshold=NEVER_CONVERGE)
    model.enable_venice_compat_loss(VeniceLossAlgebra())
    with self.assertRaisesRegex(ValueError, 'contradictory couplings'):
      optimizer.optimize()

  def test_adaptive_adam_honours_clip_weight_off_the_preset(self):
    model = AdaptivePixelModel(structural_params=_small_params(), seed=0)
    optimizer = AdaptiveAdam_Optimizer(
        model, max_iterations=1, clip_weight=LARGE_CLIP_WEIGHT)
    self.assertEqual(optimizer.clip_weight, LARGE_CLIP_WEIGHT)

  def test_adam_and_lbfgs_refuse_clip_weight_max_under_the_preset(self):
    model = _pixel_model(venice=True)
    for factory, name in (
        (Adam_Optimizer, 'Adam_Optimizer'),
        (LBFGS_Optimizer, 'LBFGS_Optimizer'),
    ):
      with self.subTest(optimizer=name):
        with self.assertRaisesRegex(ValueError, 'clip_weight_max'):
          factory(model, max_iterations=1, clip_weight_max=LARGE_CLIP_WEIGHT)
        unset = factory(model, max_iterations=1)
        self.assertIsNone(unset.clip_weight_max)


class PhysicsOnlyOptimizersRejectThePresetTest(absltest.TestCase):
  """MMA and Optimality Criteria cannot run the preset, so they refuse it.

  Neither evaluates `Model.get_total_loss`: they drive `env.objective` and the
  physics optimality step directly. A preset-carrying model handed to either
  was optimized for compliance alone with the whole semantic coupling dropped.
  """

  FACTORIES = (
      (MMA_Optimizer, 'MMA_Optimizer'),
      (OptimalityCriteria_Optimizer, 'OptimalityCriteria_Optimizer'),
  )

  def test_construction_refuses_a_preset_carrying_model(self):
    model = _pixel_model(venice=True)
    for factory, name in self.FACTORIES:
      with self.subTest(optimizer=name):
        with self.assertRaisesRegex(ValueError, 'cannot honour the Venice'):
          factory(model, max_iterations=1)

  def test_optimize_refuses_a_preset_enabled_after_construction(self):
    """The seam can be flipped on a built model, so the check runs twice.

    The refusal also has to precede `optimize`'s own imports, or a machine
    without nlopt reports a missing dependency instead of the real problem.
    """
    for factory, name in self.FACTORIES:
      with self.subTest(optimizer=name):
        model = _pixel_model(venice=False)
        optimizer = factory(model, max_iterations=1)
        model.enable_venice_compat_loss(VeniceLossAlgebra())
        with self.assertRaisesRegex(ValueError, 'cannot honour the Venice'):
          optimizer.optimize()

  def test_the_default_algebra_is_accepted(self):
    model = _pixel_model(venice=False)
    for factory, name in self.FACTORIES:
      with self.subTest(optimizer=name):
        self.assertIsNotNone(factory(model, max_iterations=1))


class FixRightWallParamTest(absltest.TestCase):
  """`fix_right_wall` reaches `multistory_building` from `StructuralParams`.

  Nothing here solves the unconstrained system. Building the `Problem` is pure
  array construction, and provoking CHOLMOD's rejection would corrupt its state
  so the next solve anywhere in the process segfaults.
  """

  def test_the_default_constrains_x_on_the_right_edge(self):
    problem = _small_params().get_problem()
    self.assertTrue(problem.normals[-1, :, 0].all())
    self.assertGreater(int(problem.normals[:, :, 0].sum()), 0)

  def test_false_reaches_the_problem_function_under_superlu(self):
    """The whole point of the field: Venice's BCs are now configurable.

    CHOLMOD is patched out rather than uninstalled, which is enough because the
    refusal reads the same flag the solver selection does and no solve runs.
    """
    with mock.patch.object(topo_autograd, 'HAS_CHOLMOD', False):
      problem = _small_params(fix_right_wall=False).get_problem()
    self.assertEqual(int(problem.normals[:, :, 0].sum()), 0)
    # Only the X constraint goes; the ground support is untouched.
    self.assertTrue(problem.normals[:, -1, 1].all())

  def test_false_is_refused_while_cholmod_is_the_active_backend(self):
    """A segfault with no traceback is not an acceptable way to learn this."""
    with mock.patch.object(topo_autograd, 'HAS_CHOLMOD', True):
      with self.assertRaisesRegex(ValueError, 'SuperLU'):
        _small_params(fix_right_wall=False).get_problem()

  def test_the_refusal_names_the_bias_the_constraint_costs(self):
    """The number a user needs to decide whether they want the legacy BCs."""
    with mock.patch.object(topo_autograd, 'HAS_CHOLMOD', True):
      with self.assertRaises(ValueError) as caught:
        _small_params(fix_right_wall=False).get_problem()
    message = str(caught.exception)
    for figure in ('0.335%', '0.365%', '0.380%'):
      with self.subTest(figure=figure):
        self.assertIn(figure, message)

  def test_a_problem_without_a_right_wall_warns_instead_of_dropping_it(self):
    """Silently discarding a field the caller clearly meant is how rmin died."""
    params = StructuralParams(
        problem_name='cantilever_beam_full', width=16, height=16,
        density=0.3, fix_right_wall=False)
    with self.assertWarnsRegex(UserWarning, 'fix_right_wall=False is ignored'):
      params.get_problem()

  def test_the_default_never_warns_and_never_raises(self):
    for name in ('multistory_building', 'cantilever_beam_full'):
      with self.subTest(problem_name=name):
        params = StructuralParams(
            problem_name=name, width=16, height=16, density=0.3)
        self.assertTrue(params.fix_right_wall)
        with warnings.catch_warnings():
          warnings.simplefilter('error', UserWarning)
          params.get_problem()

  def test_the_field_survives_a_copy(self):
    """The resolution schedule derives every stage through `copy`."""
    params = _small_params(fix_right_wall=False)
    self.assertFalse(params.copy(width=8, height=16).fix_right_wall)
    self.assertTrue(params.copy(fix_right_wall=True).fix_right_wall)

  def test_the_problem_function_default_matches_the_params_default(self):
    """Two defaults for one boundary condition; they must not drift apart."""
    walled = problems.multistory_building(
        SMALL_WIDTH, SMALL_HEIGHT, density=0.3, interval=SMALL_INTERVAL)
    from_params = _small_params().get_problem()
    np.testing.assert_array_equal(walled.normals, from_params.normals)


if __name__ == '__main__':
  absltest.main()
