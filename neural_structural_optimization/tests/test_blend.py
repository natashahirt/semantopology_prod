# lint as python3
"""Stage 7 blending: opt-in log, frozen Venice seam, grad-match mixer.

No CLIP checkpoint. Grids are small enough that a physics solve is cheap.
"""

# pylint: disable=missing-docstring
# pylint: disable=protected-access

import numpy as np
import torch
from absl.testing import absltest

from neural_structural_optimization.model.model_ada import AdaptivePixelModel
from neural_structural_optimization.model.model_base import (
    VeniceLossAlgebra,
    VeniceLossTerms,
)
from neural_structural_optimization.problem.problems import StructuralParams
from neural_structural_optimization.guidance.blend import (
    BlendMode,
    GRAD_MATCH_EMA_DECAY,
    GRAD_MATCH_WEIGHT_MAX,
    GradNormEma,
    freeze_blend_mode,
    resolve_blend_mode,
    snapshot_blend,
    unweighted_grad_norms,
)
from neural_structural_optimization.optimize.optimizers import (
    AdaptiveAdam_Optimizer,
    CLIP_DYNAMIC_WEIGHT_MAX,
)

SMALL_WIDTH = 16
SMALL_HEIGHT = 32
SMALL_INTERVAL = 8
NEVER_CONVERGE = 0.0


def _params():
    return StructuralParams(
        problem_name='multistory_building',
        width=SMALL_WIDTH,
        height=SMALL_HEIGHT,
        density=0.3,
        interval=SMALL_INTERVAL,
        filter_width=1.5,
    )


def _venice_model(clip_loss=None):
    model = AdaptivePixelModel(
        structural_params=_params(),
        clip_loss=None,
        seed=0,
        resize_num=0,
        resize_scale=2,
    )
    model.enable_venice_compat_loss(VeniceLossAlgebra(clip_alpha=10.0))
    if clip_loss is not None:
        object.__setattr__(model, 'clip_loss', clip_loss)
    return model


def _default_model(clip_loss=None, resize_num=0):
    model = AdaptivePixelModel(
        structural_params=_params(),
        clip_loss=None,
        seed=0,
        resize_num=resize_num,
        resize_scale=2,
    )
    if clip_loss is not None:
        object.__setattr__(model, 'clip_loss', clip_loss)
    return model


def _mean_clip(field):
    return field.mean()


def _scaled_clip(field):
    """Mean CLIP with enough scale that ||g_C|| / ||g_d|| stays under the cap."""
    return field.mean() * 1.0e5


def _run(model, **kwargs):
    return AdaptiveAdam_Optimizer(
        model,
        max_iterations=2,
        lr=0.2,
        convergence_threshold=NEVER_CONVERGE,
        max_resize_iteration=50,
        **kwargs,
    ).optimize()


class BlendModeFreezeTest(absltest.TestCase):
    """A requested mixer cannot retune the Venice seam."""

    def test_venice_model_resolves_to_venice(self):
        mode = resolve_blend_mode(
            _venice_model(),
            clip_weight=None,
            clip_alpha=None,
            optimizer='AdaptiveAdam_Optimizer',
        )
        self.assertEqual(mode, BlendMode.VENICE)

    def test_default_path_without_alpha_is_static(self):
        mode = resolve_blend_mode(
            _default_model(),
            clip_weight=None,
            clip_alpha=None,
            optimizer='AdaptiveAdam_Optimizer',
        )
        self.assertEqual(mode, BlendMode.STATIC)

    def test_default_path_with_alpha_is_detached_scale(self):
        mode = resolve_blend_mode(
            _default_model(),
            clip_weight=None,
            clip_alpha=10.0,
            optimizer='AdaptiveAdam_Optimizer',
        )
        self.assertEqual(mode, BlendMode.DETACHED_SCALE)

    def test_adam_clip_alpha_is_inverse(self):
        mode = resolve_blend_mode(
            _default_model(),
            clip_weight=None,
            clip_alpha=10.0,
            optimizer='Adam_Optimizer',
        )
        self.assertEqual(mode, BlendMode.INVERSE)

    def test_requesting_static_on_venice_is_refused(self):
        with self.assertRaisesRegex(ValueError, 'live algebra is .venice'):
            AdaptiveAdam_Optimizer(
                _venice_model(),
                max_iterations=1,
                blend_mode='static',
            )

    def test_requesting_inverse_on_venice_is_refused(self):
        with self.assertRaisesRegex(ValueError, 'live algebra is .venice'):
            freeze_blend_mode('inverse', BlendMode.VENICE)

    def test_requesting_venice_on_venice_is_allowed(self):
        opt = AdaptiveAdam_Optimizer(
            _venice_model(),
            max_iterations=1,
            blend_mode='venice',
        )
        self.assertEqual(opt.blend_mode, BlendMode.VENICE)

    def test_unknown_mode_is_refused(self):
        with self.assertRaisesRegex(ValueError, 'unknown blend_mode'):
            freeze_blend_mode('fifty-fifty', BlendMode.VENICE)

    def test_blend_rho_on_default_path_resolves_to_grad_match(self):
        mode = resolve_blend_mode(
            _default_model(),
            clip_weight=None,
            clip_alpha=None,
            optimizer='AdaptiveAdam_Optimizer',
            blend_rho=0.5,
        )
        self.assertEqual(mode, BlendMode.GRAD_MATCH)

    def test_blend_rho_on_venice_is_refused(self):
        with self.assertRaisesRegex(ValueError, 'gradient-norm matching'):
            AdaptiveAdam_Optimizer(
                _venice_model(),
                max_iterations=1,
                blend_rho=1.0,
            )

    def test_requesting_grad_match_on_venice_is_refused(self):
        with self.assertRaisesRegex(ValueError, 'gradient-norm matching'):
            AdaptiveAdam_Optimizer(
                _venice_model(),
                max_iterations=1,
                blend_mode='grad_match',
            )

    def test_grad_match_refuses_clip_alpha(self):
        with self.assertRaisesRegex(ValueError, 'grad-match together'):
            AdaptiveAdam_Optimizer(
                _default_model(),
                max_iterations=1,
                blend_rho=1.0,
                clip_alpha=10.0,
            )

    def test_grad_match_refuses_static_clip_weight(self):
        with self.assertRaisesRegex(ValueError, 'grad-match together'):
            AdaptiveAdam_Optimizer(
                _default_model(),
                max_iterations=1,
                blend_rho=1.0,
                clip_weight=100.0,
            )

    def test_grad_match_without_rho_is_refused(self):
        with self.assertRaisesRegex(ValueError, 'requires blend_rho'):
            AdaptiveAdam_Optimizer(
                _default_model(),
                max_iterations=1,
                blend_mode='grad_match',
            )

    def test_blend_rho_outside_unit_interval_is_refused(self):
        with self.assertRaisesRegex(ValueError, 'blend_rho must be in'):
            AdaptiveAdam_Optimizer(
                _default_model(),
                max_iterations=1,
                blend_rho=1.5,
            )

    def test_static_clip_weight_still_refused_on_venice(self):
        with self.assertRaisesRegex(ValueError, 'contradictory couplings'):
            AdaptiveAdam_Optimizer(
                _venice_model(),
                max_iterations=1,
                clip_weight=1000.0,
                blend_log=True,
            )


class BlendLogTest(absltest.TestCase):
    """Logging is additive: off keeps the schema, on does not change the loss."""

    def test_default_run_has_no_blend_columns(self):
        ds = _run(_venice_model())
        self.assertFalse(any(name.startswith('blend_') for name in ds.data_vars))
        self.assertNotIn('blend_mode', ds.attrs)

    def test_logged_run_records_raw_terms_and_the_live_mode(self):
        model = _venice_model(clip_loss=lambda field: field.new_tensor(0.4))
        ds = _run(model, blend_log=True)
        self.assertEqual(ds.attrs['blend_mode'], 'venice')
        self.assertIn('blend_compliance', ds)
        self.assertIn('blend_clip_z', ds)
        self.assertIn('blend_clip_z_weight', ds)
        np.testing.assert_allclose(
            ds['blend_clip_z'].values, np.full(2, 0.4), rtol=1e-5)
        np.testing.assert_allclose(
            ds['blend_compliance'].values, ds['compliance'].values, rtol=1e-6)
        expected_weight = 10.0 * ds['compliance'].values
        np.testing.assert_allclose(
            ds['blend_clip_z_weight'].values, expected_weight, rtol=1e-5)

    def test_logging_does_not_change_the_loss(self):
        def clip_fn(field):
            return field.mean() * 0.0 + 0.4

        off = _run(_venice_model(clip_loss=clip_fn))
        on = _run(_venice_model(clip_loss=clip_fn), blend_log=True)
        np.testing.assert_allclose(off['loss'].values, on['loss'].values)
        np.testing.assert_allclose(
            off['compliance'].values, on['compliance'].values)
        np.testing.assert_allclose(
            off['clip_loss_raw'].values, on['clip_loss_raw'].values)

    def test_occupancy_and_density_clip_appear_when_those_terms_are_on(self):
        model = _venice_model(clip_loss=lambda field: field.new_tensor(0.5))
        model.enable_sketch_prior(
            torch.full((SMALL_HEIGHT, SMALL_WIDTH), 0.5), weight=12.0)
        model.enable_physical_clip(weight=3.0)
        ds = _run(model, blend_log=True)
        self.assertEqual(ds['blend_occupancy'].sizes['step'], 2)
        self.assertTrue(np.all(np.isfinite(ds['blend_occupancy'].values)))
        np.testing.assert_allclose(
            ds['blend_occupancy_weight'].values, np.full(2, 12.0), rtol=1e-5)
        np.testing.assert_allclose(
            ds['blend_clip_rho'].values, np.full(2, 0.5), rtol=1e-5)
        np.testing.assert_allclose(
            ds['blend_clip_rho_weight'].values, np.full(2, 3.0), rtol=1e-5)

    def test_gradient_conflict_of_orthogonal_coordinates_is_zero(self):
        logits = torch.zeros(1, 2, 2, requires_grad=True)
        compliance = logits[0, 0, 0]
        clip_z = logits[0, 0, 1]
        terms = VeniceLossTerms(
            total_loss=compliance + clip_z,
            compliance_loss=compliance,
            clip_loss=clip_z,
            clip_loss_raw=clip_z,
            clip_weight=clip_z.new_tensor(1.0),
        )

        class _Empty:
            _last_physical_clip = None
            _last_physical_clip_coefficient = None
            _last_occupancy_loss = None
            _last_occupancy_weight = None

        snap = snapshot_blend(_Empty(), terms, logits, grads=True)
        self.assertAlmostEqual(snap['grad_cosine'], 0.0, places=5)
        self.assertAlmostEqual(snap['g_compliance'], 1.0, places=5)
        self.assertAlmostEqual(snap['g_clip_z'], 1.0, places=5)

    def test_blend_grads_on_a_run_are_finite(self):
        model = _venice_model(clip_loss=lambda field: field.mean())
        ds = _run(model, blend_log=True, blend_grads=True)
        self.assertIn('blend_g_compliance', ds)
        self.assertIn('blend_g_clip_z', ds)
        self.assertIn('blend_grad_cosine', ds)
        self.assertTrue(np.all(np.isfinite(ds['blend_g_compliance'].values)))
        self.assertTrue(np.all(np.isfinite(ds['blend_g_clip_z'].values)))
        cosine = ds['blend_grad_cosine'].values
        self.assertTrue(np.all(np.isfinite(cosine)))
        self.assertTrue(np.all(cosine >= -1.0))
        self.assertTrue(np.all(cosine <= 1.0))


class GradNormEmaTest(absltest.TestCase):
    """The mixer is a ratio of EMAs, capped, with a hard reset."""

    def test_cap_matches_the_inverse_normalized_ceiling(self):
        self.assertEqual(GRAD_MATCH_WEIGHT_MAX, CLIP_DYNAMIC_WEIGHT_MAX)
        self.assertEqual(GRAD_MATCH_EMA_DECAY, 0.9)

    def test_first_observation_seeds_the_ratio(self):
        ema = GradNormEma(decay=0.9)
        ema.update(10.0, 2.0)
        self.assertAlmostEqual(ema.weight(1.0), 5.0)
        self.assertAlmostEqual(ema.weight(0.5), 2.5)

    def test_rho_zero_is_zero_weight(self):
        ema = GradNormEma()
        ema.update(10.0, 2.0)
        self.assertEqual(ema.weight(0.0), 0.0)

    def test_second_observation_smooths(self):
        ema = GradNormEma(decay=0.9)
        ema.update(10.0, 2.0)
        ema.update(20.0, 2.0)
        self.assertAlmostEqual(ema.g_c, 11.0)
        self.assertAlmostEqual(ema.weight(1.0), 5.5)

    def test_near_zero_clip_gradient_is_skipped(self):
        ema = GradNormEma()
        ema.update(10.0, 0.0)
        self.assertIsNone(ema.g_c)
        self.assertEqual(ema.weight(1.0), 0.0)

    def test_ratio_is_capped(self):
        ema = GradNormEma()
        ema.update(1.0e9, 1.0)
        self.assertEqual(ema.weight(1.0), GRAD_MATCH_WEIGHT_MAX)

    def test_reset_forgets_the_coarse_grid(self):
        ema = GradNormEma(decay=0.9)
        ema.update(10.0, 2.0)
        ema.reset()
        self.assertEqual(ema.weight(1.0), 0.0)
        ema.update(4.0, 2.0)
        self.assertAlmostEqual(ema.weight(1.0), 2.0)


class GradMatchMixerTest(absltest.TestCase):
    """AdaptiveAdam's default path can equalize gradient energy."""

    def test_grad_match_implies_logging(self):
        opt = AdaptiveAdam_Optimizer(
            _default_model(),
            max_iterations=1,
            blend_rho=1.0,
        )
        self.assertTrue(opt.blend_log)
        self.assertEqual(opt.blend_mode, BlendMode.GRAD_MATCH)

    def test_rho_zero_turns_clip_off(self):
        ds = _run(_default_model(clip_loss=_mean_clip), blend_rho=0.0)
        self.assertEqual(ds.attrs['blend_mode'], 'grad_match')
        self.assertEqual(ds.attrs['blend_rho'], 0.0)
        np.testing.assert_allclose(ds['clip_weight'].values, 0.0, atol=1e-12)
        np.testing.assert_allclose(ds['clip_loss'].values, 0.0, atol=1e-12)

    def test_first_step_weight_is_rho_times_norm_ratio(self):
        ds = _run(_default_model(clip_loss=_scaled_clip), blend_rho=0.5)
        fresh = _default_model(clip_loss=_scaled_clip)
        logits = fresh()
        compliance = fresh.get_structural_loss(logits)
        semantic = fresh.get_semantic_loss(logits)
        n_c, n_d = unweighted_grad_norms(compliance, semantic, logits)
        expected = 0.5 * n_c / n_d
        self.assertLess(expected, GRAD_MATCH_WEIGHT_MAX)
        np.testing.assert_allclose(
            ds['clip_weight'].values[0], expected, rtol=1e-4)
        np.testing.assert_allclose(
            ds['blend_clip_z_weight'].values[0], expected, rtol=1e-4)

    def test_tiny_clip_gradient_hits_the_cap(self):
        ds = _run(_default_model(clip_loss=_mean_clip), blend_rho=1.0)
        np.testing.assert_allclose(
            ds['clip_weight'].values,
            np.full(ds['clip_weight'].sizes['step'], GRAD_MATCH_WEIGHT_MAX))

    def test_ema_smooths_after_the_first_step(self):
        ds = _run(
            _default_model(clip_loss=_scaled_clip),
            blend_rho=1.0,
            blend_grads=True,
        )
        inst_c = ds['blend_g_compliance'].values
        ema_c = ds['blend_ema_g_compliance'].values
        np.testing.assert_allclose(ema_c[0], inst_c[0], rtol=1e-5)
        expected = (
            GRAD_MATCH_EMA_DECAY * inst_c[0]
            + (1.0 - GRAD_MATCH_EMA_DECAY) * inst_c[1])
        np.testing.assert_allclose(ema_c[1], expected, rtol=1e-5)
        inst_d = ds['blend_g_clip_z'].values
        ema_d = ds['blend_ema_g_clip_z'].values
        expected_d = (
            GRAD_MATCH_EMA_DECAY * inst_d[0]
            + (1.0 - GRAD_MATCH_EMA_DECAY) * inst_d[1])
        np.testing.assert_allclose(ema_d[1], expected_d, rtol=1e-5)

    def test_upsample_resets_the_ema(self):
        model = _default_model(clip_loss=_scaled_clip, resize_num=1)
        ds = AdaptiveAdam_Optimizer(
            model,
            max_iterations=2,
            lr=0.2,
            convergence_threshold=NEVER_CONVERGE,
            max_resize_iteration=1,
            blend_rho=1.0,
            blend_grads=True,
        ).optimize()
        self.assertEqual(list(ds.attrs['resize_steps']), [0])
        inst = ds['blend_g_compliance'].values
        ema = ds['blend_ema_g_compliance'].values
        np.testing.assert_allclose(ema[0], inst[0], rtol=1e-5)
        np.testing.assert_allclose(ema[1], inst[1], rtol=1e-5)
        mixed = (
            GRAD_MATCH_EMA_DECAY * inst[0]
            + (1.0 - GRAD_MATCH_EMA_DECAY) * inst[1])
        self.assertGreater(abs(mixed - inst[1]), 1e-6)

    def test_occupancy_does_not_enter_the_clip_weight(self):
        off = _run(_default_model(clip_loss=_scaled_clip), blend_rho=1.0)
        on_model = _default_model(clip_loss=_scaled_clip)
        on_model.enable_sketch_prior(
            torch.full((SMALL_HEIGHT, SMALL_WIDTH), 0.5), weight=4000.0)
        on = _run(on_model, blend_rho=1.0)
        np.testing.assert_allclose(
            off['clip_weight'].values[0],
            on['clip_weight'].values[0],
            rtol=1e-4)
        self.assertTrue(np.isfinite(on['blend_occupancy'].values[0]))
        self.assertGreater(float(on['blend_occupancy_weight'].values[0]), 0.0)

    def test_as_semantic_is_the_clip_term_not_an_extra(self):
        model = _default_model(clip_loss=_mean_clip)
        model.enable_venice_compat_loss(False)
        model.enable_physical_clip(weight=0.0, as_semantic=True)
        logits = model()
        semantic = model.get_semantic_loss(logits)
        scored = model.score_physical_clip(model.get_physical_density(logits))
        self.assertAlmostEqual(float(semantic), float(scored), places=5)
        base = model.get_structural_loss(logits)
        total = model.add_sketch_term(base, logits)
        self.assertAlmostEqual(float(total), float(base), places=5)
        self.assertAlmostEqual(
            float(model._last_physical_clip), float(semantic), places=5)

    def test_as_semantic_refuses_a_second_weight(self):
        model = _default_model(clip_loss=_mean_clip)
        with self.assertRaisesRegex(ValueError, 'as_semantic'):
            model.enable_physical_clip(weight=1.0, as_semantic=True)
        with self.assertRaisesRegex(ValueError, 'as_semantic'):
            model.enable_physical_clip(
                weight=0.0, match_venice=True, as_semantic=True)

    def test_raw_z_clip_stays_on_logits_when_as_semantic(self):
        model = _default_model(clip_loss=_mean_clip)
        model.enable_physical_clip(weight=0.0, as_semantic=True)
        logits = model() + torch.linspace(
            -1.0, 1.0, SMALL_HEIGHT * SMALL_WIDTH).reshape(1, SMALL_HEIGHT, SMALL_WIDTH)
        logits = logits.detach().requires_grad_(True)
        semantic = model.get_semantic_loss(logits)
        scored = model.score_physical_clip(model.get_physical_density(logits))
        raw_z = model.get_raw_z_clip_loss(logits)
        self.assertAlmostEqual(float(semantic), float(scored), places=5)
        self.assertAlmostEqual(float(raw_z), float(_mean_clip(logits)), places=5)
        self.assertGreater(abs(float(semantic) - float(raw_z)), 1e-6)

    def test_blend_rho_z_requires_blend_rho(self):
        with self.assertRaisesRegex(ValueError, 'requires blend_rho'):
            AdaptiveAdam_Optimizer(
                _default_model(clip_loss=_mean_clip),
                max_iterations=1,
                blend_rho_z=0.25,
            )

    def test_blend_rho_z_requires_as_semantic(self):
        with self.assertRaisesRegex(ValueError, 'as_semantic'):
            AdaptiveAdam_Optimizer(
                _default_model(clip_loss=_mean_clip),
                max_iterations=1,
                blend_rho=1.0,
                blend_rho_z=0.25,
            )

    def test_blend_rho_z_outside_unit_interval_is_refused(self):
        with self.assertRaisesRegex(ValueError, 'blend_rho_z must be in'):
            AdaptiveAdam_Optimizer(
                _default_model(clip_loss=_mean_clip),
                max_iterations=1,
                blend_rho=1.0,
                blend_rho_z=1.5,
            )

    def test_dual_first_step_weights_are_independent_rho_times_norm_ratio(self):
        model = _default_model(clip_loss=_scaled_clip)
        model.enable_physical_clip(weight=0.0, as_semantic=True)
        ds = _run(model, blend_rho=1.0, blend_rho_z=0.25)
        fresh = _default_model(clip_loss=_scaled_clip)
        fresh.enable_physical_clip(weight=0.0, as_semantic=True)
        logits = fresh()
        compliance = fresh.get_structural_loss(logits)
        density = fresh.get_semantic_loss(logits)
        raw_z = fresh.get_raw_z_clip_loss(logits)
        n_c, n_d = unweighted_grad_norms(compliance, density, logits)
        n_c_z, n_z = unweighted_grad_norms(compliance, raw_z, logits)
        expected_d = n_c / n_d
        expected_z = 0.25 * n_c_z / n_z
        self.assertLess(expected_d, GRAD_MATCH_WEIGHT_MAX)
        self.assertLess(expected_z, GRAD_MATCH_WEIGHT_MAX)
        np.testing.assert_allclose(
            ds['clip_weight'].values[0], expected_d, rtol=1e-4)
        np.testing.assert_allclose(
            ds['blend_clip_raw_z_weight'].values[0], expected_z, rtol=1e-4)
        self.assertEqual(ds.attrs['blend_rho_z'], 0.25)

    def test_rho_z_changes_the_total_without_retuning_density_weight(self):
        def _as_semantic():
            model = _default_model(clip_loss=_scaled_clip)
            model.enable_physical_clip(weight=0.0, as_semantic=True)
            return model

        density_only = _run(_as_semantic(), blend_rho=1.0)
        both = _run(_as_semantic(), blend_rho=1.0, blend_rho_z=0.25)
        np.testing.assert_allclose(
            density_only['clip_weight'].values[0],
            both['clip_weight'].values[0],
            rtol=1e-4)
        self.assertGreater(
            abs(float(density_only['loss'][0]) - float(both['loss'][0])),
            1e-8)
        self.assertNotIn('blend_clip_raw_z_weight', density_only)
        self.assertTrue(np.isfinite(both['blend_clip_raw_z_weight'].values[0]))


if __name__ == '__main__':
    absltest.main()
