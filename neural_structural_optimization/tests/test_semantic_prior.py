# lint as python3
"""Unit tests for the live CLIP/SDS spatial-prior module.

No CLIP checkpoint: providers are callables or the tiny frozen SDS denoiser.
"""

from pathlib import Path
import tempfile

import numpy as np
import torch
import xarray
from absl.testing import absltest

from neural_structural_optimization.guidance.loss_semantic_prior import (
    CallableScoreProvider,
    CoadaptiveMask,
    DiffusionSDSProvider,
    FrozenDenoiser,
    SemanticSpatialPrior,
    connectivity_metrics,
    gaussian_blur2d,
    heaviside_projection,
    normalize_preference,
    preference_from_score,
    stabilize_preference,
    projected_density_view,
    report_design_metrics,
    save_design_arrays,
    scale_fracs_for_grid,
    semantic_deficit,
    density_edge_map,
)
from neural_structural_optimization.guidance.loss_sketch import (
    mass_fraction_on_occupancy,
    scaffold_spatial_mass_loss,
)
from neural_structural_optimization.guidance.loss_structural import StructuralLoss
from neural_structural_optimization.model.model_base import (
    PHYSICAL_CLIP_INK_WRAP,
    PHYSICAL_CLIP_VOID_WRAP,
    VeniceLossAlgebra,
)
from neural_structural_optimization.problem.problems import StructuralParams
from neural_structural_optimization.optimize.optimizers import (
    AdaptiveAdam_Optimizer,
    _attach_snapshot_columns,
    _snapshot_physical_clip,
)

SMALL_WIDTH = 16
SMALL_HEIGHT = 32
SMALL_INTERVAL = 8


def _blob_score(density: torch.Tensor) -> torch.Tensor:
    """Lower when the top-left quadrant is denser: CLIP-like spatial preference."""
    field = density.reshape(density.shape[-2], density.shape[-1])
    h, w = field.shape
    return -(field[: h // 2, : w // 2].mean())


def _highfreq_score(density: torch.Tensor) -> torch.Tensor:
    field = density.reshape(1, 1, density.shape[-2], density.shape[-1])
    dx = field[..., :, 1:] - field[..., :, :-1]
    return dx.abs().mean()


class PreferenceMathTest(absltest.TestCase):

    def test_preference_is_relu_negative_density_gradient(self):
        density = torch.linspace(0.1, 0.9, 16).reshape(4, 4).requires_grad_(True)
        score = -density[1, 2]
        pref = preference_from_score(density, score, retain_graph=False)
        self.assertFalse(pref.requires_grad)
        self.assertGreater(float(pref[1, 2]), 0.0)
        pref_without = pref.clone()
        pref_without[1, 2] = 0
        self.assertEqual(float(pref_without.max()), 0.0)

    def test_normalize_and_blur_are_in_unit_interval(self):
        pref = torch.tensor([[0.0, 4.0], [1.0, 0.0]])
        normed = normalize_preference(pref)
        self.assertAlmostEqual(float(normed.max()), 1.0)
        blurred = gaussian_blur2d(normed, sigma=1.0)
        self.assertGreaterEqual(float(blurred.min()), 0.0)
        self.assertLessEqual(float(blurred.max()), 1.0 + 1e-5)

    def test_zero_sigma_blur_is_identity(self):
        field = torch.rand(5, 7)
        torch.testing.assert_close(gaussian_blur2d(field, 0.0), field)

    def test_amax_is_still_the_default_scale(self):
        pref = torch.tensor([[0.0, 4.0], [1.0, 0.0]])
        torch.testing.assert_close(
            normalize_preference(pref),
            normalize_preference(pref, quantile=None))
        self.assertAlmostEqual(float(normalize_preference(pref).max()), 1.0)

    def test_quantile_does_not_let_a_spike_crush_the_map(self):
        pref = torch.ones(8, 8)
        pref[0, 0] = 100.0
        amaxed = normalize_preference(pref)
        quantiled = normalize_preference(pref, quantile=0.99)
        self.assertAlmostEqual(float(amaxed[4, 4]), 0.01, places=5)
        self.assertGreater(float(quantiled[4, 4]), 0.9)
        self.assertAlmostEqual(float(quantiled.max()), 1.0)

    def test_quantile_rejects_out_of_range(self):
        pref = torch.ones(2, 2)
        for bad in (0.0, -0.1, 1.1):
            with self.assertRaises(ValueError):
                normalize_preference(pref, quantile=bad)

    def test_stabilize_blurs_before_scaling(self):
        pref = torch.zeros(16, 16)
        pref[1, 1] = 100.0
        pref[8:12, 8:12] = 1.0
        wrong_order = gaussian_blur2d(normalize_preference(pref), 2.0)
        right_order = stabilize_preference(pref, sigma=2.0)
        self.assertGreater(
            float(right_order[10, 10]), float(wrong_order[10, 10]) * 5.0)


class EmaPriorTest(absltest.TestCase):

    def test_ema_and_detachment(self):
        provider = CallableScoreProvider(_blob_score)
        prior = SemanticSpatialPrior(
            provider,
            scale_fracs={'global': 1.0},
            weight=1.0,
            ema_decay=0.5,
            smooth_sigma=0.0,
            curriculum='global_only',
        )
        density = torch.ones(8, 8) * 0.3
        density = density + 0.2 * torch.linspace(0, 1, 8).view(8, 1)
        prior.update(density)
        first = prior.maps['global'].clone()
        density2 = density.clone()
        density2[:4, :4] = 0.9
        prior.update(density2)
        second = prior.maps['global']
        expected = 0.5 * first + 0.5 * prior.last_instant['global']
        torch.testing.assert_close(second, expected)
        occ = prior.blended_occupancy(density2)
        self.assertFalse(occ.requires_grad)

    def test_zero_weight_without_auto_ratio_is_inactive(self):
        prior = SemanticSpatialPrior(
            CallableScoreProvider(_blob_score),
            scale_fracs={'global': 1.0},
            weight=0.0,
            auto_ratio=None,
        )
        self.assertFalse(prior.is_active())

    def test_scale_separation_uses_independent_maps(self):
        provider = CallableScoreProvider(
            _blob_score,
            scale_fns={'global': _blob_score, 'storey': _highfreq_score},
        )
        prior = SemanticSpatialPrior(
            provider,
            scale_fracs={'global': 1.0, 'storey': 0.25},
            weight=1.0,
            ema_decay=0.0,
            smooth_sigma=0.0,
            curriculum='all',
        )
        density = torch.rand(16, 8)
        prior.update(density)
        self.assertIsNotNone(prior.maps['global'])
        self.assertIsNotNone(prior.maps['storey'])
        self.assertIsNone(prior.maps['member'])
        # Different objectives should not yield identical maps.
        self.assertGreater(
            float((prior.maps['global'] - prior.maps['storey']).abs().max()),
            1e-6)

    def test_hierarchical_curriculum_unlocks_scales_with_resizes(self):
        prior = SemanticSpatialPrior(
            CallableScoreProvider(_blob_score),
            scale_fracs={'global': 1.0, 'storey': 0.25, 'member': 0.0625},
            weight=1.0,
            curriculum='hierarchical',
        )
        self.assertEqual(prior.active_scales(resizes=0, resize_num=2), ('global',))
        self.assertEqual(
            prior.active_scales(resizes=1, resize_num=2), ('global', 'storey'))
        self.assertEqual(
            prior.active_scales(resizes=2, resize_num=2),
            ('global', 'storey', 'member'))

    def test_grid_fracs_match_physical_motif_scale(self):
        fracs = scale_fracs_for_grid(256, 64)
        self.assertAlmostEqual(fracs['global'], 1.0)
        self.assertAlmostEqual(fracs['storey'], 0.25)
        self.assertAlmostEqual(fracs['member'], 0.0625)


def _prefers_material(density: torch.Tensor) -> torch.Tensor:
    """Uniformly rewards material, so preference reflects only the view's slope."""
    return -density.mean()


class ProjectionTest(absltest.TestCase):
    """The sculptural view: CLIP must commit material, not lay down gray."""

    def test_zero_beta_is_identity(self):
        field = torch.rand(6, 5)
        torch.testing.assert_close(heaviside_projection(field, 0.0), field)
        torch.testing.assert_close(
            projected_density_view(field, beta=0.0, filter_sigma=0.0), field)

    def test_projection_pushes_density_toward_zero_one(self):
        field = torch.tensor([0.05, 0.5, 0.95])
        projected = heaviside_projection(field, beta=8.0, eta=0.5)
        self.assertLess(float(projected[0]), 0.02)
        self.assertAlmostEqual(float(projected[1]), 0.5, places=5)
        self.assertGreater(float(projected[2]), 0.98)

    def test_faint_density_earns_far_less_preference_than_committed(self):
        prior = SemanticSpatialPrior(
            CallableScoreProvider(_prefers_material),
            scale_fracs={'global': 1.0},
            weight=1.0,
            ema_decay=0.0,
            smooth_sigma=0.0,
            curriculum='global_only',
            projection_beta=8.0,
            projection_filter_sigma=0.0,
        )
        density = torch.full((8, 8), 0.05)
        density[:, 4:] = 0.5
        prior.update(density)
        pref = prior.maps['global']
        self.assertGreater(float(pref[:, 4:].mean()), 20 * float(pref[:, :4].mean()))

    def test_without_projection_faint_and_committed_score_alike(self):
        prior = SemanticSpatialPrior(
            CallableScoreProvider(_prefers_material),
            scale_fracs={'global': 1.0},
            weight=1.0,
            ema_decay=0.0,
            smooth_sigma=0.0,
            curriculum='global_only',
        )
        density = torch.full((8, 8), 0.05)
        density[:, 4:] = 0.5
        prior.update(density)
        pref = prior.maps['global']
        torch.testing.assert_close(pref[:, :4].mean(), pref[:, 4:].mean())

    def test_filter_erases_features_below_the_minimum_size(self):
        field = torch.zeros(16, 16)
        field[:, 1:5] = 1.0  # wider than the filter
        field[:, 12] = 1.0  # a single-cell stroke
        view = projected_density_view(field, beta=8.0, filter_sigma=2.0)
        self.assertGreater(float(view[8, 3]), 0.5)
        self.assertLess(float(view[8, 12]), 0.1)

    def test_projection_is_confined_to_the_sculptural_scales(self):
        kwargs = dict(
            scale_fracs={'global': 1.0, 'storey': 0.25},
            weight=1.0,
            ema_decay=0.0,
            smooth_sigma=0.0,
            curriculum='all',
        )
        density = torch.full((8, 8), 0.05)
        density[:, 4:] = 0.5
        baseline = SemanticSpatialPrior(
            CallableScoreProvider(_prefers_material), **kwargs)
        sculpted = SemanticSpatialPrior(
            CallableScoreProvider(_prefers_material),
            projection_beta=8.0,
            projection_scales=('global',),
            **kwargs,
        )
        baseline.update(density)
        sculpted.update(density)
        self.assertTrue(sculpted.projects('global'))
        self.assertFalse(sculpted.projects('storey'))
        self.assertFalse(sculpted.projects('member'))
        # The detail scale is untouched; only the sculptural scale changes.
        torch.testing.assert_close(sculpted.maps['storey'], baseline.maps['storey'])
        self.assertGreater(
            float((sculpted.maps['global'] - baseline.maps['global']).abs().max()),
            1e-3)

    def test_ink_fraction_separates_faint_from_committed_preference(self):
        density = torch.tensor([[0.05, 0.9]])
        faint = torch.tensor([[1.0, 0.0]])
        committed = torch.tensor([[0.0, 1.0]])
        self.assertAlmostEqual(
            SemanticSpatialPrior._ink_fraction(density, faint), 1.0)
        self.assertAlmostEqual(
            SemanticSpatialPrior._ink_fraction(density, committed), 0.0)
        self.assertEqual(
            SemanticSpatialPrior._ink_fraction(density, torch.zeros(1, 2)), 0.0)

    def test_projection_reduces_the_reported_ink_fraction(self):
        kwargs = dict(
            scale_fracs={'global': 1.0},
            weight=1.0,
            ema_decay=0.0,
            smooth_sigma=0.0,
            curriculum='global_only',
        )
        density = torch.full((8, 8), 0.05)
        density[:, 4:] = 0.5
        baseline = SemanticSpatialPrior(
            CallableScoreProvider(_prefers_material), **kwargs)
        sculpted = SemanticSpatialPrior(
            CallableScoreProvider(_prefers_material),
            projection_beta=8.0, **kwargs)
        baseline.update(density)
        sculpted.update(density)
        self.assertLess(
            sculpted.last_metrics['preference_ink_fraction'],
            baseline.last_metrics['preference_ink_fraction'])

    def test_invalid_projection_settings_are_rejected(self):
        with self.assertRaises(ValueError):
            SemanticSpatialPrior(
                CallableScoreProvider(_blob_score),
                scale_fracs={'global': 1.0},
                projection_eta=0.0,
            )
        with self.assertRaises(ValueError):
            SemanticSpatialPrior(
                CallableScoreProvider(_blob_score),
                scale_fracs={'global': 1.0},
                projection_scales=('facade',),
            )


class SdsProviderTest(absltest.TestCase):

    def test_sds_noise_bands_differ_by_scale(self):
        provider = DiffusionSDSProvider(FrozenDenoiser(channels=8, seed=0), seed=0)
        self.assertGreater(provider.timestep_bands['global'][0],
                           provider.timestep_bands['storey'][1])
        self.assertGreater(provider.timestep_bands['storey'][0],
                           provider.timestep_bands['member'][1])
        density = torch.rand(1, 8, 4, requires_grad=True)
        scores = provider.score_by_scale(density, ('global', 'member'))
        self.assertEqual(set(scores), {'global', 'member'})
        for value in scores.values():
            self.assertTrue(torch.isfinite(value))
            self.assertTrue(value.requires_grad)

    def test_sds_occupancy_is_detached(self):
        provider = DiffusionSDSProvider(FrozenDenoiser(channels=8, seed=1), seed=1)
        prior = SemanticSpatialPrior(
            provider,
            scale_fracs={'global': 1.0},
            weight=1.0,
            ema_decay=0.5,
            curriculum='global_only',
        )
        density = torch.rand(8, 4)
        prior.update(density)
        occ = prior.blended_occupancy(density)
        self.assertFalse(occ.requires_grad)
        self.assertTrue(torch.all(occ >= 0) and torch.all(occ <= 1))


class ConnectivityMetricsTest(absltest.TestCase):

    def test_column_from_top_to_bottom_is_connected(self):
        density = np.zeros((8, 4), dtype=np.float32)
        density[:, 1] = 1.0
        loads = np.zeros_like(density, dtype=bool)
        loads[0, 1] = True
        metrics = connectivity_metrics(density, loads, threshold=0.3)
        self.assertEqual(metrics['top_to_bottom_connected'], 1.0)
        self.assertEqual(metrics['support_to_load_connected'], 1.0)
        self.assertEqual(metrics['floating_mass_fraction'], 0.0)

    def test_floating_blob_is_counted(self):
        density = np.zeros((8, 4), dtype=np.float32)
        density[2:4, 1:3] = 1.0
        loads = np.zeros_like(density, dtype=bool)
        loads[0, 0] = True
        metrics = connectivity_metrics(density, loads, threshold=0.3)
        self.assertEqual(metrics['top_to_bottom_connected'], 0.0)
        self.assertGreater(metrics['floating_mass_fraction'], 0.9)


class DesignReportMetricsTest(absltest.TestCase):
    """Shared summary helper relocates validity; scaffold keys are always present."""

    def test_validity_matches_connectivity_metrics_plus_mean(self):
        density = np.zeros((8, 4), dtype=np.float32)
        density[:, 1] = 1.0
        loads = np.zeros_like(density, dtype=bool)
        loads[0, 1] = True
        expected = connectivity_metrics(density, loads, threshold=0.3)
        expected['mean_physical_density'] = float(density.mean())
        report = report_design_metrics(density, loads)
        self.assertEqual(report['validity'], expected)
        self.assertEqual(
            report['mean_physical_density'], expected['mean_physical_density'])
        self.assertIsNone(report['spatial_mass_loss'])
        self.assertIsNone(report['mass_on_scaffold'])
        self.assertIsNone(report['clip_loss'])
        self.assertIsNone(report['clip_loss_raw'])

    def test_scaffold_metrics_match_the_sketch_helpers(self):
        density = np.zeros((8, 4), dtype=np.float32)
        density[:, 2:] = 1.0
        scaffold = np.zeros((8, 4), dtype=np.float32)
        scaffold[:, :2] = 1.0
        loads = np.zeros_like(density, dtype=bool)
        report = report_design_metrics(density, loads, scaffold=scaffold)
        self.assertEqual(
            report['mass_on_scaffold'],
            mass_fraction_on_occupancy(density, scaffold))
        self.assertEqual(
            report['spatial_mass_loss'],
            scaffold_spatial_mass_loss(density, scaffold, loads))
        self.assertIn('component_count', report['validity'])
        self.assertIn('support_to_load_connected', report['validity'])

    def test_trajectory_clip_keys_are_relocated_not_recomputed(self):
        density = np.ones((2, 2), dtype=np.float32) * 0.3
        loads = np.zeros_like(density, dtype=bool)

        class _FakeDs(dict):
            pass

        ds = _FakeDs()
        ds['clip_loss'] = [1.0, 2.5]
        ds['clip_loss_raw'] = [0.4, 0.37]
        report = report_design_metrics(density, loads, ds=ds)
        self.assertEqual(report['clip_loss'], 2.5)
        self.assertEqual(report['clip_loss_raw'], 0.37)

    def test_npy_round_trip_is_bit_identical_and_unclipped(self):
        density = np.array([[0.25, 0.35], [1.2, -0.1]], dtype=np.float32)
        raw = np.array([[-11.5, 13.1], [0.0, 0.9]], dtype=np.float32)
        with tempfile.TemporaryDirectory() as tmp:
            paths = save_design_arrays(tmp, density, raw=raw)
            loaded_density = np.load(paths['physical_density'])
            loaded_raw = np.load(paths['final_design_raw'])
        np.testing.assert_array_equal(loaded_density, density)
        np.testing.assert_array_equal(loaded_raw, raw)

    def test_save_without_raw_does_not_write_raw_file(self):
        density = np.array([[0.25, 0.35]], dtype=np.float64)
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            paths = save_design_arrays(directory, density)
            self.assertIn('physical_density', paths)
            self.assertNotIn('final_design_raw', paths)
            self.assertFalse((directory / 'final_design_raw.npy').exists())
            loaded = np.load(paths['physical_density'])
        np.testing.assert_array_equal(loaded, density)
        self.assertEqual(loaded.dtype, np.float64)

    def test_saliency_and_dream_reports_share_the_same_keys(self):
        density = np.ones((4, 4), dtype=np.float32) * 0.3
        loads = np.zeros_like(density, dtype=bool)
        scaffold = np.ones_like(density)
        class _FakeDs(dict):
            pass
        ds = _FakeDs()
        ds['clip_loss'] = [0.9]
        ds['clip_loss_raw'] = [0.4]
        saliency = report_design_metrics(density, loads, ds=ds)
        dream = report_design_metrics(
            density, loads, scaffold=scaffold, ds=ds)
        self.assertEqual(set(saliency), set(dream))
        self.assertEqual(
            set(saliency),
            {
                'validity',
                'mean_physical_density',
                'clip_loss',
                'clip_loss_raw',
                'mass_on_scaffold',
                'spatial_mass_loss',
            })
        self.assertIsNone(saliency['mass_on_scaffold'])
        self.assertIsNotNone(dream['mass_on_scaffold'])


    def test_scaffold_none_stays_null_not_zero(self):
        density = np.ones((4, 4), dtype=np.float32) * 0.3
        loads = np.zeros_like(density, dtype=bool)
        report = report_design_metrics(density, loads, scaffold=None)
        self.assertIsNone(report['mass_on_scaffold'])
        self.assertIsNone(report['spatial_mass_loss'])
        self.assertIsNot(report['mass_on_scaffold'], 0.0)


def _tiny_model(clip_loss=None):
    params = StructuralParams(
        problem_name='multistory_building',
        width=SMALL_WIDTH,
        height=SMALL_HEIGHT,
        density=0.3,
        interval=SMALL_INTERVAL,
        filter_width=1.5,
    )
    from neural_structural_optimization.model.model_ada import AdaptivePixelModel
    model = AdaptivePixelModel(
        structural_params=params,
        clip_loss=None,
        seed=0,
        resize_num=0,
        resize_scale=2,
    )
    model.enable_venice_compat_loss(VeniceLossAlgebra(clip_alpha=10.0))
    if clip_loss is not None:
        object.__setattr__(model, 'clip_loss', clip_loss)
    return model


class SemanticPriorModelSeamTest(absltest.TestCase):

    def test_disabled_prior_does_not_change_total_loss(self):
        model = _tiny_model()
        logits = model()
        without = model.get_venice_compat_losses(logits).total_loss
        prior = SemanticSpatialPrior(
            CallableScoreProvider(_blob_score),
            scale_fracs={'global': 1.0},
            weight=0.0,
        )
        model.enable_semantic_prior(prior)
        with_zero = model.get_venice_compat_losses(logits).total_loss
        torch.testing.assert_close(without, with_zero)

    def test_mocked_clip_moves_mass_into_preferred_region(self):
        def clip_fn(logits):
            return (logits ** 2).mean() * 0.0

        model = _tiny_model(clip_loss=clip_fn)
        prior = SemanticSpatialPrior(
            CallableScoreProvider(_blob_score),
            scale_fracs={'global': 1.0},
            weight=50.0,
            ema_decay=0.0,
            smooth_sigma=0.0,
            curriculum='global_only',
        )
        model.enable_semantic_prior(prior)
        logits = model()
        terms = model.get_venice_compat_losses(logits)
        self.assertGreater(float(model._last_semantic_prior_loss), 0.0)
        occ = prior.blended_occupancy(model.get_physical_density(logits))
        self.assertGreater(
            float(occ[..., : SMALL_HEIGHT // 2, : SMALL_WIDTH // 2].mean()),
            float(occ[..., SMALL_HEIGHT // 2 :, SMALL_WIDTH // 2 :].mean()))
        terms.total_loss.backward()
        self.assertIsNotNone(model.z.grad)


class SemanticPriorFeaCountTest(absltest.TestCase):
    """One StructuralLoss forward per AdaptiveAdam step, prior on or off."""

    def _count_forwards(self, enable_prior: bool) -> int:
        calls = {'n': 0}
        original = StructuralLoss.forward

        def wrapped(ctx, logits, env):
            calls['n'] += 1
            return original(ctx, logits, env)

        StructuralLoss.forward = staticmethod(wrapped)
        try:
            def clip_fn(logits):
                return logits.new_tensor(0.4)

            model = _tiny_model(clip_loss=clip_fn)
            if enable_prior:
                model.enable_semantic_prior(SemanticSpatialPrior(
                    CallableScoreProvider(_blob_score),
                    scale_fracs={'global': 1.0},
                    weight=1.0,
                    curriculum='global_only',
                ))
            AdaptiveAdam_Optimizer(
                model,
                max_iterations=3,
                lr=0.2,
                convergence_threshold=0.0,
                max_resize_iteration=50,
            ).optimize()
        finally:
            StructuralLoss.forward = original
        return calls['n']

    def test_prior_does_not_add_fea_forwards(self):
        off = self._count_forwards(False)
        on = self._count_forwards(True)
        self.assertEqual(off, 3)
        self.assertEqual(on, 3)


class CoadaptiveMaskTest(absltest.TestCase):

    def test_control_receives_grad_and_is_not_a_model_parameter(self):
        occupancy = torch.full((SMALL_HEIGHT, SMALL_WIDTH), 0.5)
        mask = CoadaptiveMask(
            occupancy, control_height=4, control_width=4, lr=0.2)
        model = _tiny_model()
        model.enable_sketch_prior(occupancy, weight=1.0)
        mask.commit_to_model(model)
        param_ids = {id(p) for p in model.parameters()}
        self.assertNotIn(id(mask.control), param_ids)
        self.assertFalse(
            any(p is model.sketch_occupancy_full for p in model.parameters()))
        self.assertFalse(model.sketch_occupancy_full.requires_grad)

        density = torch.rand(SMALL_HEIGHT, SMALL_WIDTH)

        def score_fn(field):
            return (field - 0.8).pow(2).mean()

        before = mask.control.detach().clone()
        metrics = mask.step(density, score_fn, use_saliency=True)
        self.assertNotEqual(float((mask.control.detach() - before).abs().sum()), 0.0)
        self.assertIn('mask_clip', metrics)
        self.assertFalse(mask.ema.requires_grad)
        self.assertFalse(mask.last_saliency.requires_grad)
        mask.commit_to_model(model)
        self.assertFalse(
            any(p is model.sketch_occupancy_full for p in model.parameters()))

    def test_saliency_scores_the_physical_density_not_raw_logits(self):
        occupancy = torch.full((8, 8), 0.4)
        mask = CoadaptiveMask(occupancy, control_height=4, control_width=4)
        density = torch.linspace(0.1, 0.9, 64).reshape(8, 8)
        seen = []

        def score_fn(field):
            seen.append(field.detach().clone())
            return field.mean()

        mask.step(density, score_fn, use_saliency=True)
        self.assertGreaterEqual(len(seen), 2)
        torch.testing.assert_close(seen[0], density)
        self.assertGreaterEqual(float(seen[0].min()), 0.0)
        self.assertLessEqual(float(seen[0].max()), 1.0)
        self.assertGreaterEqual(float(seen[1].min()), 0.0)
        self.assertLessEqual(float(seen[1].max()), 1.0)
        self.assertEqual(tuple(mask.last_saliency.shape), (8, 8))

    def test_batched_density_saliency_is_stored_2d(self):
        occupancy = torch.full((8, 8), 0.4)
        mask = CoadaptiveMask(occupancy, control_height=4, control_width=4)
        density = torch.linspace(0.1, 0.9, 64).reshape(1, 8, 8)

        def score_fn(field):
            return field.mean()

        mask.step(density, score_fn, use_saliency=True)
        self.assertEqual(tuple(mask.last_saliency.shape), (8, 8))
        self.assertFalse(mask.last_saliency.requires_grad)

    def test_defaults_blur_and_percentile_scale_saliency(self):
        occupancy = torch.full((8, 8), 0.5)
        mask = CoadaptiveMask(occupancy, control_height=4, control_width=4)
        self.assertAlmostEqual(mask.smooth_sigma, 2.0)
        self.assertAlmostEqual(mask.saliency_quantile, 0.99)

    def test_mask_saliency_spreads_a_point_source(self):
        occupancy = torch.full((8, 8), 0.5)
        density = torch.zeros(8, 8)
        density[3, 3] = 1.0

        def score_fn(field):
            return -field[3, 3]

        sharp = CoadaptiveMask(
            occupancy, control_height=4, control_width=4,
            smooth_sigma=0.0, saliency_quantile=None)
        sharp.step(density, score_fn)
        smooth = CoadaptiveMask(
            occupancy, control_height=4, control_width=4,
            smooth_sigma=2.0, saliency_quantile=None)
        smooth.step(density, score_fn)
        self.assertGreater(
            float((smooth.last_saliency > 0.05).sum()),
            float((sharp.last_saliency > 0.05).sum()))

    def test_after_step_hook_runs_without_owning_model_parameters(self):
        model = _tiny_model()
        occupancy = torch.full((SMALL_HEIGHT, SMALL_WIDTH), 0.5)
        model.enable_sketch_prior(occupancy, weight=1.0)
        calls = []

        def after_step(updated, step, _terms):
            calls.append(int(step))
            self.assertNotIn(
                id(updated.sketch_occupancy_full),
                {id(p) for p in updated.parameters()})

        AdaptiveAdam_Optimizer(
            model,
            max_iterations=2,
            lr=0.2,
            convergence_threshold=0.0,
            max_resize_iteration=50,
        ).optimize(after_step=after_step)
        self.assertEqual(calls, [0, 1])


class PhysicalClipTermTest(absltest.TestCase):

    def test_zero_weight_does_not_change_total_loss(self):
        model = _tiny_model()
        logits = model()
        without = model.get_venice_compat_losses(logits).total_loss
        model.enable_physical_clip(weight=0.0)
        with_zero = model.get_venice_compat_losses(logits).total_loss
        torch.testing.assert_close(without, with_zero)

    def test_term_scores_physical_density_not_raw_logits(self):
        seen = []

        def clip_fn(field):
            seen.append(field.detach().clone())
            return field.new_tensor(0.4)

        model = _tiny_model(clip_loss=clip_fn)
        model.enable_physical_clip(weight=1.0)
        logits = model()
        terms = model.get_venice_compat_losses(logits)
        self.assertGreaterEqual(len(seen), 2)
        density = model.get_physical_density(logits).detach()
        torch.testing.assert_close(seen[0], logits.detach())
        torch.testing.assert_close(seen[-1], density)
        self.assertGreaterEqual(float(seen[-1].min()), 0.0)
        self.assertLessEqual(float(seen[-1].max()), 1.0 + 1e-4)

    def test_match_venice_scales_by_clip_alpha_times_compliance(self):
        seen = []

        def clip_fn(field):
            seen.append(field.detach().clone())
            return field.new_tensor(0.5)

        model = _tiny_model(clip_loss=clip_fn)
        logits = model()
        baseline = model.get_venice_compat_losses(logits).total_loss.detach()
        model.enable_physical_clip(weight=1.0, match_venice=True)
        terms = model.get_venice_compat_losses(logits)
        extra = terms.total_loss - baseline
        # Venice algebra clip_alpha is 10; CLIP(rho)=0.5; coeff = 10 * C
        expected = 10.0 * terms.compliance_loss.detach() * 0.5
        torch.testing.assert_close(extra, expected, rtol=1e-5, atol=1e-5)
        density = model.get_physical_density(logits).detach()
        torch.testing.assert_close(seen[-1], density)


class SemanticDeficitTest(absltest.TestCase):
    """The coupling that is supposed to carry the prompt around the loop."""

    def test_deficit_discounts_cells_that_already_carry_material(self):
        density = torch.tensor([[0.0, 1.0], [0.0, 1.0]])
        saliency = torch.ones_like(density)
        deficit = semantic_deficit(density, saliency)
        # Equal demand everywhere, but only the void cells are unmet.
        self.assertAlmostEqual(float(deficit[0, 0]), 1.0, places=5)
        self.assertAlmostEqual(float(deficit[0, 1]), 0.0, places=5)

    def test_deficit_is_zero_where_the_prompt_asks_for_nothing(self):
        density = torch.zeros((2, 2))
        saliency = torch.zeros((2, 2))
        self.assertAlmostEqual(
            float(semantic_deficit(density, saliency).abs().max()), 0.0)

    def test_zero_deficit_weight_reproduces_the_agreement_only_step(self):
        occupancy = torch.full((8, 8), 0.5)
        density = torch.linspace(0.0, 1.0, 64).reshape(8, 8)

        def score_fn(field):
            return (field ** 2).mean()

        def run(**kwargs):
            torch.manual_seed(0)
            mask = CoadaptiveMask(
                occupancy, control_height=4, control_width=4, **kwargs)
            mask.step(density, score_fn)
            return mask.occupancy_cpu()

        torch.testing.assert_close(run(), run(lambda_deficit=0.0))

    def test_deficit_term_raises_the_mask_over_unmet_demand(self):
        # Material on the left; the score wants material on the right, so the
        # unmet demand sits exactly where the structure has put nothing. The
        # mask starts mid-range: an occupancy of 0 or 1 saturates the sigmoid
        # control and nothing would move regardless of the term.
        density = torch.zeros((8, 8))
        density[:, :4] = 1.0
        occupancy = torch.full((8, 8), 0.5)

        def score_fn(field):
            return -field[:, 4:].mean()

        torch.manual_seed(0)
        mask = CoadaptiveMask(
            occupancy,
            control_height=8,
            control_width=8,
            lr=0.5,
            # Every other term off, so only the new coupling can move it.
            lambda_clip=0.0,
            lambda_saliency=0.0,
            lambda_overlap=0.0,
            lambda_area=0.0,
            lambda_anchor=0.0,
            lambda_deficit=1.0)
        for _ in range(10):
            mask.step(density, score_fn)
        occ = mask.occupancy_cpu()
        self.assertGreater(float(occ[:, 4:].mean()), float(occ[:, :4].mean()))
        self.assertGreater(float(occ[:, 4:].mean()), 0.5)

    def test_every_weight_at_zero_leaves_the_mask_untouched(self):
        occupancy = torch.full((8, 8), 0.5)
        mask = CoadaptiveMask(
            occupancy,
            control_height=4,
            control_width=4,
            lambda_clip=0.0,
            lambda_saliency=0.0,
            lambda_overlap=0.0,
            lambda_area=0.0,
            lambda_anchor=0.0,
            lambda_deficit=0.0)
        before = mask.occupancy_cpu().clone()
        mask.step(torch.full((8, 8), 0.3), lambda field: field.mean())
        torch.testing.assert_close(mask.occupancy_cpu(), before)

    def test_deficit_uses_saliency_even_when_the_saliency_term_is_off(self):
        occupancy = torch.full((8, 8), 0.5)
        density = torch.full((8, 8), 0.2)
        mask = CoadaptiveMask(
            occupancy,
            control_height=4,
            control_width=4,
            lambda_saliency=0.0,
            lambda_deficit=1.0)
        mask.step(density, lambda field: (field ** 2).mean())
        self.assertIsNotNone(mask.last_deficit)
        self.assertIn('mask_deficit', mask.last_metrics)


class CoadaptReleaseTest(absltest.TestCase):
    """Fading the terms that hold the mask in its starting basin."""

    def _mask(self):
        torch.manual_seed(0)
        return CoadaptiveMask(
            torch.full((8, 8), 0.5), control_height=4, control_width=4)

    def test_default_progress_keeps_both_holding_terms_at_full_strength(self):
        mask = self._mask()
        metrics = mask.step(
            torch.full((8, 8), 0.3), lambda field: field.mean())
        self.assertAlmostEqual(metrics['release'], 1.0)

    def test_progress_fades_the_holding_terms_to_zero(self):
        mask = self._mask()
        metrics = mask.step(
            torch.full((8, 8), 0.3), lambda field: field.mean(), progress=1.0)
        self.assertAlmostEqual(metrics['release'], 0.0)
        # The terms are still reported at full value; only their contribution
        # to the loss is faded, so the diagnostic stays readable.
        self.assertGreater(metrics['mask_overlap'], 0.0)

    def test_progress_is_clamped_outside_the_unit_interval(self):
        for progress, expected in ((-1.0, 1.0), (5.0, 0.0)):
            metrics = self._mask().step(
                torch.full((8, 8), 0.3),
                lambda field: field.mean(),
                progress=progress)
            self.assertAlmostEqual(metrics['release'], expected)

    def test_full_release_drops_the_holding_terms_out_of_the_loss(self):
        mask = self._mask()
        m = mask.step(
            torch.full((8, 8), 0.3), lambda field: field.mean(), progress=1.0)
        torch.testing.assert_close(
            m['mask_loss'],
            mask.lambda_clip * m['mask_clip']
            + mask.lambda_saliency * m['mask_saliency']
            + mask.lambda_deficit * m['mask_deficit']
            + mask.lambda_area * m['mask_area'],
            rtol=1e-5, atol=1e-6)

    def test_no_release_keeps_every_term_in_the_loss(self):
        mask = self._mask()
        m = mask.step(torch.full((8, 8), 0.3), lambda field: field.mean())
        torch.testing.assert_close(
            m['mask_loss'],
            mask.lambda_clip * m['mask_clip']
            + mask.lambda_saliency * m['mask_saliency']
            + mask.lambda_deficit * m['mask_deficit']
            + mask.lambda_overlap * m['mask_overlap']
            + mask.lambda_area * m['mask_area']
            + mask.lambda_anchor * m['mask_anchor'],
            rtol=1e-5, atol=1e-6)

    def test_releasing_the_anchor_lets_the_mask_drift_further(self):
        density = torch.linspace(0.0, 1.0, 64).reshape(8, 8)

        def score_fn(field):
            return -field.mean()

        def drift(progress):
            torch.manual_seed(0)
            mask = CoadaptiveMask(
                torch.full((8, 8), 0.5),
                control_height=4,
                control_width=4,
                lr=0.5,
                # Isolate the anchor: overlap also fades with release, and it
                # pushes the mask the same way the score does.
                lambda_overlap=0.0)
            for _ in range(10):
                mask.step(density, score_fn, progress=progress)
            return mask.last_metrics['mask_anchor']

        self.assertGreater(drift(1.0), drift(0.0))


class PhysicalClipLoggingTest(absltest.TestCase):
    """The density-CLIP term has to be observable to be tunable.

    Venice logs CLIP on raw z only, so without these columns the structure
    half of the co-adapt loop cannot be read from a finished run.
    """

    def test_term_value_and_coefficient_are_recorded_on_the_model(self):
        model = _tiny_model(clip_loss=lambda field: field.new_tensor(0.4))
        model.enable_physical_clip(weight=2.0)
        model.get_venice_compat_losses(model())
        self.assertAlmostEqual(float(model._last_physical_clip), 0.4, places=5)
        self.assertAlmostEqual(
            float(model._last_physical_clip_coefficient), 2.0, places=5)

    def test_inactive_term_clears_a_stale_reading(self):
        model = _tiny_model(clip_loss=lambda field: field.new_tensor(0.4))
        model.enable_physical_clip(weight=1.0)
        model.get_venice_compat_losses(model())
        self.assertIsNotNone(model._last_physical_clip)
        model.enable_physical_clip(weight=0.0)
        model.enable_sketch_prior(
            torch.full((SMALL_HEIGHT, SMALL_WIDTH), 0.5), weight=1.0)
        model.get_venice_compat_losses(model())
        self.assertIsNone(model._last_physical_clip)
        self.assertEqual(_snapshot_physical_clip(model), {})

    def test_snapshot_reports_the_weighted_contribution(self):
        model = _tiny_model(clip_loss=lambda field: field.new_tensor(0.5))
        model.enable_physical_clip(weight=3.0)
        model.get_venice_compat_losses(model())
        snap = _snapshot_physical_clip(model)
        self.assertAlmostEqual(snap['physical_clip'], 0.5, places=5)
        self.assertAlmostEqual(snap['physical_clip_coefficient'], 3.0, places=5)
        self.assertAlmostEqual(snap['physical_clip_weighted'], 1.5, places=5)
        self.assertAlmostEqual(snap['physical_clip_beta'], 0.0, places=5)
        self.assertAlmostEqual(snap['physical_clip_sigma'], 0.0, places=5)

    def test_run_dataset_carries_the_trajectory_only_when_the_term_is_on(self):
        off = _tiny_model(clip_loss=lambda field: field.new_tensor(0.4))
        ds_off = AdaptiveAdam_Optimizer(
            off, max_iterations=2, lr=0.2, convergence_threshold=0.0,
            max_resize_iteration=50).optimize()
        self.assertNotIn('physical_clip', ds_off)

        on = _tiny_model(clip_loss=lambda field: field.new_tensor(0.4))
        on.enable_physical_clip(weight=1.0)
        ds_on = AdaptiveAdam_Optimizer(
            on, max_iterations=2, lr=0.2, convergence_threshold=0.0,
            max_resize_iteration=50).optimize()
        self.assertIn('physical_clip', ds_on)
        self.assertEqual(ds_on['physical_clip'].sizes['step'], 2)
        np.testing.assert_allclose(
            ds_on['physical_clip'].values, np.full(2, 0.4), rtol=1e-5)
        self.assertIn('physical_clip_beta', ds_on)
        self.assertIn('physical_clip_sigma', ds_on)
        np.testing.assert_allclose(ds_on['physical_clip_beta'].values, 0.0)
        np.testing.assert_allclose(ds_on['physical_clip_sigma'].values, 0.0)

    def test_snapshot_columns_nan_fill_a_step_that_lacks_a_key(self):
        ds = xarray.Dataset(coords={'step': np.arange(3)})
        _attach_snapshot_columns(
            ds, [{'a': 1.0}, {}, {'a': 2.0, 'b': 5.0}], str)
        np.testing.assert_allclose(ds['a'].values, [1.0, np.nan, 2.0])
        np.testing.assert_allclose(ds['b'].values, [np.nan, np.nan, 5.0])

    def test_all_empty_snapshots_leave_the_schema_untouched(self):
        ds = xarray.Dataset(coords={'step': np.arange(2)})
        _attach_snapshot_columns(ds, [{}, {}], str)
        self.assertEqual(list(ds.data_vars), [])


class PhysicalClipRecognizabilityTest(absltest.TestCase):
    """Projection, void, and edge compose on CLIP(rho); defaults stay off."""

    def test_defaults_score_the_raw_density(self):
        seen = []

        def clip_fn(field):
            seen.append(field.detach().clone())
            return field.mean()

        model = _tiny_model(clip_loss=clip_fn)
        model.enable_physical_clip(weight=1.0)
        field = torch.tensor([[0.2, 0.8], [0.3, 0.7]])
        model.score_physical_clip(field)
        self.assertEqual(len(seen), 1)
        torch.testing.assert_close(seen[0], field)

    def test_projection_beta_anneals_from_one_to_beta_max(self):
        model = _tiny_model()
        model.enable_physical_clip(weight=1.0, projection_beta_max=8.0)
        model._opt_max_iterations = 11
        model._opt_step = 0
        self.assertAlmostEqual(model._physical_clip_projection_beta(), 1.0)
        model._opt_step = 10
        self.assertAlmostEqual(model._physical_clip_projection_beta(), 8.0)

    def test_snapshot_logs_live_beta_and_sigma(self):
        model = _tiny_model(clip_loss=lambda field: field.new_tensor(0.5))
        model.enable_physical_clip(
            weight=1.0, projection_beta_max=8.0, projection_sigma=2.0)
        model._opt_step = 10
        model._opt_max_iterations = 11
        model.get_venice_compat_losses(model())
        snap = _snapshot_physical_clip(model)
        self.assertAlmostEqual(snap['physical_clip_beta'], 8.0, places=5)
        self.assertAlmostEqual(snap['physical_clip_sigma'], 2.0, places=5)

    def test_projection_sigma_holds_when_end_is_none(self):
        model = _tiny_model()
        model.enable_physical_clip(
            weight=1.0, projection_beta_max=8.0, projection_sigma=2.0)
        model._opt_step = 10
        model._opt_max_iterations = 11
        model.resize_num = 2
        model.resizes = 2
        self.assertAlmostEqual(model._physical_clip_projection_sigma(), 2.0)

    def test_projection_sigma_anneals_with_resizes(self):
        model = _tiny_model()
        model.enable_physical_clip(
            weight=1.0, projection_sigma=2.0, projection_sigma_end=0.5)
        model.resize_num = 2
        model.resizes = 0
        self.assertAlmostEqual(model._physical_clip_projection_sigma(), 2.0)
        model.resizes = 1
        self.assertAlmostEqual(model._physical_clip_projection_sigma(), 1.25)
        model.resizes = 2
        self.assertAlmostEqual(model._physical_clip_projection_sigma(), 0.5)

    def test_projection_sigma_anneals_on_steps_without_a_resize_schedule(self):
        model = _tiny_model()
        model.enable_physical_clip(
            weight=1.0, projection_sigma=2.0, projection_sigma_end=0.5)
        model._opt_max_iterations = 11
        model._opt_step = 0
        self.assertAlmostEqual(model._physical_clip_projection_sigma(), 2.0)
        model._opt_step = 10
        self.assertAlmostEqual(model._physical_clip_projection_sigma(), 0.5)

    def test_annealed_sigma_changes_the_view_clip_sees(self):
        seen = []

        def clip_fn(field):
            seen.append(field.detach().clone())
            return field.mean()

        model = _tiny_model(clip_loss=clip_fn)
        model.enable_physical_clip(
            weight=1.0, projection_beta_max=8.0,
            projection_sigma=2.0, projection_sigma_end=0.5)
        model._opt_step = 10
        model._opt_max_iterations = 11
        model.resize_num = 2
        field = torch.tensor([[0.2, 0.8], [0.2, 0.8]])
        model.resizes = 0
        model.score_physical_clip(field)
        coarse = seen[-1]
        model.resizes = 2
        model.score_physical_clip(field)
        fine = seen[-1]
        self.assertGreater(float((coarse - fine).abs().max()), 1e-4)

    def test_zero_beta_max_skips_projection_even_when_sigma_is_set(self):
        seen = []

        def clip_fn(field):
            seen.append(field.detach().clone())
            return field.mean()

        model = _tiny_model(clip_loss=clip_fn)
        model.enable_physical_clip(
            weight=1.0, projection_beta_max=0.0, projection_sigma=2.0)
        field = torch.tensor([[0.2, 0.8], [0.2, 0.8]])
        model.score_physical_clip(field)
        torch.testing.assert_close(seen[-1], field)

    def test_late_projection_binarizes_the_view_clip_sees(self):
        seen = []

        def clip_fn(field):
            seen.append(field.detach().clone())
            return field.mean()

        model = _tiny_model(clip_loss=clip_fn)
        model.enable_physical_clip(
            weight=1.0, projection_beta_max=8.0, projection_sigma=0.0)
        model._opt_step = 10
        model._opt_max_iterations = 11
        field = torch.tensor([[0.2, 0.8], [0.2, 0.8]])
        model.score_physical_clip(field)
        view = seen[-1]
        self.assertLess(float(view[0, 0]), 0.05)
        self.assertGreater(float(view[0, 1]), 0.95)

    def test_void_term_scores_the_complement(self):
        seen = []

        def clip_fn(field):
            seen.append(field.detach().clone())
            return field.mean()

        model = _tiny_model(clip_loss=clip_fn)
        model.enable_physical_clip(weight=1.0, void_weight=1.0)
        field = torch.tensor([[0.25, 0.75], [0.0, 1.0]])
        model.score_physical_clip(field)
        self.assertEqual(len(seen), 2)
        torch.testing.assert_close(seen[0], field)
        torch.testing.assert_close(seen[1], 1.0 - field)
        self.assertEqual(
            model.physical_clip_void_prompt_wrap, PHYSICAL_CLIP_VOID_WRAP)

    def test_edge_weight_scores_a_normalized_outline(self):
        seen = []

        def clip_fn(field):
            seen.append(field.detach().clone())
            return field.mean()

        model = _tiny_model(clip_loss=clip_fn)
        model.enable_physical_clip(weight=1.0, edge_weight=1.0)
        field = torch.zeros(8, 8)
        field[:, 3:5] = 1.0
        model.score_physical_clip(field)
        self.assertEqual(len(seen), 2)
        torch.testing.assert_close(seen[0], field)
        torch.testing.assert_close(seen[1], density_edge_map(field))
        self.assertGreater(float(seen[1].max()), 0.0)

    def test_ink_wrap_constant_matches_the_agreed_template(self):
        self.assertEqual(
            PHYSICAL_CLIP_INK_WRAP,
            'a minimal ink drawing of a {prompt}')
        self.assertEqual(
            PHYSICAL_CLIP_INK_WRAP.format(prompt='human skull'),
            'a minimal ink drawing of a human skull')


class DensityEdgeMapTest(absltest.TestCase):

    def test_a_bar_produces_a_normalized_outline(self):
        field = torch.zeros(8, 8)
        field[:, 3:5] = 1.0
        edge = density_edge_map(field)
        self.assertGreater(float(edge.max()), 0.99)
        self.assertGreaterEqual(float(edge.min()), 0.0)
        self.assertLessEqual(float(edge.max()), 1.0 + 1e-6)


if __name__ == '__main__':
    absltest.main()
