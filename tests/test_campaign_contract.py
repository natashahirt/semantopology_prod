import json
import time
from types import SimpleNamespace

import numpy as np
import pytest

from recipe import campaign


def test_incomplete_final_grid_is_not_marked_done(tmp_path, monkeypatch):
    """A coarse intermediate density must never masquerade as a finished run."""
    monkeypatch.setattr(campaign, 'save_design_arrays', lambda *args, **kwargs: None)
    monkeypatch.setattr(campaign, 'save_field_png', lambda *args, **kwargs: None)
    monkeypatch.setattr(campaign, 'write_progress_gif', lambda *args, **kwargs: None)
    monkeypatch.setattr(campaign, 'git_revision', lambda: 'test-sha')
    monkeypatch.setattr(campaign, 'pip_freeze_hash', lambda: 'test-env')

    args = SimpleNamespace(
        run_id='S1/tall/fern_fronds/g',
        experiment='S1',
        group='S1',
        mode='semantic',
        structure='tall',
        prompt_sketch=False,
    )
    preset = SimpleNamespace(
        height=8,
        width=8,
        clip_prompt='fern fronds',
        sketch_path=None,
    )

    with pytest.raises(RuntimeError, match='refusing to mark this run DONE'):
        campaign._write_contract(
            tmp_path,
            args=args,
            preset=preset,
            density=np.ones((4, 4)),
            raw=None,
            scaffold=None,
            ds=None,
            summary={'_t0': time.perf_counter()},
            started=time.time(),
            status='ok',
        )

    record = json.loads((tmp_path / 'run.json').read_text())
    assert record['exit_status'] == 'incomplete_final_grid'
    assert record['reached_final_grid'] is False
    assert record['presentation_source'] == 'physical_density'
    assert record['presentation_shape'] == [512, 512]
    assert record['presentation_resampling'] == (
        'torch-bilinear-antialias-before-clamp'
    )
    assert not (tmp_path / 'DONE').exists()


def test_hybrid_contract_preserves_full_progress_gif(tmp_path, monkeypatch):
    """Contract writing must not replace the optimizer GIF with two frames."""
    monkeypatch.setattr(campaign, 'save_design_arrays', lambda *args, **kwargs: None)
    monkeypatch.setattr(campaign, 'save_field_png', lambda *args, **kwargs: None)
    monkeypatch.setattr(
        campaign,
        'write_progress_gif',
        lambda *args, **kwargs: pytest.fail('progress GIF was overwritten'),
    )
    monkeypatch.setattr(campaign, 'git_revision', lambda: 'test-sha')
    monkeypatch.setattr(campaign, 'pip_freeze_hash', lambda: 'test-env')

    progress = tmp_path / 'progress.gif'
    progress.write_bytes(b'full-trajectory')
    args = SimpleNamespace(
        run_id='H1/tall/fern_fronds/hybrid',
        experiment='H1',
        group='hybrid',
        mode='hybrid',
        structure='tall',
        prompt_sketch=False,
    )
    preset = SimpleNamespace(
        height=4,
        width=4,
        clip_prompt='fern fronds',
        sketch_path=None,
    )

    campaign._write_contract(
        tmp_path,
        args=args,
        preset=preset,
        density=np.ones((4, 4)),
        raw=np.ones((4, 4)),
        scaffold=np.zeros((4, 4)),
        ds=None,
        summary={'_t0': time.perf_counter()},
        started=time.time(),
        status='ok',
    )

    assert progress.read_bytes() == b'full-trajectory'
    assert (tmp_path / 'DONE').exists()
    record = json.loads((tmp_path / 'run.json').read_text())
    assert record['presentation_source'] == 'final_design_raw'
    assert record['presentation_shape'] == [512, 512]
    assert record['presentation_resampling'] == (
        'torch-bilinear-antialias-before-clamp'
    )
    assert (tmp_path / 'final.png').is_file()
    assert not (tmp_path / 'semantic_design.png').exists()


def _tiny_semantic_model():
    from model.model_ada import AdaptivePixelModel
    from problem.problems import StructuralParams

    model = AdaptivePixelModel(
        structural_params=StructuralParams(
            problem_name='multistory_building', width=16, height=32,
            density=0.3, interval=8, filter_width=1.5),
        clip_loss=None, seed=0, resize_num=0, resize_scale=2)
    object.__setattr__(model, 'clip_loss', lambda field: field.mean() * 1.0e5)
    model.enable_physical_clip(weight=0.0, match_venice=False, as_semantic=True)
    return model


def test_fixed_weights_reach_the_physics_optimizer():
    from dataclasses import replace
    from recipe.preset import PAPER

    preset = replace(PAPER, max_iterations=2, clip_weight=7.0, clip_weight_z=11.0)
    ds = campaign._run_physics(_tiny_semantic_model(), preset)
    np.testing.assert_array_equal(ds['clip_weight'].values, 7.0)
    assert 'blend_clip_raw_z_weight' not in ds
    assert campaign._mean(ds, 'clip_weight') == 7.0


def test_zero_raw_z_mixer_drops_only_the_raw_z_term():
    from dataclasses import replace
    from recipe.preset import PAPER

    ds = campaign._run_physics(
        _tiny_semantic_model(), replace(PAPER, max_iterations=2, blend_rho_z=0.0))
    assert ds.attrs['blend_mode'] == 'grad_match'
    assert campaign._mean(ds, 'blend_clip_raw_z_weight') is None
    assert campaign._mean(ds, 'clip_weight') > 0.0


def test_grad_match_logs_the_raw_z_weight_c2_calibrates_from():
    from dataclasses import replace
    from recipe.preset import PAPER

    ds = campaign._run_physics(
        _tiny_semantic_model(), replace(PAPER, max_iterations=2))
    assert ds.attrs['blend_mode'] == 'grad_match'
    assert campaign._mean(ds, 'blend_clip_raw_z_weight') > 0.0
    assert campaign._mean(ds, 'clip_weight') > 0.0
