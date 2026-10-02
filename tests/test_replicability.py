"""Wiring for the paper entry point. Does not run CLIP or the solver."""

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from figures import (
    native_raw_frames,
    save_field_png,
    save_sharp_ink_png,
    sharp_ink_shape,
    write_comparison,
    write_progress_gif,
)
from language.interpret import interpret_motive
from recipe.preset import PAPER, prompt_slug
from run import resolve_prompt


def test_paper_preset_matches_the_skeleton_recipe():
    assert PAPER.problem_name == 'multistory_building'
    assert (PAPER.width, PAPER.height, PAPER.density) == (128, 256, 0.3)
    assert PAPER.interval == 64
    assert PAPER.filter_width == 2.0
    assert PAPER.penal == 3.0
    assert PAPER.control_height == 32
    assert PAPER.control_width == 16
    assert PAPER.resize_num == 2
    assert PAPER.resize_scale == 2
    assert PAPER.seed == 12
    assert PAPER.init_noise_amp == 0.01
    assert PAPER.union_load_sites is False
    assert PAPER.clip_model_name == 'ViT-B/32'
    assert PAPER.clip_rn_model_name == 'RN50'
    assert PAPER.clip_prompt == 'fern fronds'  # Intentional wording change.
    assert PAPER.num_augs == 32
    assert PAPER.clip_resize_short_side == 512
    assert PAPER.clip_alpha == 10.0
    assert PAPER.compliance_weight == 1.0
    assert PAPER.lr == 0.2
    assert PAPER.max_iterations == 200
    assert PAPER.resize_threshold == 0.5
    assert PAPER.max_resize_iteration == 50
    assert PAPER.convergence_threshold == 0.05
    assert PAPER.dream_steps == 64
    assert PAPER.dream_lr == 0.2
    assert PAPER.blend_rho == 1.0
    assert PAPER.blend_rho_z == 0.75
    assert PAPER.coadapt is True
    assert PAPER.coadapt_interval == 5
    assert PAPER.coadapt_until == 0.6
    assert PAPER.coadapt_release is False
    assert PAPER.mask_lr == 0.05
    assert PAPER.ema_decay == 0.9
    assert PAPER.mask_clip_weight == 1.0
    assert PAPER.saliency_weight == 1.0
    assert PAPER.overlap_weight == 1.0
    assert PAPER.area_weight == 10.0
    assert PAPER.anchor_weight == 1.0
    assert PAPER.deficit_weight == 0.0
    assert PAPER.physical_clip_projection_beta_max == 8.0
    assert PAPER.physical_clip_projection_sigma == 2.0
    assert PAPER.physical_clip_projection_sigma_end == 0.5
    assert PAPER.tiled_scales == ()
    assert PAPER.device == 'cpu'
    assert PAPER.sketch_weight_start == 4000.0
    assert PAPER.sketch_weight_end == 400.0


def test_prompt_slug_and_clip_passthrough():
    assert prompt_slug('fern fronds') == 'fern_fronds'
    assert resolve_prompt('butterfly wing venation', None) == 'butterfly wing venation'
    with pytest.raises(ValueError):
        resolve_prompt(None, None)
    with pytest.raises(ValueError):
        resolve_prompt('ferns', 'a building like ferns')


def test_interpret_returns_motive_only():
    assert interpret_motive(
        'a tall building like unfurling ferns',
        complete=lambda _sentence: 'Image motive: unfurling ferns',
    ) == 'unfurling ferns'


def test_figures_write_strip_and_gif(tmp_path: Path):
    field = np.linspace(0.0, 1.0, 16, dtype=np.float64).reshape(4, 4)
    frames = [field, 1.0 - field]
    comparison = write_comparison(
        tmp_path / 'comparison.png',
        [('Physical density', field), ('Dream scaffold', 1.0 - field)],
    )
    gif = write_progress_gif(tmp_path / 'progress.gif', frames)
    assert comparison.is_file() and comparison.stat().st_size > 0
    assert gif.is_file() and gif.stat().st_size > 0
    with Image.open(comparison) as image:
        assert image.size == (2416, 1232)
        comparison_pixels = np.asarray(image)
        assert comparison_pixels[32, 0] == 255
        assert comparison_pixels[32, 1216] == 0
    with Image.open(gif) as image:
        assert image.size == (512, 512)
        assert image.n_frames == 2
        assert image.info['duration'] == 50


def test_presentation_png_can_be_enlarged_without_changing_array_resolution(
        tmp_path: Path):
    field = np.ones((72, 448), dtype=np.float64)
    path = save_field_png(tmp_path / 'bridge.png', field, scale=4)
    with Image.open(path) as image:
        assert image.size == (1792, 288)


def test_smooth_presentation_png_fits_max_edge(tmp_path: Path):
    field = np.linspace(0.0, 1.0, 32, dtype=np.float64).reshape(8, 4)
    path = save_field_png(
        tmp_path / 'tall.png',
        field,
        max_edge=2400,
        smooth=True,
    )
    with Image.open(path) as image:
        assert image.size == (1200, 2400)


def test_hardfork_style_sharp_ink_render_is_512_by_1024(tmp_path: Path):
    field = np.linspace(-2.0, 3.0, 256 * 128).reshape(256, 128)
    path = save_sharp_ink_png(tmp_path / 'final.png', field)
    with Image.open(path) as image:
        assert image.size == (512, 1024)
    assert sharp_ink_shape(256, 128) == (1024, 512)
    assert sharp_ink_shape(150, 300) == (512, 1024)


def test_gif_frames_from_every_stage_share_the_sharp_ink_size(tmp_path: Path):
    frames = [np.zeros((64, 32)), np.full((128, 64), 0.5), np.ones((256, 128))]
    gif = write_progress_gif(tmp_path / 'progress.gif', frames)
    with Image.open(gif) as image:
        assert image.size == (512, 1024)
        assert image.n_frames == 3


def test_native_raw_frames_recover_each_stage_grid_exactly():
    import xarray

    coarse = np.arange(8, dtype=np.float32).reshape(4, 2)
    fine = np.arange(32, dtype=np.float32).reshape(8, 4)
    stack = np.stack([np.repeat(np.repeat(coarse, 2, 0), 2, 1), fine])
    ds = xarray.Dataset({
        'design_raw': (('step', 'y', 'x'), stack),
        'design_raw_height': (('step',), [4, 8]),
        'design_raw_width': (('step',), [2, 4]),
    })
    recovered = native_raw_frames(ds)
    np.testing.assert_array_equal(recovered[0], coarse)
    np.testing.assert_array_equal(recovered[1], fine)


def test_sharp_ink_render_resizes_before_clamping(tmp_path: Path):
    from guidance.loss_clip import _resize_short_side
    import torch

    field = np.array([[-2.0, 2.0], [2.0, -2.0]], dtype=np.float32)
    path = save_sharp_ink_png(
        tmp_path / 'final.png',
        field,
        short_edge=8,
    )
    resized = _resize_short_side(
        torch.as_tensor(field)[None, None],
        8,
    ).clamp(0.0, 1.0)
    expected = (
        255.0 * (1.0 - resized[0, 0].numpy())
    ).clip(0, 255).astype(np.uint8)
    with Image.open(path) as image:
        np.testing.assert_array_equal(np.asarray(image), expected)
