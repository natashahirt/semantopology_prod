"""Wiring for the paper entry point. Does not run CLIP or the solver."""

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from figures import save_field_png, write_comparison, write_progress_gif
from language.interpret import interpret_motive
from recipe.preset import PAPER, prompt_slug
from run import resolve_prompt


def test_paper_preset_matches_the_skeleton_recipe():
    assert PAPER.problem_name == 'multistory_building'
    assert (PAPER.width, PAPER.height, PAPER.density) == (128, 256, 0.3)
    assert PAPER.control_height == 32
    assert PAPER.control_width == 16
    assert PAPER.resize_num == 2
    assert PAPER.resize_scale == 2
    assert PAPER.max_iterations == 200
    assert PAPER.blend_rho == 1.0
    assert PAPER.blend_rho_z == 0.75
    assert PAPER.coadapt is True
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
    frames = np.stack([field, 1.0 - field])
    comparison = write_comparison(
        tmp_path / 'comparison.png',
        [('Physical density', field), ('Dream scaffold', 1.0 - field)],
    )
    gif = write_progress_gif(tmp_path / 'progress.gif', frames)
    assert comparison.is_file() and comparison.stat().st_size > 0
    assert gif.is_file() and gif.stat().st_size > 0
    with Image.open(gif) as image:
        assert image.size == (8, 8)
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


def test_hardfork_style_semantic_render_is_512_by_1024(tmp_path: Path):
    field = np.ones((256, 128), dtype=np.float64)
    path = save_field_png(
        tmp_path / 'semantic.png',
        field,
        max_edge=1024,
        smooth=True,
    )
    with Image.open(path) as image:
        assert image.size == (512, 1024)
