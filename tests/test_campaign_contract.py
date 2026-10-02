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
        raw=None,
        scaffold=np.zeros((4, 4)),
        ds=None,
        summary={'_t0': time.perf_counter()},
        started=time.time(),
        status='ok',
    )

    assert progress.read_bytes() == b'full-trajectory'
    assert (tmp_path / 'DONE').exists()
