"""Post-hoc evaluator wiring, with a stub CLIP. Never loads real weights."""

from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from analysis import evaluate
from recipe.campaign_spec import EVAL_MODELS, eval_model_suffix


class _StubClip:
    """Embeds an image as its mean intensity per channel plus a model offset."""

    def __init__(self, offset: float):
        self.visual = SimpleNamespace(input_resolution=32)
        self.offset = offset

    def encode_image(self, batch: torch.Tensor) -> torch.Tensor:
        return batch.mean(dim=(2, 3)) + self.offset


def _write_attempt(root, run_id, value):
    attempt = root / run_id / 'attempt_1'
    attempt.mkdir(parents=True)
    np.save(attempt / 'physical_density.npy', np.full((16, 8), value, np.float32))
    (attempt / 'run.json').write_text(json.dumps(
        {'run_id': run_id, 'prompt': 'fern fronds', 'structure': 'tall',
         'experiment': run_id.split('/')[0]}))
    (attempt / 'DONE').write_text('')


def test_suffixes_keep_the_primary_names():
    assert EVAL_MODELS[0] == 'ViT-B/32'
    assert eval_model_suffix('ViT-B/32') == ''
    assert eval_model_suffix('ViT-L/14') == '_vit_l_14'


def test_every_evaluator_writes_aligned_outputs(tmp_path, monkeypatch):
    offsets = {'ViT-B/32': 0.0, 'ViT-L/14': 1.0}
    monkeypatch.setattr(
        evaluate, '_load_clip_model',
        lambda name, device: (_StubClip(offsets[name]), None))
    monkeypatch.setattr(
        evaluate, '_encode_texts',
        lambda model, texts, device: torch.nn.functional.normalize(
            torch.ones(len(texts), 3), dim=-1))
    results = tmp_path / 'results'
    _write_attempt(results, 'S3/tall/fern_fronds', 0.2)
    _write_attempt(results, 'C1/tall/structure', 0.8)
    out = tmp_path / 'evaluate'

    assert evaluate.main(['--results', str(results), '--out', str(out)]) == 0

    primary = json.loads((out / 'similarities.json').read_text())
    second = json.loads((out / 'similarities_vit_l_14.json').read_text())
    assert [r['run_id'] for r in primary] == [r['run_id'] for r in second]
    assert {r['evaluator'] for r in primary} == {'ViT-B/32'}
    assert {r['evaluator'] for r in second} == {'ViT-L/14'}
    assert set(primary[0]['similarities']) == set(evaluate.EVAL_PROMPTS)
    assert {'bracken', 'tree branches', 'brick wall'} <= set(evaluate.EVAL_PROMPTS)
    for record in primary:
        assert record['own_prompt_similarity'] == pytest.approx(
            record['similarities']['fern fronds'])
    assert np.load(out / 'embeddings.npy').shape == (2, 3)
    assert np.load(out / 'embeddings_vit_l_14.npy').shape == (2, 3)
    assert primary[0]['attempt'].startswith(str(results))
