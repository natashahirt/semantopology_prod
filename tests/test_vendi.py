"""Vendi helpers. No CLIP, no FEA."""

from __future__ import annotations

import numpy as np

from analysis.vendi import downsample_binary, tanimoto_kernel, vendi


def test_identical_designs_have_vendi_one():
    x = np.ones((4, 8))
    binary = np.stack([downsample_binary(x, (4, 8)).reshape(-1)] * 3)
    kernel = tanimoto_kernel(binary)
    np.testing.assert_allclose(np.diag(kernel), 1.0)
    assert abs(vendi(kernel) - 1.0) < 1e-6


def test_orthogonal_patterns_raise_vendi():
    a = np.zeros((8, 8))
    a[:4] = 1
    b = np.zeros((8, 8))
    b[:, :4] = 1
    binary = np.stack([
        downsample_binary(a, (8, 8)).reshape(-1),
        downsample_binary(b, (8, 8)).reshape(-1),
    ])
    assert vendi(tanimoto_kernel(binary)) > 1.0
