"""Vendi score and quality-weighted Vendi on saved densities."""

from __future__ import annotations

import numpy as np


def vendi(kernel: np.ndarray) -> float:
    """Vendi score from an n x n PSD similarity matrix with unit diagonal."""
    kernel = np.asarray(kernel, dtype=np.float64)
    if kernel.shape[0] == 0:
        return float('nan')
    eigenvalues = np.linalg.eigvalsh(kernel / kernel.shape[0])
    eigenvalues = eigenvalues[eigenvalues > 1e-12]
    return float(np.exp(-np.sum(eigenvalues * np.log(eigenvalues))))


def tanimoto_kernel(binary: np.ndarray) -> np.ndarray:
    """``binary`` is (n, d) {0,1}."""
    inter = binary @ binary.T
    sizes = binary.sum(1)
    denom = sizes[:, None] + sizes[None, :] - inter
    denom = np.maximum(denom, 1e-12)
    return inter / denom


def cosine_kernel(embeddings: np.ndarray) -> np.ndarray:
    vectors = np.asarray(embeddings, dtype=np.float64)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    unit = vectors / norms
    return unit @ unit.T


def quality_weighted_vendi(kernel: np.ndarray, quality: np.ndarray) -> float:
    """Quality-weighted Vendi: q_i q_j K_ij, then Vendi.

    ``quality`` is C_unguided / C, clipped to be non-negative.
    """
    quality = np.clip(np.asarray(quality, dtype=np.float64), 0.0, None)
    if quality.sum() <= 0:
        return float('nan')
    weighted = kernel * np.outer(quality, quality)
    diag = np.sqrt(np.clip(np.diag(weighted), 0.0, None))
    diag = np.maximum(diag, 1e-12)
    normalized = weighted / np.outer(diag, diag)
    np.fill_diagonal(normalized, 1.0)
    return vendi(normalized)


def downsample_binary(density: np.ndarray, shape=(32, 64)) -> np.ndarray:
    field = np.asarray(density, dtype=np.float64)
    while field.ndim > 2:
        field = field[0]
    # Nearest block average then threshold. Shape is (y, x) = (height, width).
    out_h, out_w = shape
    y_idx = (np.linspace(0, field.shape[0], out_h + 1)).astype(int)
    x_idx = (np.linspace(0, field.shape[1], out_w + 1)).astype(int)
    blocks = np.zeros((out_h, out_w), dtype=np.float64)
    for i in range(out_h):
        for j in range(out_w):
            blocks[i, j] = field[y_idx[i]:max(y_idx[i + 1], y_idx[i] + 1),
                                 x_idx[j]:max(x_idx[j + 1], x_idx[j] + 1)].mean()
    return (blocks >= 0.5).astype(np.float64)
