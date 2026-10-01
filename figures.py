"""Comparison strip and progress GIF for a dream-layout run.

Material is drawn black, matching the paper figures. No frozen Venice
reference image is required.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image


def _plane(field: np.ndarray) -> np.ndarray:
    arr = np.asarray(field, dtype=np.float64)
    while arr.ndim > 2:
        arr = arr[0]
    if arr.ndim != 2:
        raise ValueError(f'expected a 2-D field, got shape {np.asarray(field).shape}')
    return arr


def ink_image(field: np.ndarray) -> Image.Image:
    """Grayscale image with material black. ``field`` is clipped to [0, 1]."""
    arr = np.clip(_plane(field), 0.0, 1.0)
    ink = (255.0 * (1.0 - arr)).astype(np.uint8)
    return Image.fromarray(ink, mode='L')


def save_field_png(path: Path, field: np.ndarray) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ink_image(field).save(path)
    return path


def write_comparison(path: Path, panels: list[tuple[str, np.ndarray]]) -> Path:
    """Write a labeled horizontal strip. Each panel is a [0, 1] field."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    if not panels:
        raise ValueError('comparison needs at least one panel')
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(
        1, len(panels), figsize=(3.4 * len(panels), 7.2), squeeze=False)
    for ax, (title, field) in zip(axes[0], panels):
        arr = np.clip(_plane(field), 0.0, 1.0)
        ax.imshow(1.0 - arr, cmap='gray', vmin=0.0, vmax=1.0, interpolation='nearest')
        ax.set_title(title, fontsize=10)
        ax.axis('off')
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def write_progress_gif(
    path: Path,
    design: np.ndarray,
    *,
    duration_ms: int = 80,
) -> Path:
    """GIF of the physics run from the rendered design stack.

    ``design`` is ``(step, y, x)`` or ``(step, 1, y, x)``, already on the
    common final grid. The dream loop does not record frames.
    """
    arr = np.asarray(design, dtype=np.float64)
    if arr.ndim == 4 and arr.shape[1] == 1:
        arr = arr[:, 0]
    if arr.ndim != 3:
        raise ValueError(f'expected a step stack, got shape {arr.shape}')
    if arr.shape[0] < 1:
        raise ValueError('progress GIF needs at least one frame')
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frames = [ink_image(frame).convert('P') for frame in arr]
    frames[0].save(
        path,
        save_all=True,
        append_images=frames[1:],
        duration=int(duration_ms),
        loop=0,
        optimize=False,
    )
    return path
