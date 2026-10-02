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


def save_field_png(
    path: Path,
    field: np.ndarray,
    *,
    scale: int = 1,
) -> Path:
    """Save a density field, optionally enlarged without inventing detail."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    image = ink_image(field)
    if int(scale) < 1:
        raise ValueError(f'scale must be >= 1, got {scale}')
    if int(scale) > 1:
        image = image.resize(
            (image.width * int(scale), image.height * int(scale)),
            resample=Image.Resampling.NEAREST,
        )
    image.save(path)
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
    sample = _plane(panels[0][1])
    aspect = sample.shape[0] / sample.shape[1]
    if aspect >= 1.0:
        panel_width = 3.4
        panel_height = min(8.0, max(3.0, panel_width * aspect))
    else:
        panel_width = 7.2
        panel_height = max(2.4, panel_width * aspect + 0.8)
    fig, axes = plt.subplots(
        1,
        len(panels),
        figsize=(panel_width * len(panels), panel_height),
        squeeze=False,
    )
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
    duration_ms: int = 50,
    scale: int = 2,
) -> Path:
    """Write a smooth, enlarged GIF with one frame per recorded physics step.

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
    if int(scale) < 1:
        raise ValueError(f'scale must be >= 1, got {scale}')
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frames = []
    for frame in arr:
        image = ink_image(frame)
        if int(scale) > 1:
            image = image.resize(
                (image.width * int(scale), image.height * int(scale)),
                resample=Image.Resampling.NEAREST,
            )
        frames.append(image.convert('P'))
    frames[0].save(
        path,
        save_all=True,
        append_images=frames[1:],
        duration=int(duration_ms),
        loop=0,
        optimize=False,
    )
    return path
