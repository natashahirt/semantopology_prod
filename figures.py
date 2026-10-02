"""Comparison strip and progress GIF for a dream-layout run.

Material is drawn black, matching the paper figures. No frozen Venice
reference image is required.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

PRESENTATION_MAX_EDGE = 2400
SEMANTIC_SHORT_EDGE = 512


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
    max_edge: int | None = None,
    smooth: bool = False,
) -> Path:
    """Save a density field as either an exact or presentation rendering.

    ``scale`` with nearest-neighbour resampling exposes the finite-element
    pixels. ``max_edge`` fits the image to a publication-size box while
    preserving its aspect ratio; ``smooth=True`` mirrors the antialiased
    enlargement used by the legacy Venice figures. The native array remains
    the scientific result.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    image = ink_image(field)
    if int(scale) < 1:
        raise ValueError(f'scale must be >= 1, got {scale}')
    if max_edge is not None and int(max_edge) < 1:
        raise ValueError(f'max_edge must be >= 1, got {max_edge}')
    if max_edge is not None:
        factor = float(max_edge) / float(max(image.size))
        output_size = (
            max(1, int(round(image.width * factor))),
            max(1, int(round(image.height * factor))),
        )
        image = image.resize(
            output_size,
            resample=(
                Image.Resampling.BILINEAR
                if smooth else Image.Resampling.NEAREST
            ),
        )
    elif int(scale) > 1:
        image = image.resize(
            (image.width * int(scale), image.height * int(scale)),
            resample=(
                Image.Resampling.BILINEAR
                if smooth else Image.Resampling.NEAREST
            ),
        )
    image.save(path)
    return path


def save_semantic_design_png(
    path: Path,
    raw_field: np.ndarray,
    *,
    short_edge: int = SEMANTIC_SHORT_EDGE,
) -> Path:
    """Reproduce the hardfork raw-z display path exactly.

    Hardfork resized the unbounded design parameter with Torch's bilinear,
    antialiased short-edge transform *before* clamping and inverting it.
    Clipping first creates softer transition bands and is not equivalent.
    """
    import torch

    from guidance.loss_clip import _resize_short_side

    if int(short_edge) < 1:
        raise ValueError(f'short_edge must be >= 1, got {short_edge}')
    raw = np.ascontiguousarray(_plane(raw_field), dtype=np.float32)
    resized = _resize_short_side(
        torch.as_tensor(raw)[None, None],
        int(short_edge),
    ).clamp(0.0, 1.0)
    ink = (
        255.0 * (1.0 - resized[0, 0].detach().cpu().numpy())
    ).clip(0, 255).astype(np.uint8)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(ink, mode='L').save(path)
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
                resample=Image.Resampling.BILINEAR,
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
