"""Comparison strip and progress GIF for a dream-layout run.

Material is drawn black, matching the paper figures. No frozen Venice
reference image is required.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

PRESENTATION_MAX_EDGE = 2400
SEMANTIC_SHORT_EDGE = 512
COMPARISON_PANEL_MAX_EDGE = 1200


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


def write_comparison(
    path: Path,
    panels: list[tuple[str, np.ndarray]],
    *,
    panel_max_edge: int = COMPARISON_PANEL_MAX_EDGE,
) -> Path:
    """Write an uncropped labeled strip with every panel fully visible."""
    if not panels:
        raise ValueError('comparison needs at least one panel')
    if int(panel_max_edge) < 1:
        raise ValueError(
            f'panel_max_edge must be >= 1, got {panel_max_edge}')
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    measure = ImageDraw.Draw(Image.new('L', (1, 1), 255))
    rendered = []
    for title, field in panels:
        image = ink_image(field)
        factor = float(panel_max_edge) / float(max(image.size))
        image = image.resize(
            (
                max(1, int(round(image.width * factor))),
                max(1, int(round(image.height * factor))),
            ),
            resample=Image.Resampling.NEAREST,
        )
        text_box = measure.textbbox((0, 0), str(title))
        text_width = text_box[2] - text_box[0]
        text_height = text_box[3] - text_box[1]
        slot_width = max(image.width, text_width + 16)
        rendered.append(
            (str(title), image, slot_width, text_box, text_width, text_height))

    gap = 16
    label_height = 32
    image_height = max(image.height for _, image, *_ in rendered)
    canvas = Image.new(
        'L',
        (
            sum(slot_width for _, _, slot_width, *_ in rendered)
            + gap * (len(rendered) - 1),
            label_height + image_height,
        ),
        255,
    )
    draw = ImageDraw.Draw(canvas)
    x = 0
    for title, image, slot_width, box, text_width, text_height in rendered:
        image_x = x + (slot_width - image.width) // 2
        canvas.paste(image, (image_x, label_height))
        text_x = x + (slot_width - text_width) // 2 - box[0]
        text_y = (label_height - text_height) // 2 - box[1]
        draw.text((text_x, text_y), title, fill=0)
        x += slot_width + gap
    canvas.save(path)
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
