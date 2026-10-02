"""Paper renders, comparison strip, and progress GIF for a run.

Material is drawn black, matching the paper figures. ``final.png`` and every
GIF frame use the sharp-ink render of the raw design (see
:func:`sharp_ink_image`); ``physical_density.png`` keeps the native FE grid.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
from PIL import Image, ImageDraw

SHARP_INK_SHORT_EDGE = 512
SHARP_INK_RESAMPLING = 'torch-bilinear-antialias-before-clamp'
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


def sharp_ink_shape(
    height: int,
    width: int,
    short_edge: int = SHARP_INK_SHORT_EDGE,
) -> tuple[int, int]:
    """``(height, width)`` of :func:`sharp_ink_image` for a field of this size.

    Mirrors ``_resize_short_side``, including its truncation of the long edge.
    """
    height, width, short_edge = int(height), int(width), int(short_edge)
    if min(height, width) == short_edge:
        return height, width
    if width <= height:
        return int(short_edge * height / width), short_edge
    return short_edge, int(short_edge * width / height)


def sharp_ink_image(
    raw_field: np.ndarray,
    *,
    short_edge: int = SHARP_INK_SHORT_EDGE,
) -> Image.Image:
    """Hardfork's raw-design display: resize, *then* clamp, then invert.

    Torch's bilinear, antialiased short-edge resize runs on the unbounded
    design so values beyond ``[0, 1]`` sharpen the ink edges before the clamp.
    Clipping first gives softer transition bands and is not equivalent.
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
    return Image.fromarray(ink, mode='L')


def save_sharp_ink_png(
    path: Path,
    raw_field: np.ndarray,
    *,
    short_edge: int = SHARP_INK_SHORT_EDGE,
) -> Path:
    """Write :func:`sharp_ink_image` of ``raw_field`` to ``path``."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sharp_ink_image(raw_field, short_edge=short_edge).save(path)
    return path


def native_raw_frames(ds) -> list[np.ndarray]:
    """Per-step raw design on the grid each step was optimized on.

    ``design_raw`` is block-repeated to the final grid; striding by the
    repeat factor recovers every stage's native field exactly.
    """
    stack = np.asarray(ds['design_raw'].values)
    heights = np.asarray(ds['design_raw_height'].values).astype(int)
    widths = np.asarray(ds['design_raw_width'].values).astype(int)
    full_height, full_width = stack.shape[-2:]
    return [
        frame[::full_height // height, ::full_width // width]
        for frame, height, width in zip(stack, heights, widths)
    ]


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
        if 'semantic' in str(title).lower() or 'raw z' in str(title).lower():
            image = sharp_ink_image(field)
        else:
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
    raw_frames: Sequence[np.ndarray],
    *,
    duration_ms: int = 50,
    short_edge: int = SHARP_INK_SHORT_EDGE,
) -> Path:
    """Write one sharp-ink frame per recorded physics step.

    ``raw_frames`` may mix grids (one per AdaptivePixel stage); every frame
    is rendered to the same short edge, so the animation keeps one size.
    """
    if len(raw_frames) < 1:
        raise ValueError('progress GIF needs at least one frame')
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frames = [
        sharp_ink_image(frame, short_edge=short_edge).convert('P')
        for frame in raw_frames
    ]
    frames[0].save(
        path,
        save_all=True,
        append_images=frames[1:],
        duration=int(duration_ms),
        loop=0,
        optimize=False,
    )
    return path
