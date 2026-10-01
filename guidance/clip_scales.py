"""Structure-specific CLIP crops for the paper scale ablation.

The existing ``physical_scale_boxes`` path in ``loss_clip`` is a random-free
grid of elevation-fraction windows. This module is a *second* path: one
centered square tile per tall-building storey, two centered square tiles per
short-building storey, and full-depth square panels for the bridge.

Coordinates are in element space of the design grid ``(height, width)``,
with y = 0 at the top. CLIP sampling scales them onto whatever image size
the encoder currently sees.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import torch


SCALE_KEYS = ('g', 'm', 'e')


def parse_clip_scales(spec: str | Sequence[str] | None) -> tuple[str, ...]:
    """Parse ``g,m,e`` / ``('m',)`` into a unique ordered tuple of keys."""
    if spec is None or spec == '' or spec == ():
        return ()
    if isinstance(spec, str):
        parts = [p.strip().lower() for p in spec.replace('+', ',').split(',')]
    else:
        parts = [str(p).strip().lower() for p in spec]
    keys: list[str] = []
    for part in parts:
        if not part:
            continue
        if part in ('gme', 'all'):
            for key in SCALE_KEYS:
                if key not in keys:
                    keys.append(key)
            continue
        if part not in SCALE_KEYS:
            raise ValueError(
                f'unknown CLIP scale {part!r}; expected g, m, e '
                '(global / module / element)')
        if part not in keys:
            keys.append(part)
    return tuple(keys)


def scales_token(scales: Sequence[str]) -> str:
    """Filesystem token: ``g``, ``m``, ``e``, or ``gme``."""
    keys = parse_clip_scales(scales)
    if not keys:
        return 'none'
    if keys == SCALE_KEYS:
        return 'gme'
    return ''.join(keys)


@dataclass(frozen=True)
class TileLayout:
    """Module/element windows on one structure, in element pixels.

    ``module`` / ``element`` are ``(N, 4)`` arrays of ``(cx, cy, w, h)``.
    Global is the letterboxed full frame, not a box here.
    """

    height: int
    width: int
    kind: str
    module: torch.Tensor
    element: torch.Tensor
    module_width: int
    module_height: int
    element_side: int

    def boxes_for(self, key: str) -> torch.Tensor:
        if key == 'm':
            return self.module
        if key == 'e':
            return self.element
        raise KeyError(f'{key!r} is not a tiled scale (g is letterboxed)')


def _centers_1d(length: int, side: int) -> list[float]:
    """Non-overlapping square centers along one axis, leftover split at the ends."""
    count = max(1, int(length) // int(side))
    used = count * side
    pad = (int(length) - used) / 2.0
    return [pad + (i + 0.5) * side for i in range(count)]


def _boxes_from_centers(
    xs: Iterable[float],
    ys: Iterable[float],
    width: float,
    height: float,
) -> torch.Tensor:
    cx, cy, w, h = [], [], [], []
    for y in ys:
        for x in xs:
            cx.append(float(x))
            cy.append(float(y))
            w.append(float(width))
            h.append(float(height))
    return torch.tensor([cx, cy, w, h], dtype=torch.float32).T.contiguous()


def building_storey_centers(height: int, interval: int) -> list[float]:
    """Vertical centers of each storey band ``[k*interval, (k+1)*interval)``."""
    if interval < 1:
        raise ValueError(f'interval must be >= 1, got {interval}')
    n_storeys = max(1, int(height) // int(interval))
    return [(k + 0.5) * interval for k in range(n_storeys)]


def layout_for_structure(
    height: int,
    width: int,
    *,
    kind: str,
    interval: int,
) -> TileLayout:
    """Square storey/panel modules plus four element tiles per module.

    Building module side equals one storey height. A building no wider than
    two storey heights (the tall case) gets one centered square per storey;
    wider buildings (the short case) get two centered squares per storey.
    Bridge modules remain full-depth squares tiled along the span.

    Element side is one quarter of the module side. Four element squares sit
    in the module quadrants.
    """
    height = int(height)
    width = int(width)
    if height < 1 or width < 1:
        raise ValueError(f'grid must be positive, got {width}x{height}')
    if kind == 'bridge':
        module_height = min(height, width)
        module_width = module_height
        ys = [height / 2.0]
        xs = _centers_1d(width, module_width)
    elif kind == 'building':
        module_height = max(1, min(int(interval), height))
        tiles_per_storey = 1 if width <= 2 * module_height else 2
        module_width = module_height
        ys = building_storey_centers(height, int(interval))
        xs = [
            (index + 0.5) * width / tiles_per_storey
            for index in range(tiles_per_storey)
        ]
    else:
        raise ValueError(f'kind must be building or bridge, got {kind!r}')
    module = _boxes_from_centers(
        xs, ys, width=module_width, height=module_height)

    element_side = max(1, min(module_width, module_height) // 4)
    offset_x = module_width / 4.0
    offset_y = module_height / 4.0
    elem_cx, elem_cy, elem_w, elem_h = [], [], [], []
    for row in range(module.shape[0]):
        mx = float(module[row, 0])
        my = float(module[row, 1])
        for dy in (-offset_y, offset_y):
            for dx in (-offset_x, offset_x):
                elem_cx.append(mx + dx)
                elem_cy.append(my + dy)
                elem_w.append(float(element_side))
                elem_h.append(float(element_side))
    element = torch.tensor(
        [elem_cx, elem_cy, elem_w, elem_h], dtype=torch.float32,
    ).T.contiguous()
    return TileLayout(
        height=height,
        width=width,
        kind=kind,
        module=module,
        element=element,
        module_width=module_width,
        module_height=module_height,
        element_side=element_side,
    )


def scale_boxes_to_image(
    boxes: torch.Tensor,
    *,
    src_height: int,
    src_width: int,
    dst_height: int,
    dst_width: int,
    device=None,
    dtype=None,
) -> torch.Tensor:
    """Map element-grid boxes onto an image of size ``(dst_height, dst_width)``."""
    out = boxes.to(device=device, dtype=dtype or boxes.dtype).clone()
    sx = float(dst_width) / float(src_width)
    sy = float(dst_height) / float(src_height)
    out[:, 0] = out[:, 0] * sx
    out[:, 1] = out[:, 1] * sy
    out[:, 2] = out[:, 2] * sx
    out[:, 3] = out[:, 3] * sy
    return out
