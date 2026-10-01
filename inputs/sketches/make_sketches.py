"""Generate the geometric column sketches for the tall building.

The canvas is 512x1024, four pixels per element of the 128x256 domain. Floor
lines sit on the loaded rows of ``multistory_building`` (interval 64), so the
drawn floors and the physical floors coincide. Ink is black on white, matching
the hand-drawn corpus.

``col3`` and ``col6_grid`` keep half-bay overhangs (columns centered in equal
bays). ``col3_braced`` puts the outer columns flush with the domain edges so
the braces span the full width.

Run from anywhere: ``python inputs/sketches/make_sketches.py``.
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw

WIDTH, HEIGHT = 512, 1024
PX_PER_ELEMENT = WIDTH // 128
LINE = 5 * PX_PER_ELEMENT
FLOORS = (0.0, 0.25, 0.5, 0.75)
OUT = Path(__file__).resolve().parent


def column_centers(count: int, *, edge: bool = False) -> list[float]:
    """Column centerlines.

    Default: even bays with a half-bay overhang at each side. ``edge=True``
    puts the outer strokes on the canvas border (center at ``LINE/2`` and
    ``WIDTH - LINE/2``).
    """
    if count < 1:
        raise ValueError(f'count must be >= 1, got {count}')
    if not edge:
        return [WIDTH * (i + 0.5) / count for i in range(count)]
    if count == 1:
        return [WIDTH / 2.0]
    span = WIDTH - LINE
    return [LINE / 2.0 + i * span / (count - 1) for i in range(count)]


def floor_ys() -> list[float]:
    """Centerline of each floor band; the roof band sits flush with the top."""
    return [LINE / 2 if f == 0.0 else f * HEIGHT for f in FLOORS]


def frame(count: int, *, edge: bool = False) -> tuple[Image.Image, ImageDraw.ImageDraw]:
    """Columns from roof to ground plus the four floor bands."""
    image = Image.new('L', (WIDTH, HEIGHT), 255)
    draw = ImageDraw.Draw(image)
    for x in column_centers(count, edge=edge):
        draw.rectangle([x - LINE / 2, 0, x + LINE / 2, HEIGHT], fill=0)
    for y in floor_ys():
        draw.rectangle([0, y - LINE / 2, WIDTH, y + LINE / 2], fill=0)
    return image, draw


def braced(count: int) -> Image.Image:
    """Edge-flush columns plus one diagonal per bay per storey, alternating."""
    image, draw = frame(count, edge=True)
    xs = column_centers(count, edge=True)
    levels = floor_ys() + [HEIGHT]
    for storey, (top, bottom) in enumerate(zip(levels[:-1], levels[1:])):
        for bay, (left, right) in enumerate(zip(xs[:-1], xs[1:])):
            if (storey + bay) % 2 == 0:
                start, end = (left, top), (right, bottom)
            else:
                start, end = (right, top), (left, bottom)
            draw.line([start, end], fill=0, width=LINE)
    return image


def main() -> None:
    frame(6)[0].save(OUT / 'col6_grid.png')
    frame(3)[0].save(OUT / 'col3.png')
    braced(3).save(OUT / 'col3_braced.png')


if __name__ == '__main__':
    main()
