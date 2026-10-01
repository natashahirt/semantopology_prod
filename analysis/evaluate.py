"""Post-hoc CLIP evaluator. Never reuse training-time CLIP losses."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from guidance.loss_clip import (
    CLIP_PIXEL_MEAN,
    CLIP_PIXEL_STD,
    _encode_texts,
    _load_clip_model,
    letterbox_to_square,
)
from recipe.campaign_spec import COUNTER_PROMPT, PROMPTS

_REPO = Path(__file__).resolve().parents[1]
EVAL_PROMPTS = (*PROMPTS, COUNTER_PROMPT)


def _load_density(attempt: Path) -> np.ndarray | None:
    path = attempt / 'physical_density.npy'
    if not path.exists():
        return None
    field = np.load(path)
    while field.ndim > 2:
        field = field[0]
    return np.asarray(field, dtype=np.float32)


def _grid_crops(image: torch.Tensor, size: int, n: int = 3) -> torch.Tensor:
    """n x n square crops covering the letterboxed view."""
    _, _, height, width = image.shape
    side = min(height, width)
    ys = torch.linspace(side / 2.0, height - side / 2.0, n)
    xs = torch.linspace(side / 2.0, width - side / 2.0, n)
    crops = []
    for y in ys:
        for x in xs:
            top = int(round(float(y) - side / 2.0))
            left = int(round(float(x) - side / 2.0))
            crops.append(image[:, :, top:top + side, left:left + side])
    stacked = torch.cat(crops, dim=0)
    return F.interpolate(
        stacked, size=(size, size), mode='bilinear', align_corners=False)


def embed_density(density: np.ndarray, clip_model, device) -> np.ndarray:
    field = torch.from_numpy(np.clip(density, 0.0, 1.0)).float()
    image = field.view(1, 1, *field.shape[-2:]).repeat(1, 3, 1, 1).to(device)
    size = int(clip_model.visual.input_resolution)
    full = letterbox_to_square(image, size)
    crops = _grid_crops(image, size, n=3)
    batch = torch.cat([full, crops], dim=0)
    mean = torch.tensor(CLIP_PIXEL_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(CLIP_PIXEL_STD, device=device).view(1, 3, 1, 1)
    with torch.no_grad():
        z = clip_model.encode_image((batch - mean) / std).float()
        z = F.normalize(z, dim=-1)
        pooled = F.normalize(z.mean(dim=0, keepdim=True), dim=-1)
    return pooled.cpu().numpy()[0]


def walk_done(results: Path) -> list[Path]:
    return sorted(path.parent for path in results.glob('**/DONE'))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', default=str(_REPO / 'results'))
    parser.add_argument('--out', default=str(_REPO / 'analysis' / 'out' / 'evaluate'))
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    clip_model, _ = _load_clip_model('ViT-B/32', device)
    text = _encode_texts(clip_model, list(EVAL_PROMPTS), device)
    attempts = walk_done(Path(args.results))
    rows = []
    embeddings = []
    for attempt in attempts:
        density = _load_density(attempt)
        if density is None:
            continue
        vector = embed_density(density, clip_model, device)
        sims = (vector @ text.cpu().numpy().T).tolist()
        run_json = attempt / 'run.json'
        meta = json.loads(run_json.read_text()) if run_json.exists() else {}
        rows.append({
            'attempt': str(attempt.relative_to(_REPO)),
            'run_id': meta.get('run_id'),
            'prompt': meta.get('prompt'),
            'structure': meta.get('structure'),
            'experiment': meta.get('experiment'),
            'similarities': dict(zip(EVAL_PROMPTS, sims)),
        })
        embeddings.append(vector)
    if embeddings:
        np.save(out / 'embeddings.npy', np.stack(embeddings))
    (out / 'similarities.json').write_text(json.dumps(rows, indent=2) + '\n')
    print(f'evaluated {len(rows)} runs -> {out}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
