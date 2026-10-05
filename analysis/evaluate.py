"""Post-hoc CLIP evaluator. Never reuse training-time CLIP losses.

Every finished run is scored by each model in ``EVAL_MODELS`` on the same
deterministic crops (a letterboxed full frame plus a 3x3 grid of square
crops). ``ViT-B/32`` shares the training backbone but none of its
augmentation or loss; ``ViT-L/14`` is never loaded during training, so it
is the independent check.

Each model scores two images per run (``EVAL_VIEWS``): the physical density
the analysis solved (material light), and ``final.png``, the render of the raw
design z that the figures show (material dark). Scores are comparable within
a view, not across views: CLIP responds to polarity.
"""

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
from recipe.campaign_spec import (
    COUNTER_PROMPT,
    EVAL_MODELS,
    EVAL_VIEWS,
    LADDER_PROMPTS,
    PROMPTS,
    eval_model_suffix,
    eval_view_suffix,
)

_REPO = Path(__file__).resolve().parents[1]
# The L ladder runs fern fronds -> bracken -> tree branches -> lightning ->
# brick wall, so its rungs are scored as columns too.
LADDER = (PROMPTS[0], LADDER_PROMPTS[0], 'tree branches', *LADDER_PROMPTS[1:])
EVAL_PROMPTS = tuple(dict.fromkeys((*PROMPTS, COUNTER_PROMPT, *LADDER)))


def _load_density(attempt: Path) -> np.ndarray | None:
    path = attempt / 'physical_density.npy'
    if not path.exists():
        return None
    field = np.load(path)
    while field.ndim > 2:
        field = field[0]
    return np.asarray(field, dtype=np.float32)


def _load_render(attempt: Path) -> np.ndarray | None:
    """``final.png`` as a [0, 1] field, exactly as the figures show it."""
    from PIL import Image

    path = attempt / 'final.png'
    if not path.exists():
        return None
    with Image.open(path) as image:
        return np.asarray(image.convert('L'), dtype=np.float32) / 255.0


_VIEW_LOADERS = {'density': _load_density, 'z': _load_render}


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


def embed_field(field: np.ndarray, clip_model, device) -> np.ndarray:
    """Unit embedding of a [0, 1] grayscale field: full frame and 3x3 crops, pooled."""
    field = torch.from_numpy(np.clip(field, 0.0, 1.0)).float()
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


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(_REPO))
    except ValueError:
        return str(path)


def _load_runs(
        attempts: list[Path],
        view: str = 'density') -> list[tuple[Path, np.ndarray, dict]]:
    """(attempt, field, run.json) for every attempt that saved this view's image."""
    load = _VIEW_LOADERS[view]
    runs = []
    for attempt in attempts:
        field = load(attempt)
        if field is None:
            continue
        run_json = attempt / 'run.json'
        meta = json.loads(run_json.read_text()) if run_json.exists() else {}
        runs.append((attempt, field, meta))
    return runs


def score_runs(
        runs, clip_model, model_name: str, device,
        view: str = 'density') -> tuple[list[dict], list[np.ndarray]]:
    """Similarity rows and pooled embeddings for one evaluator model and view."""
    text = _encode_texts(clip_model, list(EVAL_PROMPTS), device).cpu().numpy()
    # Each run is also scored against its own prompt (P, N, L and M use
    # prompts outside EVAL_PROMPTS); every distinct prompt is encoded once.
    own_prompts = sorted({meta['prompt'] for _, _, meta in runs if meta.get('prompt')})
    own_text = dict(zip(own_prompts, (
        _encode_texts(clip_model, own_prompts, device).cpu().numpy()
        if own_prompts else [])))
    rows, embeddings = [], []
    for attempt, field, meta in runs:
        vector = embed_field(field, clip_model, device)
        prompt = meta.get('prompt')
        rows.append({
            'attempt': _display_path(attempt),
            'run_id': meta.get('run_id'),
            'prompt': prompt,
            'structure': meta.get('structure'),
            'experiment': meta.get('experiment'),
            'evaluator': model_name,
            'view': view,
            'similarities': dict(zip(EVAL_PROMPTS, (vector @ text.T).tolist())),
            'own_prompt_similarity': (
                float(vector @ own_text[prompt]) if prompt in own_text else None),
        })
        embeddings.append(vector)
    return rows, embeddings


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', default=str(_REPO / 'results'))
    parser.add_argument('--out', default=str(_REPO / 'analysis' / 'out' / 'evaluate'))
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--models', nargs='+', default=list(EVAL_MODELS))
    parser.add_argument('--views', nargs='+', default=list(EVAL_VIEWS),
                        choices=EVAL_VIEWS)
    args = parser.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    attempts = walk_done(Path(args.results))
    runs_by_view = {view: _load_runs(attempts, view) for view in args.views}
    for model_name in args.models:
        clip_model, _ = _load_clip_model(model_name, device)
        for view, runs in runs_by_view.items():
            rows, embeddings = score_runs(runs, clip_model, model_name, device, view)
            suffix = eval_model_suffix(model_name) + eval_view_suffix(view)
            if embeddings:
                np.save(out / f'embeddings{suffix}.npy', np.stack(embeddings))
            (out / f'similarities{suffix}.json').write_text(
                json.dumps(rows, indent=2) + '\n')
            print(f'{model_name} ({view}): evaluated {len(rows)} runs -> {out}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
