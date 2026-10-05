"""Contact sheets, dial curves, scale figure, prompt x structure grid."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from guidance.blend import GRAD_MATCH_WEIGHT_MAX
from recipe.campaign_spec import (
    CONTROL_PROMPTS,
    EVAL_MODELS,
    EVAL_VIEWS,
    FERN_WORDINGS,
    NONSENSE_PROMPTS,
    PROMPTS,
    SCRAMBLED_PROMPTS,
    eval_model_suffix,
    eval_view_suffix,
)
from recipe.preset import prompt_slug

# Prompts carrying no meaning: the floor any similarity claim must clear.
MEANINGLESS_PROMPTS = frozenset(
    NONSENSE_PROMPTS + SCRAMBLED_PROMPTS + (CONTROL_PROMPTS[1],))

_REPO = Path(__file__).resolve().parents[1]


def _load_runs(results: Path) -> list[dict]:
    rows = []
    for done in results.glob('**/DONE'):
        meta_path = done.parent / 'run.json'
        density_path = done.parent / 'physical_density.npy'
        if not meta_path.exists():
            continue
        meta = json.loads(meta_path.read_text())
        meta['_dir'] = str(done.parent)
        if density_path.exists():
            meta['_density'] = np.load(density_path)
        rows.append(meta)
    return rows


def _write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def compliance_table(runs: list[dict], out: Path) -> None:
    baseline = {}
    for run in runs:
        if run.get('experiment') == 'B':
            baseline[run.get('structure')] = run.get('compliance')
    rows = []
    for run in runs:
        c = run.get('compliance')
        b = baseline.get(run.get('structure'))
        ratio = None if c in (None, 0) or b in (None, 0) else float(b) / float(c)
        rows.append({
            'run_id': run.get('run_id'),
            'experiment': run.get('experiment'),
            'structure': run.get('structure'),
            'prompt': run.get('prompt'),
            'compliance': c,
            'compliance_ratio': ratio,
            'gray_fraction': run.get('gray_fraction'),
            'thresholded_compliance': run.get('thresholded_compliance'),
            'connected_components': run.get('connected_components'),
        })
    _write_csv(
        out / 'tables' / 'compliance.csv', rows,
        ['run_id', 'experiment', 'structure', 'prompt', 'compliance',
         'compliance_ratio', 'gray_fraction', 'thresholded_compliance',
         'connected_components'])


def cross_prompt_matrix(
        similarities_path: Path, out: Path, suffix: str = '') -> None:
    if not similarities_path.exists():
        return
    records = json.loads(similarities_path.read_text())
    prompts = list(PROMPTS) + ['human skull']
    grouped = defaultdict(list)
    for record in records:
        prompt = record.get('prompt') or 'unguided'
        grouped[prompt].append(record)
    # Mean block: rows = design prompt, cols = evaluator prompt.
    fieldnames = ['design_prompt'] + [f'sim:{p}' for p in prompts] + ['own_highest']
    rows = []
    for design_prompt, items in grouped.items():
        means = []
        own_hits = 0
        for item in items:
            sims = item.get('similarities') or {}
            values = [float(sims.get(p, float('nan'))) for p in prompts]
            means.append(values)
            if design_prompt in sims:
                own_hits += int(sims[design_prompt] >= max(values))
        avg = np.nanmean(np.asarray(means), axis=0) if means else []
        row = {'design_prompt': design_prompt, 'own_highest': own_hits}
        for prompt, value in zip(prompts, avg):
            row[f'sim:{prompt}'] = float(value)
        rows.append(row)
    _write_csv(
        out / 'tables' / f'cross_prompt_matrix{suffix}.csv', rows, fieldnames)


def semantic_floor(
        similarities_path: Path, out: Path, suffix: str = '') -> None:
    """Similarity to each prompt, split by what guided the design.

    The mechanical scalars cannot tell a meaningful prompt from a meaningless
    one -- compliance and gray fraction measure the cost of CLIP pressure, not
    fidelity. This is where that distinction has to show up instead: for each
    prompt, how much closer its own designs sit to it than designs pushed by a
    meaningless string, or not pushed at all. A small gap would mean the
    similarity is reporting generic CLIP pressure rather than the words.
    """
    if not similarities_path.exists():
        return
    records = json.loads(similarities_path.read_text())
    classes = {'own': [], 'meaningless': [], 'other prompt': [], 'unguided': []}
    rows = []
    for prompt in PROMPTS:
        scores = {key: [] for key in classes}
        for record in records:
            value = (record.get('similarities') or {}).get(prompt)
            if value is None:
                continue
            guide = record.get('prompt')
            if guide is None:
                key = 'unguided'
            elif guide == prompt:
                key = 'own'
            elif guide in MEANINGLESS_PROMPTS:
                key = 'meaningless'
            else:
                key = 'other prompt'
            scores[key].append(float(value))
        row = {'prompt': prompt}
        for key, values in scores.items():
            row[f'{key}_mean'] = float(np.mean(values)) if values else None
            row[f'{key}_n'] = len(values)
        own, floor = row['own_mean'], row['meaningless_mean']
        row['own_minus_meaningless'] = (
            None if own is None or floor is None else own - floor)
        row['own_minus_unguided'] = (
            None if own is None or row['unguided_mean'] is None
            else own - row['unguided_mean'])
        rows.append(row)
    _write_csv(
        out / 'tables' / f'semantic_floor{suffix}.csv', rows,
        ['prompt'] + [f'{key}_{field}' for key in classes for field in ('mean', 'n')]
        + ['own_minus_meaningless', 'own_minus_unguided'])


def weight_headroom(runs: list[dict], out: Path) -> None:
    """Grad-matched CLIP weights against the cap that would silently clip them.

    `GradNormEma.weight` truncates the gradient ratio at
    `GRAD_MATCH_WEIGHT_MAX`, so a run that spent steps at the ceiling was not
    actually running the coupling the method claims. Runs recorded before the
    per-step maximum was logged leave those columns empty; the mean is then
    only a lower bound on how close the run came.
    """
    rows = []
    for run in runs:
        if run.get('coupling') is None:
            continue
        mean = run.get('clip_weight_mean')
        peak = run.get('clip_weight_max')
        cap = float(run.get('clip_weight_cap') or GRAD_MATCH_WEIGHT_MAX)
        rows.append({
            'run_id': run.get('run_id'),
            'experiment': run.get('experiment'),
            'structure': run.get('structure'),
            'prompt': run.get('prompt'),
            'coupling': run.get('coupling'),
            'clip_weight_mean': mean,
            'clip_weight_max': peak,
            'clip_raw_z_weight_mean': run.get('clip_raw_z_weight_mean'),
            'clip_raw_z_weight_max': run.get('clip_raw_z_weight_max'),
            'cap': cap,
            'mean_fraction_of_cap': None if mean is None else float(mean) / cap,
            'max_fraction_of_cap': None if peak is None else float(peak) / cap,
            'at_cap': None if peak is None else bool(float(peak) >= cap - 1e-6),
        })
    _write_csv(
        out / 'tables' / 'weight_headroom.csv', rows,
        ['run_id', 'experiment', 'structure', 'prompt', 'coupling',
         'clip_weight_mean', 'clip_weight_max', 'clip_raw_z_weight_mean',
         'clip_raw_z_weight_max', 'cap', 'mean_fraction_of_cap',
         'max_fraction_of_cap', 'at_cap'])


def evaluator_view_gap(
        density_path: Path, render_path: Path, out: Path, suffix: str = '') -> None:
    """Write aligned own-prompt similarity differences between z and density."""
    if not density_path.exists() or not render_path.exists():
        return
    density = {
        row['attempt']: row for row in json.loads(density_path.read_text())
    }
    rendered = {
        row['attempt']: row for row in json.loads(render_path.read_text())
    }
    rows = []
    for attempt in sorted(density.keys() & rendered.keys()):
        d_row, z_row = density[attempt], rendered[attempt]
        d_score = d_row.get('own_prompt_similarity')
        z_score = z_row.get('own_prompt_similarity')
        rows.append({
            'attempt': attempt,
            'run_id': d_row.get('run_id'),
            'experiment': d_row.get('experiment'),
            'structure': d_row.get('structure'),
            'prompt': d_row.get('prompt'),
            'density_similarity': d_score,
            'z_similarity': z_score,
            'z_minus_density': (
                None if d_score is None or z_score is None
                else float(z_score) - float(d_score)),
        })
    _write_csv(
        out / 'tables' / f'evaluator_view_gap{suffix}.csv', rows,
        ['attempt', 'run_id', 'experiment', 'structure', 'prompt',
         'density_similarity', 'z_similarity', 'z_minus_density'])


def contact_sheets(runs: list[dict], out: Path) -> None:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from PIL import Image

    by_exp = defaultdict(list)
    for run in runs:
        by_exp[run.get('experiment')].append(run)
    fig_dir = out / 'figures'
    fig_dir.mkdir(parents=True, exist_ok=True)
    for experiment, items in by_exp.items():
        if not experiment:
            continue
        n = len(items)
        cols = min(6, max(1, n))
        rows = int(np.ceil(n / cols))
        fig, axes = plt.subplots(rows, cols, figsize=(2.2 * cols, 3.4 * rows), squeeze=False)
        for ax in axes.ravel():
            ax.axis('off')
        for ax, run in zip(axes.ravel(), items):
            png = Path(run['_dir']) / 'final.png'
            if png.exists():
                ax.imshow(Image.open(png), cmap='gray')
            ratio = run.get('compliance')
            ax.set_title(f"{run.get('run_id', '')}\nC={ratio}", fontsize=6)
        fig.tight_layout()
        fig.savefig(fig_dir / f'{experiment}_contact.png', dpi=120)
        plt.close(fig)


def fern_wording_panel(results: Path, out: Path) -> None:
    """Side-by-side tall finals for the S4 plural/count texts.

    Official ``fern fronds`` is the H1/S3 column so the panel does not
    rerun the campaign fern.
    """
    from PIL import Image, ImageDraw, ImageFont

    official = PROMPTS[0]
    modes = ('hybrid', 'semantic')
    cells = []
    for mode in modes:
        row = []
        for prompt in FERN_WORDINGS:
            token = prompt_slug(prompt)
            if prompt == official:
                run_id = (
                    f'H1/tall/{token}/hybrid' if mode == 'hybrid'
                    else f'S3/tall/{token}'
                )
            else:
                run_id = f'S4/tall/{token}/{mode}'
            png = None
            root = results / run_id
            attempts = sorted(root.glob('attempt_*'))
            for attempt in reversed(attempts):
                candidate = attempt / 'final.png'
                if candidate.exists() and (attempt / 'DONE').exists():
                    png = candidate
                    break
            row.append((prompt, png))
        cells.append(row)

    found = [png for row in cells for _, png in row if png is not None]
    if not found:
        return

    sample = Image.open(found[0])
    cell_w, cell_h = sample.size
    label_h = 36
    fig = Image.new(
        'RGB',
        (cell_w * len(FERN_WORDINGS), (cell_h + label_h) * len(modes)),
        'white',
    )
    draw = ImageDraw.Draw(fig)
    font = ImageFont.load_default()
    for r, row in enumerate(cells):
        for c, (prompt, png) in enumerate(row):
            x = c * cell_w
            y = r * (cell_h + label_h)
            draw.text((x + 8, y + 10), f'{modes[r]}: {prompt}', fill='black', font=font)
            if png is not None:
                fig.paste(Image.open(png).convert('RGB').resize((cell_w, cell_h)), (x, y + label_h))
    dest = out / 'figures'
    dest.mkdir(parents=True, exist_ok=True)
    fig.save(dest / 's4_fern_wording.png')


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', default=str(_REPO / 'results'))
    parser.add_argument('--out', default=str(_REPO / 'analysis' / 'out'))
    args = parser.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    runs = _load_runs(Path(args.results))
    compliance_table(runs, out)
    weight_headroom(runs, out)
    for model_name in EVAL_MODELS:
        for view in EVAL_VIEWS:
            suffix = eval_model_suffix(model_name) + eval_view_suffix(view)
            similarities = Path(args.out) / 'evaluate' / f'similarities{suffix}.json'
            cross_prompt_matrix(similarities, out, suffix=suffix)
            semantic_floor(similarities, out, suffix=suffix)
        model_suffix = eval_model_suffix(model_name)
        evaluator_view_gap(
            Path(args.out) / 'evaluate' / f'similarities{model_suffix}.json',
            Path(args.out) / 'evaluate' / f'similarities{model_suffix}_z.json',
            out, suffix=model_suffix)
    contact_sheets(runs, out)
    fern_wording_panel(Path(args.results), out)
    print(f'wrote tables and figures for {len(runs)} runs -> {out}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
