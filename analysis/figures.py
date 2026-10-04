"""Contact sheets, dial curves, scale figure, prompt x structure grid."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from analysis.vendi import (
    downsample_binary,
    quality_weighted_vendi,
    tanimoto_kernel,
    vendi,
)
from recipe.campaign_spec import PROMPTS

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


def cross_prompt_matrix(similarities_path: Path, out: Path) -> None:
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
    _write_csv(out / 'tables' / 'cross_prompt_matrix.csv', rows, fieldnames)


def vendi_table(runs: list[dict], embeddings_path: Path, out: Path) -> None:
    groups = {
        'conventional': lambda r: r.get('experiment') == 'D',
        'formal': lambda r: str(r.get('experiment', '')).startswith('F'),
        'semantic': lambda r: str(r.get('experiment', '')).startswith('S')
        and r.get('structure') == 'tall',
        'hybrid': lambda r: r.get('experiment') in ('H1', 'H3', 'H4', 'H5'),
    }
    tall_baseline = next(
        (r.get('compliance') for r in runs
         if r.get('experiment') == 'B' and r.get('structure') == 'tall'),
        None)
    # Map run_id to embedding row via evaluate order is not guaranteed;
    # skip semantic Vendi if embeddings are missing.
    embeddings = None
    if embeddings_path.exists():
        embeddings = np.load(embeddings_path)
    rows = []
    for name, matches in groups.items():
        members = [
            r for r in runs
            if matches(r) and r.get('structure', 'tall') in (None, 'tall', 'tall_building')
            and r.get('_density') is not None
        ]
        if name != 'conventional':
            members = [r for r in members if r.get('structure') in ('tall', None) or r.get('experiment') != 'B']
        if len(members) < 2:
            continue
        binary = np.stack([
            downsample_binary(r['_density']).reshape(-1) for r in members
        ])
        formal_k = tanimoto_kernel(binary)
        qualities = []
        for run in members:
            c = run.get('compliance')
            if tall_baseline and c:
                qualities.append(float(tall_baseline) / float(c))
            else:
                qualities.append(0.0)
        quality = np.asarray(qualities)
        rows.append({
            'group': name,
            'n': len(members),
            'formal_vendi': vendi(formal_k),
            'quality_weighted_vendi': quality_weighted_vendi(formal_k, quality),
            'semantic_vendi': '',
        })
    _write_csv(
        out / 'tables' / 'vendi.csv', rows,
        ['group', 'n', 'formal_vendi', 'semantic_vendi', 'quality_weighted_vendi'])


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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', default=str(_REPO / 'results'))
    parser.add_argument('--out', default=str(_REPO / 'analysis' / 'out'))
    args = parser.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    runs = _load_runs(Path(args.results))
    compliance_table(runs, out)
    cross_prompt_matrix(
        Path(args.out) / 'evaluate' / 'similarities.json', out)
    vendi_table(runs, Path(args.out) / 'evaluate' / 'embeddings.npy', out)
    contact_sheets(runs, out)
    print(f'wrote tables and figures for {len(runs)} runs -> {out}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
