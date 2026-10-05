"""Structural, geometric, and perceptual diversity of tall-building designs.

The ``solve`` command re-solves each saved physical density in its own process
and caches per-element strain energy. The ``report`` command fits one PCA per
measure on the union of comparison sets and writes maps and bootstrap summaries.
"""

from __future__ import annotations

import runtime  # noqa: F401  # pin OpenMP before NumPy/CHOLMOD load

import argparse
import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from analysis.evaluate import walk_done

_REPO = Path(__file__).resolve().parents[1]
_GRID = (32, 16)
_SET_ORDER = (
    'unguided seeds',
    'parameter sweep',
    'ours',
    'semantic prompts',
)
_MEASURES = ('structural', 'geometric', 'perceptual')


def block_reduce(field: np.ndarray, shape: tuple[int, int], reduction: str) -> np.ndarray:
    """Reduce a 2D field to ``shape`` using equal rectangular blocks."""
    field = np.asarray(field, dtype=float)
    rows, cols = shape
    if field.ndim != 2 or field.shape[0] % rows or field.shape[1] % cols:
        raise ValueError(f'{field.shape} cannot be block-reduced to {shape}')
    blocks = field.reshape(
        rows, field.shape[0] // rows, cols, field.shape[1] // cols)
    if reduction == 'sum':
        return blocks.sum(axis=(1, 3))
    if reduction == 'mean':
        return blocks.mean(axis=(1, 3))
    raise ValueError(f'unknown reduction {reduction!r}')


def load_path_map(energy: np.ndarray, shape: tuple[int, int] = _GRID) -> np.ndarray:
    """Square root of normalized block-summed energy, with unit L2 norm."""
    coarse = np.maximum(block_reduce(energy, shape, 'sum'), 0.0)
    total = coarse.sum()
    if not np.isfinite(total) or total <= 0:
        raise ValueError('strain energy must have a positive finite sum')
    return np.sqrt(coarse / total)


def geometric_map(density: np.ndarray, shape: tuple[int, int] = _GRID) -> np.ndarray:
    """Block-mean physical density."""
    return block_reduce(np.asarray(density), shape, 'mean')


def normalize_rows(values: np.ndarray) -> np.ndarray:
    """L2-normalize rows, rejecting zero or non-finite vectors."""
    values = np.asarray(values, dtype=float)
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    if np.any(~np.isfinite(norms)) or np.any(norms == 0):
        raise ValueError('cannot normalize zero or non-finite vectors')
    return values / norms


def fit_pca(values: np.ndarray, components: int = 2) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Scores, component vectors, and explained-variance ratios from one SVD."""
    values = np.asarray(values, dtype=float)
    centered = values - values.mean(axis=0, keepdims=True)
    _, singular, vh = np.linalg.svd(centered, full_matrices=False)
    count = min(components, vh.shape[0])
    scores = centered @ vh[:count].T
    variances = singular ** 2
    ratios = variances[:count] / variances.sum() if variances.sum() else np.zeros(count)
    return scores, vh[:count], ratios


def pairwise_distances(values: np.ndarray) -> np.ndarray:
    """Euclidean distances for all unordered, distinct row pairs."""
    values = np.asarray(values, dtype=float)
    if len(values) < 2:
        return np.empty(0)
    i, j = np.triu_indices(len(values), 1)
    return np.linalg.norm(values[i] - values[j], axis=1)


def bootstrap_mean_distance(
        values: np.ndarray, *, samples: int = 10_000,
        seed: int = 2027) -> tuple[float, float, float]:
    """Mean pair distance and percentile CI by resampling distinct-index pairs."""
    distances = pairwise_distances(values)
    if not len(distances):
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed)
    means = rng.choice(distances, size=(samples, len(distances)), replace=True).mean(axis=1)
    low, high = np.percentile(means, [2.5, 97.5])
    return float(distances.mean()), float(low), float(high)


def comparison_set(meta: dict) -> str | None:
    """Assign one tall-building run to a disjoint paper comparison set."""
    if meta.get('structure') != 'tall':
        return None
    experiment = meta.get('experiment')
    prompt = meta.get('prompt')
    if experiment == 'B' or (experiment == 'R' and prompt is None):
        return 'unguided seeds'
    if experiment == 'D':
        return 'parameter sweep'
    if experiment == 'M':
        return 'ours'
    if experiment in ('S3', 'P', 'L'):
        return 'semantic prompts'
    return None


def _display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(_REPO))
    except ValueError:
        return str(path.resolve())


def _cache_path(attempt: Path, cache: Path) -> Path:
    token = hashlib.sha256(_display_path(attempt).encode()).hexdigest()[:20]
    return cache / f'{token}.npz'


def _read_meta(attempt: Path) -> dict:
    path = attempt / 'run.json'
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text())


def replay_args(meta: dict, manifests: list[Path] | None = None):
    """Recover run arguments from run.json, or its immutable manifest row."""
    if meta.get('cli'):
        return SimpleNamespace(**meta['cli'])
    run_id = meta.get('run_id')
    manifests = manifests or sorted((_REPO / 'slurm').glob('*.tsv'))
    for manifest in manifests:
        for line in manifest.read_text().splitlines():
            row_id, separator, encoded = line.partition('\t')
            if separator and row_id == run_id:
                from run import build_parser
                return build_parser().parse_args(json.loads(encoded))
    raise ValueError(
        f'{run_id!r} predates run.json CLI capture and was not found in '
        f'{len(manifests)} manifest files')


def resolve_attempt(attempt: Path, destination: Path) -> dict:
    """Re-solve one design and atomically cache its energy maps and diagnostics."""
    from physics import physics
    from physics.api import Environment, specified_task
    from recipe.campaign import preset_from_args
    from recipe.dream_layout import structural_params

    meta = _read_meta(attempt)
    preset = preset_from_args(replay_args(meta))
    env = Environment(specified_task(structural_params(preset).get_problem()))
    env.args['penal'] = preset.penal
    physics.apply_pixel_gravity(env.args, preset.gravity_load)

    density = np.asarray(np.load(attempt / 'physical_density.npy')).squeeze()
    expected = (env.nely, env.nelx)
    if density.shape != expected:
        raise ValueError(f'{attempt}: density shape {density.shape}, expected {expected}')
    kwargs = {
        'penal': env.args['penal'],
        'e_min': env.args['young_min'],
        'e_0': env.args['young'],
    }

    def solve(field: np.ndarray) -> tuple[float, np.ndarray]:
        forces = physics.calculate_forces(field, env.args)
        displacement = physics.displace(
            field, env.ke, forces, env.args['freedofs'], env.args['fixdofs'],
            **kwargs)
        energy = np.asarray(
            physics.element_strain_energy(field, displacement, env.ke, **kwargs))
        return float(energy.sum()), energy

    physical_compliance, physical_energy = solve(density)
    binary_compliance, binary_energy = solve((density >= 0.5).astype(float))
    recorded = meta.get('compliance')
    relative_error = (
        abs(physical_compliance - float(recorded)) / abs(float(recorded))
        if recorded not in (None, 0) else np.nan)
    valid = bool(np.isfinite(relative_error) and relative_error <= 0.05)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix('.tmp.npz')
    np.savez_compressed(
        temporary,
        attempt=_display_path(attempt),
        physical_energy=physical_energy,
        binary_energy=binary_energy,
        physical_compliance=physical_compliance,
        binary_compliance=binary_compliance,
        recorded_compliance=np.nan if recorded is None else float(recorded),
        relative_error=relative_error,
        valid=valid,
    )
    temporary.replace(destination)
    return {
        'attempt': _display_path(attempt),
        'physical_compliance': physical_compliance,
        'binary_compliance': binary_compliance,
        'relative_error': relative_error,
        'valid': valid,
    }


def solve_all(results: Path, cache: Path, *, force: bool = False) -> list[dict]:
    """Run every selected re-solve in a fresh interpreter."""
    cache.mkdir(parents=True, exist_ok=True)
    records = []
    for attempt in walk_done(results):
        try:
            meta = _read_meta(attempt)
        except (FileNotFoundError, json.JSONDecodeError):
            continue
        if comparison_set(meta) is None or not (attempt / 'physical_density.npy').exists():
            continue
        destination = _cache_path(attempt, cache)
        if destination.exists() and not force:
            with np.load(destination) as saved:
                records.append({
                    'attempt': str(saved['attempt']),
                    'relative_error': float(saved['relative_error']),
                    'valid': bool(saved['valid']),
                    'cached': True,
                })
            continue
        command = [
            sys.executable, str(Path(__file__).resolve()), 'resolve-one',
            '--attempt', str(attempt), '--destination', str(destination),
        ]
        completed = subprocess.run(command, cwd=_REPO, text=True, capture_output=True)
        if completed.returncode:
            records.append({
                'attempt': _display_path(attempt),
                'valid': False,
                'error': completed.stderr.strip() or completed.stdout.strip(),
            })
        else:
            records.append(json.loads(completed.stdout))
    return records


def _embedding_lookup(evaluate_dir: Path) -> dict[str, np.ndarray]:
    suffix = '_vit_l_14_z'
    rows_path = evaluate_dir / f'similarities{suffix}.json'
    vectors_path = evaluate_dir / f'embeddings{suffix}.npy'
    rows = json.loads(rows_path.read_text())
    vectors = np.load(vectors_path)
    if len(rows) != len(vectors):
        raise ValueError('ViT-L/14 z metadata and embeddings have different lengths')
    return {
        _display_path(Path(row['attempt'])): vector
        for row, vector in zip(rows, vectors)
    }


def collect_points(results: Path, cache: Path, evaluate_dir: Path) -> list[dict]:
    """Load aligned features and quality data for valid selected runs."""
    embeddings = _embedding_lookup(evaluate_dir)
    points = []
    for attempt in walk_done(results):
        try:
            meta = _read_meta(attempt)
        except (FileNotFoundError, json.JSONDecodeError):
            continue
        group = comparison_set(meta)
        energy_path = _cache_path(attempt, cache)
        key = _display_path(attempt)
        if group is None or not energy_path.exists() or key not in embeddings:
            continue
        with np.load(energy_path) as saved:
            if not bool(saved['valid']):
                continue
            energy = np.asarray(saved['physical_energy'])
            binary_energy = np.asarray(saved['binary_energy'])
            compliance = float(saved['physical_compliance'])
        density = np.asarray(np.load(attempt / 'physical_density.npy')).squeeze()
        points.append({
            'attempt': key,
            'run_id': meta.get('run_id'),
            'prompt': meta.get('prompt'),
            'experiment': meta.get('experiment'),
            'set': group,
            'resolution_scale': float(meta.get('cli', {}).get('resolution_scale', 1.0)),
            'compliance': compliance,
            'structural': load_path_map(energy).reshape(-1),
            'structural_binary': load_path_map(binary_energy).reshape(-1),
            'geometric': geometric_map(density).reshape(-1),
            'perceptual': np.asarray(embeddings[key], dtype=float),
            'image': attempt / 'final.png',
        })
    if points:
        normalized = normalize_rows(np.stack([p['perceptual'] for p in points]))
        for point, vector in zip(points, normalized):
            point['perceptual'] = vector
    return points


def _write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _baseline_compliance(points: list[dict]) -> float:
    values = [
        p['compliance'] for p in points
        if p['experiment'] == 'B' and p['resolution_scale'] == 1.0
    ]
    if not values:
        raise ValueError('no full-resolution B/tall baseline for quality')
    return float(np.mean(values))


def summarize(points: list[dict], measure: str) -> tuple[list[dict], list[dict], np.ndarray]:
    """Fit joint PCA and return per-set summaries and per-design coordinates."""
    values = np.stack([point[measure] for point in points])
    scores, components, variance = fit_pca(values)
    baseline = _baseline_compliance(points)
    table, coordinates = [], []
    for index, point in enumerate(points):
        coordinates.append({
            'attempt': point['attempt'],
            'run_id': point['run_id'],
            'set': point['set'],
            'measure': measure,
            'pc1': scores[index, 0],
            'pc2': scores[index, 1],
        })
    for group in _SET_ORDER:
        indices = [i for i, point in enumerate(points) if point['set'] == group]
        if not indices:
            continue
        mean, low, high = bootstrap_mean_distance(values[indices])
        quality = [
            baseline / points[i]['compliance'] for i in indices
            if points[i]['resolution_scale'] == 1.0
        ]
        table.append({
            'set': group,
            'measure': measure,
            'n': len(indices),
            'mean_pairwise_distance': mean,
            'ci95_low': low,
            'ci95_high': high,
            'pc1_variance': variance[0],
            'pc2_variance': variance[1],
            'mean_quality': float(np.mean(quality)) if quality else '',
            'quality_n': len(quality),
        })
    return table, coordinates, components


def _plot_map(points: list[dict], coordinates: list[dict], measure: str, path: Path) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.offsetbox import AnnotationBbox, OffsetImage
    from PIL import Image

    palette = dict(zip(_SET_ORDER, ('#4c78a8', '#f58518', '#54a24b', '#b279a2')))
    figure, axis = plt.subplots(figsize=(8, 6))
    for group in _SET_ORDER:
        rows = [(p, c) for p, c in zip(points, coordinates) if p['set'] == group]
        if not rows:
            continue
        x = [row[1]['pc1'] for row in rows]
        y = [row[1]['pc2'] for row in rows]
        axis.scatter(x, y, s=28, color=palette[group], label=group, zorder=2)
        for point, coordinate in rows:
            if not point['image'].exists():
                continue
            with Image.open(point['image']) as image:
                thumbnail = np.asarray(image.convert('L').resize((18, 36)))
            artist = AnnotationBbox(
                OffsetImage(thumbnail, cmap='gray', zoom=0.65),
                (coordinate['pc1'], coordinate['pc2']),
                frameon=False, pad=0, zorder=3)
            axis.add_artist(artist)
    axis.axhline(0, color='0.85', linewidth=0.8)
    axis.axvline(0, color='0.85', linewidth=0.8)
    axis.set_xlabel('PC1')
    axis.set_ylabel('PC2')
    axis.set_title(f'{measure.capitalize()} diversity')
    axis.legend(frameon=False)
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=200)
    plt.close(figure)


def _plot_components(
        points: list[dict], measure: str, components: np.ndarray, path: Path) -> None:
    """Mean and one-score-standard-deviation excursions along the first PCs."""
    import matplotlib.pyplot as plt

    values = np.stack([point[measure] for point in points])
    mean = values.mean(axis=0)
    scores = (values - mean) @ components.T
    figure, axes = plt.subplots(2, 3, figsize=(6, 8))
    for row in range(2):
        scale = scores[:, row].std()
        fields = (mean - scale * components[row], mean, mean + scale * components[row])
        titles = (f'-PC{row + 1}', 'Mean', f'+PC{row + 1}')
        limits = (min(field.min() for field in fields), max(field.max() for field in fields))
        for axis, field, title in zip(axes[row], fields, titles):
            axis.imshow(field.reshape(_GRID), cmap='viridis', vmin=limits[0], vmax=limits[1])
            axis.set_title(title)
            axis.axis('off')
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=200)
    plt.close(figure)


def report(results: Path, cache: Path, evaluate_dir: Path, out: Path) -> None:
    points = collect_points(results, cache, evaluate_dir)
    if len(points) < 2:
        raise ValueError('at least two aligned, valid designs are required')
    table_rows, coordinate_rows = [], []
    for measure in _MEASURES:
        table, coordinates, components = summarize(points, measure)
        table_rows.extend(table)
        coordinate_rows.extend(coordinates)
        _plot_map(points, coordinates, measure, out / 'figures' / f'diversity_{measure}.png')
        if measure in ('structural', 'geometric'):
            _plot_components(
                points, measure, components,
                out / 'figures' / f'diversity_{measure}_components.png')
    binary_table, binary_coordinates, _ = summarize(points, 'structural_binary')
    for row in binary_table:
        row['measure'] = 'structural_binary'
    for row in binary_coordinates:
        row['measure'] = 'structural_binary'
    table_rows.extend(binary_table)
    coordinate_rows.extend(binary_coordinates)
    _plot_map(
        points, binary_coordinates, 'structural_binary',
        out / 'figures' / 'diversity_structural_binary.png')
    _write_csv(
        out / 'tables' / 'diversity.csv', table_rows,
        ['set', 'measure', 'n', 'mean_pairwise_distance', 'ci95_low', 'ci95_high',
         'pc1_variance', 'pc2_variance', 'mean_quality', 'quality_n'])
    _write_csv(
        out / 'tables' / 'diversity_points.csv', coordinate_rows,
        ['attempt', 'run_id', 'set', 'measure', 'pc1', 'pc2'])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest='command', required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument('--results', type=Path, default=_REPO / 'results')
    common.add_argument(
        '--cache', type=Path, default=_REPO / 'analysis' / 'out' / 'diversity' / 'energy')
    solve = subparsers.add_parser('solve', parents=[common])
    solve.add_argument('--force', action='store_true')
    report_parser = subparsers.add_parser('report', parents=[common])
    report_parser.add_argument(
        '--evaluate', type=Path, default=_REPO / 'analysis' / 'out' / 'evaluate')
    report_parser.add_argument(
        '--out', type=Path, default=_REPO / 'analysis' / 'out' / 'diversity')
    one = subparsers.add_parser('resolve-one')
    one.add_argument('--attempt', type=Path, required=True)
    one.add_argument('--destination', type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == 'resolve-one':
        print(json.dumps(resolve_attempt(args.attempt, args.destination)))
    elif args.command == 'solve':
        records = solve_all(args.results, args.cache, force=args.force)
        print(json.dumps(records, indent=2))
    else:
        report(args.results, args.cache, args.evaluate, args.out)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
