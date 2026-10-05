"""Does the prompt do the work, and is it the meaning or the surface form?

Two tests, both run on geometry rather than on a similarity score. This is
the point: guidance maximises a CLIP similarity, so judging it by a CLIP
similarity partly guarantees the answer. Nothing here consults a
vision-language model.

`specificity` asks whether a prompt imposes a repeatable geometry. Designs
guided by the same prompt at different seeds should resemble each other more
than they resemble designs guided by a different prompt, and a permutation
null says whether the margin could be chance. Note what this does NOT show:
an arbitrary text embedding is still a fixed direction, so a nonsense prompt
passes too. It establishes specificity, not meaning.

`meaning` is the test that separates the two. A paraphrase shares no word
with the prompt it restates, and a scrambled string shares every letter and
no meaning. Neither was optimised toward, so if the paraphrase designs land
nearer the original prompt's designs than the scrambled ones do, the sense of
the words is what moved the geometry rather than their characters.
"""

from __future__ import annotations

import runtime  # noqa: F401  # pin OpenMP before NumPy loads

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from analysis.diversity import (
    cache_path,
    display_path,
    geometric_map,
    load_path_map,
    read_meta,
)
from analysis.evaluate import walk_done
from recipe.campaign_spec import PARAPHRASE_PROMPTS, scrambled_origin

_REPO = Path(__file__).resolve().parents[1]
_PERMUTATIONS = 10_000
_SEED = 2027
# 'original' is the reference each other condition is measured against.
_CONDITIONS = ('original', 'paraphrase', 'scrambled')


def prompt_condition(meta: dict) -> tuple[str, str] | None:
    """(prompt family, condition) for one tall run, or None if out of scope.

    The family is the campaign prompt a run speaks to, so a paraphrase and
    the scramble of the same prompt are filed under the same name and can be
    compared against it.
    """
    if meta.get('structure') != 'tall':
        return None
    prompt = meta.get('prompt')
    if not prompt:
        return None
    experiment = meta.get('experiment')
    origins = scrambled_origin()
    paraphrases = {value: key for key, value in PARAPHRASE_PROMPTS.items()}
    # Only the unmodified recipe counts as an original: a dial or weight arm
    # would confound the comparison with a different coupling.
    if experiment in ('S3', 'R') and prompt in PARAPHRASE_PROMPTS:
        return prompt, 'original'
    if prompt in paraphrases:
        return paraphrases[prompt], 'paraphrase'
    if prompt in origins and origins[prompt] in PARAPHRASE_PROMPTS:
        return origins[prompt], 'scrambled'
    return None


def collect_designs(results: Path, cache: Path | None = None) -> list[dict]:
    """Geometric maps for every in-scope run, plus load paths where cached.

    The geometric map comes straight off `physical_density.npy`, so this
    needs no re-solve. The load-path map needs `diversity.py solve` to have
    covered the run, which its own selection may not have; those runs simply
    carry no structural vector.
    """
    designs = []
    for attempt in walk_done(results):
        try:
            meta = read_meta(attempt)
        except (FileNotFoundError, json.JSONDecodeError):
            continue
        scope = prompt_condition(meta)
        density_path = attempt / 'physical_density.npy'
        if scope is None or not density_path.exists():
            continue
        family, condition = scope
        density = np.asarray(np.load(density_path)).squeeze()
        design = {
            'attempt': display_path(attempt),
            'run_id': meta.get('run_id'),
            'prompt': meta.get('prompt'),
            'family': family,
            'condition': condition,
            'seed': meta.get('cli', {}).get('seed'),
            'geometric': geometric_map(density).reshape(-1),
        }
        if cache is not None:
            energy = cache_path(attempt, cache)
            if energy.exists():
                with np.load(energy) as saved:
                    if bool(saved['valid']):
                        design['structural'] = load_path_map(
                            np.asarray(saved['physical_energy'])).reshape(-1)
        designs.append(design)
    return designs


def _separation(values: np.ndarray, labels: np.ndarray) -> float:
    """Between-group minus within-group mean pairwise distance.

    Positive means same-label designs sit closer together than
    different-label ones, which is what a prompt-specific geometry looks
    like. NaN when either set of pairs is empty.
    """
    rows, columns = np.triu_indices(len(values), 1)
    distances = np.linalg.norm(values[rows] - values[columns], axis=1)
    same = labels[rows] == labels[columns]
    if not same.any() or same.all():
        return float('nan')
    return float(distances[~same].mean() - distances[same].mean())


def permutation_test(
        values: np.ndarray, labels: np.ndarray, *,
        permutations: int = _PERMUTATIONS, seed: int = _SEED) -> dict:
    """Separation against a null that shuffles the labels.

    Shuffling keeps the designs and the group sizes and destroys only the
    correspondence between a design and its prompt, so the null is 'the
    prompt had no effect' with everything else held fixed.
    """
    observed = _separation(values, labels)
    rng = np.random.default_rng(seed)
    permuted = np.empty(permutations)
    for index in range(permutations):
        permuted[index] = _separation(values, rng.permutation(labels))
    # +1 in both terms so a p-value is never reported as exactly zero.
    p_value = float((np.sum(permuted >= observed) + 1) / (permutations + 1))
    return {
        'separation': observed,
        'p_value': p_value,
        'null_mean': float(np.mean(permuted)),
        'null_sd': float(np.std(permuted)),
    }


def identification_accuracy(values: np.ndarray, labels: np.ndarray) -> dict:
    """Leave-one-out nearest-neighbour prompt identification.

    Each design's nearest other design votes for its own prompt. Chance is
    computed from the label distribution rather than assumed uniform,
    because the conditions do not all have the same number of runs.
    """
    distances = np.linalg.norm(values[:, None, :] - values[None, :, :], axis=2)
    np.fill_diagonal(distances, np.inf)
    predicted = labels[np.argmin(distances, axis=1)]
    _, counts = np.unique(labels, return_counts=True)
    shares = counts / counts.sum()
    return {
        'accuracy': float(np.mean(predicted == labels)),
        'chance': float(np.sum(shares ** 2)),
        'n': int(len(labels)),
    }


def specificity(designs: list[dict], measure: str) -> dict | None:
    """Do same-prompt designs cluster? Originals only, prompt as the label."""
    rows = [
        design for design in designs
        if design['condition'] == 'original' and measure in design
    ]
    families = {design['family'] for design in rows}
    if len(rows) < 4 or len(families) < 2:
        return None
    values = np.stack([design[measure] for design in rows])
    labels = np.array([design['family'] for design in rows])
    return {
        'measure': measure,
        'test': 'specificity',
        **permutation_test(values, labels),
        **identification_accuracy(values, labels),
    }


def meaning(designs: list[dict], measure: str) -> list[dict]:
    """Is a paraphrase nearer the original's designs than a scramble is?

    Distance is to the centroid of that prompt's original designs. A
    permutation over just the paraphrase and scrambled labels says whether
    the gap between them could be chance.
    """
    results = []
    for family in sorted({design['family'] for design in designs}):
        rows = [
            design for design in designs
            if design['family'] == family and measure in design
        ]
        by_condition = {
            condition: np.stack([
                design[measure] for design in rows
                if design['condition'] == condition])
            for condition in _CONDITIONS
            if any(design['condition'] == condition for design in rows)
        }
        if 'original' not in by_condition:
            continue
        centroid = by_condition['original'].mean(axis=0)
        row = {'measure': measure, 'test': 'meaning', 'family': family}
        for condition, values in by_condition.items():
            row[f'{condition}_distance'] = float(
                np.linalg.norm(values - centroid, axis=1).mean())
            row[f'{condition}_n'] = int(len(values))
        if {'paraphrase', 'scrambled'} <= by_condition.keys():
            row['scrambled_minus_paraphrase'] = (
                row['scrambled_distance'] - row['paraphrase_distance'])
            contrast = np.concatenate(
                [by_condition['paraphrase'], by_condition['scrambled']])
            labels = np.array(
                ['paraphrase'] * row['paraphrase_n']
                + ['scrambled'] * row['scrambled_n'])
            distances = np.linalg.norm(contrast - centroid, axis=1)
            row['p_value'] = _distance_gap_p_value(distances, labels)
        results.append(row)
    return results


def meaning_pooled(designs: list[dict], measure: str) -> dict | None:
    """The paraphrase-versus-scramble gap pooled over prompt families.

    Per family the panel runs three designs per condition, so the smallest
    one-sided p a permutation can return is 1/C(6,3) = 0.05 -- too weak to
    rest a claim on. Permuting within each family and pooling the gaps keeps
    every family's own centroid as its reference while multiplying the
    number of distinguishable arrangements.
    """
    strata = []
    for family in sorted({design['family'] for design in designs}):
        rows = [
            design for design in designs
            if design['family'] == family and measure in design
        ]
        by_condition = {
            condition: [
                design[measure] for design in rows
                if design['condition'] == condition]
            for condition in _CONDITIONS
        }
        if not all(by_condition[c] for c in _CONDITIONS):
            continue
        centroid = np.stack(by_condition['original']).mean(axis=0)
        contrast = np.stack(
            by_condition['paraphrase'] + by_condition['scrambled'])
        strata.append((
            np.linalg.norm(contrast - centroid, axis=1),
            np.array(
                [False] * len(by_condition['paraphrase'])
                + [True] * len(by_condition['scrambled'])),
        ))
    if len(strata) < 2:
        return None

    def gap(assignments) -> float:
        return float(np.mean([
            distances[mask].mean() - distances[~mask].mean()
            for distances, mask in zip(
                (d for d, _ in strata), assignments)
        ]))

    observed = gap([mask for _, mask in strata])
    rng = np.random.default_rng(_SEED)
    hits = 0
    for _ in range(_PERMUTATIONS):
        hits += int(gap([rng.permutation(mask) for _, mask in strata])
                    >= observed)
    return {
        'measure': measure,
        'test': 'meaning',
        'family': 'pooled (stratified)',
        'scrambled_minus_paraphrase': observed,
        'p_value': float((hits + 1) / (_PERMUTATIONS + 1)),
        'families': len(strata),
    }


def _distance_gap_p_value(
        distances: np.ndarray, labels: np.ndarray, *,
        permutations: int = _PERMUTATIONS, seed: int = _SEED) -> float:
    """One-sided p for scrambled sitting further from the centroid."""
    is_scrambled = labels == 'scrambled'
    observed = distances[is_scrambled].mean() - distances[~is_scrambled].mean()
    rng = np.random.default_rng(seed)
    hits = 0
    for _ in range(permutations):
        shuffled = rng.permutation(is_scrambled)
        gap = distances[shuffled].mean() - distances[~shuffled].mean()
        hits += int(gap >= observed)
    return float((hits + 1) / (permutations + 1))


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fields})


def report(results: Path, cache: Path, out: Path) -> dict:
    designs = collect_designs(results, cache)
    measures = ['geometric']
    if any('structural' in design for design in designs):
        measures.append('structural')
    specificity_rows = [
        row for row in (specificity(designs, m) for m in measures)
        if row is not None
    ]
    meaning_rows = []
    for measure in measures:
        meaning_rows.extend(meaning(designs, measure))
        pooled = meaning_pooled(designs, measure)
        if pooled is not None:
            meaning_rows.append(pooled)
    _write_csv(out / 'tables' / 'specificity.csv', specificity_rows)
    _write_csv(out / 'tables' / 'meaning.csv', meaning_rows)
    _write_csv(out / 'tables' / 'specificity_designs.csv', [
        {key: design[key] for key in
         ('attempt', 'run_id', 'prompt', 'family', 'condition', 'seed')}
        for design in designs
    ])
    return {
        'designs': len(designs),
        'measures': measures,
        'specificity': specificity_rows,
        'meaning': meaning_rows,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest='command', required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument('--results', type=Path, default=_REPO / 'results')
    common.add_argument(
        '--cache', type=Path,
        default=_REPO / 'analysis' / 'out' / 'diversity' / 'energy')
    # Optional: the geometric test needs no re-solve, so `solve` only buys
    # the structural measure on runs the diversity selection left out.
    solve = subparsers.add_parser('solve', parents=[common])
    solve.add_argument('--force', action='store_true')
    report_parser = subparsers.add_parser('report', parents=[common])
    report_parser.add_argument(
        '--out', type=Path, default=_REPO / 'analysis' / 'out' / 'specificity')
    args = parser.parse_args(argv)
    if args.command == 'solve':
        from analysis.diversity import solve_all
        records = solve_all(
            args.results, args.cache, force=args.force,
            select=prompt_condition)
        print(json.dumps(records, indent=2))
    else:
        print(json.dumps(
            report(args.results, args.cache, args.out), indent=2, default=float))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
