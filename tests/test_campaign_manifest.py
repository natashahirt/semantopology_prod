"""Manifest composition. Does not submit jobs."""

from __future__ import annotations

from slurm.make_manifest import experiment_rows


def test_default_count_is_109():
    rows = experiment_rows(include_s1b=False)
    assert len(rows) == 109
    ids = [row['run_id'] for row in rows]
    assert len(ids) == len(set(ids))
    assert 'S1b/short/butterfly/m' not in ids
    assert 'H4/tall/skeletons/coadapt-off' in ids
    assert 'H2/bridge/fern_fronds/hybrid' in ids
    assert 'S2/tall/butterfly/rho-0.50' in ids
    assert 'S1/tall/skeletons/gme' in ids


def test_s1b_adds_24():
    rows = experiment_rows(include_s1b=True)
    assert len(rows) == 133
    ids = [row['run_id'] for row in rows]
    assert 'S1b/short/butterfly/m' in ids
    assert 'S1b/bridge/fern_fronds/e' in ids


def test_three_prompts_on_hybrid_and_semantic():
    argv_blob = ' '.join(
        ' '.join(row['argv']) for row in experiment_rows(include_s1b=False)
        if row['run_id'].startswith('H1/') or row['run_id'].startswith('S3/tall/'))
    assert 'fern fronds' in argv_blob
    assert 'butterfly' in argv_blob
    assert 'skeletons' in argv_blob
    assert 'human skull' in argv_blob


def test_prompt_sketch_rows_keep_coadaptation_on():
    rows = [
        row for row in experiment_rows(include_s1b=False)
        if row['run_id'].startswith('H3/')
    ]
    assert len(rows) == 9
    for row in rows:
        coadapt = row['argv'].index('--coadapt')
        assert row['argv'][coadapt + 1] == 'on'


def test_non_ablation_prompt_rows_keep_the_proven_recipe():
    rows = [
        row for row in experiment_rows(include_s1b=False)
        if row['run_id'].startswith(('S3/', 'H1/', 'H2/', 'H3/'))
        and not row['run_id'].endswith('/dream_only')
    ]
    forbidden_overrides = {
        '--clip-scales',
        '--blend-rho',
        '--filter-width',
        '--penal',
        '--beta-max',
        '--resolution-scale',
        '--seed',
    }
    assert rows
    for row in rows:
        assert forbidden_overrides.isdisjoint(row['argv']), row['run_id']
