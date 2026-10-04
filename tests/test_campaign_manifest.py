"""Manifest composition. Does not submit jobs."""

from __future__ import annotations

from dataclasses import replace

from recipe.campaign import onoff, preset_from_args
from recipe.campaign_spec import GRAVITY_LOAD
from recipe.preset import PAPER
from run import build_parser
from slurm.make_manifest import experiment_rows


def test_default_count_is_118():
    rows = experiment_rows(include_s1b=False)
    assert len(rows) == 118
    ids = [row['run_id'] for row in rows]
    assert len(ids) == len(set(ids))
    assert 'S1b/short/butterfly_wing_venation/m' not in ids
    assert 'H4/tall/skeletons/coadapt-off' in ids
    assert 'H2/bridge/fern_fronds/hybrid' in ids
    assert 'H5/bridge/fern_fronds/hybrid' in ids
    assert 'S2/tall/butterfly_wing_venation/rho-0.50' in ids
    assert 'S1/tall/skeletons/gme' in ids


def test_s1b_adds_24():
    rows = experiment_rows(include_s1b=True)
    assert len(rows) == 142
    ids = [row['run_id'] for row in rows]
    assert 'S1b/short/butterfly_wing_venation/m' in ids
    assert 'S1b/bridge/fern_fronds/e' in ids


def test_three_prompts_on_hybrid_and_semantic():
    argv_blob = ' '.join(
        ' '.join(row['argv']) for row in experiment_rows(include_s1b=False)
        if row['run_id'].startswith('H1/') or row['run_id'].startswith('S3/tall/'))
    assert 'fern fronds' in argv_blob
    assert 'butterfly wing venation' in argv_blob
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


def test_h1_manifest_resolves_to_the_hardfork_recipe():
    row = next(
        row for row in experiment_rows(include_s1b=False)
        if row['run_id'] == 'H1/tall/butterfly_wing_venation/hybrid'
    )
    args = build_parser().parse_args(row['argv'])
    args.sketch_init = onoff(args.sketch_init)
    args.sketch_weight = onoff(args.sketch_weight)
    args.coadapt = onoff(
        'on' if args.coadapt is None and args.mode == 'hybrid'
        else args.coadapt
    )
    args.device = args.device or PAPER.device

    assert preset_from_args(args) == replace(
        PAPER,
        problem_name='tall_building',
        clip_prompt='butterfly wing venation',
        physics_projection_beta_max=8.0,
    )


def test_every_row_uses_the_adopted_physics_projection():
    rows = experiment_rows(include_s1b=True)
    for row in rows:
        argv = row['argv']
        assert argv.count('--physics-beta-max') == 1, row['run_id']
        cap = float(argv[argv.index('--physics-beta-max') + 1])
        assert cap == 8.0, row['run_id']
        if row['run_id'].startswith('D/'):
            assert argv.count('--beta-max') == 1, row['run_id']
            sampled = float(argv[argv.index('--beta-max') + 1])
            assert 4.0 <= sampled <= 16.0, row['run_id']
        else:
            assert '--beta-max' not in argv, row['run_id']


def test_h5_is_standard_hybrid_plus_gravity():
    rows = [
        row for row in experiment_rows(include_s1b=False)
        if row['run_id'].startswith('H5/')
    ]
    assert len(rows) == 9
    ids = {row['run_id'] for row in rows}
    assert 'H5/tall/fern_fronds/hybrid' in ids
    assert 'H5/short/butterfly_wing_venation/hybrid' in ids
    assert 'H5/bridge/skeletons/hybrid' in ids
    for row in rows:
        assert row['argv'][row['argv'].index('--mode') + 1] == 'hybrid'
        assert row['argv'][row['argv'].index('--gravity-load') + 1] == str(GRAVITY_LOAD)
    h1 = next(
        row for row in experiment_rows(include_s1b=False)
        if row['run_id'] == 'H1/tall/fern_fronds/hybrid'
    )
    assert '--gravity-load' not in h1['argv']
    args = build_parser().parse_args(rows[0]['argv'])
    args.sketch_init = onoff(args.sketch_init)
    args.sketch_weight = onoff(args.sketch_weight)
    args.coadapt = onoff(
        'on' if args.coadapt is None and args.mode == 'hybrid'
        else args.coadapt
    )
    args.device = args.device or PAPER.device
    assert preset_from_args(args).gravity_load == GRAVITY_LOAD
