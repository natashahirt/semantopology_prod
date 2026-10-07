"""Manifest composition. Does not submit jobs."""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pytest

from model.model_base import Model
from recipe.campaign import onoff, preset_from_args, run_campaign
from recipe.campaign_spec import (
    CONTROL_PROMPTS,
    DIAL_SEED_RHOS,
    DIAL_SEEDS,
    DIVERSITY_PROMPTS,
    FERN_WORDINGS,
    GRAVITY_LOAD,
    LOOSE_WEIGHT_ENDS,
    LOOSE_WEIGHT_STARTS,
    PROMPTS,
    SKETCH_WEIGHT_LOW_ENDS,
    SKETCH_WEIGHT_LOW_START,
    TYPOLOGY_PROMPTS,
    prompt_token,
)
from recipe.preset import PAPER
from run import build_parser
from slurm.make_manifest import (
    control_prompt_rows,
    dial_seed_rows,
    experiment_rows,
    extension_rows,
    fern_wording_rows,
    fixed_weight_rows,
    loose_sketch_rows,
    sketch_weight_low_rows,
    typology_rows,
)


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


def test_s4_fern_wording_is_off_the_main_table():
    campaign = experiment_rows(include_s1b=False)
    assert len(campaign) == 118
    assert all(not row['run_id'].startswith('S4/') for row in campaign)

    rows = fern_wording_rows()
    assert len(rows) == 8
    ids = [row['run_id'] for row in rows]
    assert len(ids) == len(set(ids))
    assert 'S4/tall/fern_frond/hybrid' in ids
    assert 'S4/tall/many_fern_fronds/semantic' in ids
    assert 'S4/tall/field_of_ferns/hybrid' in ids
    assert 'S4/tall/unfurling_fern_fronds/semantic' in ids
    assert 'S4/tall/fern_fronds/hybrid' not in ids
    argv_blob = ' '.join(' '.join(row['argv']) for row in rows)
    clip_texts = [
        row['argv'][row['argv'].index('--clip') + 1] for row in rows
    ]
    assert PROMPTS[0] not in clip_texts
    for prompt in FERN_WORDINGS:
        if prompt == PROMPTS[0]:
            continue
        assert prompt in argv_blob
    for row in rows:
        argv = row['argv']
        assert argv.count('--physics-beta-max') == 1
        assert argv[argv.index('--physics-beta-max') + 1] == '8'
        assert argv[argv.index('--structure') + 1] == 'tall'


def test_c1_control_prompts_match_s3_off_the_main_table():
    campaign = experiment_rows(include_s1b=False)
    assert all(not row['run_id'].startswith('C1/') for row in campaign)
    s3 = {
        row['run_id']: row['argv'] for row in campaign
        if row['run_id'].startswith('S3/')
    }

    rows = control_prompt_rows()
    assert len(rows) == 6
    ids = [row['run_id'] for row in rows]
    assert len(ids) == len(set(ids))
    assert 'C1/tall/structure' in ids
    assert 'C1/bridge/qzv_xlrp_mnek' in ids
    for row in rows:
        argv = row['argv']
        _, structure, _ = row['run_id'].split('/')
        assert argv[argv.index('--clip') + 1] in CONTROL_PROMPTS
        reference = s3[f'S3/{structure}/{prompt_token(PROMPTS[0])}']
        assert _without_identity(argv) == _without_identity(reference), row['run_id']


def test_c2_fixed_weight_rows_are_s3_plus_two_weights():
    s3 = {
        row['run_id']: row['argv'] for row in experiment_rows(include_s1b=False)
        if row['run_id'].startswith('S3/')
    }
    rows = fixed_weight_rows(12.5, 340.0)
    assert len(rows) == 9
    for row in rows:
        argv = row['argv']
        _, structure, token = row['run_id'].split('/')
        reference = s3[f'S3/{structure}/{token}']
        assert _without_identity(argv) == [
            *_without_identity(reference)[:-2],
            '--clip-weight', '12.5', '--clip-weight-z', '340',
            *_without_identity(reference)[-2:],
        ], row['run_id']
        preset = preset_from_args(build_parser().parse_args(argv))
        assert (preset.clip_weight, preset.clip_weight_z) == (12.5, 340.0)


def test_campaign_presets_keep_grad_match():
    for row in experiment_rows(include_s1b=True):
        preset = preset_from_args(build_parser().parse_args(row['argv']))
        assert preset.clip_weight is None, row['run_id']
        assert preset.clip_weight_z is None, row['run_id']


def test_fixed_weights_are_refused_outside_semantic(tmp_path):
    args = build_parser().parse_args([
        '--mode', 'hybrid', '--clip-weight', '1', '--clip-weight-z', '1'])
    with pytest.raises(ValueError, match='semantic-only'):
        run_campaign(args, tmp_path)
    args = build_parser().parse_args(['--mode', 'semantic', '--clip-weight', '1'])
    with pytest.raises(ValueError, match='set together'):
        run_campaign(args, tmp_path)


_PANEL_FLAGS = {'--blend-rho', '--blend-rho-z', '--volume-fraction', '--seed'}


def _without_panel_flag(argv: list[str]) -> list[str]:
    """Drop identity and the one flag a Phase 2c panel varies."""
    out, skip = [], False
    for value in _without_identity(argv):
        if skip:
            skip = False
        elif value in _PANEL_FLAGS:
            skip = True
        else:
            out.append(value)
    return out


def test_extensions_are_off_the_main_table_in_queue_order():
    rows = extension_rows()
    assert len(rows) == 148
    ids = [row['run_id'] for row in rows]
    assert len(ids) == len(set(ids))
    elsewhere = {
        row['run_id'] for row in (
            experiment_rows(include_s1b=True) + fern_wording_rows()
            + control_prompt_rows() + fixed_weight_rows(1.0, 1.0))
    }
    assert elsewhere.isdisjoint(ids)
    order = list(dict.fromkeys(run_id.split('/')[0] for run_id in ids))
    assert order == ['S2b', 'A', 'P', 'N', 'L', 'V', 'M', 'R']
    counts = {name: sum(i.startswith(name + '/') for i in ids) for name in order}
    assert counts == {
        'S2b': 30, 'A': 3, 'P': 9, 'N': 7, 'L': 3, 'V': 24, 'M': 24, 'R': 48}
    assert 'P/tall/honeycomb' in ids
    assert 'V/bridge/unguided/vf-0.20' in ids
    assert 'R/short/skeletons/seed-404' in ids
    assert {i.split('/')[2] for i in ids if i.startswith('M/')} == {
        prompt_token(p) for p in DIVERSITY_PROMPTS}


def test_extensions_change_only_their_panel_flag():
    campaign = {row['run_id']: row['argv'] for row in experiment_rows(include_s1b=False)}
    for row in extension_rows():
        argv = row['argv']
        experiment, structure = row['run_id'].split('/')[:2]
        mode = argv[argv.index('--mode') + 1]
        reference = campaign[
            f'B/{structure}/unguided' if mode == 'unguided'
            else f'S3/{structure}/{prompt_token(PROMPTS[0])}']
        assert _without_panel_flag(argv) == _without_identity(reference), row['run_id']
        varied = _PANEL_FLAGS.intersection(argv)
        expected = {
            'S2b': {'--blend-rho'}, 'A': {'--blend-rho-z'},
            'V': {'--volume-fraction'}, 'R': {'--seed'},
        }.get(experiment, set())
        assert varied == expected, row['run_id']


def test_extension_flags_reach_the_preset():
    rows = {row['run_id']: row['argv'] for row in extension_rows()}

    def preset(run_id):
        return preset_from_args(build_parser().parse_args(rows[run_id]))

    density_only = preset('A/tall/fern_fronds/density-only')
    assert (density_only.blend_rho, density_only.blend_rho_z) == (1.0, 0.0)
    assert preset('V/bridge/skeletons/vf-0.40').density == 0.4
    assert preset('R/tall/unguided/seed-101').seed == 101
    s2b = preset('S2b/bridge/fern_fronds/rho-0.00')
    assert (s2b.blend_rho, s2b.blend_rho_z) == (0.0, PAPER.blend_rho_z)
    for argv in rows.values():
        assert preset_from_args(build_parser().parse_args(argv)).clip_weight is None


def test_volume_and_raw_z_flags_default_to_the_campaign_recipe():
    args = build_parser().parse_args(['--mode', 'semantic', '--structure', 'bridge'])
    preset = preset_from_args(args)
    assert preset.density == 0.3
    assert preset.blend_rho_z == PAPER.blend_rho_z
    with pytest.raises(ValueError, match='volume-fraction'):
        preset_from_args(build_parser().parse_args(['--volume-fraction', '1.2']))


def test_raw_z_mixer_is_refused_with_fixed_weights(tmp_path):
    args = build_parser().parse_args([
        '--mode', 'semantic', '--clip-weight', '1', '--clip-weight-z', '1',
        '--blend-rho-z', '0'])
    with pytest.raises(ValueError, match='grad-match only'):
        run_campaign(args, tmp_path)


def test_typology_panel_is_s3_on_every_structure():
    campaign = {row['run_id']: row['argv'] for row in experiment_rows(include_s1b=False)}
    assert all(not run_id.startswith('G/') for run_id in campaign)

    rows = typology_rows()
    assert len(rows) == 18
    ids = [row['run_id'] for row in rows]
    assert len(ids) == len(set(ids))
    # Structure outer, so a cut queue still leaves tall complete.
    assert [i.split('/')[1] for i in ids[:6]] == ['tall'] * 6
    assert 'G/tall/sedimentary_rock_layers' in ids
    assert 'G/bridge/roman_aqueduct_arches' in ids
    assert 'G/short/a_spiral_staircase' in ids
    for row in rows:
        argv = row['argv']
        structure = row['run_id'].split('/')[1]
        reference = campaign[f'S3/{structure}/{prompt_token(PROMPTS[0])}']
        assert _without_identity(argv) == _without_identity(reference), row['run_id']
        assert argv[argv.index('--clip') + 1] in TYPOLOGY_PROMPTS
        preset = preset_from_args(build_parser().parse_args(argv))
        assert preset.clip_weight is None, row['run_id']
        assert preset.density == 0.3, row['run_id']


def test_typology_prompts_are_new_to_the_campaign():
    seen = {
        row['argv'][row['argv'].index('--clip') + 1]
        for row in (
            experiment_rows(include_s1b=True) + extension_rows()
            + fern_wording_rows() + control_prompt_rows())
        if '--clip' in row['argv']
    }
    assert seen.isdisjoint(TYPOLOGY_PROMPTS)


def test_dial_seed_panel_reruns_the_dial_at_new_seeds():
    rows = dial_seed_rows()
    assert len(rows) == len(DIAL_SEEDS) * len(DIAL_SEED_RHOS) == 9
    ids = [row['run_id'] for row in rows]
    assert len(ids) == len(set(ids))
    # The campaign seed is deliberately absent: S2 already ran the dial there,
    # so these rows are the extra trajectories that turn the curve into a band.
    assert 12 not in DIAL_SEEDS
    campaign = {row['run_id'] for row in experiment_rows(include_s1b=False)}
    assert all(run_id not in campaign for run_id in ids)
    for row in rows:
        argv = row['argv']
        assert argv[argv.index('--clip') + 1] == PROMPTS[0]
        assert float(argv[argv.index('--blend-rho') + 1]) in DIAL_SEED_RHOS
        assert int(argv[argv.index('--seed') + 1]) in DIAL_SEEDS
        preset = preset_from_args(build_parser().parse_args(argv))
        assert preset.clip_weight is None, row['run_id']


def test_loose_sketch_panel_sweeps_both_ends_of_the_ramp():
    rows = loose_sketch_rows()
    assert len(rows) == 10
    ids = [row['run_id'] for row in rows]
    assert len(ids) == len(set(ids))
    campaign = {row['run_id'] for row in experiment_rows(include_s1b=False)}
    assert all(run_id not in campaign for run_id in ids)
    # Every rung is looser than the default the rigid runs used, which is the
    # whole point: WEIGHT_ENDS only has one value below it.
    assert all(end < PAPER.sketch_weight_end for end in LOOSE_WEIGHT_ENDS)
    assert all(start < PAPER.sketch_weight_start for start in LOOSE_WEIGHT_STARTS)

    presets = {}
    for row in rows:
        args = build_parser().parse_args(row['argv'])
        args.sketch_init = onoff(args.sketch_init)
        args.sketch_weight = onoff(args.sketch_weight)
        args.coadapt = onoff(args.coadapt or 'off')
        args.device = args.device or PAPER.device
        presets[row['run_id']] = preset_from_args(args)

    # The end ladder moves only the end; the start rows move only the start.
    end_row = presets['F2b/tall/sketch-col3_braced/wend-0']
    assert end_row.sketch_weight_end == 0.0
    assert end_row.sketch_weight_start == PAPER.sketch_weight_start
    start_row = presets['F2b/tall/sketch-col3_braced/wstart-1000']
    assert start_row.sketch_weight_start == 1000.0
    assert start_row.sketch_weight_end == PAPER.sketch_weight_end
    assert all(preset.use_sketch_weight for preset in presets.values())


def test_decaying_to_zero_releases_the_prior_at_full_resolution():
    """`wend-0` keeps the coarse-grid prior and ends with none at all."""
    model = SimpleNamespace(
        sketch_weight_start=4000.0, sketch_weight_end=0.0,
        resize_num=4, resizes=0)
    at = Model.sketch_weight_at.__get__(model)
    assert at() == 4000.0
    model.resizes = 2
    assert at() == 2000.0
    model.resizes = 4
    assert at() == 0.0


def _without_identity(argv: list[str]) -> list[str]:
    """Drop the run id, experiment and prompt so two rows' recipes compare."""
    out, skip = [], False
    for value in argv:
        if skip:
            skip = False
        elif value in ('--run-id', '--experiment', '--clip'):
            skip = True
        else:
            out.append(value)
    return out


def test_f2b_low_sketch_weight_is_off_the_main_table():
    campaign = experiment_rows(include_s1b=False)
    assert all(not row['run_id'].startswith('F2b/') for row in campaign)
    f2 = next(row for row in campaign if row['run_id'] == 'F2/tall/sketch-12/wend-200')
    assert '--sketch-weight-start' not in f2['argv']

    rows = sketch_weight_low_rows()
    assert len(rows) == len(SKETCH_WEIGHT_LOW_ENDS)
    ends = []
    for row in rows:
        argv = row['argv']
        assert argv[argv.index('--sketch-weight-start') + 1] == str(SKETCH_WEIGHT_LOW_START)
        assert argv[argv.index('--mode') + 1] == 'sketch'
        assert argv[argv.index('--physics-beta-max') + 1] == '8'
        ends.append(float(argv[argv.index('--sketch-weight-end') + 1]))
    assert ends == list(SKETCH_WEIGHT_LOW_ENDS)
    assert 'F2b/tall/sketch-12/wstart-200/wend-0' in {row['run_id'] for row in rows}
