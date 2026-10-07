"""The CLIP-free prompt-specificity and meaning tests."""

from __future__ import annotations

import numpy as np
import pytest

from analysis.specificity import (
    identification_accuracy,
    meaning,
    meaning_pooled,
    permutation_test,
    prompt_condition,
    specificity,
    usable_measures,
)
from recipe.campaign_spec import (
    PARAPHRASE_PROMPTS,
    PROMPTS,
    SCRAMBLED_PROMPTS,
    scrambled_origin,
)


def _letters(text: str) -> list[str]:
    return sorted(text.replace(' ', ''))


def test_each_scramble_is_an_anagram_of_the_prompt_it_controls_for():
    """The control only isolates meaning if the letters really do match."""
    origins = scrambled_origin()
    assert set(origins) == set(SCRAMBLED_PROMPTS)
    assert set(origins.values()) == set(PROMPTS)
    for scrambled, original in origins.items():
        assert _letters(scrambled) == _letters(original), original
        # Word count and word lengths match too, so neither can explain a
        # difference between a prompt and its scramble.
        assert ([len(word) for word in scrambled.split()]
                == [len(word) for word in original.split()]), original


def test_paraphrases_share_no_word_with_what_they_restate():
    """A shared word would let surface form explain any transfer."""
    for original, paraphrase in PARAPHRASE_PROMPTS.items():
        assert not set(original.split()) & set(paraphrase.split())
        assert original in PROMPTS
    # butterfly wing venation has no clean paraphrase and is left out.
    assert set(PARAPHRASE_PROMPTS) < set(PROMPTS)


def test_every_campaign_prompt_is_a_family_for_specificity():
    """Regression: scoping originals to the paraphrased families cost a third.

    `specificity` labels designs by prompt, so each prompt it cannot see is
    a label removed -- which both drops its designs and raises the chance
    rate the accuracy is judged against. Keying the scope on
    PARAPHRASE_PROMPTS silently restricted it to two families and turned a
    real effect into a null.
    """
    for prompt in PROMPTS:
        for experiment in ('S3', 'R'):
            assert prompt_condition({
                'structure': 'tall', 'experiment': experiment,
                'prompt': prompt}) == (prompt, 'original'), prompt


@pytest.mark.parametrize(
    ('meta', 'expected'),
    [
        ({'structure': 'tall', 'experiment': 'S3', 'prompt': 'fern fronds'},
         ('fern fronds', 'original')),
        ({'structure': 'tall', 'experiment': 'R', 'prompt': 'skeletons'},
         ('skeletons', 'original')),
        ({'structure': 'tall', 'experiment': 'X', 'prompt': 'bracken leaves'},
         ('fern fronds', 'paraphrase')),
        ({'structure': 'tall', 'experiment': 'X', 'prompt': 'rnef dsnorf'},
         ('fern fronds', 'scrambled')),
        ({'structure': 'tall', 'experiment': 'N', 'prompt': 'ntseleosk'},
         ('skeletons', 'scrambled')),
        # In scope even with no paraphrase to contrast against: it is still a
        # family for `specificity`, and `meaning` drops it on its own.
        ({'structure': 'tall', 'experiment': 'S3',
          'prompt': 'butterfly wing venation'},
         ('butterfly wing venation', 'original')),
        ({'structure': 'tall', 'experiment': 'N',
          'prompt': 'ytlurfebt gniw ntoeiavn'},
         ('butterfly wing venation', 'scrambled')),
        # A dial arm is a different coupling, not an 'original'.
        ({'structure': 'tall', 'experiment': 'S2', 'prompt': 'fern fronds'},
         None),
        ({'structure': 'bridge', 'experiment': 'S3', 'prompt': 'fern fronds'},
         None),
        ({'structure': 'tall', 'experiment': 'B', 'prompt': None}, None),
    ],
)
def test_prompt_condition(meta, expected):
    assert prompt_condition(meta) == expected


def _clustered(groups: int, per_group: int, spread: float) -> tuple:
    """`groups` tight clusters far apart, with `spread` jitter inside each."""
    rng = np.random.default_rng(0)
    values, labels = [], []
    for group in range(groups):
        centre = np.zeros(8)
        centre[group] = 10.0
        for _ in range(per_group):
            values.append(centre + spread * rng.standard_normal(8))
            labels.append(f'prompt-{group}')
    return np.stack(values), np.array(labels)


def test_separation_is_significant_when_prompts_cluster():
    values, labels = _clustered(3, 4, spread=0.2)
    result = permutation_test(values, labels, permutations=500)
    assert result['separation'] > 0
    assert result['p_value'] < 0.01


def test_separation_is_not_significant_when_labels_are_meaningless():
    values, labels = _clustered(3, 4, spread=0.2)
    # Same designs, labels reassigned at random: the geometry no longer
    # tracks the label, so the test must stop firing.
    shuffled = np.random.default_rng(7).permutation(labels)
    result = permutation_test(values, shuffled, permutations=500)
    assert result['p_value'] > 0.05


def test_a_p_value_is_never_exactly_zero():
    values, labels = _clustered(2, 5, spread=0.01)
    assert permutation_test(values, labels, permutations=100)['p_value'] > 0


def test_identification_reports_chance_from_the_label_mix():
    values, labels = _clustered(2, 5, spread=0.1)
    result = identification_accuracy(values, labels)
    assert result['accuracy'] == 1.0
    assert result['chance'] == pytest.approx(0.5)
    assert result['n'] == 10


def _design(family, condition, vector, measure='geometric'):
    return {
        'attempt': f'{family}/{condition}/{vector[0]}',
        'family': family,
        'condition': condition,
        measure: np.asarray(vector, dtype=float),
    }


def test_specificity_needs_two_prompts_and_four_designs():
    one = [_design('fern fronds', 'original', [i, 0]) for i in range(5)]
    assert specificity(one, 'geometric') is None
    assert specificity([], 'geometric') is None


def test_meaning_puts_a_paraphrase_nearer_than_a_scramble():
    designs = [
        _design('fern fronds', 'original', [0.0, 0.0]),
        _design('fern fronds', 'original', [0.2, 0.1]),
        _design('fern fronds', 'original', [-0.2, -0.1]),
        _design('fern fronds', 'paraphrase', [0.4, 0.3]),
        _design('fern fronds', 'paraphrase', [0.5, 0.2]),
        _design('fern fronds', 'scrambled', [6.0, 5.0]),
        _design('fern fronds', 'scrambled', [5.5, 6.0]),
    ]

    rows = meaning(designs, 'geometric')

    assert len(rows) == 1
    row = rows[0]
    assert row['family'] == 'fern fronds'
    assert row['paraphrase_n'] == 2 and row['scrambled_n'] == 2
    assert row['paraphrase_distance'] < row['scrambled_distance']
    assert row['scrambled_minus_paraphrase'] > 0
    # Two paraphrases against two scrambles admits C(4,2) = 6 label splits,
    # and the observed one has the largest gap, so the smallest p the test
    # can return is 1/6. Reaching that floor is the result here: with three
    # runs per condition the panel can resolve roughly 1/20 instead.
    assert row['p_value'] == pytest.approx(1 / 6, abs=0.02)


def test_pooling_families_beats_the_per_family_p_floor():
    """Three designs per condition floors a single family at p = 0.05."""
    designs = []
    for family, offset in (('fern fronds', 0.0), ('skeletons', 20.0)):
        for index in range(3):
            designs.append(
                _design(family, 'original', [offset + 0.1 * index, 0.0]))
            designs.append(
                _design(family, 'paraphrase', [offset + 0.5, 0.1 * index]))
            designs.append(
                _design(family, 'scrambled', [offset + 7.0, 0.1 * index]))

    per_family = meaning(designs, 'geometric')
    pooled = meaning_pooled(designs, 'geometric')

    assert {row['family'] for row in per_family} == {'fern fronds', 'skeletons'}
    assert all(row['p_value'] == pytest.approx(0.05, abs=0.01)
               for row in per_family)
    assert pooled['families'] == 2
    assert pooled['scrambled_minus_paraphrase'] > 0
    assert pooled['p_value'] < 0.01


def test_pooling_needs_more_than_one_family():
    designs = [
        _design('fern fronds', 'original', [0.0, 0.0]),
        _design('fern fronds', 'paraphrase', [1.0, 0.0]),
        _design('fern fronds', 'scrambled', [5.0, 0.0]),
    ]
    assert meaning_pooled(designs, 'geometric') is None


def test_meaning_drops_a_family_with_only_one_contrast_condition():
    """Half a contrast is not a weak result, it is not a result.

    Reporting a distance with no comparison invites reading it as one, so a
    family missing either condition leaves the table rather than appearing
    with blank columns.
    """
    designs = [
        _design('skeletons', 'original', [0.0, 0.0]),
        _design('skeletons', 'paraphrase', [1.0, 1.0]),
    ]
    assert meaning(designs, 'geometric') == []


def test_meaning_skips_a_family_with_no_originals_to_centre_on():
    designs = [_design('skeletons', 'paraphrase', [1.0, 1.0])]
    assert meaning(designs, 'geometric') == []


def test_usable_measures_puts_the_sensitive_one_first():
    """Order is sensitivity, which is also inverse circularity.

    The perceptual measure answers an appearance claim and leads; the
    model-free geometric check follows it rather than replacing it.
    """
    designs = [
        {'geometric': np.zeros(2), 'perceptual': np.zeros(2),
         'structural': np.zeros(2)}
        for _ in range(4)
    ]
    assert usable_measures(designs) == [
        'perceptual', 'geometric', 'structural']


def test_usable_measures_drops_a_measure_too_few_designs_share():
    """A measure covering one run must not reach the tables.

    The energy cache is built for a different selection, so it can cover a
    single design here. Tested at one because that is what actually
    happened, and the degenerate row it produced read as a real result.
    """
    designs = [{'geometric': np.zeros(2)} for _ in range(4)]
    designs[0]['structural'] = np.zeros(2)
    assert usable_measures(designs) == ['geometric']
