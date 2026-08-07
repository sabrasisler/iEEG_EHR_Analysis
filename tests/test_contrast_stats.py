"""
Tests for the paired contrast and its permutation noise floor
(analysis/contrast_stats.py) and the 0-pain eligibility filter
(analysis/view_tables.exclude_thin_baseline_subjects).

The one that carries the most weight is `test_null_is_centred_on_zero`: the whole
`floor_ratio` convention is worthless if the null is biased, because then every
effect would be measured against a floor that is not the floor.
"""

import numpy as np
import pandas as pd
import pytest

from ieeg_ehr.analysis import contrast_stats, view_tables


def make_epochs(n_subjects=12, n_per_bin=15, effect=0.0, seed=0, regions=('M1', 'S1'),
                none_epochs=None):
    """Synthetic slope tables: one row per (subject, region, epoch).

    `effect` is added to every `high` epoch, so the true contrast is exactly it.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for s in range(n_subjects):
        subject = f'sub-{s:03d}'
        offset = rng.normal(-1.9, 0.2)          # between-subject scale
        eid = 0
        for region in regions:
            for pain_bin in ('none', 'low', 'high'):
                n = (none_epochs if (pain_bin == 'none' and none_epochs is not None)
                     else n_per_bin)
                for _ in range(n):
                    value = offset + rng.normal(0, 0.15)
                    if pain_bin == 'high':
                        value += effect
                    rows.append((subject, region, eid, pain_bin, value))
                    eid += 1
    return pd.DataFrame(rows, columns=['subject_id', 'region', 'epoch_id',
                                       'pain_bin', 'slope'])


# ---------------------------------------------------------------------------
# paired_contrast
# ---------------------------------------------------------------------------

def test_recovers_a_planted_effect():
    df = make_epochs(effect=0.05, seed=1)
    out = contrast_stats.paired_contrast(df, 'high', 'none')
    assert len(out) == 12 * 2
    assert out['value'].mean() == pytest.approx(0.05, abs=0.02)
    assert (out['contrast'] == 'high-none').all()
    assert (out['n_high'] == 15).all() and (out['n_none'] == 15).all()


def test_per_bin_floor_drops_thin_cells_only():
    df = make_epochs(n_subjects=4, n_per_bin=15, seed=2)
    # Thin out `high` for one subject/region only.
    mask = ((df['subject_id'] == 'sub-000') & (df['region'] == 'M1')
            & (df['pain_bin'] == 'high'))
    df = pd.concat([df[~mask], df[mask].head(3)], ignore_index=True)

    kept = contrast_stats.paired_contrast(df, 'high', 'none', min_a=10)
    assert len(kept) == 4 * 2 - 1
    assert not ((kept['subject_id'] == 'sub-000') & (kept['region'] == 'M1')).any()
    # With the floor lowered the cell comes back, so the drop is the threshold
    # talking rather than a broken pivot.
    assert len(contrast_stats.paired_contrast(df, 'high', 'none', min_a=3)) == 8


def test_missing_bin_is_dropped_not_treated_as_zero():
    df = make_epochs(n_subjects=3, seed=3)
    df = df[~((df['subject_id'] == 'sub-000') & (df['pain_bin'] == 'high'))]
    out = contrast_stats.paired_contrast(df, 'high', 'none', min_a=1)
    assert 'sub-000' not in set(out['subject_id'])
    assert out['value'].notna().all()


# ---------------------------------------------------------------------------
# permutation_null — the floor
# ---------------------------------------------------------------------------

def test_null_is_centred_on_zero():
    """No pain association planted, so the observed effect must sit AT the floor."""
    df = make_epochs(effect=0.0, seed=4)
    s = contrast_stats.permutation_null(df, 'high', 'none', n_perm=100, seed=0,
                                        by_region=False)
    assert len(s) == 1
    row = s.iloc[0]
    assert abs(row['observed_mean']) < 3 * row['floor_group']
    assert row['perm_p'] > 0.05
    assert row['floor_group'] > 0


def test_planted_effect_clears_the_floor():
    df = make_epochs(effect=0.10, seed=5)
    s = contrast_stats.permutation_null(df, 'high', 'none', n_perm=100, seed=0,
                                        by_region=False).iloc[0]
    assert s['observed_mean'] == pytest.approx(0.10, abs=0.03)
    assert s['floor_ratio'] > 3
    assert s['perm_p'] == pytest.approx(1 / 101, abs=1e-9)


def test_p_value_never_reports_exactly_zero():
    """1/(n_perm+1) is the smallest honest p; an exact 0 would overclaim."""
    df = make_epochs(effect=0.5, seed=6)
    s = contrast_stats.permutation_null(df, 'high', 'none', n_perm=50, seed=0,
                                        by_region=False).iloc[0]
    assert s['perm_p'] > 0
    assert s['perm_p'] == pytest.approx(1 / 51, abs=1e-9)


def test_permutation_is_reproducible_and_seed_dependent():
    df = make_epochs(effect=0.03, seed=7)
    kw = dict(n_perm=40, by_region=False)
    a = contrast_stats.permutation_null(df, 'high', 'none', seed=0, **kw).iloc[0]
    b = contrast_stats.permutation_null(df, 'high', 'none', seed=0, **kw).iloc[0]
    c = contrast_stats.permutation_null(df, 'high', 'none', seed=1, **kw).iloc[0]
    assert a['floor_group'] == b['floor_group']
    assert a['observed_mean'] == c['observed_mean']      # observed is not random
    assert a['floor_group'] != c['floor_group']


def test_by_region_gives_one_row_per_region():
    df = make_epochs(effect=0.05, seed=8, regions=('M1', 'S1', 'Insula'))
    s = contrast_stats.permutation_null(df, 'high', 'none', n_perm=30, seed=0,
                                        by_region=True)
    assert sorted(s['region']) == ['Insula', 'M1', 'S1']
    assert s['floor_group'].gt(0).all()


def test_cell_floor_exceeds_group_floor():
    """A single cell is noisier than the mean of many; the figure relies on this."""
    df = make_epochs(effect=0.0, seed=9)
    s = contrast_stats.permutation_null(df, 'high', 'none', n_perm=100, seed=0,
                                        by_region=False).iloc[0]
    assert s['floor_cell'] > s['floor_group']


# ---------------------------------------------------------------------------
# floor_ratio
# ---------------------------------------------------------------------------

def test_floor_ratio_is_absolute_and_nan_safe():
    np.testing.assert_allclose(contrast_stats.floor_ratio([0.05, -0.05], [0.01, 0.01]),
                               [5.0, 5.0])
    assert np.isnan(contrast_stats.floor_ratio([0.05], [0.0])[0])


# ---------------------------------------------------------------------------
# the 0-pain eligibility filter
# ---------------------------------------------------------------------------

def test_excludes_exactly_the_thin_baseline_subjects():
    thick = make_epochs(n_subjects=3, none_epochs=12, seed=10)
    thin = make_epochs(n_subjects=2, none_epochs=2, seed=11)
    thin['subject_id'] = thin['subject_id'].str.replace('sub-0', 'sub-9', regex=False)
    df = pd.concat([thick, thin], ignore_index=True)

    kept, excluded = view_tables.exclude_thin_baseline_subjects(df, min_none_epochs=5)
    assert excluded == ['sub-900', 'sub-901']
    assert set(kept['subject_id']) == {'sub-000', 'sub-001', 'sub-002'}


def test_subject_with_no_zero_pain_epochs_is_excluded():
    df = make_epochs(n_subjects=2, seed=12)
    df = df[~((df['subject_id'] == 'sub-000') & (df['pain_bin'] == 'none'))]
    _, excluded = view_tables.exclude_thin_baseline_subjects(df, min_none_epochs=5)
    assert excluded == ['sub-000']


def test_filter_is_subject_level_not_cell_level():
    """A subject passing overall keeps ALL their regions, even a sparse one."""
    df = make_epochs(n_subjects=1, none_epochs=12, regions=('M1', 'S1'), seed=13)
    drop = ((df['region'] == 'S1') & (df['pain_bin'] == 'none'))
    df = pd.concat([df[~drop], df[drop].head(1)], ignore_index=True)

    kept, excluded = view_tables.exclude_thin_baseline_subjects(df, min_none_epochs=5)
    assert excluded == []
    assert set(kept['region']) == {'M1', 'S1'}


def test_zero_floor_is_a_no_op():
    df = make_epochs(n_subjects=2, none_epochs=1, seed=14)
    kept, excluded = view_tables.exclude_thin_baseline_subjects(df, min_none_epochs=0)
    assert excluded == []
    assert len(kept) == len(df)
