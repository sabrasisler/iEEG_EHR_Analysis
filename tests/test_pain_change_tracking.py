"""Per-subject tracking metrics and their within-subject permutation null."""

import numpy as np
import pandas as pd

from ieeg_ehr.analysis import pain_change_tracking as track


def _pairs(pred, actual):
    return pd.DataFrame({'value': pred, 'delta_pain': actual,
                         'sum_beta2': 0.004, 'n_cells_used': 70})


def test_a_monotone_subject_scores_rho_one_above_its_null():
    actual = np.array([-3, -2, -1, 0, 0, 1, 2, 3, 4, 5])
    row, null = track.subject_metrics(_pairs(2 * actual + 0.1, actual), 500,
                                      np.random.default_rng(0))
    assert row['spearman_rho'] == 1.0 and np.isclose(row['slope'], 2.0)
    assert row['perm_percentile'] > 99 and abs(np.median(null)) < 0.1
    assert row['n_levels'] == 9 and (row['delta_min'], row['delta_max']) == (-3, 5)


def test_the_null_is_spearman_of_shuffled_ranks():
    """Each null value equals scipy's Spearman on the matching shuffle, ties included."""
    from scipy import stats
    pred = np.array([0.3, -1.2, 2.0, 0.0, 1.1, -0.4])
    actual = np.array([1.0, -2.0, 1.0, 0.0, 3.0, -2.0])
    shuffles = np.random.default_rng(1).permuted(np.tile(actual, (5, 1)), axis=1)
    mine = track._row_corr(track._ranks(pred), track._ranks(shuffles))
    expected = [stats.spearmanr(pred, s).statistic for s in shuffles]
    np.testing.assert_allclose(mine, expected)
