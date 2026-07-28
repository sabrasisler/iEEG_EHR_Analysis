"""Tests for feature-level QC (P2.1). No Oak, no NWB, no Slurm.

The load-bearing claim under test is the one the whole storage scheme rests on:
storing an ORDER STATISTIC of the per-bin z reproduces the "more than B of bins
above K" rule EXACTLY, so K and B stay sweepable without re-reading an NWB. If
that identity were even slightly off (e.g. by using an interpolated quantile),
every exclusion downstream would be subtly wrong in a way no single number would
reveal.

Runnable either way: `pytest tests/test_feature_qc.py` or
`python -m tests.test_feature_qc`.
"""
import numpy as np
import pandas as pd

from ieeg_ehr import config
from ieeg_ehr.qc import mask_projection
from ieeg_ehr.qc.feature_level import detect_power_outlier as dpo


# ------------------------------------------------------- the order-stat identity

def test_order_stat_reproduces_fraction_rule_exactly():
    """frac(z > K) > B  <=>  sorted_desc(z)[floor(B*n)] > K, for every n/B/K.

    Randomised over 5000 draws rather than a handful of hand-picked cases,
    because the failure mode is an off-by-one at particular (n, B) pairs where
    floor(B*n) lands on a boundary.
    """
    rng = np.random.default_rng(0)
    for _ in range(5000):
        n = int(rng.integers(3, 60))
        B = float(rng.choice(config.FEATURE_BIN_FRAC_GRID))
        K = float(rng.normal(2, 3))
        z = rng.normal(0, 2, size=n)
        k = int(np.floor(B * n))
        stat = np.sort(z)[n - 1 - k]
        assert (np.mean(z > K) > B) == (stat > K), (n, B, K)


def test_order_stat_boundary_is_strict():
    """Exactly B of the bins above K must NOT flag (the rule is strictly >B)."""
    n, B = 10, 0.20
    z = np.array([9.0, 9.0] + [0.0] * 8)      # exactly 2/10 = 0.20 above K=5
    k = int(np.floor(B * n))
    assert np.mean(z > 5.0) == B
    assert not (np.sort(z)[n - 1 - k] > 5.0)


def test_order_stats_matches_direct_sort():
    rng = np.random.default_rng(1)
    z = rng.normal(0, 2, size=(6, 4, 44))
    idx = {f: int(np.floor(f * 44)) for f in config.FEATURE_BIN_FRAC_GRID}
    out = dpo._order_stats(z, idx)
    assert np.allclose(out['max'], z.max(axis=-1))
    for f, k in idx.items():
        assert np.allclose(out[f], np.sort(z, axis=-1)[..., 44 - 1 - k])


def test_order_stats_monotone_in_frac():
    """z_b05 >= z_b10 >= z_b20 >= z_b50 -- what makes flooring on the smallest B
    guarantee no censoring anywhere in the grid."""
    rng = np.random.default_rng(2)
    z = rng.normal(0, 1, size=(20, 3, 44))
    idx = {f: int(np.floor(f * 44)) for f in config.FEATURE_BIN_FRAC_GRID}
    out = dpo._order_stats(z, idx)
    ordered = sorted(config.FEATURE_BIN_FRAC_GRID)
    for a, b in zip(ordered, ordered[1:]):
        assert (out[a] >= out[b]).all(), f'{a} !>= {b}'


# ----------------------------------------------------------- mask projection

def _mask(rows):
    return pd.DataFrame(rows, columns=mask_projection.MASK_COLUMNS)


def test_pair_excluded_if_either_contact_excluded():
    mask = _mask([['run-A', 'X1', 0.0, True], ['run-A', 'X2', 0.0, False],
                  ['run-A', 'X3', 0.0, False]])
    got = mask_projection.project_to_pairs(mask, 'run-A', ['X1-X2', 'X2-X3'],
                                           np.array([0.0, 30.0]))
    # X1 bad -> X1-X2 bad; X2 and X3 both fine -> X2-X3 fine.
    assert got.tolist() == [[True, False], [True, False]]


def test_cathode_side_also_excludes():
    mask = _mask([['run-A', 'X1', 0.0, False], ['run-A', 'X2', 0.0, True]])
    got = mask_projection.project_to_pairs(mask, 'run-A', ['X1-X2'], np.array([0.0]))
    assert got.tolist() == [[True]]


def test_window_inherits_its_enclosing_60s_bin():
    mask = _mask([['run-A', 'X1', 0.0, True], ['run-A', 'X1', 60.0, False],
                  ['run-A', 'X2', 0.0, False], ['run-A', 'X2', 60.0, False]])
    secs = np.array([0.0, 59.9, 60.0, 119.9])
    got = mask_projection.project_to_pairs(mask, 'run-A', ['X1-X2'], secs)
    assert got.ravel().tolist() == [True, True, False, False]


def test_bin_absent_from_mask_is_not_excluded():
    """A window whose 60s bin has no mask row at all must pass, matching the
    fillna(False) convention the raw-voltage consumers already use -- NOT be
    dropped, and NOT raise."""
    mask = _mask([['run-A', 'X1', 0.0, True], ['run-A', 'X2', 0.0, False]])
    got = mask_projection.project_to_pairs(mask, 'run-A', ['X1-X2'],
                                           np.array([0.0, 6000.0]))
    assert got.ravel().tolist() == [True, False]


def test_channel_absent_from_mask_is_not_excluded():
    mask = _mask([['run-A', 'Q9', 0.0, True]])
    got = mask_projection.project_to_pairs(mask, 'run-A', ['X1-X2'], np.array([0.0]))
    assert got.ravel().tolist() == [False]


def test_other_run_does_not_leak():
    mask = _mask([['run-B', 'X1', 0.0, True], ['run-B', 'X2', 0.0, False]])
    got = mask_projection.project_to_pairs(mask, 'run-A', ['X1-X2'], np.array([0.0]))
    assert got.ravel().tolist() == [False]


def test_none_mask_is_all_false():
    got = mask_projection.project_to_pairs(None, 'run-A', ['X1-X2', 'X2-X3'],
                                           np.array([0.0, 1.0, 2.0]))
    assert got.shape == (3, 2)
    assert not got.any()


def test_split_pair_handles_monopolar_name():
    assert mask_projection.split_pair('LAH1-LAH2') == ('LAH1', 'LAH2')
    assert mask_projection.split_pair('LAH1') == ('LAH1', None)


# ------------------------------------------------------------------ accumulator

def test_accumulator_mean_std_match_numpy():
    """Streaming sums across chunks must equal a one-shot numpy mean/std."""
    rng = np.random.default_rng(3)
    n_bins, n_pairs = 5, 2
    data = rng.normal(-11, 1.5, size=(1000, n_pairs, n_bins))
    channels = ['A-B', 'B-C']

    acc = dpo._Accumulator(n_bins)
    for t0 in range(0, 1000, 137):                     # deliberately ragged chunks
        block = data[t0:t0 + 137]
        usable = np.ones_like(block, dtype=bool)
        acc.add(channels, block, usable, ~usable)
    out = acc.finalize(min_windows=10)

    for j, ch in enumerate(channels):
        mean, std, n, _nf, degen = out[ch]
        assert not degen.any()
        assert (n == 1000).all()
        assert np.allclose(mean, data[:, j].mean(axis=0))
        assert np.allclose(std, data[:, j].std(axis=0))   # population std


def test_accumulator_ignores_unusable_windows():
    """Masked/non-finite windows must not enter the baseline at all."""
    n_bins = 3
    good = np.full((10, 1, n_bins), 2.0)
    bad = np.full((10, 1, n_bins), 1000.0)
    block = np.concatenate([good, bad])
    usable = np.concatenate([np.ones_like(good, bool), np.zeros_like(bad, bool)])

    acc = dpo._Accumulator(n_bins)
    acc.add(['A-B'], block, usable, ~usable)
    mean, std, n, _nf, _degen = acc.finalize(min_windows=1)['A-B']
    assert (n == 10).all()
    assert np.allclose(mean, 2.0)
    assert np.allclose(std, 0.0)


def test_too_few_windows_is_degenerate():
    n_bins = 3
    block = np.full((5, 1, n_bins), 2.0)
    acc = dpo._Accumulator(n_bins)
    acc.add(['A-B'], block, np.ones_like(block, bool), np.zeros_like(block, bool))
    _m, _s, _n, _nf, degen = acc.finalize(min_windows=100)['A-B']
    assert degen.all(), 'a 5-window baseline must not be trusted'


def test_zero_variance_is_degenerate():
    """std == 0 would divide z by zero; it must be flagged instead, so the
    channel fails loudly into the cascade (gross_artifact's convention)."""
    n_bins = 2
    block = np.full((200, 1, n_bins), -7.0)
    acc = dpo._Accumulator(n_bins)
    acc.add(['A-B'], block, np.ones_like(block, bool), np.zeros_like(block, bool))
    _m, std, _n, _nf, degen = acc.finalize(min_windows=100)['A-B']
    assert np.allclose(std, 0.0)
    assert degen.all()


def test_accumulator_aligns_by_channel_name_not_position():
    """Two runs exposing channels in different orders must not be averaged
    together by position."""
    n_bins = 2
    acc = dpo._Accumulator(n_bins)
    a = np.full((100, 2, n_bins), 0.0)
    a[:, 0] = 1.0
    a[:, 1] = 5.0
    acc.add(['A-B', 'B-C'], a, np.ones_like(a, bool), np.zeros_like(a, bool))
    b = np.full((100, 2, n_bins), 0.0)
    b[:, 0] = 5.0                                     # same values, swapped order
    b[:, 1] = 1.0
    acc.add(['B-C', 'A-B'], b, np.ones_like(b, bool), np.zeros_like(b, bool))

    out = acc.finalize(min_windows=10)
    assert np.allclose(out['A-B'][0], 1.0)
    assert np.allclose(out['B-C'][0], 5.0)


# ----------------------------------------------------------------------- labels

def test_exclusion_label_is_self_documenting():
    assert config.feature_exclusion_label(5.0, 0.20) == 'z5_binfrac20'
    assert config.feature_exclusion_label(4.5, 0.05) == 'z4.5_binfrac5'
    assert config.feature_exclusion_label(5.0, 0.20, 'both') == 'z5_binfrac20_both'


def test_configured_frac_is_in_the_stored_grid():
    """The operative B must be one of the stored order statistics, or the metric
    tables cannot answer the configured rule at all."""
    assert config.FEATURE_BIN_FRAC in config.FEATURE_BIN_FRAC_GRID


def test_store_floor_below_threshold():
    """K must sit above the storage floor, or the sparse table would censor rows
    the configured threshold needs."""
    assert config.FEATURE_METRIC_STORE_FLOOR < config.FEATURE_Z_THRESH


if __name__ == '__main__':
    import sys
    import pytest
    sys.exit(pytest.main([__file__, '-v']))
