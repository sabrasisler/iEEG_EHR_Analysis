"""
Tests for the 1/f slope fit (views/aperiodic.py).

The one that matters most is `test_slope_of_mean_equals_mean_of_slopes`: the whole
builder is designed around fitting the EPOCH MEAN rather than each 2 s window, and
that is only free because an OLS slope is a fixed-weight linear functional of y.
If that property ever broke, the cheap path would silently stop equalling the
expensive one.
"""

import numpy as np
import pandas as pd
import pytest

from ieeg_ehr.views import aperiodic


def bin_table(n=50, lo=1.0, hi=250.0):
    """The cache's layout: n log-spaced bins between lo and hi Hz."""
    edges = np.logspace(np.log10(lo), np.log10(hi), n + 1)
    return pd.DataFrame({'freq_bin_index': np.arange(n),
                         'bin_low_hz': edges[:-1], 'bin_high_hz': edges[1:]})


# ---------------------------------------------------------------------------
# fit_bins
# ---------------------------------------------------------------------------

def test_fit_bins_full_range_drops_line_noise():
    idx, log_f = aperiodic.fit_bins(bin_table(), 1.0, 250.0, drop_bins=[36, 37, 43])
    assert len(idx) == 47
    assert not ({36, 37, 43} & set(idx.tolist()))
    assert len(log_f) == len(idx)
    assert np.all(np.diff(log_f) > 0)


def test_fit_bins_selects_on_geometric_centre():
    bt = bin_table()
    idx, _ = aperiodic.fit_bins(bt, 30.0, 250.0)
    centres = np.sqrt(bt['bin_low_hz'] * bt['bin_high_hz'])
    assert centres[idx].min() >= 30.0
    # The bin just below the cut is excluded even though its high edge is above it.
    assert (idx.min() - 1) not in idx


def test_fit_bins_refuses_a_range_with_fewer_than_two_bins():
    with pytest.raises(ValueError, match='at least 2'):
        aperiodic.fit_bins(bin_table(), 100.0, 101.0)


# ---------------------------------------------------------------------------
# fit_slopes — recovery
# ---------------------------------------------------------------------------

def test_recovers_a_known_slope_exactly():
    _, log_f = aperiodic.fit_bins(bin_table(), 1.0, 250.0)
    truth = np.array([-2.0, -1.0, -3.5])
    offset = np.array([-8.0, -9.0, -10.0])
    values = offset[:, None] + truth[:, None] * log_f[None, :]

    out = aperiodic.fit_slopes(values, log_f)
    np.testing.assert_allclose(out['slope'], truth, atol=1e-12)
    np.testing.assert_allclose(out['intercept'], offset, atol=1e-12)
    np.testing.assert_allclose(out['r2'], 1.0, atol=1e-12)
    assert (out['n_bins'] == len(log_f)).all()


def test_matches_numpy_polyfit_on_noisy_data():
    """The closed-form centred-sums fit must agree with the reference implementation."""
    rng = np.random.default_rng(0)
    _, log_f = aperiodic.fit_bins(bin_table(), 1.0, 250.0)
    values = -9.0 - 2.1 * log_f + rng.normal(0, 0.3, size=(6, log_f.size))

    out = aperiodic.fit_slopes(values, log_f)
    for i in range(values.shape[0]):
        ref_slope, ref_intercept = np.polyfit(log_f, values[i], 1)
        assert out['slope'][i] == pytest.approx(ref_slope, rel=1e-10)
        assert out['intercept'][i] == pytest.approx(ref_intercept, rel=1e-10)


def test_r2_is_below_one_with_noise_and_nan_when_flat():
    rng = np.random.default_rng(1)
    _, log_f = aperiodic.fit_bins(bin_table(), 1.0, 250.0)
    noisy = -9.0 - 2.0 * log_f + rng.normal(0, 0.5, size=(1, log_f.size))
    flat = np.zeros((1, log_f.size))

    assert 0.0 < aperiodic.fit_slopes(noisy, log_f)['r2'][0] < 1.0
    # A perfectly flat spectrum: the line is exact but explains no variance, so r2
    # is undefined. NaN, not a fabricated 1.0.
    assert np.isnan(aperiodic.fit_slopes(flat, log_f)['r2'][0])
    assert aperiodic.fit_slopes(flat, log_f)['slope'][0] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# THE LINEARITY PROPERTY the builder relies on
# ---------------------------------------------------------------------------

def test_slope_of_mean_equals_mean_of_slopes():
    """Fitting the epoch mean IS averaging the per-window slopes.

    An OLS slope is sum_i w_i * y_i with weights that depend only on x, so it
    commutes with any average over y. This is why build_pain_epoch_slope.py fits
    the epoch-mean spectrum instead of 300 per-window spectra.
    """
    rng = np.random.default_rng(2)
    _, log_f = aperiodic.fit_bins(bin_table(), 1.0, 250.0)
    n_windows, n_channels = 300, 7
    windows = (-9.0 - 2.0 * log_f
               + rng.normal(0, 0.4, size=(n_windows, n_channels, log_f.size)))

    per_window = np.stack([aperiodic.fit_slopes(w, log_f)['slope'] for w in windows])
    mean_of_slopes = per_window.mean(axis=0)
    slope_of_mean = aperiodic.fit_slopes(windows.mean(axis=0), log_f)['slope']

    np.testing.assert_allclose(slope_of_mean, mean_of_slopes, rtol=1e-10)


def test_linearity_holds_when_whole_windows_are_masked():
    """QC blanks a window across ALL bins, which preserves the shared bin pattern.

    That is the condition the equality needs -- so masking as the pipeline
    actually does it does not break the shortcut.
    """
    rng = np.random.default_rng(3)
    _, log_f = aperiodic.fit_bins(bin_table(), 1.0, 250.0)
    windows = -9.0 - 2.0 * log_f + rng.normal(0, 0.4, size=(20, 4, log_f.size))
    windows[3, :, :] = np.nan          # one window masked for every channel
    windows[7, 2, :] = np.nan          # one channel masked in one window

    surviving = np.stack([aperiodic.fit_slopes(w, log_f)['slope'] for w in windows])
    mean_of_slopes = np.nanmean(surviving, axis=0)
    slope_of_mean = aperiodic.fit_slopes(np.nanmean(windows, axis=0), log_f)['slope']

    np.testing.assert_allclose(slope_of_mean, mean_of_slopes, rtol=1e-10)


# ---------------------------------------------------------------------------
# refusals — a channel that cannot support a fit gets NaN, never a number
# ---------------------------------------------------------------------------

def test_too_few_finite_bins_gives_nan():
    """Ten bins SPREAD ACROSS the range, so only the count floor can bite.

    Deliberately not ten adjacent low bins: those span 0.43 decades and would trip
    the span floor instead, which is the next test's job.
    """
    _, log_f = aperiodic.fit_bins(bin_table(), 1.0, 250.0)
    values = np.full((1, log_f.size), np.nan)
    spread = np.arange(0, log_f.size, 5)[:10]
    values[0, spread] = -9.0 - 2.0 * log_f[spread]

    out = aperiodic.fit_slopes(values, log_f, min_bins=30)
    assert np.isnan(out['slope'][0])
    assert out['n_bins'][0] == 10
    # ...and it IS fittable once the floor is lowered, so the NaN is the threshold
    # talking and not a broken fit.
    assert aperiodic.fit_slopes(values, log_f, min_bins=5)['slope'][0] == pytest.approx(-2.0)


def test_too_narrow_a_span_gives_nan():
    """Enough bins, but all crowded into one corner of the spectrum."""
    _, log_f = aperiodic.fit_bins(bin_table(200), 1.0, 250.0)
    values = np.full((1, log_f.size), np.nan)
    narrow = log_f < (log_f.min() + 0.5)
    values[0, narrow] = -9.0 - 2.0 * log_f[narrow]
    assert narrow.sum() >= 30

    assert np.isnan(aperiodic.fit_slopes(values, log_f, min_span_decades=1.0)['slope'][0])
    assert np.isfinite(aperiodic.fit_slopes(values, log_f, min_span_decades=0.1)['slope'][0])


def test_all_nan_channel_gives_nan_without_warning():
    _, log_f = aperiodic.fit_bins(bin_table(), 1.0, 250.0)
    values = np.vstack([np.full(log_f.size, np.nan),
                        -9.0 - 2.0 * log_f])

    with np.errstate(all='raise'):
        out = aperiodic.fit_slopes(values, log_f)
    assert np.isnan(out['slope'][0])
    assert out['slope'][1] == pytest.approx(-2.0)
    assert out['n_bins'][0] == 0


def test_shape_mismatch_is_an_error_not_a_broadcast():
    _, log_f = aperiodic.fit_bins(bin_table(), 1.0, 250.0)
    with pytest.raises(ValueError, match='does not match'):
        aperiodic.fit_slopes(np.zeros((3, 5)), log_f)


# ---------------------------------------------------------------------------
# average_by_region
# ---------------------------------------------------------------------------

def test_region_average_is_arithmetic_and_counts_contributors():
    slopes = np.array([-2.0, -3.0, -1.0, np.nan])
    channels = ['A', 'B', 'C', 'D']
    region_of = {'A': 'M1', 'B': 'M1', 'C': 'S1', 'D': 'S1'}

    out, counts = aperiodic.average_by_region(slopes, channels, region_of, ['M1', 'S1'])
    assert out[0] == pytest.approx(-2.5)      # plain mean, no linear-then-log
    assert out[1] == pytest.approx(-1.0)      # the NaN channel does not contribute
    assert counts.tolist() == [2, 1]


def test_unmapped_and_empty_regions():
    slopes = np.array([-2.0, -3.0])
    region_of = {'A': 'M1', 'B': None}         # B has no ROI
    out, counts = aperiodic.average_by_region(slopes, ['A', 'B'], region_of,
                                              ['M1', 'S1'])
    assert out[0] == pytest.approx(-2.0)
    assert np.isnan(out[1])                    # S1 has no channels at all
    assert counts.tolist() == [1, 0]
