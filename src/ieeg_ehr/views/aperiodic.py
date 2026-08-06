"""
The 1/f slope: a straight line through log10(power) vs log10(frequency).

docs/view_registry.md lists this under "Non-axis view-time computations" -- a
cheap derived quantity of the PSD cache, not an eighth axis. It consumes the same
per-window cache every view does, at the same point in the chain (after AXIS 4's
epoch average, BEFORE AXIS 5's frequency aggregation -- there is no frequency axis
left to aggregate once a spectrum has become one number).

WHY THE STORED BINS ARE THE RIGHT X
-----------------------------------
The cache's 50 bins are LOG-SPACED over 1-250 Hz, so they are already uniform in
log10(f): every bin carries roughly equal weight per octave. An OLS fit over them
is therefore not dominated by the high-frequency end the way a fit over
linearly-spaced bins would be. The bin's GEOMETRIC centre is used, for the same
reason -- an arithmetic centre drifts toward each bin's high edge.

WHY THE EPOCH MEAN IS FIT, RATHER THAN EACH WINDOW
--------------------------------------------------
It makes no difference, and that is a fact rather than an assumption. An OLS slope
is a FIXED-WEIGHT linear functional of y:

    slope = sum_i w_i * y_i ,   w_i = (x_i - xbar) / sum_j (x_j - xbar)^2

The weights depend only on x (the frequencies), never on y. So for a set of
windows sharing one finite-bin pattern,

    slope( mean_t y_t )  ==  mean_t slope( y_t )        exactly

Fitting the epoch-mean log spectrum IS averaging the per-window slopes, at 1/300th
the cost. The equality holds only while the finite-bin pattern is common to the
windows being averaged; QC masks whole windows (cache_reader.apply_mask blanks a
window across all bins), so within an epoch it holds. `tests/test_aperiodic.py`
pins it.

NUMERICS
--------
Two-pass centred sums, never `sum(x*y) - sum(x)sum(y)/n`. The naive form cancels
catastrophically when the spread is small relative to the mean, which is the
regime log-power sits in (values around -10, spread of order 1) -- the same reason
`axes.BaselineAccumulator` computes its variance two-pass. Everything runs in
config.CACHE_ACCUMULATE_DTYPE (float64); the cache is float32 (P0.6).

A SLOPE IS NOT A POWER. It averages ARITHMETICALLY over channels and regions --
there is no Jensen correction and no linear-then-log step, because the quantity is
already a log-log gradient rather than a power. Handing these values to
`axes.aggregate_regions` with the linear-then-log branch would exponentiate a
dimensionless gradient and is meaningless; `average_by_region` below exists so
that call cannot be made by accident.
"""

import logging
import warnings

import numpy as np

from ieeg_ehr import config

logger = logging.getLogger(__name__)


def fit_bins(bin_table, fit_lo_hz, fit_hi_hz, drop_bins=()):
    """Indices of the bins entering the fit, and their log10 geometric centres.

    A bin is IN when its geometric centre falls in [fit_lo_hz, fit_hi_hz], minus
    anything in `drop_bins` (the line-noise bins, normally). Selecting on the
    centre rather than on the edges matches `epochs_to_band` in
    plot_band_violin_view.py, so a bin is never claimed by two different figures'
    frequency selections.
    """
    centres = np.sqrt(bin_table['bin_low_hz'].to_numpy(dtype=np.float64)
                      * bin_table['bin_high_hz'].to_numpy(dtype=np.float64))
    keep = (centres >= fit_lo_hz) & (centres <= fit_hi_hz)
    keep[list(drop_bins)] = False
    idx = np.flatnonzero(keep)
    if idx.size < 2:
        raise ValueError(
            f'only {idx.size} frequency bin(s) fall in {fit_lo_hz}-{fit_hi_hz} Hz '
            f'after dropping {len(drop_bins)} bin(s) -- a line needs at least 2.')
    return idx, np.log10(centres[idx])


def fit_slopes(values, log_f, min_bins=None, min_span_decades=None):
    """Per-channel OLS of log-power on log10(f). NaN-aware, fully vectorized.

    `values` is (n_channels, n_fit_bins) log-power in float64, already sliced to
    the fit bins; `log_f` is the matching (n_fit_bins,) log10 centres. Returns a
    dict of (n_channels,) arrays: slope, intercept, r2, n_bins.

    A channel that cannot support a fit gets NaN, never a number. Three ways to
    fail, all of which would otherwise produce a confident-looking value out of
    nothing:

      - fewer than `min_bins` finite bins (a mostly-masked channel),
      - the surviving bins spanning less than `min_span_decades` of frequency (a
        slope from one corner of the spectrum is not the spectrum's slope, and it
        is the one that most easily reaches an extreme value),
      - zero variance in x.

    The thresholds are DEFAULTED, not hard-coded, and r2 is returned rather than
    applied: same metric/threshold split the feature-level QC uses -- store the
    metric once, threshold it cheaply and repeatedly downstream (CLAUDE.md).
    """
    min_bins = config.SLOPE_MIN_FIT_BINS if min_bins is None else min_bins
    min_span_decades = (config.SLOPE_MIN_SPAN_DECADES if min_span_decades is None
                        else min_span_decades)

    values = np.asarray(values, dtype=config.CACHE_ACCUMULATE_DTYPE)
    log_f = np.asarray(log_f, dtype=config.CACHE_ACCUMULATE_DTYPE)
    if values.ndim != 2 or values.shape[1] != log_f.size:
        raise ValueError(f'values {values.shape} does not match log_f {log_f.shape}')

    finite = np.isfinite(values)
    n = finite.sum(axis=1)

    x = np.broadcast_to(log_f, values.shape)
    xs = np.where(finite, x, np.nan)
    ys = np.where(finite, values, np.nan)

    with warnings.catch_warnings():
        # An all-masked channel is an expected outcome, not an anomaly.
        warnings.filterwarnings('ignore', message='Mean of empty slice')
        warnings.filterwarnings('ignore', message='All-NaN slice encountered')
        xbar = np.nanmean(xs, axis=1)
        ybar = np.nanmean(ys, axis=1)
        span = np.nanmax(xs, axis=1) - np.nanmin(xs, axis=1)

    # Deviations from the per-channel means, zeroed where masked so the sums below
    # ignore them without a second nan-aware reduction.
    dx = np.where(finite, xs - xbar[:, None], 0.0)
    dy = np.where(finite, ys - ybar[:, None], 0.0)
    sxx = np.einsum('ij,ij->i', dx, dx)
    sxy = np.einsum('ij,ij->i', dx, dy)
    syy = np.einsum('ij,ij->i', dy, dy)

    usable = (n >= min_bins) & (sxx > 0) & np.isfinite(span) & (span >= min_span_decades)

    with np.errstate(invalid='ignore', divide='ignore'):
        slope = np.where(usable, sxy / np.where(sxx > 0, sxx, np.nan), np.nan)
        intercept = np.where(usable, ybar - slope * xbar, np.nan)
        # r2 of a simple linear regression. syy == 0 means a perfectly flat
        # spectrum: the line is exact but explains no variance, so r2 is
        # undefined -- NaN, not 1.0.
        r2 = np.where(usable & (syy > 0),
                      sxy * sxy / np.where((sxx > 0) & (syy > 0), sxx * syy, np.nan),
                      np.nan)

    return {'slope': slope, 'intercept': intercept, 'r2': r2, 'n_bins': n}


def average_by_region(per_channel, channels, region_of, regions):
    """(n_channels,) slopes -> (n_regions,) ARITHMETIC means + contributing counts.

    Deliberately not `axes.aggregate_regions`. That function branches on
    `is_difference` to decide between an arithmetic mean and a linear-then-log
    one, and the linear-then-log branch is WRONG for a slope: it would compute
    log10(mean(10**slope)), exponentiating a dimensionless log-log gradient. The
    branch cannot be reached from here because there is no branch.

    Averaging per-channel slopes is also why the fit happens per channel in the
    first place. Region-averaging the SPECTRA first and fitting that is a
    different quantity: with raw log power the region average is linear-then-log,
    i.e. log10(mean_c 10**x_c), which is dominated by the loudest channel -- so
    the resulting "region slope" is essentially that one channel's.
    """
    by_region = {}
    for j, channel in enumerate(channels):
        roi = region_of.get(channel)
        if roi is not None:
            by_region.setdefault(roi, []).append(j)

    out = np.full(len(regions), np.nan, dtype=config.CACHE_ACCUMULATE_DTYPE)
    counts = np.zeros(len(regions), dtype=int)
    for i, roi in enumerate(regions):
        idx = by_region.get(roi)
        if not idx:
            continue
        sub = per_channel[idx]
        counts[i] = int(np.isfinite(sub).sum())
        if counts[i]:
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', message='Mean of empty slice')
                out[i] = np.nanmean(sub, dtype=config.CACHE_ACCUMULATE_DTYPE)
    return out, counts
