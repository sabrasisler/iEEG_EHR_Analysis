"""
The seven view axes as functions (docs/view_registry.md AXIS 1-7).

Everything here operates on the dense (window, pair, bin) blocks that
cache_reader yields, in float64. The chain ORDER is the registry's and is enforced
by the caller (build_pain_epoch_view), not re-litigated here:

    domain -> baseline -> normalize PER WINDOW -> epoch-average -> freq-agg
           -> region-agg -> pain binarize

The one ordering rule that must never move: NORMALIZE BEFORE AVERAGING. Averaging
then normalizing is not the same operation (Jensen), and keeping per-window
granularity in the cache exists precisely so this order is available.
"""

import logging

import numpy as np
import pandas as pd

from ieeg_ehr import config

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# AXIS 1 -- domain
# ---------------------------------------------------------------------------

def to_domain(block, domain):
    """Stored log-power -> the requested domain.

    Exponentiation runs in CACHE_LINEAR_DOMAIN_DTYPE (float64) because the worst
    stored log-power seen is ~-36.8: 10**-36.8 is representable in float32 but
    nearly subnormal, so any later scaling -- a baseline division, a band average
    -- can underflow to exactly zero and turn a quiet channel into a silent one.
    """
    if domain == 'log':
        return block
    return np.power(10.0, block.astype(config.CACHE_LINEAR_DOMAIN_DTYPE))


# ---------------------------------------------------------------------------
# AXIS 2 -- baseline
# ---------------------------------------------------------------------------

class BaselineAccumulator:
    """Per (pair, bin) mean and SD over baseline WINDOWS, accumulated across epochs.

    Numerically stable by construction: within an epoch the variance is computed
    two-pass (mean first, then squared deviations from it), and epochs are merged
    with Chan's parallel-variance formula. The naive `sum(x^2)/n - mean^2` was
    avoided deliberately -- it cancels catastrophically when the spread is small
    relative to the mean, which is the regime power values sit in.

    NaN-aware throughout: masked windows arrive as NaN, so counts are per-element
    rather than a shared `n`. A (pair, bin) cell with no surviving baseline window
    ends up count 0 -> mean/SD NaN -> every z-score against it NaN, which is the
    correct "unknown", not a zero.
    """

    def __init__(self, n_pairs, n_bins, dtype=None):
        dtype = dtype or config.CACHE_ACCUMULATE_DTYPE
        self.count = np.zeros((n_pairs, n_bins), dtype=np.int64)
        self.mean = np.zeros((n_pairs, n_bins), dtype=dtype)
        self.m2 = np.zeros((n_pairs, n_bins), dtype=dtype)
        self.n_epochs = 0

    def update(self, block, rows=None):
        """Fold in one epoch's (n_win, n_pairs, n_bins) block.

        `rows` maps the block's pair axis onto accumulator rows, so epochs from
        runs with DIFFERENT montages accumulate into the same channel's slot. The
        baseline is keyed on channel NAME across the whole session, not on
        (run, pair_index): registry AXIS 2 defines it over "the subject's 0-pain
        epoch windows", and a per-run baseline would strand every epoch in a run
        that happens to contain no 0-pain event -- which for sub-019 was 4 of 49
        epochs.
        """
        valid = np.isfinite(block)
        n_b = valid.sum(axis=0)
        if not n_b.any():
            return
        with np.errstate(invalid='ignore'):
            import warnings
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', message='Mean of empty slice')
                mean_b = np.nanmean(np.where(valid, block, np.nan), axis=0)
            dev = np.where(valid, block - mean_b, 0.0)
            m2_b = (dev * dev).sum(axis=0)
        mean_b = np.nan_to_num(mean_b, nan=0.0)

        if rows is None:
            rows = np.arange(block.shape[1])
        rows = np.asarray(rows)

        n_a = self.count[rows]
        n = n_a + n_b
        both = n > 0
        delta = np.where(both, mean_b - self.mean[rows], 0.0)
        # nb/n weighting, guarded so empty cells stay exactly where they were.
        with np.errstate(invalid='ignore', divide='ignore'):
            self.mean[rows] = np.where(
                both, self.mean[rows] + delta * np.where(both, n_b / n, 0.0),
                self.mean[rows])
            self.m2[rows] = np.where(
                both,
                self.m2[rows] + m2_b + delta * delta * np.where(both, n_a * n_b / n, 0.0),
                self.m2[rows])
        self.count[rows] = n
        self.n_epochs += 1

    def finalize(self, min_windows=2):
        """(mean, sd) with cells below `min_windows` set to NaN.

        Sample SD (ddof=1): the baseline windows are a sample of that channel's
        0-pain behaviour, not the whole population, and ddof=1 is what the
        existing z-score helper uses. min_windows=2 because ddof=1 is undefined
        at n=1 -- returning NaN is right, dividing by zero is not.
        """
        mean = np.where(self.count >= 1, self.mean, np.nan)
        with np.errstate(invalid='ignore', divide='ignore'):
            sd = np.sqrt(self.m2 / np.maximum(self.count - 1, 1))
        sd = np.where(self.count >= min_windows, sd, np.nan)
        # A zero-variance baseline cannot standardise anything; NaN rather than an
        # infinite z.
        sd = np.where(sd > 0, sd, np.nan)
        return mean, sd


def is_baseline_epoch(epoch_row):
    """AXIS 2 `zero_pain_epochs`: the subject's own 0-pain epochs."""
    score = epoch_row['pain_score']
    return pd.notna(score) and float(score) == 0.0


# ---------------------------------------------------------------------------
# AXIS 3 -- normalization (PER WINDOW, before averaging)
# ---------------------------------------------------------------------------

def normalize(block, baseline_mean, baseline_sd, normalization):
    """Apply the per-window normalization. `block` is (n_win, n_pairs, n_bins);
    the baseline arrays are (n_pairs, n_bins) and broadcast over windows."""
    if normalization == 'none':
        return block
    if normalization == 'baseline_subtract':
        return block - baseline_mean
    if normalization == 'zscore_vs_baseline':
        with np.errstate(invalid='ignore', divide='ignore'):
            return (block - baseline_mean) / baseline_sd
    raise ValueError(f'unknown normalization {normalization!r}')


# ---------------------------------------------------------------------------
# AXIS 4 -- epoch aggregation
# ---------------------------------------------------------------------------

def epoch_mean(block):
    """Mean over windows -> (n_pairs, n_bins), accumulated in float64.

    `np.nanmean` on a float64 input accumulates in float64; the block was upcast
    by cache_reader precisely so this holds. An all-NaN channel-bin yields NaN
    (with the warning suppressed -- it is an expected outcome for a fully masked
    channel, not an anomaly).
    """
    with np.errstate(invalid='ignore'):
        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', message='Mean of empty slice')
            return np.nanmean(block, axis=0, dtype=config.CACHE_ACCUMULATE_DTYPE)


# ---------------------------------------------------------------------------
# AXIS 5 -- frequency aggregation
# ---------------------------------------------------------------------------

def aggregate_bands(values, bin_table, bands=None, is_difference=True, domain='log'):
    """(n_pairs, n_bins) -> (n_pairs, n_bands), plus the band names.

    A difference of logs (z-score / baseline-subtract) averages ARITHMETICALLY --
    it is already dimensionless or already a ratio in log space. Raw log-power does
    NOT: the mean of logs is a geometric mean, so the registry's linear-then-log
    rule applies (matching bipolar_bands.aggregate_to_bands). Branching on which
    is why `is_difference` is threaded through rather than inferred here.
    """
    bands = bands or config.CANONICAL_BANDS_HZ
    centers = np.sqrt(bin_table['bin_low_hz'].to_numpy() * bin_table['bin_high_hz'].to_numpy())
    names, cols = [], []
    for band, (fmin, fmax) in bands.items():
        idx = np.flatnonzero((centers >= fmin) & (centers < fmax))
        if idx.size == 0:
            logger.warning('no freq bin centres fall in %s (%s-%s Hz); band skipped',
                           band, fmin, fmax)
            continue
        sub = values[:, idx]
        with np.errstate(invalid='ignore'):
            import warnings
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', message='Mean of empty slice')
                if is_difference or domain == 'linear':
                    agg = np.nanmean(sub, axis=1, dtype=config.CACHE_ACCUMULATE_DTYPE)
                else:
                    linear = np.power(10.0, sub.astype(config.CACHE_LINEAR_DOMAIN_DTYPE))
                    agg = np.log10(np.nanmean(linear, axis=1,
                                              dtype=config.CACHE_ACCUMULATE_DTYPE))
        names.append(band)
        cols.append(agg)
    return np.column_stack(cols) if cols else np.empty((values.shape[0], 0)), names


# ---------------------------------------------------------------------------
# AXIS 6 -- region aggregation
# ---------------------------------------------------------------------------

def aggregate_regions(values, channels, region_of, regions, is_difference=True,
                      domain='log'):
    """(n_pairs, n_cols) -> (n_regions, n_cols) + contributing channel counts.

    Same Jensen branch as the frequency axis. Channels whose ROI is None are
    dropped; the CALLER logs how many, because coverage is a confound in this
    dataset and a shrinking denominator must stay visible.
    """
    out = np.full((len(regions), values.shape[1]), np.nan,
                  dtype=config.CACHE_ACCUMULATE_DTYPE)
    counts = np.zeros(len(regions), dtype=int)
    by_region = {}
    for j, channel in enumerate(channels):
        roi = region_of.get(channel)
        if roi is not None:
            by_region.setdefault(roi, []).append(j)

    for i, roi in enumerate(regions):
        idx = by_region.get(roi)
        if not idx:
            continue
        sub = values[idx, :]
        # A channel contributes only where it has a finite value; count the
        # channels that contribute anywhere, which is what the figure annotates.
        counts[i] = int(np.isfinite(sub).any(axis=1).sum())
        with np.errstate(invalid='ignore'):
            import warnings
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', message='Mean of empty slice')
                if is_difference or domain == 'linear':
                    out[i, :] = np.nanmean(sub, axis=0,
                                           dtype=config.CACHE_ACCUMULATE_DTYPE)
                else:
                    linear = np.power(10.0, sub.astype(config.CACHE_LINEAR_DOMAIN_DTYPE))
                    out[i, :] = np.log10(np.nanmean(linear, axis=0,
                                                    dtype=config.CACHE_ACCUMULATE_DTYPE))
    return out, counts


# ---------------------------------------------------------------------------
# AXIS 7 -- pain binarization
# ---------------------------------------------------------------------------

def assign_pain_bins(defs, scheme):
    """Series of bin labels aligned to `defs`.

    `subject_relative` splits at the subject's own mean over NON-ZERO events, and
    is computed from DISTINCT (pain_event_id, pain_score) rows -- not from exploded
    channel/bin rows -- so a subject's threshold is not skewed by how many
    channels they happen to have. That is the trap
    features/common.assign_relative_pain_bins documents; here the epoch index is
    already one row per event, so the fix is inherent.
    """
    scores = defs['pain_score'].astype(float)
    if scheme == 'absolute':
        return scores.map(config.pain_bin_for_score)

    events = defs.drop_duplicates('pain_event_id')
    nonzero = events.loc[events['pain_score'].astype(float) > 0, 'pain_score'].astype(float)
    if nonzero.empty:
        # No non-zero events at all: everything is 'none' and the split is moot.
        return pd.Series(np.where(scores == 0, 'none', None), index=defs.index)
    threshold = float(nonzero.mean())
    return pd.Series(
        np.where(scores == 0, 'none', np.where(scores >= threshold, 'high', 'low')),
        index=defs.index,
    )


def subject_relative_threshold(defs):
    """The split point, for the record -- it differs per subject, so a figure that
    pools subjects is pooling different definitions of 'high'."""
    events = defs.drop_duplicates('pain_event_id')
    nonzero = events.loc[events['pain_score'].astype(float) > 0, 'pain_score'].astype(float)
    return float(nonzero.mean()) if not nonzero.empty else float('nan')
