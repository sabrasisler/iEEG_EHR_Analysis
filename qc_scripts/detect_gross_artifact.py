"""
Step 3: non-neural gross artifact — high-variance/amplitude bursts (e.g. a
lead being unplugged/replugged), detected as sub-bin variance that's
anomalously high relative to a channel's own session-wide baseline.

A fast, high-amplitude burst that oscillates symmetrically around zero
averages out to ~0 over a 1-minute window, so a DC-offset/mean-based metric
misses it entirely — that's why this uses variance (deviation around each
sub-bin's own local mean) instead, the same underlying quantity
`detect_flatline.py` already computes, just compared against a session-
relative high threshold instead of `flatline`'s fixed absolute low threshold.

Fully independent of steps 1/2 (saturation, flatline) — the session baseline
is computed from every sub-bin unconditionally, not masked to exclude
already-flagged saturated/flatlined sub-bins. This is a deliberate choice:
keeping the three detectors independent of each other's outputs matters more
here than protecting the baseline from contamination by other artifact
types (which explicit per-detector re-analysis, e.g. permutation testing,
would otherwise have to account for). A channel with a long high-variance
burst will pull this baseline's mean/std of variance upward, potentially
masking a smaller real burst elsewhere in the same channel's session — an
accepted tradeoff, not an oversight.

The session baseline can only be known after seeing every run, but that does
NOT mean the raw NWB data needs to be read twice. Each run's raw trace is
loaded once; while it's in memory we both (a) accumulate this run's
contribution to the running session baseline of sub-bin variance, and (b)
cache each 1-minute window's mean variance (a handful of floats per channel)
for later. Once every run has been accumulated and the baseline finalized,
classification against that baseline is pure arithmetic on the tiny cached
values — no second NWB read.

`metric_value` is left signed (not `abs()`'d): only anomalously *high*
variance sets `excluded=True` (an amplitude burst is only ever "too high,"
never "too low" — a suspiciously quiet window is `flatline`'s job, with its
own fixed absolute threshold), but a low, negative z-score is still recorded
per window rather than discarded, in case it's useful for later review.

Sub-bins are the same 2-second granularity used by steps 1/2 — this assumes
SAT_WINDOW_SEC == FLATLINE_WINDOW_SEC (both default 2.0s in config.py). That's
just a shared unit of time here, not a data dependency on steps 1/2's output.
"""

import numpy as np

from qc_scripts import config


def new_accumulator():
    return {'n': 0, 'sum': 0.0, 'sumsq': 0.0}


def _reshape_into_subbins(trace_v, sfreq, sub_bin_sec):
    samples_per_subbin = max(1, int(sub_bin_sec * sfreq))
    n_subbins = len(trace_v) // samples_per_subbin
    usable = n_subbins * samples_per_subbin
    return trace_v[:usable].reshape(n_subbins, samples_per_subbin), samples_per_subbin


def accumulate_and_cache_window_means(acc, trace_v, sfreq, sub_bin_sec=None, window_sec=None):
    """
    Single pass over one run's raw trace for one channel:
      - updates `acc` (n, sum, sumsq) over per-sub-bin VARIANCE values, for
        the running session baseline of "typical variance for this channel"
      - returns this run's cached 1-minute window mean-variance
        (window_start, window_end, window_variance) for later classification,
        without needing to re-read the trace.
    """
    sub_bin_sec = sub_bin_sec if sub_bin_sec is not None else config.FLATLINE_WINDOW_SEC
    window_sec = window_sec if window_sec is not None else config.GROSS_WINDOW_SEC

    subbins, _ = _reshape_into_subbins(trace_v, sfreq, sub_bin_sec)
    n_subbins = subbins.shape[0]
    subbin_var = subbins.var(axis=1) if n_subbins else subbins  # per-subbin variance, shape (n_subbins,)

    if subbin_var.size:
        acc['n'] += subbin_var.size
        acc['sum'] += float(subbin_var.sum())
        acc['sumsq'] += float(np.square(subbin_var, dtype=np.float64).sum())

    subbins_per_window = max(1, int(round(window_sec / sub_bin_sec)))
    n_windows = n_subbins // subbins_per_window

    window_start, window_end, window_variance = [], [], []
    for w in range(n_windows):
        s, e = w * subbins_per_window, (w + 1) * subbins_per_window
        window_start.append(s * sub_bin_sec)
        window_end.append(e * sub_bin_sec)
        window_variance.append(float(subbin_var[s:e].mean()))

    cached = {
        'window_start': np.array(window_start),
        'window_end': np.array(window_end),
        'window_variance': np.array(window_variance),
    }
    return acc, cached


def finalize_baseline(acc):
    """Returns (mean, std) of per-subbin variance from an accumulator, or (nan, nan) if no data."""
    if acc['n'] == 0:
        return float('nan'), float('nan')
    mean = acc['sum'] / acc['n']
    var = acc['sumsq'] / acc['n'] - mean ** 2
    std = np.sqrt(max(var, 0.0))
    return mean, std


def classify_from_cached_means(cached, session_mean, session_std, std_thresh=None):
    """
    Pure in-memory classification against the finalized session baseline —
    no raw data access. `cached` is the dict returned by
    accumulate_and_cache_window_means for one run/channel.

    metric_value is signed (session_mean/session_std are of per-subbin
    variance): a suspiciously LOW-variance window still gets its true
    (negative) z-score recorded, but only an anomalously HIGH-variance window
    (a burst) sets excluded=True — low-variance exclusion is flatline's job.
    """
    std_thresh = std_thresh if std_thresh is not None else config.GROSS_STD_THRESH

    window_variance = cached['window_variance']

    if np.isnan(session_std) or session_std == 0:
        metric_value = np.full(len(window_variance), np.nan)
        excluded = np.ones(len(window_variance), dtype=bool)
    else:
        metric_value = (window_variance - session_mean) / session_std
        excluded = metric_value > std_thresh

    return {
        'window_start': cached['window_start'],
        'window_end': cached['window_end'],
        'excluded': excluded,
        'metric_value': metric_value,
    }
