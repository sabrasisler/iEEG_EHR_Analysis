"""
Step 3: non-neural gross artifact (session-level DC-offset shifts, e.g.
unplugging/movement).

Fully independent of steps 1/2 (saturation, flatline) — the session baseline
is computed from every sub-bin unconditionally, not masked to exclude
already-flagged saturated/flatlined sub-bins. This is a deliberate choice:
keeping the three detectors independent of each other's outputs matters more
here than protecting the baseline from contamination by other artifact
types (which explicit per-detector re-analysis, e.g. permutation testing,
would otherwise have to account for). A channel with a long saturated or
flatlined stretch will pull this baseline's mean/std toward that stretch,
potentially masking smaller real DC-offset shifts elsewhere in the same
channel's session — an accepted tradeoff, not an oversight.

The session baseline can only be known after seeing every run, but that does
NOT mean the raw NWB data needs to be read twice. Each run's raw trace is
loaded once; while it's in memory we both (a) accumulate this run's
contribution to the running session baseline, and (b) cache each 1-minute
window's mean (a handful of floats per channel) for later. Once every run has
been accumulated and the baseline finalized, classification against that
baseline is pure arithmetic on the tiny cached means — no second NWB read.

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
      - updates `acc` (n, sum, sumsq) using every sub-bin, for the running
        session baseline
      - returns this run's cached 1-minute window means (window_start,
        window_end, window_mean) for later classification, without needing
        to re-read the trace.
    """
    sub_bin_sec = sub_bin_sec if sub_bin_sec is not None else config.FLATLINE_WINDOW_SEC
    window_sec = window_sec if window_sec is not None else config.GROSS_WINDOW_SEC

    subbins, _ = _reshape_into_subbins(trace_v, sfreq, sub_bin_sec)
    n_subbins = subbins.shape[0]

    if subbins.size:
        acc['n'] += subbins.size
        acc['sum'] += float(subbins.sum())
        acc['sumsq'] += float(np.square(subbins, dtype=np.float64).sum())

    subbins_per_window = max(1, int(round(window_sec / sub_bin_sec)))
    n_windows = n_subbins // subbins_per_window

    window_start, window_end, window_mean = [], [], []
    for w in range(n_windows):
        s, e = w * subbins_per_window, (w + 1) * subbins_per_window
        window_start.append(s * sub_bin_sec)
        window_end.append(e * sub_bin_sec)
        window_mean.append(float(subbins[s:e].mean()))

    cached = {
        'window_start': np.array(window_start),
        'window_end': np.array(window_end),
        'window_mean': np.array(window_mean),
    }
    return acc, cached


def finalize_baseline(acc):
    """Returns (mean, std) from an accumulator, or (nan, nan) if no data."""
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
    """
    std_thresh = std_thresh if std_thresh is not None else config.GROSS_STD_THRESH

    window_mean = cached['window_mean']

    if np.isnan(session_std) or session_std == 0:
        metric_value = np.full(len(window_mean), np.nan)
        excluded = np.ones(len(window_mean), dtype=bool)
    else:
        metric_value = (window_mean - session_mean) / session_std
        excluded = np.abs(metric_value) > std_thresh

    return {
        'window_start': cached['window_start'],
        'window_end': cached['window_end'],
        'excluded': excluded,
        'metric_value': metric_value,
    }
