"""
Step 2b: square-wave / two-level artifact.

A digital/relay-style artifact where nearly all samples in a window sit at two
discrete levels (e.g. a 0-50µV square wave). This slips through every other
detector: flatline misses it (variance is high), saturation misses it (not at
the amplifier rail), and gross_artifact misses it (a symmetric square wave is
mean-neutral and only moderate-variance).

The metric is the fraction of samples pinned within EPS_FRAC of the window's
own min/max. Because EPS is a fraction of the window's OWN range and the metric
is itself a fraction, it is dimensionless — a 50µV and a 2000µV square wave give
the identical value, so there is no amplitude tuning. Frequency is handled by
the 2s window + the downstream 60s exclusion bucketing: a fast square wave
(period <= 2s) is flagged directly per window; a slow one has its transition
windows flagged here and its flat plateaus caught by flatline, and the 60s
rollup unions them.

Like flatline/saturation this is a pure per-window function of the raw trace +
its own config — no session baseline, so no cross-detector contamination.

Following the metric/threshold split: this stores the continuous metric
(bimodal_fraction) and the window range only. The exclusion decision
(bimodal_fraction > SQUARE_FRAC_THRESH & range > SQUARE_MIN_RANGE_V) is applied
later in build_exclusions.py, not here.
"""

import numpy as np

from qc_scripts import config


def classify_square_wave(trace_v, sfreq, window_sec=None, eps_frac=None):
    """
    trace_v: 1D array, raw voltage in volts, for one channel.
    Returns a dict of per-window arrays: window_start, window_end,
    metric_value (bimodal_fraction, 0..1) and range (peak-to-peak, V).

    No `excluded` — thresholding happens in build_exclusions.py. `range` is
    carried so the exclusion step can apply the SQUARE_MIN_RANGE_V guard that
    stops a digitally-flat window (lo≈hi, fraction→1) being called a square wave.
    """
    window_sec = window_sec if window_sec is not None else config.SQUARE_WINDOW_SEC
    eps_frac = eps_frac if eps_frac is not None else config.SQUARE_EPS_FRAC

    samples_per_window = max(1, int(window_sec * sfreq))
    n_windows = len(trace_v) // samples_per_window
    usable = n_windows * samples_per_window
    windows = trace_v[:usable].reshape(n_windows, samples_per_window)

    lo = windows.min(axis=1, keepdims=True)
    hi = windows.max(axis=1, keepdims=True)
    rng = (hi - lo).ravel()

    eps = eps_frac * (hi - lo)   # (n_windows, 1), broadcasts against windows
    near = (np.abs(windows - lo) <= eps) | (np.abs(windows - hi) <= eps)
    bimodal_fraction = near.mean(axis=1)

    window_start = np.arange(n_windows) * samples_per_window / sfreq
    window_end = window_start + samples_per_window / sfreq

    return {
        'window_start': window_start,
        'window_end': window_end,
        'metric_value': bimodal_fraction,
        'range': rng,
    }
