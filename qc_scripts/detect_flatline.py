"""
Step 2: flatlined channels. No baseline — absolute variance threshold,
same default as raw_voltage_qc.FLATLINE_VAR_THRESHOLD.
"""

import numpy as np

from qc_scripts import config


def classify_flatline(trace_v, sfreq, window_sec=None, var_thresh=None):
    """
    trace_v: 1D array, raw voltage in volts, for one channel.
    Returns a dict of per-window arrays: window_start, window_end, excluded,
    metric_value (variance, V^2).
    """
    window_sec = window_sec if window_sec is not None else config.FLATLINE_WINDOW_SEC
    var_thresh = var_thresh if var_thresh is not None else config.FLATLINE_VAR_THRESH

    samples_per_window = max(1, int(window_sec * sfreq))
    n_windows = len(trace_v) // samples_per_window
    usable = n_windows * samples_per_window
    windows = trace_v[:usable].reshape(n_windows, samples_per_window)

    metric_value = windows.var(axis=1)
    excluded = metric_value < var_thresh

    window_start = np.arange(n_windows) * samples_per_window / sfreq
    window_end = window_start + samples_per_window / sfreq

    return {
        'window_start': window_start,
        'window_end': window_end,
        'excluded': excluded,
        'metric_value': metric_value,
    }
