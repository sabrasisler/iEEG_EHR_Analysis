"""
Step 1: amplifier saturation.

The rail is inferred per channel, pooled across the ENTIRE SESSION (all
runs), then checked for agreement across channels — not inferred
independently per run. Physically, one amplifier/gain setting applies to an
entire recording session, so all channels should share one rail; a channel
that rarely/never saturates can't infer a rail from its own data alone, but
should still adopt the session's agreed rail rather than being left with
none.

This is a 3-step process, split across functions so the expensive part (a
single read per run) only needs to compute cheap per-run summary stats,
while the session-wide decision is pure arithmetic on those tiny stats:

  1. local_extreme_stats(trace_v)      -- per run, per channel (pass 1)
  2. session_rail_for_channel(...)     -- per channel, combining all its runs
  3. resolve_session_rails(...)        -- once per session, cross-channel agreement
  4. classify_saturation_with_rail(...) -- per run, per channel, using the
                                           resolved rail (pass 2 — a genuine
                                           second read, since the agreed rail
                                           can differ from what a given run's
                                           own local stats would suggest)
"""

import numpy as np

from qc_scripts import config

RTOL = 1e-5  # float tolerance for "same value" comparisons throughout this module


def local_extreme_stats(trace_v):
    """
    trace_v: 1D array, raw voltage in volts, for one channel, one run.
    Returns (abs_max, count_at_abs_max) — the run's own extreme value and how
    many samples hit it. Cheap: this is all pass 1 needs per channel per run.
    """
    abs_max = float(np.max(np.abs(trace_v)))
    if abs_max == 0:
        return 0.0, 0
    count = int(np.isclose(np.abs(trace_v), abs_max, rtol=RTOL, atol=0).sum())
    return abs_max, count


def session_rail_for_channel(per_run_stats):
    """
    per_run_stats: list of (abs_max, count_at_abs_max) tuples, one per run,
    for a single channel across a whole session.
    Returns (session_abs_max, total_count_at_session_max). The count only
    sums contributions from runs whose OWN local abs_max equals the session
    abs_max — any run with a strictly lower local max necessarily has zero
    samples at the session max, with no need to look at its raw data.
    """
    if not per_run_stats:
        return 0.0, 0
    session_abs_max = max(abs_max for abs_max, _ in per_run_stats)
    total_count = sum(count for abs_max, count in per_run_stats
                       if np.isclose(abs_max, session_abs_max, rtol=RTOL, atol=0))
    return session_abs_max, total_count


def resolve_session_rails(per_channel_session_stats, agreement_threshold=None, min_repeats=None,
                           fallback_threshold_uv=None):
    """
    per_channel_session_stats: dict {channel: (session_abs_max, total_count_at_session_max)},
    one entry per channel in the session (from session_rail_for_channel).

    Returns dict {channel: (rail_value_or_None, rail_source)}, where
    rail_source is one of 'session_agreement', 'session_individual', 'fallback', 'none'.

    First checks for cross-channel agreement: groups channels by their
    candidate abs_max (float-tolerance grouping); if the largest group's size
    is >= agreement_threshold * (total channel count), every channel in the
    session gets that group's value, regardless of its own individual stats
    (this is how a channel that never itself saturates still gets a usable
    rail). Otherwise, falls back to each channel's own session-wide inference,
    gated by min_repeats.
    """
    agreement_threshold = (agreement_threshold if agreement_threshold is not None
                            else config.SAT_AGREEMENT_THRESHOLD)
    min_repeats = min_repeats if min_repeats is not None else config.SAT_MIN_REPEATS
    fallback_threshold_uv = (fallback_threshold_uv if fallback_threshold_uv is not None
                              else config.SAT_FALLBACK_THRESHOLD_UV)

    channels = list(per_channel_session_stats.keys())
    n_channels = len(channels)
    results = {}

    if n_channels == 0:
        return results

    # Group channels by candidate abs_max (float-tolerance union-find via sorting).
    candidates = sorted(((per_channel_session_stats[ch][0], ch) for ch in channels),
                         key=lambda x: x[0])
    groups = []  # list of (representative_value, [channels])
    for value, ch in candidates:
        placed = False
        for i, (rep_value, members) in enumerate(groups):
            if np.isclose(value, rep_value, rtol=RTOL, atol=0):
                members.append(ch)
                placed = True
                break
        if not placed:
            groups.append((value, [ch]))

    best_value, best_members = max(groups, key=lambda g: len(g[1]))
    if len(best_members) / n_channels >= agreement_threshold:
        for ch in channels:
            results[ch] = (best_value, 'session_agreement')
        return results

    # No cross-channel agreement — fall back to each channel's own inference.
    for ch in channels:
        abs_max, count = per_channel_session_stats[ch]
        if count >= min_repeats:
            results[ch] = (abs_max, 'session_individual')
        elif fallback_threshold_uv is not None:
            results[ch] = (fallback_threshold_uv * 1e-6, 'fallback')
        else:
            results[ch] = (None, 'none')
    return results


def classify_saturation_with_rail(trace_v, sfreq, rail, window_sec=None, min_samples=None):
    """
    Pure classification against an already-resolved rail — no inference here.
    trace_v: 1D array, raw voltage in volts, for one channel, one run.
    Returns a dict of per-window arrays: window_start, window_end, excluded,
    metric_value (fraction of samples in the window at/beyond the rail).
    """
    window_sec = window_sec if window_sec is not None else config.SAT_WINDOW_SEC
    min_samples = min_samples if min_samples is not None else config.SAT_MIN_SAMPLES

    samples_per_window = max(1, int(window_sec * sfreq))
    n_windows = len(trace_v) // samples_per_window
    usable = n_windows * samples_per_window
    windows = trace_v[:usable].reshape(n_windows, samples_per_window)

    if rail is None:
        metric_value = np.zeros(n_windows)
        excluded = np.zeros(n_windows, dtype=bool)
    else:
        saturated_samples = np.abs(windows) >= rail
        metric_value = saturated_samples.mean(axis=1)  # fraction of samples saturated
        excluded = saturated_samples.sum(axis=1) >= min_samples

    window_start = np.arange(n_windows) * samples_per_window / sfreq
    window_end = window_start + samples_per_window / sfreq

    return {
        'window_start': window_start,
        'window_end': window_end,
        'excluded': excluded,
        'metric_value': metric_value,
    }


def zero_result_like(result):
    """
    A same-shaped result with everything unexcluded/zero — for a run whose
    own local peak is strictly below the resolved session rail, where we
    already know (without touching the raw samples) that zero windows can
    possibly be saturated.
    """
    n = len(result['window_start'])
    return {
        'window_start': result['window_start'],
        'window_end': result['window_end'],
        'excluded': np.zeros(n, dtype=bool),
        'metric_value': np.zeros(n),
    }
