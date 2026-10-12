"""
Shared computational core for bipolar re-referencing + PSD: pair derivation,
re-referencing, per-2s variance, and the 50-log-bin reduction that the retired
50-bin family stored and that the bit-exact audits still replay.

Pure functions operating on already-loaded in-memory arrays -- no NWB I/O here
(that lives in run_pipeline_bipolar.py). The bipolar time series itself is
never persisted: it's cheap enough to recompute from raw NWB later, so this
module treats it as strictly transient (callers `del` it after use).

Pair-derivation logic (parse_electrode_shaft/create_bipolar_pairs/
create_bipolar_electrode_table) is carried forward here from the now-archived
outdated/preprocessing/preprocess_ieeg.py, unchanged in behavior. New active
code shouldn't import from outdated/, so it's reimplemented directly rather
than imported.
"""

import re

import numpy as np
import pandas as pd
from scipy import signal


# ============================================================================
# Bipolar pair derivation (carried forward from outdated/preprocessing/preprocess_ieeg.py)
# ============================================================================

def parse_electrode_shaft(location):
    """Extract shaft name and contact number from a location string, e.g.
    'LOF3' -> ('LOF', 3)."""
    match = re.match(r'^([A-Za-z]+)(\d+)$', location)
    if match:
        return match.group(1), int(match.group(2))
    return None, None


def create_bipolar_pairs(elec_df):
    """Adjacent-contact bipolar pairs within each shaft (cathode contact_num -
    anode contact_num == 1). Returns (pairs, filtered_elec_df)."""
    elec_df = elec_df.copy()
    elec_df['shaft'] = None
    elec_df['contact_num'] = None

    for idx, row in elec_df.iterrows():
        shaft, num = parse_electrode_shaft(row['location'])
        elec_df.at[idx, 'shaft'] = shaft
        elec_df.at[idx, 'contact_num'] = num

    valid_parse = elec_df['shaft'].notna()
    if (~valid_parse).sum() > 0:
        print(f"  Warning: {(~valid_parse).sum()} electrodes have unparseable locations",
              flush=True)
        elec_df = elec_df[valid_parse].copy()

    pairs = []
    for shaft_name, shaft_df in elec_df.groupby('shaft'):
        shaft_df = shaft_df.sort_values('contact_num')
        contacts = [(idx, row) for idx, row in shaft_df.iterrows()]
        for i in range(len(contacts) - 1):
            anode_idx, anode = contacts[i]
            cathode_idx, cathode = contacts[i + 1]
            if cathode['contact_num'] - anode['contact_num'] == 1:
                pairs.append({
                    'anode_idx': anode_idx,
                    'cathode_idx': cathode_idx,
                    'anode_location': anode['location'],
                    'cathode_location': cathode['location'],
                    'location': f"{anode['location']}-{cathode['location']}",
                    'shaft': shaft_name,
                })
    return pairs, elec_df


def create_bipolar_electrode_table(elec_df, pairs):
    """Output electrode table for bipolar pairs: coordinate columns averaged
    anode/cathode, everything else kept as _anode/_cathode-suffixed pairs."""
    coord_systems = ['MNI', 'LEPTO', 'MGRID', 'subINF', 'fsaverageINF', 'ScannerNativeRAS']
    coord_columns = [f"{cs}{ax}" for cs in coord_systems for ax in ('_coord_1', '_coord_2', '_coord_3')
                     if f"{cs}{ax}" in elec_df.columns]
    single_columns = ['group', 'group_name']

    rows = []
    for pair in pairs:
        anode_row = elec_df.loc[pair['anode_idx']]
        cathode_row = elec_df.loc[pair['cathode_idx']]
        new_row = {'location': pair['location']}

        for col in single_columns:
            if col in elec_df.columns:
                new_row[col] = anode_row[col]

        for col in coord_columns:
            a, c = anode_row[col], cathode_row[col]
            if pd.notna(a) and pd.notna(c):
                new_row[col] = (a + c) / 2
            elif pd.notna(a):
                new_row[col] = a
            elif pd.notna(c):
                new_row[col] = c
            else:
                new_row[col] = np.nan

        for col in elec_df.columns:
            if col not in single_columns and col not in coord_columns and col != 'location':
                new_row[f"{col}_anode"] = anode_row[col]
                new_row[f"{col}_cathode"] = cathode_row[col]

        rows.append(new_row)
    return pd.DataFrame(rows)


def derive_pairs(elec_df):
    """Thin wrapper around this module's own create_bipolar_pairs. Returns
    (pairs, filtered_elec_df)."""
    return create_bipolar_pairs(elec_df)


def pairs_signature(pairs):
    """Hashable signature of a pairs list -- sorted (anode_location,
    cathode_location) tuples -- used to detect whether a session's runs agree
    on their bipolar pairs."""
    return tuple(sorted((p['anode_location'], p['cathode_location']) for p in pairs))


# ============================================================================
# Re-referencing (transient -- never persisted)
# ============================================================================

def rereference(data_v, elec_indices, pairs):
    """anode_col - cathode_col for every pair. Returns bipolar_v
    (n_samples, n_pairs) float32. Caller must not persist this array."""
    elec_indices = np.asarray(elec_indices)
    n_samples = data_v.shape[0]
    bipolar_v = np.zeros((n_samples, len(pairs)), dtype=np.float32)
    for i, pair in enumerate(pairs):
        anode_col = np.where(elec_indices == pair['anode_idx'])[0][0]
        cathode_col = np.where(elec_indices == pair['cathode_idx'])[0][0]
        bipolar_v[:, i] = data_v[:, anode_col] - data_v[:, cathode_col]
    return bipolar_v


# ============================================================================
# Per-2s variance metric (feeds the mask-aware exclusion step, ieeg_ehr/qc/
# build_bipolar_exclusions.py -- independent granularity from the PSD below)
# ============================================================================

def compute_variance_windows(bipolar_v, sfreq, window_sec=2.0):
    """
    Per-non-overlapping-window variance per bipolar channel. Returns dict with
    'window_start'/'window_end' (n_windows,) and 'metric_value'
    (n_windows, n_pairs) arrays (variance, V^2).
    """
    n_samples, n_pairs = bipolar_v.shape
    samples_per_window = max(1, int(round(window_sec * sfreq)))
    n_windows = n_samples // samples_per_window
    usable = n_windows * samples_per_window

    if n_windows == 0:
        return {'window_start': np.array([]), 'window_end': np.array([]),
                'metric_value': np.zeros((0, n_pairs))}

    reshaped = bipolar_v[:usable].reshape(n_windows, samples_per_window, n_pairs)
    metric_value = reshaped.var(axis=1)   # (n_windows, n_pairs)

    window_start = np.arange(n_windows) * window_sec
    window_end = window_start + window_sec
    return {'window_start': window_start, 'window_end': window_end, 'metric_value': metric_value}


# ============================================================================
# Frequency binning
# ============================================================================

def log_bin_edges(n_bins, f_min, f_max):
    return np.logspace(np.log10(f_min), np.log10(f_max), n_bins + 1)


def line_noise_mask(bin_edges, line_freqs, guard_hz):
    """bool per bin: does [edge_lo, edge_hi] overlap any line_freq +/- guard_hz."""
    n_bins = len(bin_edges) - 1
    flagged = np.zeros(n_bins, dtype=bool)
    for lf in line_freqs:
        lo, hi = lf - guard_hz, lf + guard_hz
        flagged |= (bin_edges[:-1] < hi) & (bin_edges[1:] > lo)
    return flagged


def _band_average_linear(freqs, psd, bin_edges):
    """
    Average linear PSD within each of len(bin_edges)-1 bins. If a bin contains
    no raw frequency sample (common for the narrowest low-frequency log bins,
    since bin width there is smaller than the raw FFT resolution), fall back
    to the single nearest raw frequency's power rather than NaN.
    Returns (n_bins,) linear power, for one window/channel's PSD.
    """
    n_bins = len(bin_edges) - 1
    out = np.empty(n_bins, dtype=np.float64)
    for b in range(n_bins):
        lo, hi = bin_edges[b], bin_edges[b + 1]
        mask = (freqs >= lo) & (freqs < hi)
        if mask.any():
            out[b] = psd[mask].mean()
        else:
            nearest = np.argmin(np.abs(freqs - 0.5 * (lo + hi)))
            out[b] = psd[nearest]
    return out


# ============================================================================
# Welch PSD, band-averaged into log-spaced bins
# ============================================================================

#: scipy.signal.spectrogram's own defaults, made EXPLICIT rather than inherited.
#: Numerically inert -- these are exactly what the calls below were already
#: getting -- but a default is not provenance:
#:   - `window` was the literal 'hann' while `config.PSD_WINDOW_FN` was what got
#:     RECORDED, so changing the constant would have produced wrong provenance
#:     for unchanged output.
#:   - `detrend='constant'` means every spectrum this project has ever stored has
#:     had its per-2s-segment mean removed, and that was written down nowhere. A
#:     reader reconstructing the method from a sidecar could not recover it, and a
#:     future scipy default change would silently alter the pipeline.
#: Parameters rather than a config import because this module is deliberately
#: config-free: `_welch_one_channel` is pickled to ProcessPoolExecutor workers,
#: and the caller (`run_pipeline_bipolar`) is where config belongs. (2026-09-15)
DEFAULT_PSD_WINDOW_FN = 'hann'
DEFAULT_PSD_DETREND = 'constant'


def _welch_one_channel(channel_col, sfreq, nperseg, noverlap, bin_edges,
                       window=DEFAULT_PSD_WINDOW_FN, detrend=DEFAULT_PSD_DETREND):
    """Continuous overlapping PSD (band-averaged into log bins) for ONE
    channel's full time series -- single-level windowing (no outer/inner
    split): each `nperseg`-sample window (default 2s) is its own periodogram-
    style estimate (single segment, hann-windowed), stepped by `nperseg -
    noverlap` samples (default 50% overlap -> 1s hop). One scipy call per
    channel via signal.spectrogram (returns every window's PSD at once,
    rather than looping per-window in Python). Runs in a worker process when
    n_workers > 1 -- kept as a free function (not a closure) so it's
    picklable for ProcessPoolExecutor."""
    freqs, times, Sxx = signal.spectrogram(
        channel_col, fs=sfreq, nperseg=nperseg, noverlap=noverlap,
        window=window, detrend=detrend, scaling='density', mode='psd')
    # Sxx: (n_freqs, n_windows) linear power. Band-average each time-slice's
    # spectrum into the log bins, then log10 (average linear, THEN log --
    # averaging log first would bias the estimate low, Jensen's inequality).
    n_windows = Sxx.shape[1]
    n_bins = len(bin_edges) - 1
    out = np.empty((n_windows, n_bins), dtype=np.float32)
    for w in range(n_windows):
        linear_bins = _band_average_linear(freqs, Sxx[:, w], bin_edges)
        with np.errstate(divide='ignore'):
            out[w, :] = np.log10(linear_bins)
    return out, times
