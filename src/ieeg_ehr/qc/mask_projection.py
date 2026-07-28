"""
Project a monopolar, 60s-binned raw-voltage QC mask onto bipolar pairs at
arbitrary window times.

WHY THIS IS ITS OWN MODULE
-------------------------
The raw-voltage mask is keyed on (run_id, MONOPOLAR channel, 60s bin_start), but
everything downstream of bipolar re-referencing is keyed on PAIRS ('LAH1-LAH2')
at the PSD's window rate (1 s hop for the default 2 s / 50% overlap grid). Two
different consumers need the same translation:

  - the feature-level power-outlier detector, which must drop already-flagged
    windows from its baseline (config.FEATURE_BASELINE_EXCLUDES_RAW_VOLTAGE);
  - the view layer, which applies the mask to the epoch cache at load time now
    that the cache no longer bakes it in (2026-07-27 decision).

This used to live as `_excluded_mask` inside build_pain_epoch_power.py, private
to the cache builder that no longer does masking at all. One implementation,
because two subtly different ones would mean the baseline and the analysis
disagreed about which windows are usable — a difference that would be invisible
in any single number and would quietly bias every effect size.

THE PAIR RULE
-------------
A bipolar pair is excluded if EITHER of its contributing monopolar contacts is
excluded. That is the only defensible direction: the pair's signal is a
difference, so an artifact on one leg is an artifact in the difference.

BIN ALIGNMENT
-------------
Windows inherit the verdict of the 60s bin they fall in
(`bin_start = floor(run_seconds / 60) * 60`), matching how build_exclusions.py
buckets the 2s detectors up to 60s in the first place. Note the asymmetry this
implies: exclusion is coarse (60s) relative to the window grid (1s), so a
1-second artifact removes its whole enclosing minute. That is inherited from the
raw-voltage tree's granularity, not a choice made here.
"""

import numpy as np
import pandas as pd

BIN_SEC = 60.0

MASK_COLUMNS = ['run_id', 'channel', 'bin_start', 'excluded']


def split_pair(channel):
    """('LAH1-LAH2') -> ('LAH1', 'LAH2'). Returns (channel, None) if the name is
    not a pair, so a monopolar-named channel degrades to "look me up directly"
    rather than raising — some derivative trees mix the two."""
    if '-' not in channel:
        return channel, None
    anode, cathode = channel.split('-', 1)
    return anode, cathode


def load_mask(mask_path, run_id=None):
    """Read one subject/session's raw-voltage mask CSV, optionally for one run.

    Returns None if the file does not exist. That is deliberately not an error:
    build_mask.py drops subject/sessions that are missing any artifact type (the
    sub-236 gap, docs/qc_context.md), so a missing file means "this
    subject/session has no mask at this label" and the caller decides whether to
    proceed unmasked or skip. Callers MUST log which they did.
    """
    mask_path = pd.io.common.stringify_path(mask_path)
    try:
        df = pd.read_csv(mask_path, usecols=MASK_COLUMNS)
    except (FileNotFoundError, OSError):
        return None
    if run_id is not None:
        df = df[df['run_id'] == run_id]
    return df


def project_to_pairs(mask_df, run_id, channel_names, run_seconds):
    """(n_windows, n_pairs) bool: True where this window/pair is mask-excluded.

    mask_df       long mask table for this subject/session (any run); the run
                  filter is applied here. None -> all-False (unmasked).
    channel_names bipolar pair names, in the order the caller's data columns are.
    run_seconds   run-relative seconds of each window (per PSD row).

    Implementation note: this does NOT build the (n_windows x n_pairs) cross
    product as a DataFrame merge. A long session is ~10^5 windows x ~200 pairs =
    2x10^7 rows, and merging that per run was the slow part of the original.
    Instead the mask is pivoted to a small (n_60s_bins x n_monopolar) matrix, OR'd
    down to pairs once, then broadcast to windows by integer indexing — the
    per-window step is a gather, not a join.
    """
    n_win = len(run_seconds)
    n_pairs = len(channel_names)
    if mask_df is None or n_win == 0 or n_pairs == 0:
        return np.zeros((n_win, n_pairs), dtype=bool)

    run_mask = mask_df[mask_df['run_id'] == run_id]
    if run_mask.empty:
        return np.zeros((n_win, n_pairs), dtype=bool)

    # Small dense matrix: 60s bin x monopolar channel.
    pivot = (run_mask.pivot_table(index='bin_start', columns='channel',
                                  values='excluded', aggfunc='any')
             .fillna(False).astype(bool))
    bin_starts = pivot.index.to_numpy(dtype=np.float64)
    mono_cols = {name: i for i, name in enumerate(pivot.columns)}
    mono = pivot.to_numpy()                                  # (n_bins, n_mono)

    # OR anode/cathode into a per-pair matrix, once per run rather than per window.
    absent = np.zeros(len(bin_starts), dtype=bool)
    pair_by_bin = np.empty((len(bin_starts), n_pairs), dtype=bool)
    for j, channel in enumerate(channel_names):
        anode, cathode = split_pair(channel)
        a = mono[:, mono_cols[anode]] if anode in mono_cols else absent
        c = mono[:, mono_cols[cathode]] if cathode in mono_cols else absent
        pair_by_bin[:, j] = a | c

    # Each window -> the row of pair_by_bin for its enclosing 60s bin. Windows
    # whose bin is absent from the mask entirely map to an all-False row appended
    # at the end, so "no mask row" means "not excluded" (matching the fillna(False)
    # convention the raw-voltage consumers already use) rather than dropping the
    # window or raising.
    want = np.floor(np.asarray(run_seconds, dtype=np.float64) / BIN_SEC) * BIN_SEC
    pos = np.searchsorted(bin_starts, want)
    pos = np.clip(pos, 0, len(bin_starts) - 1)
    missing = bin_starts[pos] != want
    pair_by_bin = np.vstack([pair_by_bin, np.zeros((1, n_pairs), dtype=bool)])
    pos = np.where(missing, len(bin_starts), pos)
    return pair_by_bin[pos]


def excluded_fraction(excluded):
    """Per-pair fraction of windows excluded — for logging/QC of the projection
    itself, so a run whose mask covers nothing (a silent join failure) is visible
    as an exactly-0.0 rate rather than passing for a clean recording."""
    if excluded.size == 0:
        return np.zeros(excluded.shape[1] if excluded.ndim == 2 else 0)
    return excluded.mean(axis=0)
