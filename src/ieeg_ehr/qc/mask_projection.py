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

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

BIN_SEC = 60.0

MASK_COLUMNS = ['run_id', 'channel', 'bin_start', 'excluded']

# The join key for a per-subject (multi-session) mask lookup. session_id is part
# of it because load_mask_lookup() concatenates across a subject's sessions and
# run_id is only unique WITHIN a session -- joining without it would let one
# session's verdict leak onto an identically-named run in another.
LOOKUP_KEY = ['session_id', 'run_id', 'channel', 'bin_start']


def split_pair(channel):
    """('LAH1-LAH2') -> ('LAH1', 'LAH2'). Returns (channel, None) if the name is
    not a pair, so a monopolar-named channel degrades to "look me up directly"
    rather than raising — some derivative trees mix the two."""
    if '-' not in channel:
        return channel, None
    anode, cathode = channel.split('-', 1)
    return anode, cathode


def load_mask(mask_path, run_id=None, columns=None):
    """Read one subject/session's mask table, optionally for one run.

    Dispatches on the file extension, because the two levels store their masks
    differently and both are legitimate: the raw-voltage tree is CSV (it predates
    the Parquet convention and works -- docs/io_conventions.md §7), while the
    bipolar tree is Parquet (a new artifact, so it follows the current rule). A
    caller should be able to name a mask without knowing which.

    Returns None if the file does not exist. That is deliberately not an error:
    build_mask.py drops subject/sessions that are missing any artifact type (the
    sub-236 gap, docs/qc_context.md), so a missing file means "this
    subject/session has no mask at this label" and the caller decides whether to
    proceed unmasked or skip. Callers MUST log which they did -- an absent mask
    silently keeps artifact windows, which inflates a baseline std, deflates z,
    and makes a detector LESS sensitive for exactly the subjects whose QC inputs
    were incomplete.

    NOTE ON KEYING: this returns the table as stored and does NOT tell you whether
    `channel` holds monopolar contacts or bipolar pairs. That is the caller's job
    to know, and it decides which projector is correct --
    project_to_pairs (monopolar) vs project_pair_mask_to_windows (pair-keyed).
    Guessing wrong returns all-False rather than failing, hence
    project_pair_mask_to_windows' docstring.
    """
    path = Path(mask_path)
    if not path.exists():
        return None
    cols = list(columns) if columns is not None else MASK_COLUMNS
    if path.suffix == '.parquet':
        # Parquet carries dtypes, so `excluded` arrives as real bool rather than
        # the object/str a CSV round-trip can produce.
        df = pd.read_parquet(path, columns=cols)
    else:
        df = pd.read_csv(path, usecols=cols)
    if run_id is not None:
        df = df[df['run_id'] == run_id]
    return df


def load_mask_lookup(mask_dir, subject_id):
    """All of ONE subject's mask rows, across every session, as a merge key.

    Session lives in the FILENAME at the raw-voltage level (`sub-019_ses-01.csv`,
    per the 2026-07-14 filename migration) rather than in a column, so it is
    recovered here and inserted -- see LOOKUP_KEY for why joining without it is
    unsafe.

    Returns an EMPTY frame with the right columns when nothing matches, never
    None. Callers that treat "no mask" as "nothing excluded" MUST say so in a log
    line: an absent mask silently keeps artifact windows, which inflates a
    baseline std, deflates z, and makes a detector LESS sensitive for exactly the
    subjects whose QC inputs were incomplete (docs/labnotebook/2026-07-28.md).
    """
    frames = []
    for mask_path in sorted(Path(mask_dir).glob(f'{subject_id}_ses-*.csv')):
        session_id = mask_path.stem.split('_ses-', 1)[1].split('_')[0]
        df = pd.read_csv(mask_path, usecols=MASK_COLUMNS)
        df.insert(0, 'session_id', f'ses-{session_id}')
        frames.append(df)
    if not frames:
        return pd.DataFrame(columns=['session_id'] + MASK_COLUMNS)
    return pd.concat(frames, ignore_index=True)


def or_pair_flags_60s(df, mask_df, channel_col):
    """Per-row bool: is this row's 60s bin excluded for the contact named in
    `channel_col`?

    THE PAIR RULE at 60s-bin granularity. Call it twice -- once with
    'anode_channel', once with 'cathode_channel' -- and OR the results; that is
    the same rule project_to_pairs applies at window granularity, which is the
    whole reason both live in this module.

    `df` must already carry a `_bin` column (floor of run seconds to BIN_SEC).
    `mask_df` is a load_mask_lookup() frame, keyed on MONOPOLAR contact.

    Returns a numpy array POSITIONALLY aligned to `df`, so a caller can assign it
    straight onto a column. That alignment holds only if the merge cannot
    multiply rows, hence the duplicate-key check: a duplicated mask key would
    silently shift every row after the first duplicate, which is invisible in any
    single number and would quietly bias every downstream effect size.
    """
    if mask_df is None or len(mask_df) == 0:
        return np.zeros(len(df), dtype=bool)

    n_dupes = int(mask_df.duplicated(LOOKUP_KEY).sum())
    if n_dupes:
        raise ValueError(
            f'mask lookup has {n_dupes} duplicate {LOOKUP_KEY} rows; a left merge '
            'on it would multiply rows and break positional alignment. Fix the '
            'mask source rather than de-duplicating here -- duplicates mean two '
            'disagreeing verdicts for one bin, and silently keeping either is wrong.'
        )

    on = ['session_id', 'run_id', channel_col, '_bin']
    merged = df[on].merge(
        mask_df.rename(columns={'channel': channel_col, 'bin_start': '_bin'}),
        on=on, how='left',
    )
    # .eq(True) rather than .fillna(False).astype(bool): the merged column is
    # object dtype (bool + NaN from unmatched rows), and fillna-then-downcast on
    # object is deprecated in pandas and changes behaviour on upgrade.
    # Unmatched -> NaN -> False, which is the "no mask row means not excluded"
    # convention every raw-voltage consumer already uses.
    return merged['excluded'].eq(True).to_numpy()


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

    return _gather_by_bin(bin_starts, pair_by_bin, run_seconds, n_pairs)


def _gather_by_bin(bin_starts, by_bin, run_seconds, n_cols):
    """Broadcast a per-60s-bin verdict matrix out to per-window rows.

    Each window takes the row of `by_bin` for its enclosing 60s bin. Windows whose
    bin is absent from the mask entirely map to an all-False row appended at the
    end, so "no mask row" means "not excluded" -- matching the convention the
    raw-voltage consumers already use -- rather than dropping the window or
    raising. The per-window step is a gather, not a join, which is what keeps this
    cheap for ~10^5 windows x ~200 pairs.
    """
    want = np.floor(np.asarray(run_seconds, dtype=np.float64) / BIN_SEC) * BIN_SEC
    pos = np.searchsorted(bin_starts, want)
    pos = np.clip(pos, 0, len(bin_starts) - 1)
    missing = bin_starts[pos] != want
    by_bin = np.vstack([by_bin, np.zeros((1, n_cols), dtype=bool)])
    pos = np.where(missing, len(bin_starts), pos)
    return by_bin[pos]


def project_pair_mask_to_windows(mask_df, run_id, channel_names, run_seconds):
    """(n_windows, n_pairs) bool from an ALREADY-PAIR-KEYED mask, e.g. one from
    qc/bipolar/masks/ or qc/bipolar/exclusions/bipolar_variance/.

    The pair-level twin of project_to_pairs, and deliberately a SEPARATE entry
    point rather than an auto-detecting branch inside it. Handing a pair-keyed
    table to project_to_pairs does not fail loudly: it splits 'LA1-LA2' into
    'LA1'/'LA2', finds neither among the table's pair-named channels, and returns
    all-False for every window -- a silent "nothing is excluded", which is the
    worst possible failure mode for a QC mask. Making the caller state which
    keying it holds is the point.

    Unlike project_to_pairs there is no anode/cathode OR here, because a
    pair-keyed mask has already applied the pair rule.
    """
    n_win = len(run_seconds)
    n_pairs = len(channel_names)
    if mask_df is None or n_win == 0 or n_pairs == 0:
        return np.zeros((n_win, n_pairs), dtype=bool)

    run_mask = mask_df[mask_df['run_id'] == run_id]
    if run_mask.empty:
        return np.zeros((n_win, n_pairs), dtype=bool)

    pivot = (run_mask.pivot_table(index='bin_start', columns='channel',
                                  values='excluded', aggfunc='any')
             .fillna(False).astype(bool))
    bin_starts = pivot.index.to_numpy(dtype=np.float64)
    cols = {name: i for i, name in enumerate(pivot.columns)}
    by_pair = pivot.to_numpy()

    absent = np.zeros(len(bin_starts), dtype=bool)
    missing_pairs = [c for c in channel_names if c not in cols]
    if missing_pairs:
        # Loud, because an absent pair reads as "clean" and there is no other
        # signal that it was never actually evaluated.
        logger.warning(
            'run %s: %d/%d pairs absent from the pair-keyed mask and treated as '
            'NOT excluded (e.g. %s). If this is most of the montage, the mask '
            'probably does not match this run\'s channel naming.',
            run_id, len(missing_pairs), n_pairs, missing_pairs[:5],
        )

    out_by_bin = np.empty((len(bin_starts), n_pairs), dtype=bool)
    for j, channel in enumerate(channel_names):
        out_by_bin[:, j] = by_pair[:, cols[channel]] if channel in cols else absent

    return _gather_by_bin(bin_starts, out_by_bin, run_seconds, n_pairs)


def excluded_fraction(excluded):
    """Per-pair fraction of windows excluded — for logging/QC of the projection
    itself, so a run whose mask covers nothing (a silent join failure) is visible
    as an exactly-0.0 rate rather than passing for a clean recording."""
    if excluded.size == 0:
        return np.zeros(excluded.shape[1] if excluded.ndim == 2 else 0)
    return excluded.mean(axis=0)
