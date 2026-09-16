"""Model frames for one region x frequency cell, from the FULL-RESOLUTION view.

The load-path sibling of `run_mixed_model_pilot.load_cell_frames`, reading
`features/pain/psd_epochs_fullres/<epoch>/views/fullresmean-*` instead of the
50-log-bin per-channel view. Everything downstream of the load is SHARED and
unchanged: `mixed_model.build_cell_frame` tidies a cell, `mixed_model.fit_cell`
fits it, and the ROI map comes from the same `channel_meta` tables the 50-bin
analysis used -- deliberately, so "Insula" means the same set of contacts in both
runs and the two maps are comparable.

THREE THINGS THE WIDE LAYOUT CHANGES, which is why this is a separate module
rather than a `--source` flag on the old loader:

  - The view is WIDE: one row per (epoch, channel), frequency in 499 float32
    columns f000..f498. The 50-bin view is LONG, with a `freq_bin_index` COLUMN,
    so `df[df.freq_bin_index.isin(wanted)]` has no counterpart here -- selecting
    frequencies means selecting columns, and it happens on the way off disk.
  - Rows carry NO subject_id and NO region. Subject comes from the FILENAME,
    region from the channel_meta join; both are done here.
  - The frequency axis comes from the unit's MANIFEST, never from a config
    constant (see `views/fullres_reader`), and the line-noise notch is a
    VIEW-TIME choice with a sweepable width rather than a flag baked into the
    stored artifact.

A region's cells are loaded ONCE as an (n_rows, n_bins) float32 matrix plus an
index frame, and each cell is sliced out of it one column at a time. Building 463
DataFrames up front would hold the same numbers several times over in Python
objects; the matrix is ~75 MB for the largest region.

WHAT THE EPOCH MEAN IS HERE, AND HOW IT DIFFERS FROM THE 50-BIN VIEW. This view
averages LINEAR power over windows and then logs it (`log10(mean(10**x))`); the
50-bin chan view averaged the log values directly (`mean(x)`), which is a
GEOMETRIC mean. They are different quantities. The gap between them is close to
constant for a fixed number of averaged windows, and a channel's constant is
absorbed by the channel random intercept, so the pain SLOPE should barely move --
but "barely" is an expectation, not an identity, and it belongs in the METHODS of
any run that compares the two maps.
"""

import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from ieeg_ehr import config, io
from ieeg_ehr.analysis import mixed_model as mm
from ieeg_ehr.views import fullres_reader, view_config as vc

logger = logging.getLogger(__name__)

#: Index columns of the epoch-mean view, i.e. everything that is not a frequency.
INDEX_COLUMNS = ('epoch_id', 'channel', 'pain_score')


# ============================================================================
# WHERE THE VIEW IS
# ============================================================================

def resolve_view_dir(explicit=None, *, mask_label=None, max_excluded_frac=None,
                     epoch_minutes=None):
    """The epoch-mean view directory, resolved the way its BUILDER resolves it.

    Never a hard-coded config_hash: the hash is recomputed from
    `build_pain_epoch_fullres_mean.mean_params`, so the builder and every reader
    go through one code path and cannot disagree about which directory is meant.
    If the hash moves because that params dict changed, this raises with the
    directories that DO exist rather than silently globbing up a stale one.
    """
    if explicit:
        return Path(explicit)

    from ieeg_ehr.views import build_pain_epoch_fullres_mean as frm

    view = vc.ViewConfig(normalization='none', domain='log', epoch_agg='mean',
                         freq='fullres', region='none', pain_bins='subject_relative',
                         mask_level='bipolar', mask_label=mask_label,
                         max_excluded_frac=max_excluded_frac,
                         epoch_minutes=epoch_minutes).resolved()
    n_freqs = fullres_reader.n_freqs(view.epoch_minutes)
    params = frm.mean_params(view, n_freqs)
    out = config.fullres_epoch_views_dir(
        f'{frm.VIEW_LABEL_PREFIX}-{view.scheme_code}', io.config_hash(params),
        view.epoch_minutes)
    if not out.exists():
        siblings = sorted(p.name for p in out.parent.glob('*') if p.is_dir())
        raise SystemExit(
            f'no epoch-mean view at {out}\n'
            f'  params hashed: {params}\n'
            f'  views present: {siblings}\n'
            'Build it first:  sbatch sbatch/build_fullres_mean_array.sbatch')
    return out


def subject_paths(view_dir):
    paths = sorted(Path(view_dir).glob('mean_sub-*_ses-*.parquet'))
    if not paths:
        raise SystemExit(
            f'no mean_sub-*.parquet in {view_dir}.\n'
            'Build the epoch-mean view first:\n'
            '    sbatch sbatch/build_fullres_mean_array.sbatch')
    return paths


def subject_session_of(path):
    """('085', '01') from mean_sub-085_ses-01.parquet."""
    stem = Path(path).stem
    return stem.split('sub-')[1].split('_')[0], stem.split('ses-')[1].split('_')[0]


def load_epoch_scores(paths):
    """(subject_id, epoch_id, pain_score), one row per epoch.

    Two narrow columns off disk per subject. The subject id is NOT in the table
    -- the epoch-mean view is written per subject-session and its rows say
    nothing about whose they are -- so it comes from the filename, which is the
    only place it exists.
    """
    frames = []
    for p in paths:
        subject, _ = subject_session_of(p)
        df = io.read_table(p, columns=['epoch_id', 'pain_score'], on_stale='warn')
        df = df.drop_duplicates('epoch_id')
        df.insert(0, 'subject_id', f'sub-{subject}')
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


# ============================================================================
# THE FREQUENCY AXIS
# ============================================================================

def analysis_freq_table(notch_half_width_hz=None, epoch_minutes=None):
    """(bin_table indexed by freq_bin_index, dropped notch indices).

    The full native axis from the manifest, minus the line-noise notch. Both the
    axis and the notch come from `fullres_reader` -- the axis from the unit's
    manifest and never from `config.PSD_FULLRES_N_FREQS`, the notch from a
    half-width that is a VIEW parameter here rather than a stored flag.

    The notch costs 36 of 499 bins at the default +/-2 Hz -- 9 per harmonic,
    since the comparison is inclusive at both ends: 58.0-62.0, 118.0-122.0,
    178.0-182.0, 238.0-242.0 Hz, 18 Hz of spectrum. The 50-log-bin analysis this
    replaces dropped 6 of 50 bins, where 60 Hz alone cost 13.2 Hz.
    """
    table = fullres_reader.freq_table(epoch_minutes).set_index('freq_bin_index')
    dropped = [int(b) for b in fullres_reader.notch_freqs(
        half_width_hz=notch_half_width_hz, epoch_minutes=epoch_minutes)]
    freqs = fullres_reader.freqs_hz(epoch_minutes)
    logger.info('frequency axis: %d native bins, notch drops %d '
                '(%s Hz) -> %d bins fitted',
                len(table), len(dropped),
                ', '.join(f'{freqs[b]:.1f}' for b in dropped),
                len(table) - len(dropped))
    return table.drop(index=[b for b in dropped if b in table.index]), dropped


# ============================================================================
# LOADING ONE REGION
# ============================================================================

def load_region_matrix(paths, subjects, region, roi_by_subject, bin_indices,
                       epoch_minutes=None):
    """(index frame, (n_rows, n_bins) float32 matrix, stats) for one region.

    One subject's file at a time, filtered to this region's channels right after
    the read so peak memory is one subject's slice. Only the wanted frequency
    COLUMNS are decoded -- the payoff of the wide layout, and the thing the long
    50-bin layout could not do.

    Rows stay float32, per P0.6 "store narrow, compute wide": the upcast to
    float64 happens in `mixed_model.build_cell_frame`, once per cell, where the
    reductions actually are.
    """
    all_cols = fullres_reader.freq_columns(epoch_minutes)
    cols = [all_cols[int(b)] for b in bin_indices]

    idx_parts, val_parts = [], []
    stats = {'n_subjects': 0, 'n_files': 0, 'n_channels': 0, 'n_nonfinite': 0,
             'n_rows': 0}
    for p in paths:
        subject, _ = subject_session_of(p)
        sid = f'sub-{subject}'
        if sid not in subjects or sid not in roi_by_subject:
            continue
        mapping = roi_by_subject[sid]
        wanted = {c for c, r in mapping.items() if r == region}
        if not wanted:
            continue

        df = io.read_table(p, columns=list(INDEX_COLUMNS) + cols, on_stale='warn')
        # Channels present in the view but absent from channel_meta belong to no
        # region at all, so they cannot be noticed here as a smaller region --
        # `unmapped_channels()` counts them once at prepare time instead.
        df = df[df['channel'].isin(wanted)]
        if df.empty:
            continue

        values = df[cols].to_numpy(dtype=config.CACHE_FLOAT_DTYPE)
        index = df[list(INDEX_COLUMNS)].copy()
        index.insert(0, 'subject_id', sid)
        idx_parts.append(index)
        val_parts.append(values)
        stats['n_subjects'] += 1
        stats['n_files'] += 1
        stats['n_channels'] += int(df['channel'].nunique())
        stats['n_nonfinite'] += int((~np.isfinite(values)).sum())

    if not idx_parts:
        return (pd.DataFrame(columns=['subject_id', *INDEX_COLUMNS]),
                np.empty((0, len(cols)), dtype=config.CACHE_FLOAT_DTYPE), stats)

    index = pd.concat(idx_parts, ignore_index=True)
    values = np.concatenate(val_parts, axis=0)
    stats['n_rows'] = int(len(index))
    return index, values, stats


def channel_audit(paths, subjects, roi_by_subject):
    """{subject_id: {'absent_from_meta': [...], 'n_non_roi': int}} per subject.

    TWO DIFFERENT THINGS, kept apart because only one of them is a problem:

      `absent_from_meta` -- the view has a bipolar pair `channel_meta` has never
          heard of. That IS a problem: the fullres unit derives its pair list
          from the raw file while channel_meta's came from the PSD NWB, and some
          sessions carry `pairs_diverged_from_session_first_run`, so a silent
          divergence here would mislabel contacts rather than merely drop them.
      `n_non_roi` -- the pair is in channel_meta and its DK label maps to a
          non-ROI category (white matter, CSF, Exclude, Other). EXPECTED, in
          every subject, and the same channels the 50-bin analysis dropped. It is
          counted so the two numbers cannot be read as one.

    Run once at prepare time rather than per region: a pair in neither bucket is
    in no region at all, so it does not show up as a smaller region -- it shows
    up as nothing.
    """
    from ieeg_ehr.views import channel_meta

    out = {}
    for p in paths:
        subject, session = subject_session_of(p)
        sid = f'sub-{subject}'
        if sid not in subjects:
            continue
        try:
            known = set(channel_meta.build(subject, session, [])['channel'].unique())
        except FileNotFoundError:
            known = set()
        in_view = set(io.read_table(p, columns=['channel'],
                                    on_stale='ignore')['channel'].unique())
        with_roi = set(roi_by_subject.get(sid, {}))
        absent = sorted(in_view - known)
        n_non_roi = len((in_view & known) - with_roi)
        if absent or n_non_roi:
            out[sid] = {'absent_from_meta': absent, 'n_non_roi': int(n_non_roi),
                        'n_in_view': len(in_view), 'n_with_roi': len(in_view & with_roi)}
    return out


# ============================================================================
# PER-SUBJECT SLOPE MAPS, for the two-stage cluster test
# ============================================================================
# The wide-layout counterpart of `pain_coef.subject_coef_matrix`. Same estimand
# -- one OLS slope of ROI power on pain score, per subject per cell -- and the
# same `pain_coef.coef_from_predictor` identity behind it, but the input is the
# 499-column epoch-mean view rather than a long table, and the output carries the
# pre-blocked structure the permutation null needs.

def roi_epoch_matrix(path, mapping, regions, freq_cols):
    """((n_epochs, n_region, n_freq) float64, pain scores, n_channels per region).

    CHANNELS ARE AVERAGED WITHIN AN ROI AS A MEAN OF LOGS, not linear-then-log.
    That is a deliberate departure from registry AXIS 6 and from the 2026-08-08
    two-stage reference run, and the reason is what this matrix is FOR: it backs
    the mixed-model map, and the mixed model keeps per-channel log values and
    absorbs each contact's level in a random intercept -- which makes its fixed
    effect an average of per-contact log slopes, i.e. the slope of a mean of logs.
    A linear-then-log ROI value is dominated by the loudest contact, so its slope
    is a different quantity and would not be the thing the grid estimated.

    NaN-aware on the channel axis: a channel-epoch the QC mask blanked contributes
    nothing, and an ROI whose every contact was blanked for an epoch comes out NaN
    rather than zero.
    """
    df = io.read_table(path, columns=list(INDEX_COLUMNS) + list(freq_cols),
                       on_stale='warn')
    df = df.assign(roi=df['channel'].map(mapping)).dropna(subset=['roi'])
    if df.empty:
        return None, None, None

    epochs = sorted(df['epoch_id'].unique())
    e_idx = {e: i for i, e in enumerate(epochs)}
    r_idx = {r: i for i, r in enumerate(regions)}
    df = df[df['roi'].isin(r_idx)]
    if df.empty:
        return None, None, None

    grouped = df.groupby(['epoch_id', 'roi'], sort=False)[list(freq_cols)].mean()
    counts = df.groupby(['epoch_id', 'roi'], sort=False)['channel'].nunique()

    Y = np.full((len(epochs), len(regions), len(freq_cols)), np.nan)
    ei = np.fromiter((e_idx[e] for e, _ in grouped.index), int, len(grouped))
    ri = np.fromiter((r_idx[r] for _, r in grouped.index), int, len(grouped))
    Y[ei, ri, :] = grouped.to_numpy(dtype=np.float64)

    n_chan = np.zeros(len(regions), dtype=int)
    for (_, roi), c in counts.items():
        n_chan[r_idx[roi]] = max(n_chan[r_idx[roi]], int(c))

    x = (df.drop_duplicates('epoch_id').set_index('epoch_id')
         .loc[epochs, 'pain_score'].to_numpy(dtype=np.float64))
    return Y, x, n_chan


def coef_blocks(Y2d, min_rows=3):
    """[(row indices, column indices, Y block)] grouping columns by MISSINGNESS.

    The permutation null recomputes a subject's whole slope map thousands of
    times, and `pain_coef.coef_from_predictor` falls back to a PER-COLUMN Python
    loop for any column with a missing epoch. At 10,479 cells x ~50 subjects x
    2,000 permutations that fallback is ~10^9 iterations, which is the difference
    between a minute and a week.

    It is avoidable because the missingness has structure: the QC mask blanks
    whole (window, channel) cells and broadcasts over frequency, so an ROI mean is
    NaN for ALL frequencies of an epoch at once, and a subject has at most
    n_region distinct patterns rather than n_cells. Grouping columns by their
    exact finite pattern therefore collapses the loop to a handful of matmuls --
    and it does so WITHOUT assuming that structure: `np.unique` finds whatever
    patterns are actually there, so the uncharacterized non-finite values in the
    cache can only cost speed, never correctness.

    Columns with fewer than `min_rows` finite epochs are dropped entirely: a slope
    on two points is noise wearing a number. They stay NaN in the output map.
    """
    finite = np.isfinite(Y2d)
    patterns, inverse = np.unique(finite, axis=1, return_inverse=True)
    blocks = []
    for k in range(patterns.shape[1]):
        rows = np.flatnonzero(patterns[:, k])
        if rows.size < min_rows:
            continue
        cols = np.flatnonzero(inverse == k)
        blocks.append((rows, cols, np.ascontiguousarray(Y2d[np.ix_(rows, cols)])))
    return blocks


def coef_from_blocks(x, blocks, n_cells):
    """The slope map for one subject, from pre-blocked data. NaN where unestimable.

    `x` is the subject's pain scores IN EPOCH ORDER -- permuted or not. Each block
    re-derives the OLS weights over exactly its own surviving rows, which is the
    same correctness argument `pain_coef.coef_from_predictor` makes for its slow
    path; the blocking only changes how many times that has to happen.
    """
    from ieeg_ehr.analysis import pain_coef

    out = np.full(n_cells, np.nan)
    for rows, cols, block in blocks:
        w = pain_coef.regression_weights(x[rows])
        if w is None:
            continue
        out[cols] = w @ block
    return out


def subject_coef_matrix(paths, subjects, roi_by_subject, regions, freq_cols,
                        min_rows=3):
    """(coef (n_subject, n_region, n_freq), subject ids, {subject: (x, blocks)},
    per-subject region coverage).

    A cell the subject has no coverage for is NaN, never 0 -- 0 is a real
    coefficient meaning "no relationship", and conflating the two would feed
    fabricated nulls into the group mean.
    """
    n_region, n_freq = len(regions), len(freq_cols)

    # ONE ROW PER SUBJECT, NOT PER SUBJECT-SESSION. Two subjects in this view have
    # two sessions, and iterating paths would enter such a subject TWICE in the
    # across-subject t -- double-weighting one patient and breaking the n=51 the
    # cohort assertion just verified.
    by_subject = {}
    for p in paths:
        subject, _ = subject_session_of(p)
        by_subject.setdefault(f'sub-{subject}', []).append(p)
    multi = {s: len(v) for s, v in by_subject.items() if len(v) > 1}
    if multi:
        logger.info('%d subject(s) contribute more than one session and are POOLED '
                    'with within-session centring: %s', len(multi), multi)

    kept, per_subject, coverage = [], {}, []
    for sid, sid_paths in sorted(by_subject.items()):
        if sid not in subjects or sid not in roi_by_subject:
            continue
        parts_y, parts_x, n_chan_tot = [], [], np.zeros(n_region, dtype=int)
        for p in sid_paths:
            Y, x, n_chan = roi_epoch_matrix(p, roi_by_subject[sid], regions,
                                            freq_cols)
            if Y is None:
                continue
            # CENTRE WITHIN SESSION, both sides. Two sessions of one patient have
            # different contacts, so each carries its own ROI power offset; stacking
            # them raw would let a between-session difference in pain level be read
            # as a within-patient slope. Demeaning x and y inside each session is
            # exactly a session fixed effect, and it makes the pooled OLS weights
            # the within-session slope. A single-session subject is unaffected --
            # centring is what the slope already removes.
            Y2 = Y.reshape(len(x), n_region * n_freq)
            if len(sid_paths) > 1:
                with warnings.catch_warnings():
                    # All-NaN columns are the EXPECTED case -- they are the regions
                    # this subject has no contact in -- and NaN is the answer we
                    # want there. Left as a warning it fires once per uncovered
                    # region per session and buries the log.
                    warnings.filterwarnings('ignore', message='Mean of empty slice',
                                            category=RuntimeWarning)
                    Y2 = Y2 - np.nanmean(Y2, axis=0, keepdims=True)
                x = x - x.mean()
            parts_y.append(Y2)
            parts_x.append(x)
            n_chan_tot = np.maximum(n_chan_tot, n_chan)
        if not parts_y:
            logger.warning('%s: no ROI-labelled channel in the view, skipped', sid)
            continue

        Y2d = np.vstack(parts_y)
        x_all = np.concatenate(parts_x)
        blocks = coef_blocks(Y2d, min_rows=min_rows)
        if not blocks:
            logger.warning('%s: no cell has >=%d finite epochs, skipped', sid, min_rows)
            continue
        kept.append(sid)
        per_subject[sid] = (x_all, blocks)
        coverage.append({'subject_id': sid, 'n_sessions': len(parts_y),
                         'n_epochs': len(x_all),
                         'n_regions_covered': int((n_chan_tot > 0).sum()),
                         'n_channels': int(n_chan_tot.sum()),
                         'n_missingness_patterns': len(blocks)})

    if not kept:
        raise SystemExit('no subject produced a slope map')

    coef = np.full((len(kept), n_region, n_freq), np.nan)
    for i, sid in enumerate(kept):
        x, blocks = per_subject[sid]
        coef[i] = coef_from_blocks(x, blocks, n_region * n_freq).reshape(
            n_region, n_freq)
    logger.info('slope maps: %d subjects x %d regions x %d bins; %d/%d cells '
                'estimable', len(kept), n_region, n_freq,
                int(np.isfinite(coef).sum()), coef.size)
    return coef, kept, per_subject, pd.DataFrame(coverage)


def cell_frame(index, values, column, *, region, freq_bin_index):
    """One cell's model frame, sliced out of the region matrix.

    `column` is the POSITION in `values`, not the frequency bin index -- the two
    differ as soon as the notch removes a bin, and conflating them would silently
    shift every cell's frequency label by the number of dropped bins below it.
    """
    rows = pd.DataFrame({
        'subject_id': index['subject_id'].to_numpy(),
        'channel': index['channel'].to_numpy(),
        'epoch_id': index['epoch_id'].to_numpy(),
        'pain_score': index['pain_score'].to_numpy(),
        'value': values[:, column],
    })
    return mm.build_cell_frame(rows, region=region, freq_bin_index=freq_bin_index)
