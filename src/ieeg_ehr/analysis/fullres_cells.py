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
