"""Reader for the full-resolution epoch cache (`features/pain/psd_epochs_fullres/`).

A SIBLING of `cache_reader`, not a replacement. The two caches have genuinely
different physical layouts -- psd_epochs is LONG (a `bin` column, 50 rows per
window-channel), psd_epochs_fullres is WIDE (499 float32 columns, one row per
window-channel) -- so the read path differs. Everything downstream of the read
does NOT: masking, all seven view axes, band aggregation and the slope fit all
take a `(n_win, n_pairs, n_freq)` block and a `bin_table`, and are agnostic to
how long the frequency axis is.

So this module deliberately implements only what is layout-specific, and
re-exports the rest:

    load_defs            <- cache_reader (the epoch index, copied into this unit)
    load_mask            <- cache_reader (keyed on run/channel/60s bin)
    epoch_excluded       <- cache_reader (per window x channel, freq-agnostic)
    apply_mask           <- cache_reader (broadcasts over the bin axis, any length)

THE FREQUENCY AXIS COMES FROM THE MANIFEST, NEVER FROM CONFIG. `n_freqs` is
`len(manifest['freqs_hz'])`. This is the one thing psd_epochs got wrong:
`views/build_pain_epoch_view.py:57` takes `n_bins = config.PSD_N_LOG_BINS`, so
moving that constant makes every cache already on disk raise `CacheLayoutError` --
loud, but for the wrong reason. A unit written under an older config stays
readable here.

`freq_table()` returns the same `(freq_bin_index, bin_low_hz, bin_high_hz)` shape
`cache_reader.bin_edges()` returns, with each native FFT frequency treated as a
bin of width df centred on itself. That is what lets `views.axes.aggregate_bands`
and `views.aperiodic.fit_bins` consume this cache with NO changes: both select by
geometric bin centre from exactly those columns.
"""

import logging

import numpy as np
import pandas as pd

from ieeg_ehr import config, io
from ieeg_ehr.views.cache_reader import (CacheLayoutError, apply_mask,  # noqa: F401
                                         epoch_excluded, load_defs, load_mask)

logger = logging.getLogger(__name__)


def manifest(epoch_minutes=None):
    """The unit's manifest, or raise -- the frequency axis lives only here."""
    unit = config.fullres_epoch_unit_dir(epoch_minutes)
    man = io.read_manifest(unit)
    if man is None:
        raise FileNotFoundError(
            f'no manifest.json at {unit}. The frequency axis is not recoverable '
            'without it: the cache columns are positional (f000..), so nothing '
            'else on disk says what f000 is in Hz.')
    return man


def freqs_hz(epoch_minutes=None):
    """The frequency axis in Hz, (n_freqs,) float64. THE source of truth."""
    man = manifest(epoch_minutes)
    try:
        return np.asarray(man['freqs_hz'], dtype=float)
    except KeyError:
        raise CacheLayoutError(
            f"manifest at {config.fullres_epoch_unit_dir(epoch_minutes)} has no "
            "'freqs_hz'; it was not written by build_pain_epoch_fullres_psd.")


def n_freqs(epoch_minutes=None):
    return len(freqs_hz(epoch_minutes))


def freq_columns(epoch_minutes=None):
    """The cache's frequency column names, in axis order."""
    from ieeg_ehr.features.build_pain_epoch_fullres_psd import freq_column_names
    return freq_column_names(n_freqs(epoch_minutes))


def freq_table(epoch_minutes=None):
    """(freq_bin_index, bin_low_hz, bin_high_hz) -- the shape the axes expect.

    Each native frequency f becomes a bin [f - df/2, f + df/2). Half-width rather
    than zero-width because `axes.aggregate_bands` selects on the GEOMETRIC centre
    `sqrt(lo*hi)` and `aperiodic.fit_bins` does too; a degenerate lo == hi would
    give centre == f (fine) but a `bin_high_hz - bin_low_hz` width of 0, which
    silently zeroes every weight under `band_weighting='width'`.

    Note the geometric centre of [f-df/2, f+df/2) is very slightly below f
    (sqrt(59.75*60.25) = 59.9987 for f=60). Irrelevant at 0.5 Hz spacing -- it
    shifts a band edge decision by <2 mHz -- but it is why `notch_freqs` below
    selects on the true frequency and not on the table.
    """
    f = freqs_hz(epoch_minutes)
    df = float(manifest(epoch_minutes)['params']['df_hz'])
    return pd.DataFrame({'freq_bin_index': np.arange(len(f)),
                         'bin_low_hz': f - df / 2.0,
                         'bin_high_hz': f + df / 2.0})


def notch_freqs(half_width_hz=None, line_freqs=None, epoch_minutes=None):
    """Indices of frequencies within +/- half_width of a line-noise harmonic.

    The full-resolution counterpart of psd_epochs' stored `contains_line_noise`
    flags -- but a VIEW DECISION, not a stored one, which is the whole point of
    this unit. On the 50-log-bin axis the notch was 6 of 50 bins and cost 13.2 Hz
    around 60 Hz alone; here the default +/-2 Hz takes 9 bins per harmonic
    (58.0-62.0 inclusive, 4.5 Hz of the axis) -- 36 of 499 cohort-wide -- and the
    width is free to sweep. The comparison is INCLUSIVE at both ends, so a
    half-width of h takes 2h/df + 1 bins, not 2h/df; an earlier version of this
    docstring said 8 and was off by one per harmonic (corrected 2026-09-16).

    Selects on the true frequency rather than on `freq_table`'s geometric centres
    -- see that docstring. Returns an int array, matching
    `cache_reader.line_noise_bins()` so a caller can swap one for the other.
    """
    half = (config.PSD_NOTCH_HALF_WIDTH_HZ if half_width_hz is None
            else half_width_hz)
    lines = config.PSD_LINE_NOISE_FREQS_HZ if line_freqs is None else line_freqs
    f = freqs_hz(epoch_minutes)
    flagged = np.zeros(len(f), dtype=bool)
    for lf in lines:
        flagged |= np.abs(f - lf) <= half
    return np.flatnonzero(flagged)


def open_cache(subject, session, epoch_minutes=None):
    import pyarrow.parquet as pq
    path = config.fullres_epoch_cache_path(subject, session, epoch_minutes)
    if not path.exists():
        raise FileNotFoundError(f'no full-res epoch cache at {path}')
    return pq.ParquetFile(path), path


def verify_layout(parquet_file, defs, n_freq, epochs_present=None):
    """Map each epoch to its CONTIGUOUS RANGE of row groups, proving the layout.

    Same hazard as `cache_reader.verify_layout` and the same remedy: the builder
    calls `write_table` once per epoch, but pyarrow SPLITS a call above its
    default row-group size, so epoch:row-group is not 1:1. Assuming it were would
    read the wrong slice and still produce a plausible figure.

    The row arithmetic differs from psd_epochs because the layout does: one row
    per (window, channel), with frequency in columns, so an epoch is
    `n_windows * n_channels` rows -- NOT times n_freq. That also means the
    ~1,048,576-row split ceiling is reached 499x less often: a 300-window epoch
    needs 3,496 pairs to cross it, so in practice every epoch here is exactly one
    row group. The general walk is kept anyway, because "in practice" is what the
    50-bin cache's 1:1 assumption also looked like.

    `epochs_present` restricts the expected set, for the case an epoch was skipped
    at build time (a short raw read) and so is in `defs` but not in the cache --
    the builder's sidecar records `n_epochs` vs `n_epochs_in_defs`.

    Returns {epoch_id: [row_group_index, ...]}.
    """
    defs = defs.sort_values('epoch_id').reset_index(drop=True)
    if epochs_present is not None:
        defs = defs[defs['epoch_id'].isin(set(epochs_present))].reset_index(drop=True)

    rg_rows = [parquet_file.metadata.row_group(i).num_rows
               for i in range(parquet_file.num_row_groups)]
    expected = int((defs['n_windows'] * defs['n_channels']).sum())
    if sum(rg_rows) != expected:
        raise CacheLayoutError(
            f'cache has {sum(rg_rows)} rows but defs implies {expected} '
            f'(sum of n_windows x n_channels). If the builder skipped an epoch on '
            'a short raw read, pass epochs_present; otherwise cache and '
            'epoch_defs are out of sync and this should be rebuilt, not guessed.')

    n_cols = parquet_file.metadata.num_columns
    if n_cols != n_freq + len(_INDEX_COLUMNS):
        raise CacheLayoutError(
            f'cache has {n_cols} columns, expected {n_freq} frequencies plus '
            f'{len(_INDEX_COLUMNS)} index columns. The manifest and the cache '
            'disagree about the frequency axis.')

    mapping, cursor = {}, 0
    for i in range(len(defs)):
        want = int(defs.at[i, 'n_windows']) * int(defs.at[i, 'n_channels'])
        got, groups = 0, []
        while got < want:
            if cursor >= len(rg_rows):
                raise CacheLayoutError(
                    f'ran out of row groups assembling epoch '
                    f'{int(defs.at[i, "epoch_id"])}: needed {want}, found {got}')
            got += rg_rows[cursor]
            groups.append(cursor)
            cursor += 1
        if got != want:
            raise CacheLayoutError(
                f'epoch {int(defs.at[i, "epoch_id"])} needs {want} rows but its '
                f'row groups hold {got}; an epoch boundary does not coincide with '
                'a row-group boundary, so a read would straddle two epochs.')
        mapping[int(defs.at[i, 'epoch_id'])] = groups
    if cursor != len(rg_rows):
        raise CacheLayoutError(f'{len(rg_rows) - cursor} row groups left unassigned')

    # Ravel-order proof on the first epoch: row r is (win = r // n_pairs,
    # pair = r % n_pairs). Getting this backwards would transpose windows and
    # channels and still average to something plausible.
    n_win = int(defs.at[0, 'n_windows'])
    n_pairs = int(defs.at[0, 'n_channels'])
    head = parquet_file.read_row_groups(mapping[int(defs.at[0, 'epoch_id'])],
                                        columns=['epoch_id', 'window_idx'])
    if not (np.asarray(head.column('epoch_id')) == int(defs.at[0, 'epoch_id'])).all():
        raise CacheLayoutError('epoch 0 row groups do not all hold the same epoch_id')
    if not np.array_equal(np.asarray(head.column('window_idx')),
                          np.repeat(np.arange(n_win), n_pairs)):
        raise CacheLayoutError(
            'window_idx is not a C-order ravel of (window, pair); the reshape in '
            'read_epoch() would transpose channels and windows into each other\'s '
            'slots and still produce a plausible figure. Refusing to continue.')

    n_split = sum(1 for g in mapping.values() if len(g) > 1)
    logger.info('fullres layout verified: %d epochs over %d row groups '
                '(%d epochs span >1), %d frequencies',
                len(mapping), len(rg_rows), n_split, n_freq)
    return mapping


_INDEX_COLUMNS = ('epoch_id', 'window_idx', 'channel')


def read_epoch(parquet_file, epoch_row, row_groups, columns=None,
               epoch_minutes=None):
    """One epoch as (n_windows, n_pairs, n_freq) float64.

    `columns` is a list of frequency column NAMES to read (see `freq_columns`);
    None reads all of them. This is the payoff of the wide layout and of Parquet
    generally -- a view that only needs alpha reads 8 columns off disk instead of
    499, which the long layout could not do at all.

    Upcasts to CACHE_ACCUMULATE_DTYPE after the reshape, one contiguous pass, per
    P0.6: store narrow, compute wide. Every reduction downstream needs float64,
    and numpy will NOT do this for you.
    """
    n_win = int(epoch_row['n_windows'])
    n_pairs = int(epoch_row['n_channels'])
    cols = list(columns) if columns is not None else freq_columns(epoch_minutes)

    table = parquet_file.read_row_groups(row_groups, columns=cols)
    if table.num_rows != n_win * n_pairs:
        raise CacheLayoutError(
            f'epoch {epoch_row["epoch_id"]}: {table.num_rows} rows, expected '
            f'{n_win * n_pairs} (n_windows x n_channels)')

    # Column-major -> (n_freq, n_rows) -> transpose. Building it this way rather
    # than via to_pandas avoids materialising a 499-column DataFrame per epoch.
    out = np.empty((len(cols), table.num_rows),
                   dtype=config.CACHE_ACCUMULATE_DTYPE)
    for j, name in enumerate(cols):
        out[j] = table.column(name).to_numpy(zero_copy_only=False)
    return out.T.reshape(n_win, n_pairs, len(cols))


def iter_epochs(parquet_file, defs, row_group_map, mask=None, channels_by_run=None,
                view_config=None, columns=None, epoch_filter=None,
                epoch_minutes=None):
    """Yield (epoch_row, block, kept, excluded_frac) per epoch, masked.

    Mirrors `cache_reader.iter_epochs` so a caller can switch caches by switching
    module. Streams one epoch at a time: the whole cache is ~3.6 GB per
    subject-session and does not want to be resident.
    """
    for _, ep in defs.sort_values('epoch_id').iterrows():
        epoch_id = int(ep['epoch_id'])
        if epoch_id not in row_group_map:
            continue
        if epoch_filter is not None and not epoch_filter(ep):
            continue

        block = read_epoch(parquet_file, ep, row_group_map[epoch_id],
                           columns=columns, epoch_minutes=epoch_minutes)
        kept = np.ones(block.shape[1], dtype=bool)
        frac = np.zeros(block.shape[1])
        if mask is not None and view_config is not None:
            channels = (channels_by_run or {}).get(ep['run_id'])
            if channels is None:
                raise CacheLayoutError(
                    f'epoch {epoch_id}: masking was requested but no channel list '
                    f'was supplied for {ep["run_id"]}. An empty list would read as '
                    '"nothing excluded", which is the failure mode that makes '
                    'incomplete QC look like clean data.')
            excluded = epoch_excluded(mask, ep, channels, view_config)
            block, kept, frac = apply_mask(block, excluded,
                                           view_config.resolved().max_excluded_frac)
        yield ep, block, kept, frac
