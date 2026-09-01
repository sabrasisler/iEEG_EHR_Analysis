"""
Stream the per-window cache one epoch at a time as a dense (window, pair, bin)
array, and apply the QC mask.

THE PERFORMANCE IDEA
--------------------
build_pain_epoch_power.py writes ONE PARQUET ROW GROUP PER EPOCH, and within a
row group the rows are a C-order ravel of (window, pair, bin) -- it builds them
with `np.repeat`/`np.tile` in exactly that order. So an epoch can be recovered by
reading ONE COLUMN of ONE ROW GROUP and reshaping:

    log_power = f.read_row_group(i, columns=['log_power'])   ->  (n_win*n_pairs*n_bins,)
    block     = log_power.reshape(n_win, n_pairs, n_bins)

No groupby, no pivot, no join. That is the difference between a few seconds and
many minutes per subject: the largest subject is 409M rows / 1.6 GB, which as a
pandas long frame would not fit comfortably in memory at all, while one epoch is
300 x ~200 x 50 float64 = ~24 MB.

...AND WHY IT IS GUARDED, NOT ASSUMED
-------------------------------------
That reshape is a silent-corruption risk: if the ravel order were ever different,
every channel and frequency would be transposed into the wrong slot and the
output would still be a plausible-looking heatmap. So `verify_layout()` checks the
index columns against the expected repeat/tile pattern before any data is trusted,
and every row group's row count is checked against n_win*n_pairs*n_bins. Cheap
(index columns only, one row group) relative to being wrong.

PRECISION (P0.6)
----------------
Blocks are upcast to config.CACHE_ACCUMULATE_DTYPE (float64) on the way out. The
cache is float32 and numpy does NOT widen accumulators for you: a float32 mean
over ~300 windows holds only ~6 significant figures, the largest precision loss
anywhere in this pipeline. Exponentiating to linear likewise runs in float64,
because the worst stored log-power (~-36.8) is one decade above float32's smallest
normal and a later baseline division would underflow to exactly zero.
"""

import logging

import numpy as np
import pandas as pd

from ieeg_ehr import config, io
from ieeg_ehr.qc import mask_projection

logger = logging.getLogger(__name__)


class CacheLayoutError(RuntimeError):
    """The cache is not laid out the way the fast path requires.

    Its own exception type because the correct response is to STOP, never to fall
    back to a slow path that might disagree numerically.
    """


def open_cache(subject, session, epoch_minutes=None):
    import pyarrow.parquet as pq
    path = config.pain_epoch_cache_path(subject, session, epoch_minutes)
    if not path.exists():
        raise FileNotFoundError(f'no epoch cache at {path}')
    return pq.ParquetFile(path), path


def load_defs(subject, session, epoch_minutes=None, require_timing=True):
    """The epoch index, with the run-timing columns the mask join needs."""
    path = config.pain_epoch_defs_path(subject, session, epoch_minutes)
    defs = io.read_table(path, on_stale='ignore')
    missing = [c for c in ('epoch_start_sec', 'hop_sec') if c not in defs.columns]
    if missing and require_timing:
        raise CacheLayoutError(
            f'{path.name} lacks {missing}. Run '
            '`python -m ieeg_ehr.features.backfill_epoch_defs_timing` first -- '
            'without run-relative seconds the 60s QC mask cannot be aligned, and '
            'guessing the hop would silently mis-mask every window.'
        )
    return defs


def verify_layout(parquet_file, defs, n_bins):
    """Map each epoch to its CONTIGUOUS RANGE of row groups, proving the layout.

    NOT one row group per epoch. The builder calls `ParquetWriter.write_table`
    once per epoch, and pyarrow starts a new row group per call -- but it also
    SPLITS a single call that exceeds its default row-group size (~1,048,576
    rows). An epoch is n_windows x n_pairs x n_bins values, so a 300-window
    subject crosses that ceiling at 70 pairs: sub-019 (35 pairs, 525k rows) is one
    row group per epoch, sub-039 (85 pairs, 1.275M rows) is two. Assuming 1:1
    would have read the wrong slice for most of the cohort and still produced a
    plausible heatmap.

    What is guaranteed, and what this relies on, is that epoch boundaries always
    coincide with row-group boundaries (a new write_table always starts a new row
    group). So epochs map to consecutive runs of row groups, verified by exact
    cumulative row-count alignment -- if a boundary ever failed to line up, the
    walk below raises instead of silently straddling two epochs.

    Returns {epoch_id: [row_group_index, ...]}.
    """
    defs = defs.sort_values('epoch_id').reset_index(drop=True)
    n_epochs = len(defs)
    if list(defs['epoch_id']) != list(range(n_epochs)):
        raise CacheLayoutError(
            f'epoch_id is not 0..{n_epochs - 1}; the epoch <-> row-group mapping is '
            'positional, so a gap would pair data with the wrong pain score.'
        )

    rg_rows = [parquet_file.metadata.row_group(i).num_rows
               for i in range(parquet_file.num_row_groups)]
    total_expected = int((defs['n_windows'] * defs['n_channels'] * n_bins).sum())
    if sum(rg_rows) != total_expected:
        raise CacheLayoutError(
            f'cache has {sum(rg_rows)} rows but defs implies {total_expected} '
            f'(sum of n_windows x n_channels x {n_bins}) -- cache and epoch_defs are '
            'out of sync; rebuild rather than guess.'
        )

    mapping, cursor = {}, 0
    for i in range(n_epochs):
        want = int(defs.at[i, 'n_windows']) * int(defs.at[i, 'n_channels']) * n_bins
        got, groups = 0, []
        while got < want:
            if cursor >= len(rg_rows):
                raise CacheLayoutError(
                    f'ran out of row groups while assembling epoch {i}: needed {want} '
                    f'rows, found {got}'
                )
            got += rg_rows[cursor]
            groups.append(cursor)
            cursor += 1
        if got != want:
            raise CacheLayoutError(
                f'epoch {i} needs {want} rows but the row groups assigned to it hold '
                f'{got}; an epoch boundary does not coincide with a row-group boundary, '
                'so a read would straddle two epochs. Refusing to continue.'
            )
        mapping[int(defs.at[i, 'epoch_id'])] = groups
    if cursor != len(rg_rows):
        raise CacheLayoutError(f'{len(rg_rows) - cursor} row groups left unassigned')

    # Ravel-order proof, on the first epoch (all of its row groups, so the full
    # pattern is seen even when it was split).
    n_win = int(defs.at[0, 'n_windows'])
    n_pairs = int(defs.at[0, 'n_channels'])
    head = parquet_file.read_row_groups(mapping[0], columns=['epoch_id', 'window_idx', 'bin'])
    if not (np.asarray(head.column('epoch_id')) == 0).all():
        raise CacheLayoutError('the row groups assigned to epoch 0 do not all hold epoch_id 0')
    if not np.array_equal(np.asarray(head.column('window_idx')),
                          np.repeat(np.arange(n_win), n_pairs * n_bins)):
        raise CacheLayoutError(
            'window_idx is not a C-order ravel of (window, pair, bin); the reshape in '
            'read_epoch() would transpose channels/frequencies into the wrong slots '
            'and still produce a plausible figure. Refusing to continue.'
        )
    if not np.array_equal(np.asarray(head.column('bin')),
                          np.tile(np.arange(n_bins), n_win * n_pairs)):
        raise CacheLayoutError('`bin` is not the innermost ravel axis; see above.')

    n_split = sum(1 for g in mapping.values() if len(g) > 1)
    logger.info('layout verified: %d epochs over %d row groups (%d epochs span >1)',
                n_epochs, len(rg_rows), n_split)
    return mapping


def read_epoch(parquet_file, epoch_row, n_bins, row_groups):
    """One epoch as (n_windows, n_pairs, n_bins) float64.

    Reads ONLY `log_power` over the epoch's row-group range -- the index columns
    are implied by the verified layout, so they are not paid for per epoch.
    """
    n_win = int(epoch_row['n_windows'])
    n_pairs = int(epoch_row['n_channels'])
    table = parquet_file.read_row_groups(row_groups, columns=['log_power'])
    flat = table.column('log_power').to_numpy(zero_copy_only=False)
    if flat.size != n_win * n_pairs * n_bins:
        raise CacheLayoutError(
            f'epoch {epoch_row["epoch_id"]}: {flat.size} values, expected '
            f'{n_win * n_pairs * n_bins}'
        )
    # Upcast AFTER the reshape so the cast is one contiguous pass (P0.6: store
    # narrow, compute wide -- every reduction downstream needs float64).
    return flat.reshape(n_win, n_pairs, n_bins).astype(config.CACHE_ACCUMULATE_DTYPE)


# ---------------------------------------------------------------------------
# Masking
# ---------------------------------------------------------------------------

def load_mask(subject, session, view_config):
    """The pair-keyed mask table for one subject/session, or None if unmasked.

    RAISES when a mask was requested but is absent. That is deliberate and is the
    open TASKS.md item: three existing code paths fall back to an unmasked
    baseline with only a warning, and the failure is invisible in the output --
    keeping artifact windows inflates a baseline std, deflates z, and makes
    everything LOOK cleaner for exactly the subjects whose QC was incomplete. A
    view is the worst place to inherit that, so the caller must pass
    mask_level='none' to mean it.
    """
    if view_config.mask_level == 'none':
        logger.warning('sub-%s ses-%s: NO QC MASK APPLIED (mask_level=none) -- '
                       'artifact windows are included', subject, session)
        return None

    if view_config.mask_level == 'bipolar':
        path = config.bipolar_mask_path(subject, session, view_config.mask_label)
        if not path.exists():
            raise FileNotFoundError(
                f'no bipolar mask at {path}. Build it with '
                '`python -m ieeg_ehr.qc.build_bipolar_mask`, or pass '
                '--mask-level none to deliberately run unmasked.'
            )
        mask = io.read_table(path, on_stale='warn')
        return mask[['run_id', 'channel', 'bin_start', 'excluded']]

    # raw_voltage: monopolar, needs the anode/cathode OR at window time.
    path = config.mask_csv(subject, session, view_config.mask_label)
    if not path.exists():
        raise FileNotFoundError(f'no raw-voltage mask at {path}')
    return mask_projection.load_mask(path)


def epoch_excluded(mask, epoch_row, channels, view_config):
    """(n_windows, n_pairs) bool for one epoch. All-False when unmasked.

    Window times come from epoch_defs' epoch_start_sec/hop_sec, so no NWB is
    touched here.
    """
    n_win = int(epoch_row['n_windows'])
    if mask is None:
        return np.zeros((n_win, len(channels)), dtype=bool)

    run_seconds = (float(epoch_row['epoch_start_sec'])
                   + np.arange(n_win) * float(epoch_row['hop_sec']))
    if view_config.mask_level == 'bipolar':
        # Already pair-keyed: the pair rule was applied when the mask was rolled up.
        return mask_projection.project_pair_mask_to_windows(
            mask, epoch_row['run_id'], channels, run_seconds)
    # Monopolar: OR anode|cathode per pair, here.
    return mask_projection.project_to_pairs(
        mask, epoch_row['run_id'], channels, run_seconds)


def apply_mask(block, excluded, max_excluded_frac):
    """Blank masked windows to NaN and drop under-covered channel-epochs.

    Returns (block, kept_channels_bool, excluded_frac_per_channel).

    NaN rather than deletion so every pair keeps its slot in the (window, pair,
    bin) grid -- dropping rows would desynchronise `channels` from the array's
    second axis, which is exactly the class of error verify_layout() exists to
    prevent. Every reduction downstream is therefore a nan-aware one.

    A channel whose surviving fraction is too low is dropped ENTIRELY for this
    epoch (not averaged over whatever is left), because an epoch mean over a
    handful of surviving windows is not the 5-minute average it claims to be.
    """
    block = block.copy()
    block[excluded, :] = np.nan
    frac = excluded.mean(axis=0)
    kept = frac <= max_excluded_frac
    block[:, ~kept, :] = np.nan
    return block, kept, frac


def iter_epochs(parquet_file, defs, n_bins, mask, channels_by_run, view_config,
                row_group_map, epoch_filter=None):
    """Yield (epoch_row, masked_block, kept, frac) for each epoch.

    `epoch_filter` restricts to a subset (used for the baseline pass, which only
    reads 0-pain epochs) without re-verifying or re-opening anything.
    `row_group_map` comes from verify_layout -- passed in rather than recomputed so
    the layout is proven exactly once per subject.
    """
    for _, epoch_row in defs.sort_values('epoch_id').iterrows():
        if epoch_filter is not None and not epoch_filter(epoch_row):
            continue
        channels = channels_by_run.get(epoch_row['run_id'])
        if channels is None:
            logger.error('epoch %s: no channel metadata for %s, skipping',
                         epoch_row['epoch_id'], epoch_row['run_id'])
            continue
        if len(channels) != int(epoch_row['n_channels']):
            raise CacheLayoutError(
                f'epoch {epoch_row["epoch_id"]} ({epoch_row["run_id"]}): defs says '
                f'{epoch_row["n_channels"]} channels but channel_meta has '
                f'{len(channels)}. The reshape would mislabel every channel.'
            )
        block = read_epoch(parquet_file, epoch_row, n_bins,
                           row_group_map[int(epoch_row['epoch_id'])])
        excluded = epoch_excluded(mask, epoch_row, channels, view_config)
        block, kept, frac = apply_mask(block, excluded, view_config.max_excluded_frac)
        yield epoch_row, block, kept, frac


def line_noise_bins(epoch_minutes=None):
    """Indices of bins flagged contains_line_noise in the unit manifest."""
    unit = config.pain_epoch_unit_dir(epoch_minutes)
    manifest = io.read_manifest(unit)
    flags = manifest.get('contains_line_noise')
    if flags is None:
        return np.array([], dtype=int)
    return np.flatnonzero(np.asarray(flags, dtype=bool))


def bin_edges(epoch_minutes=None):
    """Bin edge frequencies from the unit manifest, as a (low, high) frame."""
    unit = config.pain_epoch_unit_dir(epoch_minutes)
    edges = np.asarray(io.read_manifest(unit)['bin_edges_hz'], dtype=float)
    return pd.DataFrame({'freq_bin_index': np.arange(len(edges) - 1),
                         'bin_low_hz': edges[:-1], 'bin_high_hz': edges[1:]})


def unresolvable_bins(epoch_minutes=None):
    """Bins narrower than the FFT can resolve, which carry DUPLICATED values.

    The frequency axis is 50 log-spaced bins from 1 Hz, but the PSD comes from
    2-second windows, so the underlying resolution is a flat 0.5 Hz. Below about
    4.7 Hz a log bin is narrower than that, and several consecutive bins contain
    no FFT frequency at all: bin 1 spans [1.117, 1.247) Hz and there is simply no
    measurement in there.

    The cache builder gives every bin its NEAREST FFT frequency, so those bins are
    not empty -- they are COPIES of a neighbour. At 5-minute epochs that makes
    bins {1, 2, 4, 5, 7, 10} exact duplicates, verified against the built view:
    every region has 38 distinct values across its 44 non-line-noise bins.

    This matters wherever bins are counted as independent. A map looks like it has
    smooth broadband low-frequency structure when it is one number repeated; a
    multiple-comparison family is padded with copies; and "significant across nine
    consecutive bins from 1-2.7 Hz" is really three measurements.

    Returns the indices to DROP. The bins kept are the ones whose range actually
    contains the frequency they report, so a retained bin never has a nominal
    range that excludes its own measurement.
    """
    unit = config.pain_epoch_unit_dir(epoch_minutes)
    manifest = io.read_manifest(unit)
    window_sec = float(manifest['params']['window_sec'])
    edges = np.asarray(manifest['bin_edges_hz'], dtype=float)

    # Rayleigh resolution of the PSD window: the FFT reports at multiples of this.
    df = 1.0 / window_sec
    fft_freqs = np.arange(0.0, edges[-1] + df, df)

    drop = [i for i in range(len(edges) - 1)
            if not np.any((fft_freqs >= edges[i]) & (fft_freqs < edges[i + 1]))]
    return np.asarray(drop, dtype=int)
