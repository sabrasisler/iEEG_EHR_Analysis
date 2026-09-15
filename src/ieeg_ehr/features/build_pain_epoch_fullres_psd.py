"""Full-resolution per-window PSD for one subject-session's pain epochs.

    python -m ieeg_ehr.features.build_pain_epoch_fullres_psd --subjects 019

Same epochs, same 2s/1s window grid, same bipolar pairs as `psd_epochs` -- but
the frequency axis is the NATIVE FFT grid (0.5 Hz, 1-250 Hz, 499 bins) instead of
50 log-spaced bins, so no frequency reduction is baked into storage.

WHY THIS EXISTS. The 50-log-bin reduction is irreversible and wrong at BOTH ends
of the spectrum (DECISIONS 2026-09-15):

  - HIGH FREQUENCY: a log bin is far WIDER than the line noise it has to exclude.
    Bins 36+37 span 53.3-66.4 Hz, so dropping 60 Hz costs 13.2 Hz of real
    spectrum where the contamination is ~4 Hz. Bin 49 alone spans 26 Hz. Six of
    50 bins are flagged, i.e. ~12% of the axis is discarded to remove ~1.6%.
  - LOW FREQUENCY: below ~4.7 Hz a log bin is NARROWER than the 2s window's own
    0.5 Hz resolution, so `bipolar_reref._band_average_linear`'s nearest-frequency
    fallback fills it with a COPY OF A NEIGHBOUR. Bins {1,2,4,5,7,10} are exact
    duplicates: the 44 non-line-noise bins carry only 38 distinct values
    (`views.cache_reader.unresolvable_bins`).
  - And ~2 samples per oscillatory peak is unfittable, which is what blocked
    specparam (BG.3) and left the 1/f number a broadband tilt rather than an
    aperiodic exponent (`config/psd_params.py` SLOPE_* block).

Storing the native grid moves EVERY frequency scheme -- log_bins_50,
canonical_bands, paper_bands_6, any notch half-width -- into the view layer as a
free recompute. Nothing here is baked in that a view could have decided.

SOURCE IS THE RAW SIGNAL, NOT `preprocessed/bipolar_fft`. The full-resolution PSD
is computed today inside `bipolar_reref._welch_one_channel` and immediately
discarded by the binning step; it exists nowhere on disk and cannot be recovered
from the stored log bins. So this re-reads raw.

WHY EPOCHS ONLY, not a continuous family. Re-running the continuous extraction at
499 bins would be ~10x of an 835 GB tree and 26 x 24h array tasks. V1 raw NWB is
chunked `(10000, n_channels)` -- along TIME -- so a 5-min full-width read is the
FAST direction (measured 1.04 s). Same considered departure from
architecture.md's layer model as `features/build_pain_epoch_bandpass.py`, and
recorded in `config.fullres_epoch_unit_dir`'s docstring.

TWO NUMBERS THAT MUST BE RIGHT, both asserted below rather than trusted:

  1. THE READ IS (n_windows + 1) SECONDS, NOT n_windows. PSD row `i` of a run
     covers raw samples [i*hop, i*hop + nperseg). Windows row_start..row_stop-1
     therefore need samples [row_start*hop, (row_stop+1)*hop) -- 301 s for a
     300-window 5-min epoch. `build_pain_epoch_bandpass.py` reads exactly
     epoch_minutes*60 = 300 s, which is right for its RMS purpose but yields 299
     windows here, silently one fewer than the existing cache.
  2. THE GRID IS 0.5 Hz AT EVERY SAMPLING RATE, because df = sfreq/nperseg =
     1/PSD_WINDOW_SEC and nperseg is resolved per-run from that run's own sfreq.
     500, 1000 and 2000 Hz subjects land on an IDENTICAL frequency axis with no
     resampling, which is the only reason a fixed axis is possible cohort-wide.

DELIBERATELY MATCHES PRODUCTION FLOAT32. The spectrogram runs on the float32
bipolar trace and is called PER CHANNEL with the same 1-D signature
`_welch_one_channel` uses, not vectorized over an axis. That costs ~2 s per epoch
and buys a genuinely BIT-EXACT correctness gate: `fullres_psd_audit.py` re-bins
this output through the real `_band_average_linear` and must reproduce the
on-disk 50-bin cache exactly. A batched FFT or a float64 input would make that
comparison approximate, and an approximate gate cannot distinguish a 1-ulp
difference from a one-window time misalignment.

READS V1 (`iEEG_EHR/iEEG_NWB`), DELIBERATELY. V2 has 3 subjects, no `ehr/` folder
and therefore no pain scores, and every derivative this depends on -- the epoch
definitions, the QC masks, the channel metadata -- was built from V1. Beyond
availability, V2's `(120 s, 1 channel)` chunking makes the short full-width read
this does the SLOW direction (docs/dataset_v2.md §4.1).
"""

import argparse
import logging
import sys
import time

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from scipy import signal

from ieeg_ehr import config, io
from ieeg_ehr.io import nwb as nwb_io
from ieeg_ehr.preprocessing import bipolar_reref
from ieeg_ehr.views import cache_reader

logger = logging.getLogger(__name__)

SCRIPT = 'ieeg_ehr/features/build_pain_epoch_fullres_psd.py'

#: The index columns. The frequency axis is COLUMNS f000..f{n-1}, not a `bin`
#: column: long format would be ~75 billion rows cohort-wide, and would also
#: overflow psd_epochs' `bin: int8` (a 127-value ceiling).
INDEX_COLUMNS = ['epoch_id', 'window_idx', 'channel']


def freq_column_names(n_freqs):
    """f000, f001, ... -- POSITIONAL, with the Hz values in the manifest.

    Positional because a float in a column name is a formatting decision that
    would have to round-trip exactly forever ('f60.0' vs 'f60' vs 'f60.00'), and
    because the manifest already owns the frequency axis for this unit the same
    way `bin_edges_hz` owns it for psd_epochs. Readers zip these against
    `manifest['extra']['freqs_hz']`; nothing parses the name.
    """
    width = max(3, len(str(n_freqs - 1)))
    return [f'f{i:0{width}d}' for i in range(n_freqs)]


def cache_schema(n_freqs):
    return pa.schema(
        [('epoch_id', pa.int32()),
         ('window_idx', pa.int16()),
         ('channel', pa.dictionary(pa.int16(), pa.string()))]
        + [(name, pa.float32()) for name in freq_column_names(n_freqs)])


#: Parquet compression + encoding. MEASURED 2026-09-15 on real sub-019 data, as a
#: fraction of the raw float32 payload, with the cohort projection at 156,905,430
#: rows x 499 frequencies:
#:
#:   snappy + dictionary (pyarrow DEFAULT)  1.49x   465 GB
#:   snappy, no dictionary                  1.00x   313 GB
#:   zstd, no dictionary                    0.82x   256 GB
#:   snappy + BYTE_STREAM_SPLIT             0.80x   249 GB
#:   zstd   + BYTE_STREAM_SPLIT             0.72x   226 GB   <- chosen
#:
#: THE DEFAULT IS ACTIVELY BAD HERE. pyarrow tries dictionary encoding on every
#: column, and a float32 column of ~10k near-unique log-power values produces a
#: dictionary about as large as the data PLUS the indices -- so it INFLATES by
#: ~49%. That is the single biggest storage decision in this unit and it is a
#: writer flag, not anything about the science.
#:
#: BYTE_STREAM_SPLIT transposes the four bytes of each float into separate
#: streams, which groups the (highly repetitive) sign/exponent bytes together and
#: gives the compressor something to work with. It is LOSSLESS -- it only reorders
#: bytes -- and that was verified bitwise, including NaN and -inf, rather than
#: assumed: P0.6 validated float32 round-trip through DEFAULT Parquet, so a new
#: encoding gets the same check (`tests/test_fullres_psd.py`).
#:
#: zstd over snappy costs ~2x decode (0.138 s vs 0.068 s per epoch; 6.9 vs 3.4
#: min for a full-cohort pass) and saves 23 GB now, ~70 GB at the ~250-subject
#: cohort. Taken because sub-second-per-epoch reads are not the bottleneck for
#: anything here, and Oak was at 64% on 2026-09-15 having grown 14 TB in 11 days.
#:
#: Dictionary encoding is kept for `channel` ALONE, where it is the right call:
#: ~200 repeated short strings per epoch. The two integer index columns take
#: pyarrow's default (RLE), which is already near-optimal for a repeat/tile pattern.
CACHE_COMPRESSION = 'zstd'
CACHE_FREQ_ENCODING = 'BYTE_STREAM_SPLIT'
CACHE_DICTIONARY_COLUMNS = ['channel']


def parquet_options(freq_columns):
    """Writer kwargs for the cache. See CACHE_COMPRESSION for the measurements."""
    return dict(
        compression=CACHE_COMPRESSION,
        use_dictionary=list(CACHE_DICTIONARY_COLUMNS),
        column_encoding={c: CACHE_FREQ_ENCODING for c in freq_columns},
    )


def window_geometry(sfreq, window_sec=None, overlap_frac=None):
    """(nperseg, noverlap, hop_samples) for one run, resolved from ITS sfreq.

    Asserts `nperseg - noverlap == hop_samples`, the one latent fragility in the
    production windowing: the stored NWB `rate` comes from the float expression
    `window_sec * (1 - overlap_frac)` while scipy's real hop is the integer
    `nperseg - noverlap`. They agree for every sfreq present in this dataset
    because nperseg is even, but nothing structurally enforces it, and a
    half-sample-per-window drift would misalign every epoch in a run.
    """
    window_sec = config.PSD_WINDOW_SEC if window_sec is None else window_sec
    overlap_frac = config.PSD_OVERLAP_FRAC if overlap_frac is None else overlap_frac

    nperseg = max(1, int(round(window_sec * sfreq)))
    noverlap = int(nperseg * overlap_frac)
    hop_samples = int(round(window_sec * (1.0 - overlap_frac) * sfreq))
    if nperseg - noverlap != hop_samples:
        raise ValueError(
            f'hop mismatch at sfreq={sfreq}: scipy steps {nperseg - noverlap} '
            f'samples but the stored PSD rate implies {hop_samples}. Every epoch '
            'in this run would be misaligned.')
    return nperseg, noverlap, hop_samples


def freq_slice(freqs, f_min=None, f_max=None, df=None):
    """A contiguous slice of `freqs` covering [f_min, f_max] INCLUSIVE.

    Computed by index rather than by comparing floats: `rfftfreq` values are not
    exactly representable, so `freqs <= 250.0` can drop the top bin depending on
    the last bit. The slice is then verified against the analytic grid, so a
    disagreement raises instead of silently shifting the axis by one bin.
    """
    f_min = config.PSD_FULLRES_FREQ_MIN_HZ if f_min is None else f_min
    f_max = config.PSD_FULLRES_FREQ_MAX_HZ if f_max is None else f_max
    df = config.PSD_FULLRES_DF_HZ if df is None else df

    i0, i1 = int(round(f_min / df)), int(round(f_max / df))
    if i1 >= len(freqs):
        raise ValueError(
            f'f_max={f_max} Hz needs index {i1} of a {len(freqs)}-long grid; '
            f'Nyquist here is {freqs[-1]} Hz.')
    sl = slice(i0, i1 + 1)
    expected = np.arange(i0, i1 + 1) * df
    if not np.allclose(freqs[sl], expected, rtol=0, atol=1e-9):
        raise ValueError(f'frequency grid is not {df} Hz-spaced as assumed: '
                         f'got {freqs[sl][:3]}..., expected {expected[:3]}...')
    return sl, expected


def analytic_freqs(sfreq, nperseg):
    """The frequency grid `signal.spectrogram` will return, without computing one.

    `rfftfreq(nperseg, 1/sfreq)` is exactly what scipy uses. Knowing it up front
    is what lets `epoch_spectrogram` allocate only the frequencies it will keep --
    at 2000 Hz the full grid is 2001 bins and we store 499, so allocating the
    full one for every channel would cost 4x the memory for data that is
    immediately discarded. The values are verified against the real grid inside
    `epoch_spectrogram`, so this is an optimisation and not a second source of
    truth.
    """
    return np.fft.rfftfreq(nperseg, d=1.0 / sfreq)


def epoch_spectrogram(bipolar_v, sfreq, nperseg, noverlap, freq_sl=None):
    """(freqs, Sxx) with Sxx (n_freq, n_win, n_pairs) -- float32, like production.

    One 1-D `signal.spectrogram` call per channel, identical in signature and
    dtype to `bipolar_reref._welch_one_channel` -- see the module docstring on why
    this is not vectorized over an axis.

    `freq_sl` is a slice of frequencies to KEEP, applied per channel so the full
    grid is never held for all channels at once. `None` returns the full grid,
    which is what `fullres_psd_audit` needs to feed the production binning
    function. `freqs` is always the full grid either way, so a caller can still
    interpret the slice.

    `detrend='constant'` and `window=PSD_WINDOW_FN` are passed EXPLICITLY though
    both match what production gets by default. Per-segment mean removal has been
    in effect for every stored spectrum in this project via scipy's default and is
    recorded nowhere; a future scipy default change would silently alter the
    pipeline, and a reader reconstructing the method from a sidecar cannot
    currently recover it.
    """
    n_pairs = bipolar_v.shape[1]
    keep = freq_sl if freq_sl is not None else slice(None)
    freqs = None
    out = None
    for ch in range(n_pairs):
        freqs, _times, Sxx = signal.spectrogram(
            bipolar_v[:, ch], fs=sfreq, nperseg=nperseg, noverlap=noverlap,
            window=config.PSD_WINDOW_FN, detrend='constant',
            scaling='density', mode='psd')
        if out is None:
            n_keep = len(freqs[keep])
            out = np.empty((n_keep, Sxx.shape[1], n_pairs), dtype=Sxx.dtype)
        out[:, :, ch] = Sxx[keep]
    return freqs, out


def _run_paths(subject, session):
    """{run_id (bare, no 'run-' prefix): raw nwb path} from the file registry."""
    reg = pd.read_csv(config.FILE_REGISTRY_CSV)
    sel = reg[(reg.sub_id == f'sub-{subject}') & (reg.ses_id == f'ses-{session}')]
    return {str(r).replace('run-', ''): p
            for r, p in zip(sel.run_id, sel.raw_file_path)}


def _expected_pairs(channel_meta, run_id):
    """The pair names psd_epochs recorded for this run, in cache column order.

    Returns None when there is no channel_meta for the run, which makes the
    cross-check SKIP rather than pass -- an absent table must not read as
    agreement. `views.channel_meta.COLUMNS` fixes the schema as
    (run_id, pair_index, channel, ...), and pair_index IS the cache's channel-axis
    position, so sorting by it is what makes the comparison meaningful.
    """
    if channel_meta is None:
        return None
    sel = channel_meta[channel_meta['run_id'] == f'run-{run_id}']
    if sel.empty:
        return None
    return list(sel.sort_values('pair_index')['channel'])


def build_subject_session(subject, session, epoch_minutes=None, overwrite=False,
                          f_min=None, f_max=None):
    """Write one subject-session's full-resolution epoch cache.

    Returns (cache_path, freqs_hz, stats) or None if nothing was written.
    """
    t0 = time.time()
    epoch_minutes = config.EPOCH_MINUTES_BEFORE if epoch_minutes is None else epoch_minutes

    cache_path = config.fullres_epoch_cache_path(subject, session, epoch_minutes)
    if cache_path.exists() and not overwrite:
        logger.info('sub-%s ses-%s: exists, skipping (--overwrite to rebuild)',
                    subject, session)
        return None

    # The epoch definitions are REUSED verbatim, not recomputed. load_defs refuses
    # defs lacking epoch_start_sec/hop_sec, because guessing the hop would
    # silently mis-locate every window.
    #
    # A MISSING defs file is skipped, not fatal: the subject list has 87 candidates
    # against 83 built units, so 4 array tasks would otherwise die on subjects that
    # simply have no 50-bin unit. Warned loudly rather than passed over silently,
    # because the cause is upstream (no psd_epochs build) and worth knowing.
    defs_src = config.pain_epoch_defs_path(subject, session, epoch_minutes)
    if not defs_src.exists():
        logger.warning('sub-%s ses-%s: no epoch_defs at %s -- skipping. Build the '
                       '50-bin unit first (features.build_pain_epoch_power).',
                       subject, session, defs_src.name)
        return None
    defs = cache_reader.load_defs(subject, session, epoch_minutes)
    defs = defs.sort_values('epoch_id').reset_index(drop=True)
    if defs.empty:
        logger.warning('sub-%s ses-%s: epoch_defs is empty -- skipping',
                       subject, session)
        return None

    meta_path = config.pain_epoch_channel_meta_path(subject, session, epoch_minutes)
    if meta_path.exists():
        channel_meta = io.read_table(meta_path, on_stale='ignore')
    else:
        channel_meta = None
        logger.warning('sub-%s ses-%s: no channel_meta at %s; the pair-order '
                       'cross-check will be SKIPPED for every run. Build it with '
                       '`python -m ieeg_ehr.views.channel_meta --subjects %s`',
                       subject, session, meta_path.name, subject)

    runs = _run_paths(subject, session)

    by_run = {}
    for _, ep in defs.iterrows():
        by_run.setdefault(str(ep['run_id']).replace('run-', ''), []).append(ep)

    stats = {'n_epochs': 0, 'n_short_read': 0, 'n_missing_run': 0,
             'n_nonfinite': 0, 'n_rows': 0, 'missing_runs': set(),
             'sfreqs': set(), 'n_pairs': 0}
    freqs_hz = None
    writer = None
    schema = None
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        for run_id, run_epochs in by_run.items():
            raw_path = runs.get(run_id)
            if raw_path is None:
                stats['n_missing_run'] += len(run_epochs)
                stats['missing_runs'].add(run_id)
                continue

            pairs = elec_indices = channels = None
            geom = None

            for ep in run_epochs:
                n_windows = int(ep['n_windows'])
                hop_sec = float(ep['hop_sec'])
                start_sec = float(ep['epoch_start_sec'])

                # (n_windows + 1) * hop_sec -- see the module docstring. Reading
                # n_windows * hop_sec yields n_windows - 1 complete windows.
                dur_sec = (n_windows + 1) * hop_sec
                data_v, contact_names, sfreq, elec_df, _ = (
                    nwb_io.load_window_with_electrodes(
                        raw_path, start_sec, dur_sec, series_name=None))

                if geom is None:
                    geom = window_geometry(sfreq)
                nperseg, noverlap, hop_samples = geom
                want = (n_windows + 1) * hop_samples
                if data_v.shape[0] != want:
                    # The loader clips at the run end rather than raising. A short
                    # read means fewer windows than the 50-bin cache holds for the
                    # same epoch, so the two would not be comparable -- skip and
                    # count rather than write a differently-shaped epoch.
                    logger.warning(
                        'sub-%s ses-%s epoch %s: read %d samples, wanted %d '
                        '(%.1f s short) -- skipping', subject, session,
                        ep['epoch_id'], data_v.shape[0], want,
                        (want - data_v.shape[0]) / sfreq)
                    stats['n_short_read'] += 1
                    continue

                if pairs is None:
                    # Derived once per run: create_bipolar_pairs prints a warning
                    # per unparseable location, and this loop runs thousands of
                    # times per cohort.
                    pairs, _ = bipolar_reref.create_bipolar_pairs(elec_df)
                    if not pairs:
                        raise RuntimeError(
                            f'sub-{subject} run-{run_id}: no bipolar pairs could '
                            'be derived; check the electrode naming')
                    elec_indices = elec_df.index.to_numpy()
                    channels = [p['location'] for p in pairs]

                    expected = _expected_pairs(channel_meta, run_id)
                    if expected is None:
                        logger.warning('run-%s: no channel_meta rows, pair-order '
                                       'cross-check skipped', run_id)
                    elif list(expected) != channels:
                        raise RuntimeError(
                            f'sub-{subject} run-{run_id}: bipolar pairs disagree '
                            f'with psd_epochs channel_meta. Got {len(channels)} '
                            f'pairs starting {channels[:3]}, expected '
                            f'{len(expected)} starting {list(expected)[:3]}. '
                            'Proceeding would mislabel every channel; some runs '
                            'carry pairs_diverged_from_session_first_run.')
                    stats['n_pairs'] = len(channels)

                # Resolve the frequency slice BEFORE the spectrogram so only the
                # kept frequencies are ever allocated (at 2000 Hz the full grid is
                # 2001 bins against the 499 stored).
                sl, expected_freqs = freq_slice(
                    analytic_freqs(sfreq, nperseg), f_min, f_max)
                if freqs_hz is None:
                    freqs_hz = expected_freqs
                    schema = cache_schema(len(freqs_hz))
                    writer = pq.ParquetWriter(cache_path, schema, **parquet_options(
                        freq_column_names(len(freqs_hz))))
                elif len(expected_freqs) != len(freqs_hz):
                    raise RuntimeError(
                        f'sub-{subject} run-{run_id}: frequency axis changed '
                        f'mid-subject ({len(freqs_hz)} -> {len(expected_freqs)}). '
                        'Runs of one session must share a grid.')

                bipolar_v = bipolar_reref.rereference(data_v, elec_indices, pairs)
                del data_v

                freqs_full, cube = epoch_spectrogram(bipolar_v, sfreq, nperseg,
                                                     noverlap, freq_sl=sl)
                del bipolar_v
                if not np.allclose(freqs_full[sl], expected_freqs, rtol=0, atol=1e-9):
                    raise RuntimeError(
                        f'sub-{subject} run-{run_id}: the real frequency grid does '
                        'not match the analytic one used to size the slice.')
                if cube.shape[1] != n_windows:
                    raise RuntimeError(
                        f'sub-{subject} epoch {ep["epoch_id"]}: spectrogram gave '
                        f'{cube.shape[1]} windows, epoch_defs says {n_windows}. '
                        'The read length or the window geometry is wrong.')

                with np.errstate(divide='ignore'):
                    # IN PLACE: log10 is elementwise, so float32 in -> float32 out
                    # loses nothing an accumulator would, and there is no reason to
                    # hold the linear and log copies at once. -inf for an
                    # exactly-zero bin matches what the 50-bin path stores.
                    np.log10(cube, out=cube)

                # (n_freq, n_win, n_pairs) -> (n_freq, n_win*n_pairs). The last two
                # axes are contiguous, so this is a free VIEW, and each frequency
                # row is contiguous for pyarrow. Row r is
                # (win = r // n_pairs, pair = r % n_pairs).
                n_pairs_here = len(channels)
                cols = cube.reshape(cube.shape[0], n_windows * n_pairs_here)
                stats['n_nonfinite'] += int((~np.isfinite(cols)).sum())
                n_rows = cols.shape[1]

                arrays = [
                    pa.array(np.full(n_rows, int(ep['epoch_id']), dtype=np.int32)),
                    pa.array(np.repeat(np.arange(n_windows, dtype=np.int16),
                                       n_pairs_here)),
                    pa.array(np.tile(np.asarray(channels), n_windows)
                             ).dictionary_encode(),
                ] + [pa.array(cols[j]) for j in range(cols.shape[0])]
                writer.write_table(pa.Table.from_arrays(arrays, schema=schema))
                del arrays, cols, cube

                # pyarrow's pool RETAINS freed buffers by default, so without this
                # RSS ratchets up across epochs and a 199-pair subject sits at the
                # cgroup ceiling for the whole run (measured: sub-256 at 16.00 GB
                # of 16.00 GB, 2026-09-15).
                pa.default_memory_pool().release_unused()

                stats['n_epochs'] += 1
                stats['n_rows'] += n_rows
                stats['sfreqs'].add(float(sfreq))
    finally:
        if writer is not None:
            writer.close()

    if writer is None:
        logger.warning('sub-%s ses-%s: no epochs written (%d missing run, '
                       '%d short read)', subject, session,
                       stats['n_missing_run'], stats['n_short_read'])
        return None

    params = _unit_params(epoch_minutes, freqs_hz)
    io.write_sidecar(
        cache_path, kind='table', script=SCRIPT, params=params,
        parents=[str(runs[r]) for r in by_run if r in runs],
        subjects=[f'sub-{subject}'],
        extra={'n_rows': int(stats['n_rows']), 'n_epochs': stats['n_epochs'],
               'n_epochs_in_defs': int(len(defs)),
               'n_pairs': stats['n_pairs'],
               'n_short_read': stats['n_short_read'],
               'n_missing_run': stats['n_missing_run'],
               'missing_runs': sorted(stats['missing_runs']),
               'n_nonfinite_values': int(stats['n_nonfinite']),
               'sfreq_hz': sorted(stats['sfreqs']),
               'elapsed_sec': round(time.time() - t0, 1)})

    # epoch_defs and channel_meta are COPIED into this unit, not referenced, so
    # it is readable on its own; the psd_epochs manifest rides along as a parent
    # so the two stay provably linked.
    io.write_table(defs, config.fullres_epoch_defs_path(subject, session, epoch_minutes),
                   kind='table', script=SCRIPT, params=params,
                   parents=[str(config.pain_epoch_defs_path(subject, session, epoch_minutes))],
                   subjects=[f'sub-{subject}'],
                   extra={'n_epochs_written': stats['n_epochs'],
                          'note': 'The FULL psd_epochs index, copied verbatim. '
                                  'n_epochs_written may be smaller: an epoch whose '
                                  'raw read came up short is listed here but absent '
                                  'from the cache. Read the cache for what exists.'})
    if channel_meta is not None:
        io.write_table(channel_meta,
                       config.fullres_epoch_channel_meta_path(subject, session, epoch_minutes),
                       kind='table', script=SCRIPT, params=params,
                       parents=[str(config.pain_epoch_channel_meta_path(
                           subject, session, epoch_minutes))],
                       subjects=[f'sub-{subject}'])

    logger.info('sub-%s ses-%s: %d/%d epochs, %d pairs, %d rows -> %s (%.2f GB, %.0fs)%s',
                subject, session, stats['n_epochs'], len(defs), stats['n_pairs'],
                stats['n_rows'], cache_path.name,
                cache_path.stat().st_size / 1e9, time.time() - t0,
                f"  [{stats['n_nonfinite']} non-finite]" if stats['n_nonfinite'] else '')
    return cache_path, freqs_hz, stats


def _unit_params(epoch_minutes, freqs_hz):
    """The config that CHANGES THE OUTPUT, and nothing else -- this is hashed into
    config_hash, which is what staleness compares."""
    return {
        'epoch_minutes_before': epoch_minutes,
        'anchor': 'pain_score_time',
        'window_sec': config.PSD_WINDOW_SEC,
        'overlap_frac': config.PSD_OVERLAP_FRAC,
        'window_fn': config.PSD_WINDOW_FN,
        'detrend': 'constant',
        'scaling': 'density',
        'mode': 'psd',
        'df_hz': float(config.PSD_FULLRES_DF_HZ),
        'freq_min_hz': float(freqs_hz[0]),
        'freq_max_hz': float(freqs_hz[-1]),
        'n_freqs': int(len(freqs_hz)),
        'domain': 'log',
        'dtype': str(config.CACHE_FLOAT_DTYPE),
        'source': 'raw',
        'rereferencing_method': 'bipolar_adjacent',
        'index_columns': INDEX_COLUMNS,
        # Recorded because it is a large, deliberate, MEASURED choice (2.1x the
        # cohort footprint against pyarrow's default) -- not because it changes a
        # value. Both encodings are bitwise lossless, so this does NOT belong in
        # the hashed set on correctness grounds; it is here so a reader can tell
        # why two same-shaped units differ in size.
        'compression': CACHE_COMPRESSION,
        'freq_column_encoding': CACHE_FREQ_ENCODING,
        'masked': False, 'averaged': False, 'normalized': False,
        'line_noise_applied': False,
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--subjects', nargs='+', required=True,
                    help='Subject IDs, e.g. 019 256 (or sub-019).')
    ap.add_argument('--session', default='01',
                    help="Session, or 'all' for every session in the registry.")
    ap.add_argument('--epoch-minutes', type=float, default=None)
    ap.add_argument('--f-min', type=float, default=None,
                    help=f'default: config {config.PSD_FULLRES_FREQ_MIN_HZ}')
    ap.add_argument('--f-max', type=float, default=None,
                    help=f'default: config {config.PSD_FULLRES_FREQ_MAX_HZ}')
    ap.add_argument('--overwrite', action='store_true')
    ap.add_argument('--write-manifest', action='store_true',
                    help='Write the base unit manifest.json. ONE task in an array '
                         'should set this; it describes the unit, not the subject.')
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)s %(message)s')
    io.warn_if_dirty()

    epoch_minutes = (config.EPOCH_MINUTES_BEFORE if args.epoch_minutes is None
                     else args.epoch_minutes)
    freqs_hz = None
    n_ok = 0
    for s in args.subjects:
        subject = s.replace('sub-', '')
        sessions = ([args.session] if args.session != 'all'
                    else _sessions_for(subject))
        for session in sessions:
            result = build_subject_session(subject, session, epoch_minutes,
                                           overwrite=args.overwrite,
                                           f_min=args.f_min, f_max=args.f_max)
            if result is not None:
                n_ok += 1
                if freqs_hz is None:
                    freqs_hz = result[1]

    if args.write_manifest:
        unit = config.fullres_epoch_unit_dir(epoch_minutes)
        if freqs_hz is None:
            # Nothing was built this run -- normally because the subject already
            # existed and was skipped. That is fine IF the unit is already
            # described; it is only an error if the unit has no manifest, since
            # the frequency axis would then be unrecoverable (the cache columns
            # are positional). Task 0 of an array hits this the moment its own
            # subject is re-run, which is not a reason to fail the task.
            existing = io.read_manifest(unit)
            if existing and existing.get('freqs_hz'):
                logger.info('nothing built this run; unit manifest already present '
                            'with %d frequencies -> %s',
                            len(existing['freqs_hz']), unit / 'manifest.json')
                return 0
            logger.error('--write-manifest given, nothing was built, and there is '
                         'no existing manifest at %s. The frequency axis is '
                         'therefore unknown, and a manifest describing a unit '
                         'nobody measured is worse than none. Re-run with '
                         '--overwrite on a subject that has epoch_defs.', unit)
            return 1
        io.write_manifest(
            unit, script=SCRIPT, params=_unit_params(epoch_minutes, freqs_hz),
            extra={'freqs_hz': [float(x) for x in freqs_hz],
                   'note': 'Cache is RAW slices at the NATIVE FFT resolution: no '
                           'QC mask, no line-noise notch, no averaging, no '
                           'normalization, no frequency binning. freqs_hz is the '
                           'SINGLE SOURCE OF TRUTH for the frequency axis -- '
                           'derive n_freqs from its length, never from a config '
                           'constant. Line-noise status is a view-time decision '
                           '(config.PSD_NOTCH_HALF_WIDTH_HZ), not a stored flag. '
                           'AXIS 2 whole_session baselines are UNAVAILABLE here: '
                           'this unit holds only epoch windows.'})
        logger.info('wrote unit manifest -> %s', unit / 'manifest.json')

    if n_ok:
        io.log_analysis(
            f'full-res pain epoch PSD cache, {n_ok} subject-session(s)',
            config.fullres_epoch_unit_dir(epoch_minutes))
    return 0


def _sessions_for(subject):
    reg = pd.read_csv(config.FILE_REGISTRY_CSV)
    ses = reg[reg.sub_id == f'sub-{subject}'].ses_id.unique()
    return [str(s).replace('ses-', '') for s in ses]


if __name__ == '__main__':
    sys.exit(main())
