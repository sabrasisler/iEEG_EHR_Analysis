"""
P0.6 dtype audit — is float32 a safe storage dtype for the per-window cache?

Answers the three questions PLANNING.md P0.6 asks, as four legs:

  A. INVENTORY   What dtype is actually on disk today, in both the raw NWB
                 (ElectricalSeries) and the stored PSD NWB (DecompositionSeries)?
                 Plus value ranges, to confirm float32's exponent range has
                 headroom in the log domain AND after a view exponentiates back
                 to linear.
  B. ROUND-TRIP  Does a float32 array survive write->read bit-exactly, in HDF5
                 (what bipolar_fft uses today) and Parquet (what the P1.1 cache
                 will use)?
  C. ACCUMULATE  Averaging a real 5-minute epoch of stored float32 values: how
                 far does a float32 accumulator drift from a float64 one? This
                 is a question about the VIEW layer, not about storage.
  D. RECOMPUTE   The headline test. Recompute one run's PSD in float64 all the
                 way through, and compare its epoch averages against the
                 production float32 path's. This is the actual cost of the
                 float32 storage decision, measured rather than assumed.

WHAT IS AND ISN'T UNDER TEST. Leg D holds the bipolar time series fixed at
float32 and varies ONLY the dtype of the log-power output. That is deliberate:
the float32 voltage cast in `io/nwb.py` is a separate, earlier decision, and
leg A reports the raw dtype so you can see whether that cast is itself lossless
(it is, for integer raw data of <=24 bits). P0.6 is about the CACHE dtype, so
the cache dtype is the only thing that moves.

WHY REPORT SIG FIGS *AND* LINEAR-DOMAIN ERROR. The stored quantity is
log10(V^2/Hz), so a relative error on it is not the physically interesting
number. An absolute error of d in log10-power is a factor 10**d in power, so
`linear_frac_error = 10**d - 1` is what a downstream effect size would actually
see. Both are reported; the second is the one to quote.

Outputs (Oak, never the repo) land under
  qc/feature_level/validation/dtype_audit/<label>_<timestamp>/
as dtype_audit.json + summary.txt + epoch_average_errors.csv, with a provenance
sidecar. Feature-level QC is the right home: this is a choice-independent fact
about the feature layer, inherited by every view.

Run via Slurm, never the login node:
    sbatch sbatch/dtype_audit.sbatch
or interactively:
    module load python/3.12
    source $GROUP_HOME/venvs/ieeg_ehr_analysis/bin/activate
    python -m ieeg_ehr.features.dtype_audit --subjects 085 071 --label smoke
"""

import argparse
import json
import logging
import os
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from scipy import signal

from ieeg_ehr import config
from ieeg_ehr.io import nwb as io_utils
from ieeg_ehr.io.analysis_log import log_analysis
from ieeg_ehr.io.provenance import git_provenance, run_timestamp, warn_if_dirty
from ieeg_ehr.preprocessing import bipolar_reref

# Production epoch selection, reused rather than reimplemented so leg C measures
# the epochs the pipeline actually builds. Private names, same package.
from ieeg_ehr.features.build_pain_epoch_power import (
    _excluded_mask,
    _find_matching_run,
    _load_run_psd,
    load_mask,
    load_pain_scores,
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

# HDF5 paths inside a bipolar_psd.nwb. Read with h5py rather than pynwb because
# pynwb hands back a materialised numpy array and hides exactly what this leg is
# asking about: the dtype/chunking/compression as written on disk.
PSD_DATA_H5 = '/processing/ecephys/psd_log_bins/data'
BROADBAND_DATA_H5 = '/processing/ecephys/broadband_log_power/data'
RAW_SERIES_H5 = '/acquisition/ElectricalSeries_sEEG/data'

# float32 has a 24-bit mantissa, so the largest relative spacing between
# representable neighbours is 2**-24. Every error below is quoted against this.
FLOAT32_EPS = float(np.finfo(np.float32).eps)      # 2**-23, the full ulp
FLOAT32_HALF_ULP = FLOAT32_EPS / 2.0               # 2**-24, worst-case rounding

# The bar P0.6 sets: epoch averages must agree to at least this many significant
# figures. float32 itself only carries ~7.2, so 6 leaves a real margin.
SIG_FIG_TARGET = 6.0


# ============================================================================
# shared helpers
# ============================================================================

def _sig_figs(rel_error):
    """Significant decimal figures implied by a relative error.

    inf for an exact match (rel_error == 0), which is the expected result for
    the bit-exactness legs.
    """
    rel_error = float(rel_error)
    if rel_error <= 0.0:
        return float('inf')
    return float(-np.log10(rel_error))


def _error_stats(ref64, test64, *, label):
    """Compare two float64 arrays elementwise.

    ref64 is the reference (the more precise arm). Both relative error on the
    stored log-power and the fractional error it implies in LINEAR power are
    reported -- see the module docstring on why the second is the one to quote.
    Only finite pairs are compared; log10(0) = -inf is a real occurrence in this
    data (dead channel at one bin) and production drops those channel-epochs
    outright, so they carry no information about dtype.
    """
    ref64 = np.asarray(ref64, dtype=np.float64).ravel()
    test64 = np.asarray(test64, dtype=np.float64).ravel()
    finite = np.isfinite(ref64) & np.isfinite(test64)
    n_compared = int(finite.sum())
    if n_compared == 0:
        return {'label': label, 'n_compared': 0, 'note': 'no finite pairs to compare'}

    ref64, test64 = ref64[finite], test64[finite]
    abs_err = np.abs(test64 - ref64)
    # Guarded denominator: log-power sits around -10 to -14 here so |ref| is
    # nowhere near zero, but a hard-coded guard beats a NaN in a decision record.
    rel_err = abs_err / np.maximum(np.abs(ref64), np.finfo(np.float64).tiny)
    max_rel = float(rel_err.max())

    return {
        'label': label,
        'n_compared': n_compared,
        'bit_exact': bool(np.array_equal(test64, ref64)),
        'max_abs_error_log10power': float(abs_err.max()),
        'median_abs_error_log10power': float(np.median(abs_err)),
        'max_rel_error': max_rel,
        'median_rel_error': float(np.median(rel_err)),
        'max_rel_error_in_float32_half_ulps': max_rel / FLOAT32_HALF_ULP,
        # 10**d - 1: what the worst-case log-domain error costs in linear power.
        'max_linear_frac_error': float(np.expm1(abs_err.max() * np.log(10.0))),
        'min_sig_figs': _sig_figs(max_rel),
        'median_sig_figs': _sig_figs(float(np.median(rel_err))),
        'meets_sig_fig_target': _sig_figs(max_rel) >= SIG_FIG_TARGET,
        'ref_min': float(ref64.min()),
        'ref_max': float(ref64.max()),
    }


def _tmp_dir():
    """Node-local SSD for the round-trip scratch files, per CLAUDE.md.

    $L_SCRATCH is wiped at job end, which is exactly right for throwaway
    round-trip probes -- nothing here is a derivative worth keeping. Falls back
    to $SCRATCH, then to the job's tmp, so an interactive run still works.
    """
    for var in ('L_SCRATCH', 'SCRATCH'):
        root = os.environ.get(var)
        if root and Path(root).is_dir():
            d = Path(root) / 'dtype_audit_tmp'
            d.mkdir(parents=True, exist_ok=True)
            return d
    d = Path(os.environ.get('TMPDIR', '/tmp')) / 'dtype_audit_tmp'
    d.mkdir(parents=True, exist_ok=True)
    return d


def _subjects_with_psd(subjects):
    """Keep only subjects that actually have a bipolar_fft tree on Oak."""
    kept = []
    for s in subjects:
        if (config.BIPOLAR_PSD_DERIV_ROOT / f'sub-{s}').is_dir():
            kept.append(s)
        else:
            logger.warning('sub-%s: no bipolar_fft tree, dropping from the audit sample', s)
    return kept


def _psd_nwb_paths(subject, max_runs):
    """Existing PSD NWBs for a subject, registry order, capped at max_runs."""
    out = []
    for session, run, _raw in io_utils.get_session_runs(subject):
        path = config.bipolar_psd_nwb_path(subject, session, run)
        if path.exists():
            out.append((session, run, path))
        if len(out) >= max_runs:
            break
    return out


# ============================================================================
# LEG A — what dtype is on disk right now
# ============================================================================

def _h5_dataset_facts(h5_path, dataset_path):
    """dtype/shape/chunking/compression/size of one HDF5 dataset, or None."""
    with h5py.File(str(h5_path), 'r') as fh:
        if dataset_path not in fh:
            return None
        dset = fh[dataset_path]
        n_values = int(np.prod(dset.shape)) if dset.shape else 0
        stored_bytes = int(dset.id.get_storage_size())
        facts = {
            'dataset': dataset_path,
            'dtype': str(dset.dtype),
            'itemsize_bytes': int(dset.dtype.itemsize),
            'shape': list(dset.shape),
            'n_values': n_values,
            'chunks': list(dset.chunks) if dset.chunks else None,
            'compression': dset.compression,
            'compression_opts': dset.compression_opts,
            'uncompressed_bytes': n_values * int(dset.dtype.itemsize),
            'stored_bytes': stored_bytes,
        }
        if stored_bytes > 0:
            facts['compression_ratio'] = facts['uncompressed_bytes'] / stored_bytes
        # `conversion` is the volts scaling on a raw ElectricalSeries; recording
        # it makes the raw-dtype question answerable (an int16 series is exactly
        # representable in float32, so io/nwb.py's cast is lossless there).
        parent = dset.parent
        for attr in ('conversion', 'unit', 'resolution'):
            if attr in parent:
                try:
                    facts[f'series_{attr}'] = np.asarray(parent[attr][()]).item()
                except Exception:                                   # noqa: BLE001
                    facts[f'series_{attr}'] = str(parent[attr][()])
        return facts


def _raw_float32_cast_facts(raw_path, n_samples=20000):
    """Is io/nwb.py's float32 cast of the raw voltage actually lossless?

    The raw ElectricalSeries turns out to be stored as float64, so this cannot
    be settled by inspecting the dtype alone the way an integer series could.
    But float64 here is a container, not information: the underlying signal is
    ADC-quantised (16-bit-class hardware), so the distinct values present are
    expected to be exactly representable in float32 even though the container
    is not. Checked empirically on a leading sample rather than assumed --
    round-trip float32 and compare bit-for-bit against the float64 original.

    This is background for P0.6, not part of its verdict: the voltage cast is
    `io/nwb.py`'s pre-existing decision, upstream of the cache dtype. Recorded
    so the audit says something definite about it instead of leaving it open.
    """
    with h5py.File(str(raw_path), 'r') as fh:
        if RAW_SERIES_H5 not in fh:
            return None
        dset = fh[RAW_SERIES_H5]
        take = min(n_samples, dset.shape[0])
        sample = dset[:take]
    sample64 = np.asarray(sample, dtype=np.float64)
    round_tripped = sample64.astype(np.float32).astype(np.float64)
    finite = np.isfinite(sample64)
    exact = bool(np.array_equal(round_tripped[finite], sample64[finite]))
    abs_err = np.abs(round_tripped[finite] - sample64[finite])
    return {
        'n_samples_checked': int(take),
        'n_values_checked': int(finite.sum()),
        'float32_cast_bit_exact': exact,
        'max_abs_cast_error': float(abs_err.max()) if abs_err.size else 0.0,
        'n_distinct_values': int(np.unique(sample64[finite]).size),
    }


def _value_range_facts(log_power):
    """Range facts for stored log-power, in both the log and linear domain.

    The linear-domain check matters because views may exponentiate (10**x) --
    a log-power of -40 is fine in float32, but 10**-40 is subnormal, and
    10**-45 underflows to zero. Reported so the view layer can be written
    knowing whether that is a live hazard for this data or a theoretical one.
    """
    flat = np.asarray(log_power).ravel()
    finite = flat[np.isfinite(flat)]
    facts = {
        'n_values': int(flat.size),
        'n_nonfinite': int(flat.size - finite.size),
        'frac_nonfinite': float((flat.size - finite.size) / flat.size) if flat.size else 0.0,
        'n_neg_inf': int(np.sum(np.isneginf(flat))),
        'n_nan': int(np.sum(np.isnan(flat))),
    }
    if finite.size:
        facts.update({
            'log10power_min': float(finite.min()),
            'log10power_max': float(finite.max()),
            'log10power_median': float(np.median(finite)),
            # float32 normals bottom out at ~1.18e-38 (log10 ~ -37.9) and
            # subnormals at ~1.4e-45 (log10 ~ -44.85).
            'float32_linear_headroom_decades': float(
                finite.min() - np.log10(np.finfo(np.float32).tiny)
            ),
            'linear_underflows_float32': bool(
                finite.min() < np.log10(np.finfo(np.float32).smallest_subnormal)
            ),
            'linear_subnormal_in_float32': bool(
                finite.min() < np.log10(np.finfo(np.float32).tiny)
            ),
        })
    return facts


def leg_a_inventory(subjects, max_runs):
    """On-disk dtypes for the PSD NWBs and their source raw NWBs."""
    logger.info('LEG A: on-disk dtype inventory')
    psd_files, raw_files, range_facts, degenerate_runs = [], [], [], []
    seen_raw = set()

    for subject in subjects:
        for session, run, psd_path in _psd_nwb_paths(subject, max_runs):
            facts = _h5_dataset_facts(psd_path, PSD_DATA_H5)
            if facts is None:
                logger.warning('%s: no %s dataset', psd_path, PSD_DATA_H5)
                continue
            facts.update({'subject': subject, 'session': session, 'run': run,
                          'path': str(psd_path)})
            broadband = _h5_dataset_facts(psd_path, BROADBAND_DATA_H5)
            if broadband is not None:
                facts['broadband_dtype'] = broadband['dtype']
                facts['broadband_shape'] = broadband['shape']
            psd_files.append(facts)
            logger.info('  %s psd_log_bins dtype=%s shape=%s chunks=%s %s',
                        psd_path.name, facts['dtype'], facts['shape'],
                        facts['chunks'], facts['compression'])

            # Value ranges from one run per subject -- reading every run's full
            # array would be a lot of IO for a range check that does not vary.
            # Skip all-non-finite runs: a dead run (constant voltage -> PSD 0 ->
            # log10 = -inf everywhere) has no range to report, and taking it as
            # the subject's sample yields a NaN row that says nothing about
            # float32. Counted separately, since an entirely -inf stored run is
            # a data-quality fact worth surfacing even though it is not P0.6's.
            if not any(r['subject'] == subject for r in range_facts):
                with h5py.File(str(psd_path), 'r') as fh:
                    sample = fh[PSD_DATA_H5][:]
                rf = _value_range_facts(sample)
                del sample
                rf.update({'subject': subject, 'session': session, 'run': run})
                if rf['n_nonfinite'] == rf['n_values']:
                    degenerate_runs.append({'subject': subject, 'session': session,
                                            'run': run, 'n_values': rf['n_values'],
                                            'note': 'stored PSD is entirely non-finite'})
                else:
                    range_facts.append(rf)

        # Raw side: one run per subject is enough to establish the raw dtype.
        for session, run, raw_path in io_utils.get_session_runs(subject)[:1]:
            if raw_path in seen_raw or not Path(raw_path).exists():
                continue
            seen_raw.add(raw_path)
            facts = _h5_dataset_facts(raw_path, RAW_SERIES_H5)
            if facts is None:
                continue
            facts.update({'subject': subject, 'session': session, 'run': run,
                          'path': str(raw_path)})
            # An integer raw dtype narrower than float32's 24-bit mantissa is
            # exactly representable, so io/nwb.py's float32 cast loses nothing.
            dt = np.dtype(facts['dtype'])
            facts['integer_raw'] = bool(np.issubdtype(dt, np.integer))
            if facts['integer_raw']:
                facts['float32_cast_lossless'] = bool(dt.itemsize * 8 <= 24)
            else:
                # Float raw: dtype alone settles nothing, so measure it.
                cast = _raw_float32_cast_facts(raw_path)
                if cast is not None:
                    facts['float32_cast_probe'] = cast
                    facts['float32_cast_lossless'] = cast['float32_cast_bit_exact']
            raw_files.append(facts)
            logger.info('  raw %s ElectricalSeries dtype=%s', Path(raw_path).name, facts['dtype'])

    dtypes = sorted({f['dtype'] for f in psd_files})
    # Tightest float32 margin seen if a view exponentiates log-power back to
    # linear. Small margins are the argument for exponentiating in float64.
    headrooms = [r['float32_linear_headroom_decades'] for r in range_facts
                 if 'float32_linear_headroom_decades' in r]
    return {
        'psd_files': psd_files,
        'raw_files': raw_files,
        'value_ranges': range_facts,
        'degenerate_runs': degenerate_runs,
        'psd_dtypes_seen': dtypes,
        'psd_dtype_uniform': len(dtypes) <= 1,
        'raw_dtypes_seen': sorted({f['dtype'] for f in raw_files}),
        'min_float32_linear_headroom_decades': min(headrooms) if headrooms else None,
    }


# ============================================================================
# LEG B — does float32 survive a storage round-trip bit-exactly
# ============================================================================

def leg_b_round_trip(sample_log_power, tmp_dir):
    """Write real float32 log-power out and read it back; demand bit-exactness.

    HDF5 (gzip 4, production's settings) is what bipolar_fft writes today.
    Parquet is what the P1.1 cache will write, in the long layout the
    architecture doc specifies. Parquet is skipped with a clear note if pyarrow
    is absent -- installing it is P0.3, a separate task.
    """
    logger.info('LEG B: storage round-trip bit-exactness')
    arr32 = np.asarray(sample_log_power, dtype=np.float32)
    results = {'sample_shape': list(arr32.shape), 'sample_dtype': str(arr32.dtype)}

    h5_path = tmp_dir / 'round_trip.h5'
    with h5py.File(str(h5_path), 'w') as fh:
        fh.create_dataset('log_power', data=arr32, compression='gzip', compression_opts=4)
    with h5py.File(str(h5_path), 'r') as fh:
        back = fh['log_power'][:]
    results['hdf5'] = {
        'available': True,
        'dtype_preserved': str(back.dtype) == 'float32',
        # equal_nan: -inf/NaN are real values here and must survive too.
        'bit_exact': bool(np.array_equal(back, arr32, equal_nan=True)),
        'stored_bytes': int(h5_path.stat().st_size),
        'uncompressed_bytes': int(arr32.nbytes),
    }
    h5_path.unlink(missing_ok=True)
    logger.info('  HDF5: dtype_preserved=%s bit_exact=%s',
                results['hdf5']['dtype_preserved'], results['hdf5']['bit_exact'])

    # Long layout matching docs/architecture.md's cache columns, so this probes
    # the format the cache will actually use rather than a bare array dump.
    try:
        import pyarrow                                            # noqa: F401
    except ImportError:
        results['parquet'] = {
            'available': False,
            'note': 'pyarrow not installed in the venv (P0.3 installs it). '
                    'Parquet stores IEEE-754 binary32 natively, so a float32 '
                    'column is expected to round-trip bit-exactly; re-run this '
                    'leg once pyarrow lands to confirm rather than assume.',
        }
        logger.warning('  Parquet leg SKIPPED: pyarrow not installed (P0.3)')
    else:
        n_win, n_pairs, n_bins = arr32.shape
        win_i, pair_i, bin_i = np.meshgrid(
            np.arange(n_win), np.arange(n_pairs), np.arange(n_bins), indexing='ij')
        df = pd.DataFrame({
            'window_idx': win_i.ravel().astype(np.int32),
            'channel_idx': pair_i.ravel().astype(np.int16),
            'bin': bin_i.ravel().astype(np.int16),
            'log_power': arr32.ravel(),
        })
        pq_path = tmp_dir / 'round_trip.parquet'
        df.to_parquet(pq_path, index=False, compression='snappy')
        back_df = pd.read_parquet(pq_path)
        col = back_df['log_power'].to_numpy()
        results['parquet'] = {
            'available': True,
            'n_rows': int(len(df)),
            'dtype_preserved': str(col.dtype) == 'float32',
            'bit_exact': bool(np.array_equal(col, arr32.ravel(), equal_nan=True)),
            'stored_bytes': int(pq_path.stat().st_size),
            'uncompressed_bytes': int(arr32.nbytes),
        }
        pq_path.unlink(missing_ok=True)
        logger.info('  Parquet: dtype_preserved=%s bit_exact=%s',
                    results['parquet']['dtype_preserved'], results['parquet']['bit_exact'])
    return results


# ============================================================================
# LEG C — float32 vs float64 accumulator when averaging a real epoch
# ============================================================================

def _leg_c_one_run(subject, session, run, run_data, pain_df, mask_df,
                    epoch_minutes, max_excluded_frac, rows):
    """Compare accumulators over every epoch that falls wholly inside one run.

    Returns how many epochs were processed. Passing a single-run list to
    `_find_matching_run` keeps production's matching logic verbatim while
    naturally skipping the 'boundary' and 'no_match' cases, which production
    drops anyway.
    """
    n_epochs = 0
    for _, pain_row in pain_df.iterrows():
        if pain_row['pain_bin'] is None:
            continue
        pain_time = pain_row['date']
        window_start = pain_time - pd.Timedelta(minutes=epoch_minutes)
        status, matched = _find_matching_run(pain_time, window_start, [run_data])
        if status != 'match':
            continue
        dts = matched['run_datetimes']
        epoch_rows = np.where((dts >= window_start) & (dts < pain_time))[0]
        if len(epoch_rows) == 0:
            continue

        excluded = _excluded_mask(matched, matched['run_id_full'], epoch_rows, mask_df)
        epoch_lp = matched['log_power'][epoch_rows]
        n_epoch_rows = epoch_lp.shape[0]
        n_epochs += 1

        for pair_i, channel in enumerate(matched['channel_names']):
            row_excluded = excluded[:, pair_i]
            n_kept = int((~row_excluded).sum())
            if 1.0 - (n_kept / n_epoch_rows) > max_excluded_frac:
                continue
            kept = epoch_lp[~row_excluded, pair_i, :]
            if not np.all(np.isfinite(kept)):
                continue    # production drops these channel-epochs too

            mean64 = kept.astype(np.float64).mean(axis=0)
            mean32_default = kept.mean(axis=0)                    # numpy default
            mean32_forced = kept.mean(axis=0, dtype=np.float32)   # naive accumulator

            for arm, mean_test in (('numpy_default_float32', mean32_default),
                                    ('forced_float32_accumulator', mean32_forced)):
                abs_err = np.abs(mean_test.astype(np.float64) - mean64)
                rel_err = abs_err / np.maximum(np.abs(mean64), np.finfo(np.float64).tiny)
                rows.append({
                    'subject': subject, 'session': session, 'run': run,
                    'pain_event_id': int(pain_row['pain_event_id']),
                    'channel': channel, 'arm': arm,
                    'n_windows_averaged': n_kept,
                    'max_abs_error': float(abs_err.max()),
                    'max_rel_error': float(rel_err.max()),
                    'min_sig_figs': _sig_figs(float(rel_err.max())),
                })
    return n_epochs


def leg_c_accumulation(subjects, mask_label, epoch_minutes, max_excluded_frac,
                        max_sessions_per_subject, max_runs, max_channel_epochs):
    """Average real 5-minute channel-epochs three ways and compare.

    Reference is a float64 accumulator over the stored float32 values, which is
    exact to ~1e-16 and so isolates accumulator error from storage error (leg D
    covers storage). The two test arms are numpy's default for float32 input
    (float32 accumulator, pairwise summation) and a forced-float32 accumulator.

    This is a VIEW-layer question: it decides whether the view chain must upcast
    before it averages, which is free to do and independent of what the cache
    stores.

    Runs are loaded ONE AT A TIME and freed, unlike build_pain_epoch_power which
    holds a whole session in memory: a session here can be 80+ runs (~15 GB of
    log-power), and this leg needs a sample of epochs, not all of them. Both
    caps are logged when they bite, so a truncated sample never reads as a
    complete one.
    """
    logger.info('LEG C: epoch-average accumulator dtype, on real epochs')
    rows = []
    n_epochs_seen = 0
    truncated = []

    for subject in subjects:
        sessions = sorted({s for s, _r, _p in io_utils.get_session_runs(subject)})
        if len(sessions) > max_sessions_per_subject:
            truncated.append(f'sub-{subject}: {len(sessions)} sessions, used '
                             f'{max_sessions_per_subject}')
        for session in sessions[:max_sessions_per_subject]:
            pain_df = load_pain_scores(subject, session)
            if pain_df is None or pain_df.empty:
                logger.info('  sub-%s ses-%s: no pain scores, skipping', subject, session)
                continue
            mask_df = load_mask(subject, session, mask_label)

            available = [(run, config.bipolar_psd_nwb_path(subject, session, run))
                         for _s, run, _raw in io_utils.get_session_runs(subject, session)]
            available = [(run, p) for run, p in available if p.exists()]
            if not available:
                logger.warning('  sub-%s ses-%s: no usable PSD runs', subject, session)
                continue
            if len(available) > max_runs:
                truncated.append(f'sub-{subject} ses-{session}: {len(available)} PSD runs, '
                                 f'used {max_runs}')

            for run, path in available[:max_runs]:
                run_data = _load_run_psd(path)
                run_data['run'] = run
                run_data['run_id_full'] = f'run-{run}'
                n_epochs_seen += _leg_c_one_run(
                    subject, session, run, run_data, pain_df, mask_df,
                    epoch_minutes, max_excluded_frac, rows)
                del run_data
                if len(rows) >= 2 * max_channel_epochs:      # two arms per channel-epoch
                    truncated.append(f'sub-{subject} ses-{session}: hit the '
                                     f'{max_channel_epochs} channel-epoch cap, stopped early')
                    break
            if len(rows) >= 2 * max_channel_epochs:
                break
        if len(rows) >= 2 * max_channel_epochs:
            break

    for note in truncated:
        logger.info('  coverage note — %s', note)

    if not rows:
        return {'n_channel_epochs': 0, 'n_epochs_seen': n_epochs_seen,
                'coverage_notes': truncated,
                'note': 'no surviving channel-epochs found in the sampled subjects'}

    df = pd.DataFrame(rows)
    summary = {'n_epochs_seen': n_epochs_seen,
               'n_channel_epoch_comparisons': int(len(df)),
               'n_subjects': int(df['subject'].nunique()),
               'windows_per_epoch_median': float(df['n_windows_averaged'].median()),
               'windows_per_epoch_max': int(df['n_windows_averaged'].max()),
               'coverage_notes': truncated,
               'arms': {}}
    for arm, sub in df.groupby('arm'):
        worst = float(sub['max_rel_error'].max())
        summary['arms'][arm] = {
            'n': int(len(sub)),
            'max_rel_error': worst,
            'median_rel_error': float(sub['max_rel_error'].median()),
            'max_abs_error_log10power': float(sub['max_abs_error'].max()),
            'max_linear_frac_error': float(np.expm1(float(sub['max_abs_error'].max()) * np.log(10.0))),
            'min_sig_figs': _sig_figs(worst),
            'median_sig_figs': float(sub['min_sig_figs'].median()),
            'meets_sig_fig_target': _sig_figs(worst) >= SIG_FIG_TARGET,
        }
        logger.info('  %s: worst rel err %.3e (%.1f sig figs) over %d channel-epochs',
                    arm, worst, _sig_figs(worst), len(sub))
    return summary, df


# ============================================================================
# LEG D — float64 recompute vs the production float32 path
# ============================================================================

def _welch_one_channel_float64(channel_col, sfreq, nperseg, noverlap, bin_edges):
    """float64 mirror of bipolar_reref._welch_one_channel.

    Identical arithmetic -- same scipy call, same production
    `_band_average_linear` (already float64), same log10 -- differing ONLY in
    that the result is not cast down to float32. `leg_d_recompute` asserts
    `production_float32 == mirror.astype(float32)` bit-for-bit, which proves the
    mirror is faithful: both round the same float64 log10 result to float32 with
    the same round-to-nearest, so any drift between the two functions would show
    up as an assertion failure rather than as a quietly wrong headline number.
    """
    freqs, times, Sxx = signal.spectrogram(
        channel_col, fs=sfreq, nperseg=nperseg, noverlap=noverlap,
        window='hann', scaling='density', mode='psd')
    n_windows = Sxx.shape[1]
    n_bins = len(bin_edges) - 1
    out = np.empty((n_windows, n_bins), dtype=np.float64)
    for w in range(n_windows):
        linear_bins = bipolar_reref._band_average_linear(freqs, Sxx[:, w], bin_edges)
        with np.errstate(divide='ignore'):
            out[w, :] = np.log10(linear_bins)
    return out, times


def _raw_run_candidates(subject, epoch_minutes, max_candidates=4):
    """Raw runs for leg D, cheapest first, with the degenerate ones filtered out.

    Cheapest-first keeps the one unavoidable raw read in this audit small:
    precision is a property of the arithmetic, not of run length. But "smallest"
    alone picks badly. The smallest run in this dataset is often a dead one --
    sub-071's is a single constant value repeated, so every PSD bin is
    log10(0) = -inf and there is nothing to compare. So require the run to be at
    least one epoch long, then let the caller try candidates in order until one
    yields finite data.
    """
    registry = pd.read_csv(config.FILE_REGISTRY_CSV)
    rows = registry[registry['sub_id'] == f'sub-{subject}'].copy()
    if rows.empty:
        return []
    long_enough = rows['duration_minutes'] >= epoch_minutes
    if long_enough.any():
        rows = rows[long_enough]
    rows = rows.sort_values('raw_file_size_mb')

    out = []
    for _, row in rows.iterrows():
        if not Path(row['raw_file_path']).exists():
            continue
        out.append({
            'session': row['ses_id'].replace('ses-', ''),
            'run': row['run_id'].replace('run-', ''),
            'path': row['raw_file_path'],
            'size_mb': float(row['raw_file_size_mb']),
            'duration_minutes': float(row['duration_minutes']),
        })
        if len(out) >= max_candidates:
            break
    return out


def leg_d_recompute(subject, n_channels, epoch_minutes, window_sec, overlap_frac,
                     bin_edges):
    """Recompute one run's PSD in float64 and compare epoch averages to float32.

    The headline P0.6 number. Epoch averages are taken over contiguous blocks of
    windows the size of a real epoch (5 min / 1 s hop = 300 windows by default),
    which is what a view will do, so the reported error is the error a real
    effect size would carry.

    All frequency bins are compared, including the line-noise-flagged ones the
    cache drops -- a superset of what reaches the cache, since dtype behaviour
    does not depend on which bin a value came from.
    """
    logger.info('LEG D: float64 recompute vs production float32 path')
    candidates = _raw_run_candidates(subject, epoch_minutes)
    if not candidates:
        return {'ran': False, 'note': f'no readable raw run found for sub-{subject}'}

    attempts = []
    for cand in candidates:
        result = _leg_d_one_run(subject, cand, n_channels, epoch_minutes,
                                window_sec, overlap_frac, bin_edges)
        if result.get('ran'):
            result['candidates_tried'] = attempts + [
                {'run': cand['run'], 'size_mb': cand['size_mb'], 'outcome': 'used'}]
            return result
        attempts.append({'run': cand['run'], 'size_mb': cand['size_mb'],
                         'outcome': result.get('note')})
        logger.warning('  sub-%s run-%s unusable for leg D (%s); trying the next candidate',
                       subject, cand['run'], result.get('note'))
    return {'ran': False, 'candidates_tried': attempts,
            'note': f'no usable raw run for sub-{subject} after {len(attempts)} candidate(s)'}


def _leg_d_one_run(subject, cand, n_channels, epoch_minutes, window_sec, overlap_frac,
                    bin_edges):
    """One candidate run's float64-vs-float32 comparison; see leg_d_recompute."""
    session, run, raw_path = cand['session'], cand['run'], cand['path']
    logger.info('  trying sub-%s ses-%s run-%s (%.0f MB raw, %.0f min)',
                subject, session, run, cand['size_mb'], cand['duration_minutes'])

    data_v, _channel_names, sfreq, elec_df, elec_indices = \
        io_utils.load_all_channels_with_electrodes(raw_path)
    pairs, _filtered = bipolar_reref.derive_pairs(elec_df)
    if not pairs:
        del data_v
        return {'ran': False, 'note': f'no bipolar pairs derivable for run-{run}'}

    # Only the pairs actually compared need re-referencing; the rest would be
    # memory and time spent on channels leg D never looks at.
    pairs_used = pairs[:n_channels]
    bipolar_v = bipolar_reref.rereference(data_v, elec_indices, pairs_used)
    del data_v

    # A dead run (constant voltage) gives PSD == 0 -> log10 == -inf everywhere,
    # which carries no dtype information. Detect it here rather than discovering
    # it as an empty comparison further down.
    if not np.any(np.diff(bipolar_v[:, :min(4, bipolar_v.shape[1])], axis=0) != 0):
        del bipolar_v
        return {'ran': False, 'note': f'run-{run} is flat/constant in its first pairs'}

    nperseg = max(1, int(round(window_sec * sfreq)))
    noverlap = int(nperseg * overlap_frac)
    hop_sec = window_sec * (1.0 - overlap_frac)
    windows_per_epoch = max(1, int(round(epoch_minutes * 60.0 / hop_sec)))

    per_channel_stats, mirror_faithful = [], True
    all_ref, all_test = [], []

    for ch in range(bipolar_v.shape[1]):
        col = bipolar_v[:, ch]
        prod32, _t32 = bipolar_reref._welch_one_channel(col, sfreq, nperseg, noverlap, bin_edges)
        ref64, _t64 = _welch_one_channel_float64(col, sfreq, nperseg, noverlap, bin_edges)

        # Proves the mirror is production's arithmetic minus the downcast.
        faithful = bool(np.array_equal(ref64.astype(np.float32), prod32, equal_nan=True))
        mirror_faithful = mirror_faithful and faithful

        n_win = ref64.shape[0]
        n_blocks = n_win // windows_per_epoch
        if n_blocks == 0:
            # Run shorter than one epoch: average whatever windows exist rather
            # than skip the channel, so short runs still contribute.
            blocks = [(0, n_win)]
        else:
            blocks = [(b * windows_per_epoch, (b + 1) * windows_per_epoch)
                      for b in range(n_blocks)]

        ch_ref, ch_test = [], []
        for lo, hi in blocks:
            block_ref, block_prod = ref64[lo:hi], prod32[lo:hi]
            # Drop bins that are non-finite anywhere in the block, matching
            # production's "drop the whole channel-epoch on any -inf" rule at
            # bin granularity.
            good = np.all(np.isfinite(block_ref), axis=0) & np.all(np.isfinite(block_prod), axis=0)
            if not good.any():
                continue
            ch_ref.append(block_ref[:, good].mean(axis=0))
            ch_test.append(block_prod[:, good].astype(np.float64).mean(axis=0))
        if not ch_ref:
            continue

        ch_ref = np.concatenate(ch_ref)
        ch_test = np.concatenate(ch_test)
        stats = _error_stats(ch_ref, ch_test, label=f'pair_{ch}')
        stats['mirror_bit_faithful'] = faithful
        stats['n_windows'] = int(n_win)
        stats['n_epoch_blocks'] = len(blocks)
        per_channel_stats.append(stats)
        all_ref.append(ch_ref)
        all_test.append(ch_test)

    del bipolar_v

    if not all_ref:
        return {'ran': False, 'note': 'every channel was non-finite across all bins'}

    overall = _error_stats(np.concatenate(all_ref), np.concatenate(all_test),
                            label='all_channels_epoch_averages')
    logger.info('  worst rel err %.3e (%.1f sig figs); linear-power error %.2e; mirror faithful=%s',
                overall['max_rel_error'], overall['min_sig_figs'],
                overall['max_linear_frac_error'], mirror_faithful)
    return {
        'ran': True,
        'subject': subject, 'session': session, 'run': run,
        'raw_path': str(raw_path), 'raw_size_mb': cand['size_mb'],
        'raw_duration_minutes': cand['duration_minutes'],
        'sfreq_hz': float(sfreq),
        'n_pairs_available': len(pairs),
        'n_pairs_compared': len(pairs_used),
        'nperseg': int(nperseg), 'noverlap': int(noverlap),
        'hop_sec': float(hop_sec),
        'windows_per_epoch_block': int(windows_per_epoch),
        'mirror_bit_faithful_all_channels': mirror_faithful,
        'per_channel': per_channel_stats,
        'overall': overall,
    }


# ============================================================================
# report assembly
# ============================================================================

def _verdict(report):
    """Roll the four legs into a verdict on P0.6.

    Two questions are kept apart on purpose, because conflating them produces a
    wrong answer:

      STORAGE (what P0.6 decides) — is float32 the right dtype for the cache?
        Legs A, B, D. Storing float32 must round-trip exactly and must not cost
        meaningful precision in an epoch average versus a float64 pipeline.

      VIEW-LAYER REQUIREMENTS (what the audit discovered along the way) — how
        must code that READS the cache behave? Leg C, plus leg A's linear-domain
        headroom. A float32 accumulator failing the sig-fig bar is not an
        argument against float32 storage; it is an argument for upcasting before
        averaging, which is free. Recorded as requirements, not failures, so
        they cannot be mistaken for a reason to store float64.
    """
    storage, requirements, notes = {}, {}, []

    inv = report['leg_a_inventory']
    storage['psd_stored_as_float32'] = inv['psd_dtypes_seen'] == ['float32']
    storage['psd_dtype_uniform_across_files'] = inv['psd_dtype_uniform']
    storage['no_float32_linear_underflow'] = all(
        not r.get('linear_underflows_float32', False) for r in inv['value_ranges'])

    rt = report['leg_b_round_trip']
    storage['hdf5_round_trip_bit_exact'] = bool(rt['hdf5']['bit_exact'])
    if rt['parquet'].get('available'):
        storage['parquet_round_trip_bit_exact'] = bool(rt['parquet']['bit_exact'])
    else:
        notes.append('Parquet round-trip UNVERIFIED — pyarrow absent (P0.3). Re-run '
                     'this audit once it is installed to close the one storage '
                     'question this pass could not answer directly.')

    rec = report['leg_d_recompute']
    if rec.get('ran'):
        storage['recompute_mirror_faithful'] = bool(rec['mirror_bit_faithful_all_channels'])
        storage[f'recompute_epoch_avg_meets_{int(SIG_FIG_TARGET)}_sig_figs'] = \
            bool(rec['overall']['meets_sig_fig_target'])
    else:
        notes.append(f"Leg D did NOT run ({rec.get('note')}) — the end-to-end "
                     f"float64-vs-float32 comparison is the headline test, so the "
                     f"storage verdict is incomplete without it.")

    acc = report['leg_c_accumulation']
    if acc.get('arms'):
        float32_arms_ok = all(s['meets_sig_fig_target'] for s in acc['arms'].values())
        worst = max(s['max_rel_error'] for s in acc['arms'].values())
        requirements['views_must_accumulate_in_float64'] = not float32_arms_ok
        requirements['float32_accumulator_sig_figs'] = _sig_figs(worst)
        if not float32_arms_ok:
            notes.append(f'A float32 accumulator holds only '
                         f'{_sig_figs(worst):.1f} sig figs over a ~5 min epoch, below the '
                         f'{SIG_FIG_TARGET:.0f} target. Views MUST upcast to float64 before '
                         f'averaging. This constrains the VIEW layer, not the cache dtype.')

    headroom = inv.get('min_float32_linear_headroom_decades')
    if headroom is not None:
        # Under ~6 decades of margin is close enough to float32's subnormal floor
        # that a baseline-divided or otherwise scaled linear view could underflow.
        requirements['views_must_exponentiate_in_float64'] = bool(headroom < 6.0)
        requirements['min_float32_linear_headroom_decades'] = headroom
        if headroom < 6.0:
            notes.append(f'Worst stored log-power leaves only {headroom:.1f} decades above '
                         f'float32\'s smallest normal, so 10**log_power in float32 is close '
                         f'to underflow. Linear-domain views MUST exponentiate in float64.')

    if inv.get('degenerate_runs'):
        notes.append(f"{len(inv['degenerate_runs'])} sampled run(s) have an entirely "
                     f"non-finite stored PSD (dead/constant channels). A data-quality "
                     f"observation, not a dtype one — see SCRATCHPAD.")

    failures = [k for k, v in storage.items() if not v]
    return {
        'storage_checks': storage,
        'failed_storage_checks': failures,
        'float32_storage_validated': (not failures) and bool(rec.get('ran')),
        'view_layer_requirements': requirements,
        'notes': notes,
        'sig_fig_target': SIG_FIG_TARGET,
    }


def _summary_text(report):
    lines = ['P0.6 dtype audit — is float32 safe for the per-window cache?', '']
    v = report['verdict']
    lines.append(f"VERDICT: float32 STORAGE "
                 f"{'VALIDATED' if v['float32_storage_validated'] else 'NOT VALIDATED'} "
                 f"(target {v['sig_fig_target']:.0f}+ sig figs on epoch averages)")
    if v['failed_storage_checks']:
        lines.append(f"  FAILED STORAGE CHECKS: {', '.join(v['failed_storage_checks'])}")
    req = v.get('view_layer_requirements', {})
    if req:
        lines.append('  VIEW-LAYER REQUIREMENTS this audit imposes '
                     '(constraints on readers, not on the cache dtype):')
        if req.get('views_must_accumulate_in_float64'):
            lines.append(f"    - upcast to float64 BEFORE epoch-averaging "
                         f"(float32 accumulator holds only "
                         f"{req['float32_accumulator_sig_figs']:.1f} sig figs)")
        if req.get('views_must_exponentiate_in_float64'):
            lines.append(f"    - exponentiate to linear in float64 (only "
                         f"{req['min_float32_linear_headroom_decades']:.1f} decades of "
                         f"float32 headroom at the worst stored value)")
    for note in v.get('notes', []):
        lines.append(f"  NOTE: {note}")
    lines.append('')

    inv = report['leg_a_inventory']
    lines.append('LEG A — on-disk dtype')
    lines.append(f"  PSD psd_log_bins dtype(s): {inv['psd_dtypes_seen']} "
                 f"across {len(inv['psd_files'])} file(s)")
    lines.append(f"  raw ElectricalSeries dtype(s): {inv['raw_dtypes_seen']}")
    for r in inv['raw_files'][:3]:
        kind = 'integer raw' if r.get('integer_raw') else f"float raw ({r['dtype']})"
        detail = ''
        probe = r.get('float32_cast_probe')
        if probe:
            detail = (f" (probed {probe['n_values_checked']} values, "
                      f"{probe['n_distinct_values']} distinct)")
        lines.append(f"    sub-{r['subject']}: {kind}, io/nwb.py float32 cast "
                     f"lossless = {r.get('float32_cast_lossless')}{detail}")
    for r in inv['value_ranges']:
        lines.append(f"    sub-{r['subject']} log10-power range "
                     f"[{r.get('log10power_min', float('nan')):.3f}, "
                     f"{r.get('log10power_max', float('nan')):.3f}], "
                     f"{r['frac_nonfinite'] * 100:.4f}% non-finite, "
                     f"{r.get('float32_linear_headroom_decades', float('nan')):.1f} decades "
                     f"of float32 headroom if exponentiated")
    for d in inv.get('degenerate_runs', []):
        lines.append(f"    sub-{d['subject']} run-{d['run']}: {d['note']} "
                     f"({d['n_values']} values) — excluded from the range check")
    lines.append('')

    rt = report['leg_b_round_trip']
    lines.append('LEG B — storage round-trip')
    lines.append(f"  HDF5 gzip4: dtype preserved={rt['hdf5']['dtype_preserved']}, "
                 f"bit exact={rt['hdf5']['bit_exact']}")
    if rt['parquet'].get('available'):
        lines.append(f"  Parquet snappy: dtype preserved={rt['parquet']['dtype_preserved']}, "
                     f"bit exact={rt['parquet']['bit_exact']}, "
                     f"{rt['parquet']['n_rows']} rows")
    else:
        lines.append(f"  Parquet: SKIPPED — {rt['parquet']['note']}")
    lines.append('')

    acc = report['leg_c_accumulation']
    lines.append('LEG C — epoch-average accumulator (reference: float64 over stored float32)')
    if acc.get('arms'):
        lines.append(f"  {acc['n_channel_epoch_comparisons']} channel-epoch comparisons, "
                     f"{acc['n_subjects']} subject(s), "
                     f"median {acc['windows_per_epoch_median']:.0f} windows averaged")
        for arm, s in acc['arms'].items():
            lines.append(f"  {arm}: worst rel err {s['max_rel_error']:.3e} "
                         f"({s['min_sig_figs']:.1f} sig figs), "
                         f"worst linear-power err {s['max_linear_frac_error']:.2e}")
    else:
        lines.append(f"  {acc.get('note', 'no comparisons')}")
    for note in acc.get('coverage_notes', []):
        lines.append(f"  coverage: {note}")
    lines.append('')

    rec = report['leg_d_recompute']
    lines.append('LEG D — float64 recompute vs production float32 (the headline test)')
    if rec.get('ran'):
        o = rec['overall']
        lines.append(f"  sub-{rec['subject']} ses-{rec['session']} run-{rec['run']}, "
                     f"{rec['n_pairs_compared']}/{rec['n_pairs_available']} pairs, "
                     f"{rec['sfreq_hz']:.0f} Hz, "
                     f"{rec['windows_per_epoch_block']} windows per epoch block")
        lines.append(f"  float64 mirror bit-faithful to production: "
                     f"{rec['mirror_bit_faithful_all_channels']}")
        lines.append(f"  {o['n_compared']} epoch-average values compared")
        lines.append(f"  worst rel err {o['max_rel_error']:.3e} "
                     f"({o['min_sig_figs']:.1f} sig figs), "
                     f"median {o['median_rel_error']:.3e} ({o['median_sig_figs']:.1f} sig figs)")
        lines.append(f"  worst abs err {o['max_abs_error_log10power']:.3e} log10-power "
                     f"= {o['max_linear_frac_error']:.2e} fractional error in LINEAR power")
        lines.append(f"  worst error is {o['max_rel_error_in_float32_half_ulps']:.2f} "
                     f"float32 half-ulps")
    else:
        lines.append(f"  NOT RUN — {rec.get('note')}")
    for c in rec.get('candidates_tried', []):
        lines.append(f"    candidate run-{c['run']} ({c['size_mb']:.0f} MB): {c['outcome']}")
    lines.append('')
    return '\n'.join(lines) + '\n'


def _write_outputs(report, accumulation_df, out_dir, description):
    out_dir.mkdir(parents=True, exist_ok=True)

    (out_dir / 'dtype_audit.json').write_text(json.dumps(report, indent=2, default=str))
    summary = _summary_text(report)
    (out_dir / 'summary.txt').write_text(summary)
    if accumulation_df is not None and len(accumulation_df):
        accumulation_df.to_csv(out_dir / 'epoch_average_errors.csv', index=False)

    (out_dir / 'provenance.json').write_text(json.dumps({
        'script': 'ieeg_ehr/features/dtype_audit.py',
        'git': report['run']['git'],
        'run_timestamp': report['run']['run_timestamp'],
        'params': report['run']['params'],
        'subjects': report['run']['subjects'],
        'inputs': {
            'bipolar_psd_deriv_root': str(config.BIPOLAR_PSD_DERIV_ROOT),
            'file_registry_csv': str(config.FILE_REGISTRY_CSV),
            'mask_dir': str(config.raw_voltage_mask_dir(report['run']['params']['mask_label'])),
        },
        'verdict': report['verdict'],
    }, indent=2, default=str))

    log_analysis(description, out_dir)
    print()
    print(summary, flush=True)
    logger.info('wrote audit outputs to %s', out_dir)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--subjects', nargs='+', default=None,
                         help='Subject IDs without the sub- prefix. Default: the first '
                              '--n-subjects of config.exploratory_subjects() that have a '
                              'bipolar_fft tree.')
    parser.add_argument('--n-subjects', type=int, default=3,
                         help='How many subjects to sample when --subjects is not given. '
                              'Precision is a property of the arithmetic, so a handful of '
                              'subjects across different sfreq/channel counts is enough.')
    parser.add_argument('--max-runs-per-subject', type=int, default=3,
                         help='Cap on PSD NWBs inspected per subject in leg A.')
    parser.add_argument('--max-sessions-per-subject', type=int, default=1,
                         help='Cap on sessions used for the leg C epoch sweep.')
    parser.add_argument('--max-runs-leg-c', type=int, default=8,
                         help='Cap on PSD runs loaded per session in leg C. A session can '
                              'be 80+ runs; each is loaded and freed one at a time.')
    parser.add_argument('--max-channel-epochs', type=int, default=4000,
                         help='Cap on channel-epochs compared in leg C. Reported in the '
                              'output when it bites.')
    parser.add_argument('--recompute-channels', type=int, default=8,
                         help='Bipolar pairs re-referenced and recomputed in leg D.')
    parser.add_argument('--skip-recompute', action='store_true',
                         help='Skip leg D (the only leg that reads raw data).')
    parser.add_argument('--mask-label', default=None,
                         help=f'Raw-voltage mask label (default: {config.CANONICAL_MASK_LABEL}).')
    parser.add_argument('--epoch-minutes', type=float, default=config.EPOCH_MINUTES_BEFORE)
    parser.add_argument('--max-excluded-frac', type=float, default=config.EPOCH_MAX_EXCLUDED_FRAC)
    parser.add_argument('--label', default='p0.6',
                         help='Human label for the output run directory.')
    args = parser.parse_args()

    prov = warn_if_dirty()
    ts = run_timestamp()

    subjects = args.subjects or config.exploratory_subjects()
    subjects = _subjects_with_psd(subjects)[:args.n_subjects] if not args.subjects \
        else _subjects_with_psd(subjects)
    if not subjects:
        raise SystemExit('No sampled subject has a bipolar_fft tree — nothing to audit.')
    mask_label = args.mask_label or config.CANONICAL_MASK_LABEL
    logger.info('auditing %d subject(s): %s (mask %s)', len(subjects), subjects, mask_label)

    bin_edges = bipolar_reref.log_bin_edges(
        config.PSD_N_LOG_BINS, config.PSD_FREQ_MIN_HZ, config.PSD_FREQ_MAX_HZ)

    report = {'run': {
        'script': 'ieeg_ehr/features/dtype_audit.py',
        'git': prov,
        'run_timestamp': ts,
        'subjects': subjects,
        'params': {
            'mask_label': mask_label,
            'epoch_minutes': args.epoch_minutes,
            'max_excluded_frac': args.max_excluded_frac,
            'max_runs_per_subject': args.max_runs_per_subject,
            'max_sessions_per_subject': args.max_sessions_per_subject,
            'max_runs_leg_c': args.max_runs_leg_c,
            'max_channel_epochs': args.max_channel_epochs,
            'recompute_channels': args.recompute_channels,
            'skip_recompute': args.skip_recompute,
            'psd_window_sec': config.PSD_WINDOW_SEC,
            'psd_overlap_frac': config.PSD_OVERLAP_FRAC,
            'n_log_bins': config.PSD_N_LOG_BINS,
            'sig_fig_target': SIG_FIG_TARGET,
            'float32_eps': FLOAT32_EPS,
        },
    }}

    report['leg_a_inventory'] = leg_a_inventory(subjects, args.max_runs_per_subject)

    # Leg B works on real stored values, not synthetic ones, so the -inf/NaN
    # cases in this data are part of what has to survive the round-trip.
    first_psd = _psd_nwb_paths(subjects[0], 1)
    if first_psd:
        with h5py.File(str(first_psd[0][2]), 'r') as fh:
            sample = fh[PSD_DATA_H5][:2000]
        report['leg_b_round_trip'] = leg_b_round_trip(sample, _tmp_dir())
        del sample
    else:
        report['leg_b_round_trip'] = {'hdf5': {'bit_exact': False, 'available': False},
                                       'parquet': {'available': False},
                                       'note': 'no PSD NWB available to sample'}

    acc = leg_c_accumulation(subjects, mask_label, args.epoch_minutes,
                              args.max_excluded_frac, args.max_sessions_per_subject,
                              args.max_runs_leg_c, args.max_channel_epochs)
    if isinstance(acc, tuple):
        report['leg_c_accumulation'], accumulation_df = acc
    else:
        report['leg_c_accumulation'], accumulation_df = acc, None

    if args.skip_recompute:
        report['leg_d_recompute'] = {'ran': False, 'note': 'skipped via --skip-recompute'}
    else:
        report['leg_d_recompute'] = leg_d_recompute(
            subjects[0], args.recompute_channels, args.epoch_minutes,
            config.PSD_WINDOW_SEC, config.PSD_OVERLAP_FRAC, bin_edges)

    report['verdict'] = _verdict(report)

    out_dir = (config.validation_dir(config.FEATURE_LEVEL_ROOT) / 'dtype_audit'
               / f"{args.label}_{ts[:19].replace(':', '')}")
    _write_outputs(
        report, accumulation_df, out_dir,
        f"P0.6 dtype audit: float32 cache round-trip + epoch-average precision, "
        f"{len(subjects)} subject(s)")

    # Non-zero exit only when the STORAGE question fails, which is the one P0.6
    # asks. View-layer requirements are findings to act on in P1.1/P1.3, not
    # reasons to call this run a failure.
    if not report['verdict']['float32_storage_validated']:
        raise SystemExit('dtype audit did NOT validate float32 storage: '
                         f"failed={report['verdict']['failed_storage_checks']}, "
                         f"leg_d_ran={report['leg_d_recompute'].get('ran')}")


if __name__ == '__main__':
    main()
