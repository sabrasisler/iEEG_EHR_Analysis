"""The correctness gate for the full-resolution epoch PSD. Run BEFORE the array.

    python -m ieeg_ehr.features.fullres_psd_audit --subjects 019 --n-epochs 3

Two SEPARATE checks, because they test different things and only one of them has
a tolerance. Do not merge them into a single "close enough" comparison: that is
precisely what would let a one-window time misalignment hide behind float32
rounding.

CHECK A -- EXACT REPRODUCTION.  Re-read the raw signal the way
`build_pain_epoch_fullres_psd` does, then reduce the resulting float64-accumulated
spectrogram to 50 log bins with THE PRODUCTION FUNCTIONS
(`bipolar_reref.log_bin_edges` + `_band_average_linear`), and compare against the
on-disk `psd_epochs` cache. **This must be BIT-EXACT.**

  It is the single highest-value test in this change, because one comparison
  proves all of: the (n_windows + 1)-second read length, the epoch's position in
  the run, the bipolar pair ordering, `detrend='constant'`, `scaling='density'`,
  the Hann window, and the log transform. If the read were one window short, or
  the pairs in a different order, or the epoch off by a second, this check fails
  loudly. A mismatch here is a BUG, not a tolerance question.

CHECK B -- ROUND-TRIP THROUGH THE STORED GRID.  Re-bin from the float32 log values
this project actually STORES (`10**x` in float64, average in float64, `log10`
back) and compare to the same target. This is NOT bit-exact: it adds one float32
log-domain round trip that check A does not have. It answers a different question
-- "does storing the native grid at float32 cost anything a band average would
notice" -- and P0.6 already puts the expected scale at ~2.5e-07 fractional error
in linear power. Reported, with a loose assertion, rather than asserted tight.

WHY CHECK A CAN BE BIT-EXACT AT ALL. The extractor deliberately calls
`signal.spectrogram` per channel on the float32 bipolar trace, exactly as
`bipolar_reref._welch_one_channel` does. Feeding float64 or batching the FFT over
an axis would both be defensible and slightly more accurate, and would both turn
this gate into an approximate one. Matching production was worth more.

NOTE ON THE FLOAT32 ACCUMULATOR. `_band_average_linear` writes into a float64
array but computes `psd[mask].mean()` on a float32 slice, so the within-bin
average accumulates in float32 -- the exact pattern `config/cache_params.py`
warns views about. That is REPRODUCED here, not fixed: this module measures the
existing pipeline, it does not improve it.
"""

import argparse
import logging
import sys

import numpy as np
import pandas as pd

from ieeg_ehr import config, io
from ieeg_ehr.features import build_pain_epoch_fullres_psd as fullres
from ieeg_ehr.io import nwb as nwb_io
from ieeg_ehr.preprocessing import bipolar_reref
from ieeg_ehr.views import cache_reader

logger = logging.getLogger(__name__)

#: Check B's ceiling, in log10 power units, on the PER-WINDOW comparison.
#:
#: MEASURED 2026-09-15 on sub-019: 2.25e-06 in log10, i.e. 5.2e-06 fractional in
#: linear power. Do NOT compare that directly against P0.6's 2.5e-07 and conclude
#: something is wrong -- they measure different things, and they agree:
#:
#:   - P0.6 measured EPOCH AVERAGES. Per-window float32 rounding is independent,
#:     so it averages DOWN over ~300 windows: 5.2e-06 / sqrt(300) = 3.0e-07,
#:     which reproduces P0.6's number almost exactly. `check_b_max_frac_linear`
#:     and P0.6 are the same measurement at two stages of the chain.
#:   - The per-window floor is set by float32's ~7 SIGNIFICANT digits against a
#:     stored log10 value of magnitude ~20 (iEEG log power in V^2/Hz is very
#:     negative). That leaves ~1e-06 of absolute resolution in log10 units, so
#:     2.25e-06 is at the representation floor, not above it. Nothing is
#:     recoverable here by being more careful; only float64 storage would move it,
#:     and P0.6 already decided that trade.
#:
#: 1e-5 is therefore ~4x the measured floor: tight enough to catch a real problem
#: (a wrong bin, a dropped frequency, a shifted axis all produce errors orders of
#: magnitude larger) and loose enough not to fail on a platform's log10 last bit.
CHECK_B_MAX_ABS_LOG10 = 1e-5

#: The epoch-averaged counterpart, which is the number REAL USE cares about --
#: every view averages over the epoch before anything looks at it, so this is the
#: error that actually reaches a figure. Expected ~3e-07 fractional; 2e-06 in
#: log10 is generous headroom.
CHECK_B_MAX_ABS_LOG10_EPOCH_MEAN = 2e-6


def _open_target(subject, session, defs, n_bins, epoch_minutes=None):
    """(ParquetFile, {epoch_id: [row_group, ...]}) for the EXISTING 50-bin cache.

    `verify_layout` is called ONCE per subject-session, not per epoch: it walks
    every row group and reads the first epoch's index columns to prove the ravel
    order. It also returns a MAP, because an epoch is not one row group -- pyarrow
    splits a `write_table` call above ~1,048,576 rows, so a 300-window subject
    spans two row groups per epoch from 70 pairs upward.
    """
    import pyarrow.parquet as pq

    path = config.pain_epoch_cache_path(subject, session, epoch_minutes)
    if not path.exists():
        raise FileNotFoundError(f'no 50-bin cache at {path}; check A has no target.')
    pf = pq.ParquetFile(path)
    return pf, cache_reader.verify_layout(pf, defs, n_bins)


def _target_block(pf, row_group_map, epoch_row, n_bins):
    """One epoch of the 50-bin cache as (n_win, n_pairs, n_bins) float64.

    Goes through `cache_reader.read_epoch` rather than reading the Parquet
    directly, so the C-order ravel is decoded by the same code the view layer
    uses -- a bespoke reshape here could agree with itself and disagree with
    production, which is the one failure this whole module exists to catch.
    """
    epoch_id = int(epoch_row['epoch_id'])
    if epoch_id not in row_group_map:
        raise KeyError(f'epoch {epoch_id} has no row groups in the 50-bin cache')
    return cache_reader.read_epoch(pf, epoch_row, n_bins, row_group_map[epoch_id])


def _rebin_from_sxx(freqs_full, sxx, bin_edges):
    """(n_freq, n_win, n_pairs) float32 linear PSD -> (n_win, n_pairs, n_bins) log10.

    Calls the PRODUCTION `_band_average_linear` per (window, channel), which is
    what makes check A exact. Slow -- a Python loop over n_win * n_pairs -- and
    that is fine: this runs on a handful of epochs, not on the cohort.
    """
    n_freq, n_win, n_pairs = sxx.shape
    n_bins = len(bin_edges) - 1
    out = np.empty((n_win, n_pairs, n_bins), dtype=np.float32)
    for w in range(n_win):
        for c in range(n_pairs):
            linear = bipolar_reref._band_average_linear(
                freqs_full, sxx[:, w, c], bin_edges)
            with np.errstate(divide='ignore'):
                out[w, c, :] = np.log10(linear)
    return out


def _rebin_from_stored_log(freqs_stored, stored_log, bin_edges):
    """Check B's path: re-bin from the float32 LOG values the cache stores.

    Exponentiate in CACHE_LINEAR_DOMAIN_DTYPE and average in
    CACHE_ACCUMULATE_DTYPE -- the two float64 rules views are held to. Note this
    is deliberately NOT bit-comparable to `_rebin_from_sxx`: that one reproduces
    production's float32 within-bin accumulator, this one does the numerically
    correct thing. The gap between them IS the measurement.

    The nearest-frequency fallback is reproduced too, including production's use
    of the ARITHMETIC bin midpoint `0.5*(lo+hi)` even though every "bin centre"
    elsewhere in the codebase is the geometric `sqrt(lo*hi)`.
    """
    n_freq, n_win, n_pairs = stored_log.shape
    n_bins = len(bin_edges) - 1
    linear = np.power(10.0, stored_log.astype(config.CACHE_LINEAR_DOMAIN_DTYPE))
    out = np.empty((n_win, n_pairs, n_bins), dtype=config.CACHE_ACCUMULATE_DTYPE)
    for b in range(n_bins):
        lo, hi = bin_edges[b], bin_edges[b + 1]
        mask = (freqs_stored >= lo) & (freqs_stored < hi)
        if mask.any():
            # linear is (n_freq, n_win, n_pairs); mean over the selected
            # frequencies already leaves (n_win, n_pairs), which is out's slice
            # shape. No transpose.
            out[:, :, b] = linear[mask].mean(
                axis=0, dtype=config.CACHE_ACCUMULATE_DTYPE)
        else:
            nearest = int(np.argmin(np.abs(freqs_stored - 0.5 * (lo + hi))))
            out[:, :, b] = linear[nearest]
    with np.errstate(divide='ignore'):
        return np.log10(out)


def audit_subject(subject, session='01', n_epochs=3, epoch_minutes=None):
    """Returns a dict of measured results. Does not assert; the caller decides."""
    epoch_minutes = (config.EPOCH_MINUTES_BEFORE if epoch_minutes is None
                     else epoch_minutes)

    manifest = io.read_manifest(config.pain_epoch_unit_dir(epoch_minutes))
    if manifest is None:
        raise FileNotFoundError(
            f'no manifest for the 50-bin unit at '
            f'{config.pain_epoch_unit_dir(epoch_minutes)}; check A has no target.')
    # From the manifest, NOT config.PSD_N_LOG_BINS: the target cache was written
    # under whatever the constant said THEN, and this audit must describe what is
    # on disk. (This is the fix views/build_pain_epoch_view.py:57 still needs.)
    bin_edges = np.asarray(manifest['bin_edges_hz'], dtype=float)
    n_bins = len(bin_edges) - 1

    defs = cache_reader.load_defs(subject, session, epoch_minutes)
    defs = defs.sort_values('epoch_id').reset_index(drop=True)
    runs = fullres._run_paths(subject, session)

    meta_path = config.pain_epoch_channel_meta_path(subject, session, epoch_minutes)
    channel_meta = io.read_table(meta_path, on_stale='ignore') if meta_path.exists() else None

    pf, row_group_map = _open_target(subject, session, defs, n_bins, epoch_minutes)

    results = []
    for _, ep in defs.head(n_epochs).iterrows():
        epoch_id = int(ep['epoch_id'])
        run_id = str(ep['run_id']).replace('run-', '')
        raw_path = runs.get(run_id)
        if raw_path is None:
            logger.warning('epoch %d: run-%s not in the registry, skipping',
                           epoch_id, run_id)
            continue

        n_windows = int(ep['n_windows'])
        hop_sec = float(ep['hop_sec'])
        dur_sec = (n_windows + 1) * hop_sec

        data_v, _, sfreq, elec_df, _ = nwb_io.load_window_with_electrodes(
            raw_path, float(ep['epoch_start_sec']), dur_sec, series_name=None)
        nperseg, noverlap, hop_samples = fullres.window_geometry(sfreq)
        if data_v.shape[0] != (n_windows + 1) * hop_samples:
            logger.warning('epoch %d: short read, skipping', epoch_id)
            continue

        pairs, _ = bipolar_reref.create_bipolar_pairs(elec_df)
        channels = [p['location'] for p in pairs]
        expected = fullres._expected_pairs(channel_meta, run_id)
        pair_order_ok = None if expected is None else (list(expected) == channels)

        bipolar_v = bipolar_reref.rereference(data_v, elec_df.index.to_numpy(), pairs)
        del data_v
        freqs_full, sxx = fullres.epoch_spectrogram(bipolar_v, sfreq, nperseg, noverlap)
        del bipolar_v

        target = _target_block(pf, row_group_map, ep, n_bins)
        if target.shape != (n_windows, len(channels), n_bins):
            raise RuntimeError(
                f'epoch {epoch_id}: target block is {target.shape}, expected '
                f'{(n_windows, len(channels), n_bins)}. Shapes must agree before '
                'any value comparison is meaningful.')

        # ---- CHECK A: bit-exact ----
        rebin_a = _rebin_from_sxx(freqs_full, sxx, bin_edges)
        finite = np.isfinite(target) & np.isfinite(rebin_a)
        exact = bool(np.array_equal(rebin_a[finite], target[finite]))
        diff_a = (np.abs(rebin_a[finite].astype(np.float64)
                         - target[finite].astype(np.float64)).max()
                  if finite.any() else 0.0)
        ulps_a = _max_ulps(rebin_a[finite], target[finite]) if finite.any() else 0

        # ---- CHECK B: float32 round trip through the stored 499 grid ----
        sl, freqs_stored = fullres.freq_slice(freqs_full)
        with np.errstate(divide='ignore'):
            stored = np.log10(sxx[sl]).astype(config.CACHE_FLOAT_DTYPE)
        rebin_b = _rebin_from_stored_log(freqs_stored, stored, bin_edges)
        # Only bins the stored 1-250 Hz grid can actually represent. Bin 49 is
        # [223.86, 250.0) and IS covered; a bin straddling the edge would not be.
        cmp_b = finite & np.isfinite(rebin_b)
        diff_b = (np.abs(rebin_b[cmp_b] - target[cmp_b].astype(np.float64)).max()
                  if cmp_b.any() else 0.0)
        frac_b = _max_frac_linear(rebin_b[cmp_b], target[cmp_b]) if cmp_b.any() else 0.0

        # The EPOCH-AVERAGED version of check B, which is the error that actually
        # reaches a figure: every view averages over the epoch before anything
        # looks at it, and independent per-window rounding averages down by
        # ~sqrt(n_windows). Done in float64 on both sides (10**log -> mean ->
        # log10), the same linear-then-log the view layer uses, so this is not
        # just the per-window number scaled.
        mean_b = _epoch_mean_log(rebin_b)
        mean_t = _epoch_mean_log(target.astype(np.float64))
        cmp_m = np.isfinite(mean_b) & np.isfinite(mean_t)
        diff_bm = float(np.abs(mean_b[cmp_m] - mean_t[cmp_m]).max()) if cmp_m.any() else 0.0
        frac_bm = _max_frac_linear(mean_b[cmp_m], mean_t[cmp_m]) if cmp_m.any() else 0.0

        results.append({
            'subject': subject, 'session': session, 'epoch_id': epoch_id,
            'run_id': run_id, 'sfreq': sfreq, 'n_windows': n_windows,
            'n_pairs': len(channels), 'n_bins': n_bins,
            'n_freqs_stored': len(freqs_stored),
            'pair_order_ok': pair_order_ok,
            'check_a_bit_exact': exact,
            'check_a_max_abs_log10': float(diff_a),
            'check_a_max_ulps': int(ulps_a),
            'check_b_max_abs_log10': float(diff_b),
            'check_b_max_frac_linear': float(frac_b),
            'check_b_epochmean_max_abs_log10': diff_bm,
            'check_b_epochmean_max_frac_linear': float(frac_bm),
            'n_nonfinite_target': int((~np.isfinite(target)).sum()),
        })
        logger.info(
            'epoch %d (run-%s, %d win x %d pairs @ %g Hz): '
            'A bit-exact=%s (max|dlog10|=%.3g, %d ulp) | B per-window %.3g '
            '(frac %.3g), epoch-mean %.3g (frac %.3g) | pair order %s',
            epoch_id, run_id, n_windows, len(channels), sfreq,
            exact, diff_a, ulps_a, diff_b, frac_b, diff_bm, frac_bm,
            {True: 'OK', False: 'MISMATCH', None: 'unchecked'}[pair_order_ok])
        del sxx, rebin_a, rebin_b, stored, target

    return results


def _max_ulps(a, b):
    """Max float32 ulp distance, so a near-miss reads as '2 ulp' not '3e-07'.

    The distinction matters: a handful of ulps is FFT/compiler noise, while a
    time misalignment or a pair swap produces differences of order the signal.
    """
    a32 = np.asarray(a, dtype=np.float32).view(np.int32).astype(np.int64)
    b32 = np.asarray(b, dtype=np.float32).view(np.int32).astype(np.int64)
    # Map sign-magnitude to a monotone ordering so the subtraction is meaningful
    # across zero.
    a32 = np.where(a32 < 0, np.int64(-2)**31 - a32, a32)
    b32 = np.where(b32 < 0, np.int64(-2)**31 - b32, b32)
    return int(np.abs(a32 - b32).max()) if a32.size else 0


def _epoch_mean_log(block_log):
    """(n_win, n_pairs, n_bins) log10 -> (n_pairs, n_bins) log10 of the mean.

    Linear-then-log in float64, the same order `views.axes.epoch_mean` uses for
    raw log power -- averaging the logs directly would be a geometric mean and a
    different quantity, so comparing one against the other would measure the
    Jensen gap rather than the float32 round trip.
    """
    linear = np.power(10.0, np.asarray(block_log, dtype=np.float64))
    with np.errstate(divide='ignore'):
        return np.log10(np.nanmean(linear, axis=0, dtype=np.float64))


def _max_frac_linear(log_a, log_b):
    """Max fractional difference in LINEAR power, the unit P0.6 reported in."""
    d = np.abs(np.asarray(log_a, dtype=np.float64)
               - np.asarray(log_b, dtype=np.float64))
    return float(np.max(np.power(10.0, d) - 1.0)) if d.size else 0.0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--subjects', nargs='+', required=True)
    ap.add_argument('--session', default='01')
    ap.add_argument('--n-epochs', type=int, default=3,
                    help='Epochs per subject-session to audit (default 3). Each '
                         'runs a Python loop over n_win x n_pairs bin averages, '
                         'so this is deliberately not the whole cache.')
    ap.add_argument('--epoch-minutes', type=float, default=None)
    ap.add_argument('--allow-inexact', action='store_true',
                    help='Report check A instead of failing on it. For diagnosing '
                         'a mismatch, NEVER for clearing the gate.')
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)s %(message)s')

    rows = []
    for s in args.subjects:
        rows += audit_subject(s.replace('sub-', ''), args.session,
                              n_epochs=args.n_epochs,
                              epoch_minutes=args.epoch_minutes)

    if not rows:
        logger.error('no epochs audited -- the gate has not been cleared')
        return 2

    df = pd.DataFrame(rows)
    print()
    print(df.to_string(index=False))
    print()

    failures = []
    if not df['check_a_bit_exact'].all():
        bad = df[~df['check_a_bit_exact']]
        failures.append(
            f'CHECK A NOT BIT-EXACT for {len(bad)}/{len(df)} epochs '
            f'(max {bad["check_a_max_ulps"].max()} ulp, '
            f'max|dlog10| {bad["check_a_max_abs_log10"].max():.3g}). '
            'If the ulp count is small this is FFT/library noise; if it is large, '
            'suspect the read length, the epoch offset, or the pair order.')
    if (df['pair_order_ok'] == False).any():  # noqa: E712 -- None must not match
        failures.append('PAIR ORDER MISMATCH against channel_meta.')
    if df['pair_order_ok'].isna().any():
        logger.warning('%d epochs had no channel_meta to check pair order against',
                       int(df['pair_order_ok'].isna().sum()))
    worst_b = df['check_b_max_abs_log10'].max()
    worst_bm = df['check_b_epochmean_max_abs_log10'].max()
    if worst_b > CHECK_B_MAX_ABS_LOG10:
        failures.append(
            f'CHECK B per-window max|dlog10| {worst_b:.3g} exceeds '
            f'{CHECK_B_MAX_ABS_LOG10:.0e}. The float32 floor here is ~2e-06 (7 '
            'significant digits against a stored log10 magnitude of ~20), so an '
            'excess of this size is a wrong bin or a dropped frequency, not rounding.')
    if worst_bm > CHECK_B_MAX_ABS_LOG10_EPOCH_MEAN:
        failures.append(
            f'CHECK B epoch-mean max|dlog10| {worst_bm:.3g} exceeds '
            f'{CHECK_B_MAX_ABS_LOG10_EPOCH_MEAN:.0e}. This is the error that '
            'reaches a figure, so it is the one to take seriously.')

    n_win = int(df['n_windows'].median())
    print(f'CHECK A  bit-exact: {int(df["check_a_bit_exact"].sum())}/{len(df)} epochs'
          f'  (max {int(df["check_a_max_ulps"].max())} ulp)')
    print(f'CHECK B  per-window  max |dlog10| = {worst_b:.3g}   '
          f'(frac linear {df["check_b_max_frac_linear"].max():.3g})')
    print(f'         epoch-mean  max |dlog10| = {worst_bm:.3g}   '
          f'(frac linear {df["check_b_epochmean_max_frac_linear"].max():.3g})')
    print(f'         per-window / sqrt({n_win}) = '
          f'{df["check_b_max_frac_linear"].max() / np.sqrt(n_win):.3g}  '
          f'-- P0.6 measured 2.5e-07 on epoch averages, so these should agree: '
          'independent per-window rounding averages down over the epoch.')
    print()

    if failures:
        for f in failures:
            logger.error(f)
        if args.allow_inexact and all('CHECK A' in f for f in failures):
            logger.warning('--allow-inexact given; NOT treating this as a pass')
        return 1

    print('GATE CLEARED.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
