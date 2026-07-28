#!/usr/bin/env python3
"""
Feature-level QC, metric stage (P2.1): per-channel/per-bin session baselines of
log-power, plus the per-window order statistics that make the K and B thresholds
sweepable without ever re-reading an NWB.

THE RULE this serves (config/feature_qc_params.py owns the numbers):

    z[window, channel, bin] = (log_power - mean[channel, bin]) / std[channel, bin]
    window flagged  <=>  frac(z > K) > B          K = 5.0, B = 0.20

WHY TWO PASSES OVER EACH RUN
----------------------------
The baseline is session-wide, so no window's z is knowable until every window has
been seen. Unlike detect_gross_artifact.py — which dodges its second pass by
caching a handful of floats per channel/window while the trace is in memory — the
quantity needed here is per (window, channel, BIN), i.e. as large as the PSD
itself. Caching it would mean holding the whole session in memory. So: pass 1
accumulates sums, pass 2 computes z. The input is the already-cheap stored PSD
(not raw voltage), and HDF5 slicing means neither pass ever holds more than one
time-chunk, so the second read is affordable in a way it would not be upstream.

WHAT LANDS ON DISK  (four tables; see config/paths.py FEATURE-LEVEL QC)
----------------------------------------------------------------------
  baseline/    per (channel, bin) -- mean, std, counts, degenerate flag. The
               metric proper. Tiny.
  per_window/  per (run, channel, window) -- z order statistics, SPARSE: only
               rows whose z_max exceeds FEATURE_METRIC_STORE_FLOOR.
  summary/     per (run, channel) -- n_windows / n_stored / n_nonfinite /
               n_mask_excluded. The denominator for every rate, so the sparse table
               above cannot silently overstate cleanliness.
  zhist/       per (run, channel, mask_excluded) -- histogram of the operative z
               order statistic on a fixed grid, so distribution SHAPE (the knee
               P2.1 needs for structural threshold-setting) survives sparsity.

WHICH MASK IS SUBTRACTED
------------------------
Windows already excluded by the UPSTREAM mask are DROPPED FROM THE BASELINE
(config.FEATURE_BASELINE_EXCLUDES_RAW_VOLTAGE) but still get metrics computed and
a `mask_excluded` flag recorded — that is what makes "how much of what this
detector flags was already flagged upstream?" answerable, which is one of the
structural criteria for setting K.

`--mask-level` chooses that upstream mask, and DEFAULTS TO `bipolar`:

    bipolar      qc/bipolar/masks/<label>/*.parquet, keyed on bipolar PAIRS
                 excluded = (raw_voltage[anode] | raw_voltage[cathode])
                            | bipolar_variance
    raw_voltage  qc/raw_voltage/masks/<label>/*.csv, keyed on MONOPOLAR contacts

The bipolar mask is a strict SUPERSET (verified on real data: 16.60% vs 13.91% of
windows on sub-039 run-DA1003FI, with zero windows excluded upstream but missed
here) and is already keyed the way the PSD is, so no anode/cathode translation is
applied. The metric tables are scoped on disk by BOTH level and label
(`bp-<label>` vs `rv-<label>`), because the same channel gets a different mean/std
depending on which mask was subtracted first — so the two cannot collide.

Hence `mask_excluded` rather than `rv_excluded`: with the default level the flag
means "the bipolar mask excluded this window", which is not the same claim.

Run via Slurm (never the login node):
    sbatch sbatch/detect_power_outlier_array.sbatch
    python -m ieeg_ehr.qc.feature_level.detect_power_outlier --subjects 085
"""

import argparse
import json
import logging

import numpy as np
import pandas as pd
from pynwb import NWBHDF5IO

from ieeg_ehr import config, io
from ieeg_ehr.qc import mask_projection

logger = logging.getLogger(__name__)

# Windows per HDF5 slice. Bounds memory to roughly
# CHUNK * n_pairs * n_bins * 8 bytes for the float64 working copy -- ~330 MB at
# 4096 x 200 x 50, which is why this is a constant and not the whole run.
CHUNK_WINDOWS = 4096


def _open_psd(nwb_path):
    """Open one run's PSD NWB and return (handle, decomposition, metadata).

    The handle stays OPEN so the caller can slice `decomp.data` lazily; closing it
    would invalidate the dataset. Caller must close.
    """
    handle = NWBHDF5IO(str(nwb_path), 'r')
    nwb = handle.read()
    decomp = nwb.processing['ecephys']['psd_log_bins']
    bands = decomp.bands.to_dataframe()
    lo = bands['band_limits'].apply(lambda t: t[0]).to_numpy(dtype=float)
    hi = bands['band_limits'].apply(lambda t: t[1]).to_numpy(dtype=float)
    elec = nwb.electrodes.to_dataframe()
    meta = {
        'channels': list(elec['location']),
        'bin_low_hz': lo,
        'bin_high_hz': hi,
        'contains_line_noise': bands['contains_line_noise'].to_numpy(dtype=bool),
        'rate': float(decomp.rate),
        'starting_time': float(decomp.starting_time),
        'n_time': int(decomp.data.shape[0]),
    }
    meta['run_seconds'] = meta['starting_time'] + np.arange(meta['n_time']) / meta['rate']
    return handle, decomp, meta


def _chunks(n_time, size=CHUNK_WINDOWS):
    for t0 in range(0, n_time, size):
        yield t0, min(t0 + size, n_time)


def expected_psd_rate():
    """The PSD row rate implied by the current window/overlap design (1.0 Hz for
    2s windows at 50% overlap)."""
    hop_sec = config.PSD_WINDOW_SEC * (1.0 - config.PSD_OVERLAP_FRAC)
    return 1.0 / hop_sec


def resolve_mask_path(subject, session, mask_level, mask_label):
    """Where this level's mask for one subject/session lives.

    The two levels differ in BOTH location and file format, and load_mask
    dispatches on the extension:
      raw_voltage  masks/<label>/sub-X_ses-Y.csv       (predates the Parquet rule)
      bipolar      masks/<label>/sub-X_ses-Y.parquet   (new artifact)
    """
    if mask_level == 'bipolar':
        return config.bipolar_mask_path(subject, session, mask_label)
    if mask_level == 'raw_voltage':
        return config.mask_csv(subject, session, mask_label)
    raise ValueError(f'unknown mask level: {mask_level}')


def resolve_mask(subject, session, mask_level, mask_label):
    """(mask_df, projector, path) for one subject/session; mask_df None if absent.

    Owns the level -> (path, keying, projector) mapping in ONE place, because the
    two levels differ in all three at once and picking the wrong projector fails
    silently rather than loudly:

      raw_voltage  CSV,     MONOPOLAR contacts -> project_to_pairs (anode|cathode)
      bipolar      Parquet, bipolar PAIRS      -> project_pair_mask_to_windows

    Handing a pair-keyed table to project_to_pairs would split 'LA1-LA2' into
    'LA1'/'LA2', match neither, and return all-False -- "nothing is excluded",
    the worst failure mode for a QC mask. Hence one resolver, not a flag threaded
    through the call sites.
    """
    path = resolve_mask_path(subject, session, mask_level, mask_label)
    projector = (mask_projection.project_pair_mask_to_windows if mask_level == 'bipolar'
                 else mask_projection.project_to_pairs)
    return mask_projection.load_mask(path), projector, path


class _Accumulator:
    """Per (channel, bin) sum / sumsq / count over usable windows.

    Keyed by channel NAME, not column index: a session's runs are not guaranteed
    to expose the same electrode table, and silently aligning by position would
    average two different contacts together.
    """

    def __init__(self, n_bins):
        self.n_bins = n_bins
        self.n = {}
        self.total = {}
        self.sq = {}
        self.n_nonfinite = {}

    def _slot(self, channel):
        if channel not in self.n:
            self.n[channel] = np.zeros(self.n_bins, dtype=np.int64)
            self.total[channel] = np.zeros(self.n_bins, dtype=np.float64)
            self.sq[channel] = np.zeros(self.n_bins, dtype=np.float64)
            self.n_nonfinite[channel] = np.zeros(self.n_bins, dtype=np.int64)
        return channel

    def add(self, channels, block, usable, nonfinite):
        """block/usable/nonfinite: (n_win, n_pairs, n_bins)."""
        x = np.where(usable, block, 0.0)
        n_add = usable.sum(axis=0)
        sum_add = x.sum(axis=0)
        sq_add = np.square(x).sum(axis=0)
        nf_add = nonfinite.sum(axis=0)
        for j, channel in enumerate(channels):
            self._slot(channel)
            self.n[channel] += n_add[j]
            self.total[channel] += sum_add[j]
            self.sq[channel] += sq_add[j]
            self.n_nonfinite[channel] += nf_add[j]

    def finalize(self, min_windows):
        """{channel: (mean, std, n, n_nonfinite, degenerate)} with mean/std as
        (n_bins,) float64 arrays.

        A channel-bin with fewer than `min_windows` usable windows, or with
        std <= 0, is marked degenerate. Same convention as gross_artifact's
        "degenerate std -> excluded" (build_exclusions.py:236): downstream treats
        a degenerate baseline as ALWAYS flagged, so a channel with no usable
        baseline fails loudly into the cascade instead of producing NaN z-scores
        that comparison operators quietly swallow.
        """
        out = {}
        for channel, n in self.n.items():
            with np.errstate(invalid='ignore', divide='ignore'):
                mean = np.where(n > 0, self.total[channel] / np.maximum(n, 1), np.nan)
                var = np.where(n > 1, self.sq[channel] / np.maximum(n, 1) - mean ** 2, np.nan)
                std = np.sqrt(np.clip(var, 0.0, None))
            degenerate = (n < min_windows) | ~np.isfinite(std) | (std <= 0)
            out[channel] = (mean, std, n, self.n_nonfinite[channel], degenerate)
        return out


def _order_stats(z, idx_by_frac):
    """Descending order statistics of z over its last axis.

    z: (n_win, n_pairs, n_keep). Returns {frac: (n_win, n_pairs)} plus 'max'.

    The exact identity being used: with n usable bins,
        frac(z > K) > B   <=>   sorted_desc(z)[floor(B * n)] > K
    because sorted_desc[k] > K implies exactly k+1 bins exceed K, and
    (floor(B*n) + 1) / n > B for every n. So comparing the stored order statistic
    to K reproduces the fraction rule exactly -- no interpolated quantile, which
    would NOT reproduce it.
    """
    asc = np.sort(z, axis=-1)
    n_keep = asc.shape[-1]
    out = {'max': asc[..., -1]}
    for frac, k in idx_by_frac.items():
        out[frac] = asc[..., n_keep - 1 - k]
    return out


def process_subject_session(subject, session, mask_label, mask_level=None, overwrite=False):
    z_thresh = config.FEATURE_Z_THRESH
    bin_frac = config.FEATURE_BIN_FRAC
    frac_grid = tuple(config.FEATURE_BIN_FRAC_GRID)

    baseline_path = config.feature_metrics_path('baseline', subject, session,
                                                mask_label, mask_level)
    if baseline_path.exists() and not overwrite:
        logger.info('sub-%s ses-%s: metrics exist, skipping (use --overwrite)', subject, session)
        return None

    registry = pd.read_csv(config.FILE_REGISTRY_CSV)
    rows = registry[(registry.sub_id == f'sub-{subject}') & (registry.ses_id == f'ses-{session}')]
    run_ids = [str(r).replace('run-', '') for r in rows.run_id.unique()]
    runs = [(rid, config.bipolar_psd_nwb_path(subject, session, rid)) for rid in run_ids]
    runs = [(rid, p) for rid, p in runs if p.exists()]
    if not runs:
        logger.warning('sub-%s ses-%s: no bipolar_fft runs on disk, skipping', subject, session)
        return None

    mask_df, project, mask_path = resolve_mask(subject, session, mask_level, mask_label)
    if mask_label and mask_df is None:
        logger.warning('sub-%s ses-%s: no %s mask at %s -- baseline will be UNMASKED for this '
                       'subject/session, which INFLATES its std and makes the detector LESS '
                       'sensitive here than elsewhere', subject, session, mask_level, mask_path)
    if not mask_label:
        project = mask_projection.project_to_pairs      # all-False on mask_df=None

    # ---------------------------------------------------------------- pass 1
    acc = None
    bin_meta = None
    per_run_counts = {}
    # Observed PSD rate per run, recorded rather than gated on: the hop audit is
    # done and the stale-hop subjects are being re-run, but a rate that is not the
    # expected one should still be visible in run_info rather than invisible.
    run_rates = {}
    for run_id, path in runs:
        handle, decomp, meta = _open_psd(path)
        try:
            if acc is None:
                acc = _Accumulator(len(meta['contains_line_noise']))
                bin_meta = meta
            excl = project(mask_df, f'run-{run_id}',
                           meta['channels'], meta['run_seconds'])
            run_rates[f'run-{run_id}'] = meta['rate']
            counts = per_run_counts.setdefault(run_id, {})
            for t0, t1 in _chunks(meta['n_time']):
                block = np.asarray(decomp.data[t0:t1], dtype=np.float64)
                nonfinite = ~np.isfinite(block)
                usable = ~nonfinite & ~excl[t0:t1, :, None]
                acc.add(meta['channels'], block, usable, nonfinite)
            for j, channel in enumerate(meta['channels']):
                counts[channel] = {
                    'n_windows': int(meta['n_time']),
                    'n_mask_excluded': int(excl[:, j].sum()),
                }
        finally:
            handle.close()

    baseline = acc.finalize(config.FEATURE_MIN_BASELINE_WINDOWS)

    keep_bins = np.ones(acc.n_bins, dtype=bool)
    if config.FEATURE_EXCLUDE_LINE_NOISE_BINS:
        keep_bins = ~bin_meta['contains_line_noise']
    n_keep = int(keep_bins.sum())
    if n_keep == 0:
        logger.error('sub-%s ses-%s: every bin is line-noise flagged, nothing to threshold',
                     subject, session)
        return None
    idx_by_frac = {f: int(np.floor(f * n_keep)) for f in frac_grid}

    hist_edges = np.linspace(*config.FEATURE_ZHIST_RANGE, config.FEATURE_ZHIST_BINS + 1)

    # ---------------------------------------------------------------- pass 2
    window_rows = []
    zhist = {}
    flag_counts = {}          # (run_id, channel) -> [n_flagged, n_flagged_not_rv]
    for run_id, path in runs:
        handle, decomp, meta = _open_psd(path)
        try:
            channels = meta['channels']
            excl = project(mask_df, f'run-{run_id}', channels, meta['run_seconds'])
            mean = np.stack([baseline[c][0] for c in channels])[:, keep_bins]
            std = np.stack([baseline[c][1] for c in channels])[:, keep_bins]
            degen = np.stack([baseline[c][4] for c in channels])[:, keep_bins]
            safe_std = np.where(degen, 1.0, std)

            for t0, t1 in _chunks(meta['n_time']):
                block = np.asarray(decomp.data[t0:t1], dtype=np.float64)[:, :, keep_bins]
                nonfinite = ~np.isfinite(block)
                z = (block - mean[None]) / safe_std[None]
                if config.FEATURE_Z_SIDE == 'both':
                    z = np.abs(z)
                # Degenerate baseline or non-finite power -> treat as exceeding any
                # K, so it propagates through the cascade as flagged rather than
                # vanishing into a NaN comparison.
                z = np.where(nonfinite | degen[None], np.inf, z)

                stats = _order_stats(z, idx_by_frac)
                zop = stats[bin_frac]
                zmax = stats['max']

                # Flagged at the CONFIGURED (K, B), for the summary table. Counted
                # over every window, and again over only those the raw-voltage mask
                # had not already removed -- the second is this detector's
                # INCREMENTAL yield, which is one of the structural criteria for
                # setting K (P2.1) and is not recoverable from the sparse table
                # alone once rows are floored away.
                flagged = zop > z_thresh
                rv_chunk = excl[t0:t1]
                # A window with ANY non-finite bin is invalid data rather than a
                # high-power outlier, so the 20%-of-bins rule can pass it (15% of
                # bins at -inf does not trip B=0.20). Counted separately so the
                # size of that population is known before deciding whether it
                # deserves its own artifact type at this level.
                any_nonfinite = nonfinite.any(axis=-1)
                for j, channel in enumerate(channels):
                    key = (f'run-{run_id}', channel)
                    fl = flag_counts.setdefault(key, [0, 0, 0])
                    fl[0] += int(flagged[:, j].sum())
                    fl[1] += int((flagged[:, j] & ~rv_chunk[:, j]).sum())
                    fl[2] += int(any_nonfinite[:, j].sum())

                # Floor on the LARGEST stored order statistic (the smallest B in the
                # grid), not on z_max. Two reasons, both measured:
                #   - z_max is the max over ~44 bins, so it clears a 2 SD floor for
                #     most windows by construction (32% of sub-039's channel-windows
                #     on the first run of this script) -- it barely sparsifies.
                #   - Flooring on the smallest-B statistic guarantees NO CENSORING
                #     anywhere in FEATURE_BIN_FRAC_GRID: the stats are monotone in B
                #     (z_b05 >= z_b10 >= z_b20 >= z_b50), so a row whose largest
                #     statistic is below the floor cannot be flagged at any grid B
                #     for any K above the floor.
                keep = stats[min(frac_grid)] > config.FEATURE_METRIC_STORE_FLOOR
                if keep.any():
                    wi, pi = np.nonzero(keep)
                    rec = {
                        'run_id': np.full(wi.size, f'run-{run_id}'),
                        'channel': np.asarray(channels, dtype=object)[pi],
                        'window_idx': (t0 + wi).astype(np.int32),
                        'window_start_time': meta['run_seconds'][t0 + wi],
                        'z_max': zmax[keep].astype(np.float32),
                        'n_bins_nonfinite': nonfinite.sum(axis=-1)[keep].astype(np.int16),
                        'mask_excluded': excl[t0:t1][keep],
                    }
                    for f in frac_grid:
                        rec[f'z_b{f * 100:g}'] = stats[f][keep].astype(np.float32)
                    window_rows.append(pd.DataFrame(rec))

                # Histograms over the operative statistic, split by whether the
                # raw-voltage mask already excluded the window -- the two
                # populations answer different questions and must not be pooled.
                finite_zop = np.where(np.isfinite(zop), zop, config.FEATURE_ZHIST_RANGE[1])
                for j, channel in enumerate(channels):
                    for flag in (False, True):
                        sel = (excl[t0:t1, j] == flag)
                        if not sel.any():
                            continue
                        h, _ = np.histogram(finite_zop[sel, j], bins=hist_edges)
                        key = (f'run-{run_id}', channel, flag)
                        if key in zhist:
                            zhist[key] += h
                        else:
                            zhist[key] = h
        finally:
            handle.close()

    return _write_outputs(subject, session, mask_label, mask_level, baseline, bin_meta, keep_bins,
                          per_run_counts, window_rows, zhist, hist_edges,
                          z_thresh, bin_frac, frac_grid, idx_by_frac, runs, flag_counts,
                          mask_df is not None, run_rates)


def _write_outputs(subject, session, mask_label, mask_level, baseline, bin_meta, keep_bins,
                   per_run_counts, window_rows, zhist, hist_edges,
                   z_thresh, bin_frac, frac_grid, idx_by_frac, runs, flag_counts,
                   mask_applied, run_rates):
    params = {
        'z_thresh': z_thresh, 'bin_frac': bin_frac,
        'bin_frac_grid': list(frac_grid), 'z_side': config.FEATURE_Z_SIDE,
        'baseline_stat': 'mean_std',
        'baseline_excludes_raw_voltage': bool(config.FEATURE_BASELINE_EXCLUDES_RAW_VOLTAGE),
        'mask_level': mask_level,
        'mask_label': mask_label,
        # Whether that mask was actually FOUND for this subject/session. A missing
        # mask file is not fatal (build_mask.py legitimately drops subject/sessions
        # that lack an artifact type -- the sub-236 gap), but recording only the
        # requested label while silently computing an unmasked baseline would make
        # the sidecar assert something false. This also lands in config_hash, so
        # unmasked subject/sessions are findable rather than merely logged.
        'raw_voltage_mask_applied': bool(mask_applied),
        'min_baseline_windows': config.FEATURE_MIN_BASELINE_WINDOWS,
        'exclude_line_noise_bins': bool(config.FEATURE_EXCLUDE_LINE_NOISE_BINS),
        'store_floor': config.FEATURE_METRIC_STORE_FLOOR,
        'n_bins_kept': int(keep_bins.sum()),
        'order_stat_index_by_frac': {str(k): v for k, v in idx_by_frac.items()},
    }
    parents = [str(p) for _rid, p in runs]
    if mask_label:
        parents.append(str(resolve_mask_path(subject, session, mask_level, mask_label)))
    subjects = [f'sub-{subject}']

    # baseline: per (channel, bin)
    n_bins = len(bin_meta['contains_line_noise'])
    frames = []
    for channel, (mean, std, n, n_nf, degen) in baseline.items():
        frames.append(pd.DataFrame({
            'channel': channel,
            'bin': np.arange(n_bins, dtype=np.int16),
            'bin_low_hz': bin_meta['bin_low_hz'],
            'bin_high_hz': bin_meta['bin_high_hz'],
            'contains_line_noise': bin_meta['contains_line_noise'],
            'used_for_threshold': keep_bins,
            'mean_log_power': mean, 'std_log_power': std,
            'n_windows_used': n, 'n_nonfinite': n_nf, 'degenerate': degen,
        }))
    baseline_df = pd.concat(frames, ignore_index=True)
    io.write_table(baseline_df, config.feature_metrics_path('baseline', subject, session, mask_label, mask_level),
                   params=params, parents=parents, subjects=subjects)

    # summary: per (run, channel)
    stored = (pd.concat(window_rows, ignore_index=True)
              if window_rows else pd.DataFrame(columns=['run_id', 'channel']))
    n_stored = (stored.groupby(['run_id', 'channel']).size().rename('n_stored')
                if len(stored) else pd.Series(dtype=np.int64, name='n_stored'))
    srows = []
    for run_id, per_channel in per_run_counts.items():
        for channel, c in per_channel.items():
            _, _, n_used, n_nf, degen = baseline[channel]
            n_flagged, n_flagged_not_rv, n_any_nf = flag_counts.get(
                (f'run-{run_id}', channel), (0, 0, 0))
            srows.append({
                'run_id': f'run-{run_id}', 'channel': channel,
                'n_windows': c['n_windows'], 'n_mask_excluded': c['n_mask_excluded'],
                'n_nonfinite_values': int(n_nf.sum()),
                'n_bins_degenerate': int(degen.sum()),
                'n_baseline_windows_min': int(n_used.min()) if len(n_used) else 0,
                # At the CONFIGURED (K, B) recorded in this table's sidecar params.
                'n_flagged': int(n_flagged),
                'n_flagged_not_mask_excluded': int(n_flagged_not_rv),
                'n_windows_any_nonfinite': int(n_any_nf),
            })
    summary_df = pd.DataFrame(srows)
    if len(n_stored):
        summary_df = summary_df.merge(n_stored.reset_index(), on=['run_id', 'channel'], how='left')
    else:
        summary_df['n_stored'] = 0
    summary_df['n_stored'] = summary_df['n_stored'].fillna(0).astype(np.int64)
    io.write_table(summary_df, config.feature_metrics_path('summary', subject, session, mask_label, mask_level),
                   params=params, parents=parents, subjects=subjects)

    # per_window: sparse tail
    if len(stored):
        stored = stored.sort_values(['run_id', 'channel', 'window_idx']).reset_index(drop=True)
    io.write_table(stored, config.feature_metrics_path('per_window', subject, session, mask_label, mask_level),
                   params=params, parents=parents, subjects=subjects)

    # zhist: per (run, channel, mask_excluded)
    hrows = []
    centers = (hist_edges[:-1] + hist_edges[1:]) / 2
    for (run_id, channel, flag), counts in zhist.items():
        nz = np.nonzero(counts)[0]
        hrows.append(pd.DataFrame({
            'run_id': run_id, 'channel': channel, 'mask_excluded': flag,
            'z_bin_idx': nz.astype(np.int16), 'z_bin_center': centers[nz],
            'count': counts[nz].astype(np.int64),
        }))
    zhist_df = (pd.concat(hrows, ignore_index=True) if hrows
                else pd.DataFrame(columns=['run_id', 'channel', 'mask_excluded',
                                           'z_bin_idx', 'z_bin_center', 'count']))
    io.write_table(zhist_df, config.feature_metrics_path('zhist', subject, session, mask_label, mask_level),
                   params=dict(params, zhist_range=list(config.FEATURE_ZHIST_RANGE),
                               zhist_bins=config.FEATURE_ZHIST_BINS),
                   parents=parents, subjects=subjects)

    info_path = config.feature_metrics_run_info_path(subject, session)
    info_path.parent.mkdir(parents=True, exist_ok=True)
    info_path.write_text(json.dumps({
        'subject': subject, 'session': session, 'params': params,
        'runs': [str(p) for _rid, p in runs],
        'git': config.git_provenance(), 'run_timestamp': config.run_timestamp(),
        'n_channels': len(baseline), 'n_window_rows_stored': int(len(stored)),
        'psd_rate_hz_by_run': run_rates,
        'expected_psd_rate_hz': expected_psd_rate(),
    }, indent=2, default=str))

    n_total = int(summary_df['n_windows'].sum()) if len(summary_df) else 0
    n_flag = int(summary_df['n_flagged'].sum()) if len(summary_df) else 0
    n_flag_inc = int(summary_df['n_flagged_not_mask_excluded'].sum()) if len(summary_df) else 0
    n_rv = int(summary_df['n_mask_excluded'].sum()) if len(summary_df) else 0
    pct = (lambda x: 100.0 * x / n_total if n_total else 0.0)
    logger.info(
        'sub-%s ses-%s: %d channels, %d channel-windows | FLAGGED at K=%g B=%g: %d (%.3f%%), '
        'of which not already upstream-excluded: %d (%.3f%%) | %s-mask excluded: %d (%.3f%%) '
        '| any-nonfinite-bin: %d (%.3f%%) | stored %d rows (%.2f%% above floor %.1f) '
        '| %d degenerate channel-bins',
        subject, session, len(baseline), n_total, z_thresh, bin_frac,
        n_flag, pct(n_flag), n_flag_inc, pct(n_flag_inc), mask_level, n_rv, pct(n_rv),
        int(summary_df['n_windows_any_nonfinite'].sum()) if len(summary_df) else 0,
        pct(int(summary_df['n_windows_any_nonfinite'].sum()) if len(summary_df) else 0),
        len(stored), pct(len(stored)), config.FEATURE_METRIC_STORE_FLOOR,
        int(summary_df['n_bins_degenerate'].sum()) if len(summary_df) else 0)
    return config.feature_metrics_path('baseline', subject, session, mask_label, mask_level)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--subjects', required=True,
                    help='Comma- or space-separated subject IDs, e.g. 071,085')
    ap.add_argument('--session', default='all',
                    help="Session ID, or 'all' (default) for every session in the registry.")
    ap.add_argument('--mask-level', default=config.FEATURE_BASELINE_MASK_LEVEL,
                    choices=sorted(config.FEATURE_MASK_LEVEL_PREFIX),
                    help='Which QC level supplies the mask whose excluded windows are dropped '
                         f'from the baseline (default: {config.FEATURE_BASELINE_MASK_LEVEL}). '
                         '"bipolar" is the superset: (raw_voltage[anode] | '
                         'raw_voltage[cathode]) | bipolar_variance.')
    ap.add_argument('--mask-label', default=None,
                    help='Mask label within that level. Default depends on the level: '
                         f'raw_voltage -> {config.CANONICAL_MASK_LABEL}; bipolar -> '
                         f'{config.bipolar_mask_label(config.FEATURE_BASELINE_BIPOLAR_VARIANCE_LABEL)}. '
                         'Pass "none" to compute an UNMASKED baseline.')
    ap.add_argument('--overwrite', action='store_true')
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

    mask_level = args.mask_level
    if args.mask_label is None:
        mask_label = (config.bipolar_mask_label(config.FEATURE_BASELINE_BIPOLAR_VARIANCE_LABEL)
                      if mask_level == 'bipolar' else config.CANONICAL_MASK_LABEL)
    else:
        mask_label = args.mask_label
    if mask_label == 'none':
        mask_label = None

    subjects = [s for s in args.subjects.replace(',', ' ').split() if s]
    io.warn_if_dirty()
    logger.info('feature-level power-outlier metrics: %d subjects, mask=%s/%s, K=%g, B=%g',
                len(subjects), mask_level, mask_label,
                config.FEATURE_Z_THRESH, config.FEATURE_BIN_FRAC)

    registry = pd.read_csv(config.FILE_REGISTRY_CSV)
    failed = []
    for s in subjects:
        subject = s.replace('sub-', '')
        if args.session == 'all':
            sessions = [str(x).replace('ses-', '')
                        for x in registry[registry.sub_id == f'sub-{subject}'].ses_id.unique()]
        else:
            sessions = [args.session]
        for session in sessions:
            try:
                process_subject_session(subject, session, mask_label, mask_level,
                                        overwrite=args.overwrite)
            except Exception:
                # One malformed subject must not take down the array task's whole
                # batch -- log the traceback, keep going, summarize at the end.
                logger.exception('sub-%s ses-%s: unhandled error, skipping', subject, session)
                failed.append(f'sub-{subject}_ses-{session}')

    if failed:
        logger.warning('%d subject/sessions failed: %s', len(failed), failed)


if __name__ == '__main__':
    main()
