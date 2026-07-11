#!/usr/bin/env python3
"""
Downstream canonical-band aggregation from an already-computed bipolar PSD
NWB (preprocessing/run_pipeline_bipolar.py's output). Reads ONLY the stored
log-power bins + bands table -- no raw NWB re-read, no new Welch computation.

Kept deliberately separate from the fused reref+PSD pass (see plan): canonical
band definitions are cheap to retune (edges, which harmonics to dodge) without
ever re-running the expensive raw-data pass, matching the metric/threshold-
split philosophy used throughout this repo.

Usage:
  python -m preprocessing.bipolar_bands --nwb-path <psd.nwb> --out-csv <path>
  python -m preprocessing.bipolar_bands --glob '<pattern>' --out-csv <path>
"""

import argparse
import glob as glob_module

import numpy as np
import pandas as pd
from pynwb import NWBHDF5IO

from qc_scripts import config


def load_psd_bins(nwb_path):
    """Returns (log_power (n_time, n_pairs, n_bins), bin_edges (n_bins+1,),
    contains_line_noise (n_bins,) bool, channel_names (n_pairs,), rate)."""
    io = NWBHDF5IO(str(nwb_path), 'r')
    nwb = io.read()
    decomp = nwb.processing['ecephys']['psd_log_bins']

    log_power = decomp.data[:]
    bands = decomp.bands.to_dataframe()
    lo = bands['band_limits'].apply(lambda t: t[0]).to_numpy()
    hi = bands['band_limits'].apply(lambda t: t[1]).to_numpy()
    bin_edges = np.concatenate([lo, hi[-1:]])
    contains_line_noise = bands['contains_line_noise'].to_numpy(dtype=bool)

    # NOTE: DecompositionSeries.source_channels does NOT survive an NWB
    # write/read round-trip in the pynwb version installed here (confirmed via
    # a Sherlock smoke test -- it reads back None even though it was set at
    # write time). nwb.electrodes itself DOES survive the round-trip, and its
    # row order matches the PSD's channel axis (electrode_table_region was
    # built with region=list(range(len(pairs))) in the same order pairs were
    # re-referenced), so read channel names from there instead.
    channel_names = list(nwb.electrodes.to_dataframe()['location'])
    rate = float(decomp.rate)
    io.close()
    return log_power, bin_edges, contains_line_noise, channel_names, rate


def aggregate_to_bands(log_power, bin_edges, contains_line_noise, bands=None):
    """
    bands: dict name -> (fmin, fmax), default config.CANONICAL_BANDS_HZ. For
    each band: average LINEAR power (10**log_power) across bins whose center
    falls in [fmin, fmax) AND are NOT contains_line_noise, then log10 -- same
    linear-then-log ordering as the fused script's broadband_log_power, for
    consistency. Returns dict name -> (n_time, n_pairs) array.
    """
    bands = bands or config.CANONICAL_BANDS_HZ
    bin_centers = np.sqrt(bin_edges[:-1] * bin_edges[1:])   # geometric center, matches band_mean in the NWB
    linear_power = 10.0 ** log_power   # (n_time, n_pairs, n_bins)

    out = {}
    for name, (fmin, fmax) in bands.items():
        mask = (bin_centers >= fmin) & (bin_centers < fmax) & (~contains_line_noise)
        if not mask.any():
            out[name] = np.full(log_power.shape[:2], np.nan)
            continue
        with np.errstate(divide='ignore'):
            out[name] = np.log10(linear_power[:, :, mask].mean(axis=2))
    return out


def _to_long_df(subject_id, session_id, run_id, channel_names, band_arrays, rate):
    rows = []
    for band_name, arr in band_arrays.items():
        n_time, n_pairs = arr.shape
        times = np.arange(n_time) / rate
        for ch_idx, ch_name in enumerate(channel_names):
            for t_idx, t in enumerate(times):
                rows.append({
                    'subject_id': subject_id, 'session_id': session_id, 'run_id': run_id,
                    'channel': ch_name, 'time': t, 'band': band_name,
                    'log_power': arr[t_idx, ch_idx],
                })
    return pd.DataFrame(rows)


def _parse_ids_from_nwb_name(nwb_path):
    stem = nwb_path.name if hasattr(nwb_path, 'name') else str(nwb_path).split('/')[-1]
    parts = stem.split('_')
    sub = next((p for p in parts if p.startswith('sub-')), 'sub-unknown')
    ses = next((p for p in parts if p.startswith('ses-')), 'ses-unknown')
    run = next((p for p in parts if p.startswith('run-')), 'run-unknown')
    return sub, ses, run


def aggregate_nwb_file(nwb_path, bands=None):
    from pathlib import Path
    nwb_path = Path(nwb_path)
    log_power, bin_edges, contains_line_noise, channel_names, rate = load_psd_bins(nwb_path)
    band_arrays = aggregate_to_bands(log_power, bin_edges, contains_line_noise, bands=bands)
    subject_id, session_id, run_id = _parse_ids_from_nwb_name(nwb_path)
    return _to_long_df(subject_id, session_id, run_id, channel_names, band_arrays, rate)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--nwb-path', default=None, help='One PSD NWB file')
    ap.add_argument('--glob', default=None, help='Glob pattern for batch aggregation across many run-level NWBs')
    ap.add_argument('--out-csv', required=True)
    args = ap.parse_args()

    if not args.nwb_path and not args.glob:
        raise SystemExit('Provide --nwb-path or --glob')

    paths = [args.nwb_path] if args.nwb_path else sorted(glob_module.glob(args.glob))
    if not paths:
        raise SystemExit('No NWB files matched.')

    dfs = []
    for p in paths:
        print(f"  aggregating {p}", flush=True)
        dfs.append(aggregate_nwb_file(p))
    out = pd.concat(dfs, ignore_index=True)
    out.to_csv(args.out_csv, index=False)
    print(f"Wrote {args.out_csv} ({len(out)} rows)", flush=True)


if __name__ == '__main__':
    main()
