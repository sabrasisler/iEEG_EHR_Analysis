#!/usr/bin/env python3
"""
Step A of the metric/threshold split: apply ONE artifact type's threshold to the
stored continuous metrics and roll the result up to 60s exclusion bins. Cheap,
CSV-only, touches no raw NWB data — so a threshold sweep on one type re-runs only
this (per that type), never the raw pass and never the other types.

Detection stores metrics only (run_pipeline.py); this owns the threshold. Per
type the per-window boolean is:
  flatline:      variance          < var_thresh
  square_wave:   bimodal_fraction  > frac_thresh  AND  range > min_range
  saturation:    fraction_at_rail  > sat_frac_thresh   (0 when below the rail)
  gross_artifact: z=(var-mean)/std > std_thresh        (degenerate std -> excluded)

For the 2s types (flatline/square_wave/saturation) the per-2s-window booleans are
OR'd within each enclosing 60s bin (bin = floor(window_start_time/60)); the
gross_artifact metric is already on 60s windows so it maps through 1:1 on the same
grid. Output is one compact 60s table per subject:
  subject_id, session_id, run_id, channel, bin_start, bin_end, excluded
written to exclusions/<artifact_type>/<label>/ with a params.json (thresholds +
git provenance + parent metrics manifest ref).

Usage:
  python -m qc_scripts.build_exclusions --level-root <path> --artifact-type gross_artifact --label std4 --std-thresh 4
  python -m qc_scripts.build_exclusions --level-root <path> --artifact-type all --label default
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from qc_scripts import config

BIN_SEC = 60.0

# Columns each type needs from the metric CSV (keep reads lean).
USECOLS = {
    'flatline':      ['subject_id', 'session_id', 'run_id', 'channel', 'window_start_time', 'metric_value'],
    'square_wave':   ['subject_id', 'session_id', 'run_id', 'channel', 'window_start_time', 'metric_value', 'range'],
    'saturation':    ['subject_id', 'session_id', 'run_id', 'channel', 'window_start_time', 'metric_value'],
    'gross_artifact':['subject_id', 'session_id', 'run_id', 'channel', 'window_start_time', 'metric_value',
                      'session_mean', 'session_std'],
}


def label_for(artifact_type, p):
    """A self-documenting folder label derived from the type's threshold(s), so
    you can read the parameters off the path instead of an opaque 'default'."""
    if artifact_type == 'flatline':
        return f"var{p['var_thresh']:g}"
    if artifact_type == 'square_wave':
        return f"frac{p['frac_thresh']:g}"
    if artifact_type == 'saturation':
        return f"pct{p['sat_frac_thresh']:g}"
    if artifact_type == 'gross_artifact':
        return f"std{p['std_thresh']:g}"
    raise ValueError(artifact_type)


def default_params(artifact_type):
    if artifact_type == 'flatline':
        return {'var_thresh': config.FLATLINE_VAR_THRESH}
    if artifact_type == 'square_wave':
        return {'frac_thresh': config.SQUARE_FRAC_THRESH, 'min_range': config.SQUARE_MIN_RANGE_V}
    if artifact_type == 'saturation':
        return {'sat_frac_thresh': 0.0}   # >0 fraction == >=1 saturated sample (old SAT_MIN_SAMPLES=1)
    if artifact_type == 'gross_artifact':
        return {'std_thresh': config.GROSS_STD_THRESH}
    raise ValueError(artifact_type)


def compute_excluded(artifact_type, chunk, p):
    """Per-row boolean exclusion for the chunk (before 60s bucketing)."""
    if artifact_type == 'flatline':
        return chunk['metric_value'] < p['var_thresh']
    if artifact_type == 'square_wave':
        return (chunk['metric_value'] > p['frac_thresh']) & (chunk['range'] > p['min_range'])
    if artifact_type == 'saturation':
        return chunk['metric_value'] > p['sat_frac_thresh']
    if artifact_type == 'gross_artifact':
        std = chunk['session_std']
        with np.errstate(divide='ignore', invalid='ignore'):
            z = (chunk['metric_value'] - chunk['session_mean']) / std
        excl = (z > p['std_thresh']).fillna(False)
        degenerate = ~np.isfinite(std) | (std <= 0)   # no usable baseline -> flag the whole channel
        return (excl | degenerate).astype(bool)
    raise ValueError(artifact_type)


def build_one_subject(metric_csv, artifact_type, p, chunksize=500_000):
    """Stream one subject's metric CSV, threshold + OR into 60s bins. Returns a
    tidy DataFrame; memory is bounded by the (30x smaller) bin count via a dict
    OR-accumulator that survives chunk-boundary splits."""
    acc = {}   # (session_id, run_id, channel, bin) -> excluded bool
    for chunk in pd.read_csv(metric_csv, usecols=USECOLS[artifact_type], chunksize=chunksize):
        chunk = chunk.copy()
        chunk['_excl'] = compute_excluded(artifact_type, chunk, p)
        chunk['_bin'] = (chunk['window_start_time'] // BIN_SEC).astype(int)
        grouped = chunk.groupby(['session_id', 'run_id', 'channel', '_bin'])['_excl'].any()
        for key, val in grouped.items():
            if val or key not in acc:
                acc[key] = acc.get(key, False) or bool(val)
    if not acc:
        return None
    subject_id = metric_csv.name.split('_')[0]   # 'sub-XXX'
    rows = [{'subject_id': subject_id, 'session_id': s, 'run_id': r, 'channel': c,
             'bin_start': b * BIN_SEC, 'bin_end': (b + 1) * BIN_SEC, 'excluded': e}
            for (s, r, c, b), e in acc.items()]
    return pd.DataFrame(rows).sort_values(['run_id', 'channel', 'bin_start']).reset_index(drop=True)


def run_type(level_root, artifact_type, label, params):
    metrics_dir = config.metrics_per_window_dir(level_root)
    out_dir = config.exclusion_dir(level_root, artifact_type, label)
    out_dir.mkdir(parents=True, exist_ok=True)

    metric_csvs = sorted(metrics_dir.glob(f'sub-*_{artifact_type}.csv'))
    if not metric_csvs:
        print(f"  [{artifact_type}] no metric CSVs in {metrics_dir}, skipping.", flush=True)
        return

    for metric_csv in metric_csvs:
        df = build_one_subject(metric_csv, artifact_type, params)
        subject_id = metric_csv.name.split('_')[0]
        if df is None:
            print(f"  [{artifact_type}] {subject_id}: no rows, skipping.", flush=True)
            continue
        out_path = out_dir / f'{subject_id}.csv'
        df.to_csv(out_path, index=False)
        n_excl = int(df['excluded'].sum())
        print(f"  [{artifact_type}] {subject_id}: {len(df)} bins, {n_excl} excluded -> {out_path}",
              flush=True)

    prov = config.warn_if_dirty()
    params_out = {
        'artifact_type': artifact_type,
        'label': label,
        'bin_sec': BIN_SEC,
        'thresholds': params,
        'metrics_manifest': str(config.metrics_manifest_path(level_root)),
        'git': prov,
    }
    with open(out_dir / 'params.json', 'w') as f:
        json.dump(params_out, f, indent=2, default=str)
    print(f"  [{artifact_type}] wrote {out_dir / 'params.json'}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--level-root', default=str(config.DEFAULT_LEVEL_ROOT),
                     help=f'QC level root (default: {config.DEFAULT_LEVEL_ROOT})')
    ap.add_argument('--artifact-type', required=True,
                     help="One of saturation/flatline/square_wave/gross_artifact, or 'all'")
    ap.add_argument('--label', default=None,
                     help='Folder label (default: auto from the threshold, e.g. std5 / var5e-13 / '
                          'frac0.9 / pct0 — self-documenting rather than an opaque "default")')
    # per-type threshold overrides (else config defaults)
    ap.add_argument('--var-thresh', type=float, default=None)
    ap.add_argument('--frac-thresh', type=float, default=None)
    ap.add_argument('--min-range', type=float, default=None)
    ap.add_argument('--sat-frac-thresh', type=float, default=None)
    ap.add_argument('--std-thresh', type=float, default=None)
    args = ap.parse_args()

    overrides = {'var_thresh': args.var_thresh, 'frac_thresh': args.frac_thresh,
                 'min_range': args.min_range, 'sat_frac_thresh': args.sat_frac_thresh,
                 'std_thresh': args.std_thresh}

    types = config.ARTIFACT_TYPES if args.artifact_type == 'all' else [args.artifact_type]
    for artifact_type in types:
        params = default_params(artifact_type)
        for k in params:
            if overrides.get(k) is not None:
                params[k] = overrides[k]
        label = args.label or label_for(artifact_type, params)
        print(f"=== build_exclusions: {artifact_type} (label={label}) params={params} ===", flush=True)
        run_type(args.level_root, artifact_type, label, params)


if __name__ == '__main__':
    main()
