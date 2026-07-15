#!/usr/bin/env python3
"""
Step A of the metric/threshold split: apply ONE artifact type's threshold to the
stored continuous metrics and roll the result up to 60s exclusion bins. Cheap,
CSV-only, touches no raw NWB data — so a threshold sweep on one type re-runs only
this (per that type), never the raw pass and never the other types.

Detection stores metrics only (run_pipeline.py); this owns the threshold. Per
type the per-window boolean is:
  flatline:      variance          < var_thresh   (absolute, default)
                 OR z=(log10(var)-channel_mean)/channel_std < -std_thresh  (if
                 --std-thresh given: per-channel-relative mode -- see below)
  square_wave:   bimodal_fraction  > frac_thresh  AND  range > min_range
  saturation:    fraction_at_rail  > sat_frac_thresh   (0 when below the rail)
  gross_artifact: z=(var-mean)/std > std_thresh        (degenerate std -> excluded)

Flatline's default mode (var_thresh alone) is a single global absolute V²
floor -- fine for most channels, but some individual channels/runs run
"quietly" (naturally low background amplitude, not disconnected) whose normal
baseline sits close to that floor; loosening the floor to catch more truly
flat channels then also eats real low-amplitude signal on those. The
per-channel-relative mode (--std-thresh) fixes this: it computes each
channel's own baseline mean/std of log10(variance) -- pooled over its whole
session, i.e. every run for that channel in one pass, mirroring how
gross_artifact's session_mean/session_std are already pooled per channel --
then flags a window only if it's abnormally low RELATIVE TO THAT CHANNEL's
own typical variance (one-sided low z, same convention as gross_artifact's
one-sided high z, just flipped and on log-variance since variance spans many
orders of magnitude / is lognormal-shaped, not linear). The absolute
var_thresh floor stays active alongside it (OR'd) as a backstop for a channel
that's dead for its ENTIRE session -- with no real "normal" baseline, a
relative z-score can't help there. Computed with two streaming passes over the
existing per-window CSV (first: channel baseline stats; second: exclusion) --
still no raw NWB re-read, still lives entirely in this "rollup" step.

For the 2s types (flatline/square_wave/saturation) the per-2s-window booleans are
OR'd within each enclosing 60s bin (bin = floor(window_start_time/60)); the
gross_artifact metric is already on 60s windows so it maps through 1:1 on the same
grid. Output is one compact 60s table per subject:
  subject_id, session_id, run_id, channel, bin_start, bin_end, excluded
written to exclusions/<artifact_type>/<label>/ with a params.json (thresholds +
git provenance + parent metrics manifest ref).

Usage:
  python -m qc_scripts.build_exclusions --level-root <path> --artifact-type gross_artifact --label std4 --std-thresh 4
  python -m qc_scripts.build_exclusions --level-root <path> --artifact-type flatline --std-thresh 3   # -> logz3
  python -m qc_scripts.build_exclusions --level-root <path> --artifact-type all --label default

Note: --std-thresh is shared by gross_artifact and flatline's relative mode (same
override mechanism, since both types happen to have a 'std_thresh' key) -- fine
when sweeping one type at a time, but --artifact-type all --std-thresh N applies
N to BOTH simultaneously; run them separately if you want different values.
"""

import argparse
import os
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from qc_scripts import config

BIN_SEC = 60.0

# Columns each type needs from the metric CSV (keep reads lean). subject_id/
# session_id are NOT columns -- one metric CSV already covers exactly one
# subject/session (see run_pipeline._output_path), so they're parsed from the
# filename (_parse_subject_session) instead of repeated on every row.
USECOLS = {
    'flatline':      ['run_id', 'channel', 'window_start_time', 'metric_value'],
    'square_wave':   ['run_id', 'channel', 'window_start_time', 'metric_value', 'range'],
    'saturation':    ['run_id', 'channel', 'window_start_time', 'metric_value',
                      'window_max_abs', 'rail_value'],
    'gross_artifact':['run_id', 'channel', 'window_start_time', 'metric_value',
                      'session_mean', 'session_std'],
}

_METRIC_CSV_RE = re.compile(r'^sub-(?P<subject>[^_]+)_ses-(?P<session>[^_]+)_(?P<artifact_type>.+)$')


def _parse_subject_session(metric_csv_path, artifact_type):
    """Recover (subject_id, session_id) e.g. ('sub-039', 'ses-01') from a
    metrics filename like sub-039_ses-01_saturation.csv."""
    m = _METRIC_CSV_RE.match(metric_csv_path.stem)
    if not m or m.group('artifact_type') != artifact_type:
        raise ValueError(f"Unexpected metrics filename: {metric_csv_path.name}")
    return f"sub-{m.group('subject')}", f"ses-{m.group('session')}"


def label_for(artifact_type, p):
    """A self-documenting folder label derived from the type's threshold(s), so
    you can read the parameters off the path instead of an opaque 'default'."""
    if artifact_type == 'flatline':
        if p.get('std_thresh') is not None:
            base = f"linz{p['std_thresh']:g}" if p.get('linear') else f"logz{p['std_thresh']:g}"
            if p.get('mask_label'):
                base += f"_masked-{p['mask_label']}"
            return base
        return f"var{p['var_thresh']:g}"
    if artifact_type == 'square_wave':
        return f"frac{p['frac_thresh']:g}"
    if artifact_type == 'saturation':
        base = f"pct{p['sat_frac_thresh']:g}"
        if p.get('rail_margin_frac'):
            base += f"_marginfrac{p['rail_margin_frac']:g}"
        return base
    if artifact_type == 'gross_artifact':
        return f"std{p['std_thresh']:g}"
    raise ValueError(artifact_type)


def default_params(artifact_type):
    if artifact_type == 'flatline':
        # std_thresh=None -> absolute-only mode (current behavior, unchanged);
        # set (e.g. via --std-thresh) to additionally enable the per-channel
        # relative variance z-score mode described above. linear=True computes
        # that channel baseline on raw metric_value instead of log10(metric_value)
        # (see --linear below). mask_label, when set, is informational only here
        # (populated by run_type/main from --mask-from-label) -- it does not
        # affect thresholding, just gets folded into the label/params.json.
        return {'var_thresh': config.FLATLINE_VAR_THRESH, 'std_thresh': None,
                'linear': False, 'mask_label': None}
    if artifact_type == 'square_wave':
        return {'frac_thresh': config.SQUARE_FRAC_THRESH, 'min_range': config.SQUARE_MIN_RANGE_V}
    if artifact_type == 'saturation':
        # >0 fraction == >=1 saturated sample (old SAT_MIN_SAMPLES=1). rail_margin_frac is opt-in
        # (None = off): when set, a window is ALSO flagged if its own peak comes within that
        # fraction of the rail, even if no sample actually reached it (e.g. 0.10 -> peak >= 90%
        # of rail_value counts too) -- catches near-clipping bursts without re-reading raw NWB,
        # since window_max_abs/rail_value are already stored per 2s window.
        return {'sat_frac_thresh': 0.0, 'rail_margin_frac': None}
    if artifact_type == 'gross_artifact':
        return {'std_thresh': config.GROSS_STD_THRESH}
    raise ValueError(artifact_type)


def load_mask_lookup(level_root, mask_label, tag):
    """Load masks/<mask_label>/<tag>.csv (from build_mask.py) and return
    {(run_id, channel, bin_start): excluded_bool} for this subject/session --
    used to keep OTHER artifact types' already-known-bad 60s bins out of
    flatline's own per-channel baseline-stats pass (they'd otherwise inflate/
    skew mean and std of a channel that's mostly fine but has a few
    saturation/square-wave/gross-artifact bursts). Does not affect flatline's
    final excluded verdict -- only which windows feed the baseline."""
    mask_csv = config.mask_dir(level_root, mask_label) / f'{tag}.csv'
    if not mask_csv.exists():
        raise FileNotFoundError(f"--mask-from-label {mask_label}: missing {mask_csv}")
    df = pd.read_csv(mask_csv, usecols=['run_id', 'channel', 'bin_start', 'excluded'])
    return {(r, c, b): bool(e) for r, c, b, e in
            zip(df['run_id'], df['channel'], df['bin_start'], df['excluded'])}


def flatline_channel_stats(metric_csv, linear=False, mask_lookup=None, chunksize=500_000):
    """First pass for flatline's per-channel-relative mode: stream the metric
    CSV once and return {channel: (mean, std)}, pooled over every row for that
    channel in this subject/session (i.e. across all its runs) -- same pooling
    convention as gross_artifact's session_mean/std, just computed here instead
    of during the expensive detection pass.

    linear=False (default): stats are on log10(variance), since flatline
    variance spans many orders of magnitude (lognormal-shaped) -- see the
    module docstring. linear=True: stats are on raw metric_value instead --
    simpler/reuses the already-stored value as-is, at the cost of a right tail
    that can dominate the std (see CONTEXT.md flatline logz discussion).

    mask_lookup, if given, is a {(run_id, channel, bin_start): excluded} dict
    (from load_mask_lookup) -- rows whose 60s bin (window_start_time // 60 * 60)
    is excluded there are dropped from this baseline pass entirely."""
    acc = {}   # channel -> [n, sum, sumsq]
    for chunk in pd.read_csv(metric_csv, usecols=['run_id', 'channel', 'window_start_time', 'metric_value'],
                              chunksize=chunksize):
        if mask_lookup:
            bins = (chunk['window_start_time'] // 60.0) * 60.0
            keep = [not mask_lookup.get((r, c, b), False)
                    for r, c, b in zip(chunk['run_id'], chunk['channel'], bins)]
            chunk = chunk[np.array(keep)]
            if chunk.empty:
                continue
        v = chunk['metric_value'].to_numpy()
        val = v if linear else np.log10(np.clip(v, 1e-20, None))
        df = pd.DataFrame({'channel': chunk['channel'].to_numpy(), 'val': val})
        g = df.groupby('channel')['val'].agg(n='size', s='sum', ss=lambda x: float((x**2).sum()))
        for ch, row in g.iterrows():
            a = acc.setdefault(ch, [0, 0.0, 0.0])
            a[0] += int(row['n']); a[1] += row['s']; a[2] += row['ss']
    stats = {}
    for ch, (n, s, ss) in acc.items():
        mean = s / n
        var = ss / n - mean**2
        stats[ch] = (mean, var**0.5 if var > 0 else 0.0)
    return stats


def compute_excluded(artifact_type, chunk, p):
    """Per-row boolean exclusion for the chunk (before 60s bucketing)."""
    if artifact_type == 'flatline':
        excl_floor = chunk['metric_value'] < p['var_thresh']
        if p.get('std_thresh') is None:
            return excl_floor
        stats = p['_channel_stats']
        mean_v = chunk['channel'].map(lambda c: stats.get(c, (np.nan, np.nan))[0]).to_numpy()
        std_v = chunk['channel'].map(lambda c: stats.get(c, (np.nan, np.nan))[1]).to_numpy()
        raw = chunk['metric_value'].to_numpy()
        val = raw if p.get('linear') else np.log10(np.clip(raw, 1e-20, None))
        with np.errstate(divide='ignore', invalid='ignore'):
            z = (val - mean_v) / std_v
        excl_relative = pd.Series(z < -p['std_thresh']).fillna(False).to_numpy()
        degenerate = ~np.isfinite(std_v) | (std_v <= 0)  # no usable per-channel baseline
        return excl_floor.to_numpy() | excl_relative | degenerate
    if artifact_type == 'square_wave':
        return (chunk['metric_value'] > p['frac_thresh']) & (chunk['range'] > p['min_range'])
    if artifact_type == 'saturation':
        excl = chunk['metric_value'] > p['sat_frac_thresh']
        if p.get('rail_margin_frac'):
            near_rail = chunk['window_max_abs'] >= chunk['rail_value'] * (1 - p['rail_margin_frac'])
            excl = excl | near_rail
        return excl
    if artifact_type == 'gross_artifact':
        std = chunk['session_std']
        with np.errstate(divide='ignore', invalid='ignore'):
            z = (chunk['metric_value'] - chunk['session_mean']) / std
        excl = (z > p['std_thresh']).fillna(False)
        degenerate = ~np.isfinite(std) | (std <= 0)   # no usable baseline -> flag the whole channel
        return (excl | degenerate).astype(bool)
    raise ValueError(artifact_type)


def build_one_subject_session(metric_csv, artifact_type, p, chunksize=500_000):
    """Stream one subject/session's metric CSV, threshold + OR into 60s bins.
    Returns a tidy DataFrame; memory is bounded by the (30x smaller) bin count
    via a dict OR-accumulator that survives chunk-boundary splits. The file
    already covers exactly one subject/session, so no subject/session grouping
    is needed here (unlike before the per-session file split)."""
    acc = {}   # (run_id, channel, bin) -> excluded bool
    for chunk in pd.read_csv(metric_csv, usecols=USECOLS[artifact_type], chunksize=chunksize):
        chunk = chunk.copy()
        chunk['_excl'] = compute_excluded(artifact_type, chunk, p)
        chunk['_bin'] = (chunk['window_start_time'] // BIN_SEC).astype(int)
        grouped = chunk.groupby(['run_id', 'channel', '_bin'])['_excl'].any()
        for key, val in grouped.items():
            if val or key not in acc:
                acc[key] = acc.get(key, False) or bool(val)
    if not acc:
        return None
    rows = [{'run_id': r, 'channel': c, 'bin_start': b * BIN_SEC, 'bin_end': (b + 1) * BIN_SEC,
             'excluded': e}
            for (r, c, b), e in acc.items()]
    return pd.DataFrame(rows).sort_values(['run_id', 'channel', 'bin_start']).reset_index(drop=True)


def run_type(level_root, artifact_type, label, params, subjects=None):
    metrics_dir = config.metrics_per_window_dir(level_root)
    out_dir = config.exclusion_dir(level_root, artifact_type, label)
    out_dir.mkdir(parents=True, exist_ok=True)

    metric_csvs = sorted(metrics_dir.glob(f'sub-*_ses-*_{artifact_type}.csv'))
    if subjects:
        wanted = {f'sub-{s.replace("sub-", "")}' for s in subjects}
        metric_csvs = [p for p in metric_csvs
                       if _parse_subject_session(p, artifact_type)[0] in wanted]
    if not metric_csvs:
        print(f"  [{artifact_type}] no metric CSVs in {metrics_dir}, skipping.", flush=True)
        return

    zmode = artifact_type == 'flatline' and params.get('std_thresh') is not None
    mask_label = params.get('mask_label') if artifact_type == 'flatline' else None
    for metric_csv in metric_csvs:
        subject_id, session_id = _parse_subject_session(metric_csv, artifact_type)
        file_params = params
        if zmode:
            mask_lookup = None
            if mask_label:
                mask_lookup = load_mask_lookup(level_root, mask_label, f'{subject_id}_{session_id}')
            # first pass: this subject/session's own per-channel baseline (pooled
            # across all its runs, minus any masked-out bins), before the second
            # (thresholding) pass below.
            stats = flatline_channel_stats(metric_csv, linear=params.get('linear', False),
                                            mask_lookup=mask_lookup)
            file_params = dict(params, _channel_stats=stats)
        df = build_one_subject_session(metric_csv, artifact_type, file_params)
        tag = f'{subject_id}_{session_id}'
        if df is None:
            print(f"  [{artifact_type}] {tag}: no rows, skipping.", flush=True)
            continue
        out_path = out_dir / f'{tag}.csv'
        df.to_csv(out_path, index=False)
        n_excl = int(df['excluded'].sum())
        print(f"  [{artifact_type}] {tag}: {len(df)} bins, {n_excl} excluded -> {out_path}",
              flush=True)

    prov = config.warn_if_dirty()
    params_out = {
        'artifact_type': artifact_type,
        'label': label,
        'bin_sec': BIN_SEC,
        'thresholds': params,
        'metrics_per_window_dir': str(metrics_dir),
        'metrics_run_info': str(config.metrics_run_info_dir(level_root)),
        'run_timestamp': config.run_timestamp(),
        'git': prov,
        # explicit top-level flag (in addition to thresholds.mask_label) so it's
        # obvious at a glance whether this label's baseline pass was masked, and
        # by what, without having to know to look inside `thresholds`.
        'masked_by': ({'mask_label': mask_label, 'mask_dir': str(config.mask_dir(level_root, mask_label))}
                       if mask_label else None),
    }
    # atomic write: concurrent array tasks (one subject each) all write this same
    # per-(type,label) file — tmp+replace prevents an interleaved/corrupted JSON.
    tmp = out_dir / f'params.json.{os.getpid()}.tmp'
    with open(tmp, 'w') as f:
        json.dump(params_out, f, indent=2, default=str)
    os.replace(tmp, out_dir / 'params.json')
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
    ap.add_argument('--rail-margin-frac', type=float, default=None,
                     help='Saturation only: also flag a window if window_max_abs >= '
                          'rail_value * (1 - this), i.e. its peak came within this fraction '
                          'of the rail even without a sample actually crossing it (e.g. 0.10 '
                          'catches peaks at >=90%% of the rail). Default: off (exact-rail-only).')
    ap.add_argument('--std-thresh', type=float, default=None)
    ap.add_argument('--linear', action='store_true',
                     help='Flatline relative mode only: compute the per-channel baseline z-score '
                          'on raw metric_value instead of log10(metric_value) -- label linz<N> '
                          'instead of logz<N>. See build_exclusions.flatline_channel_stats docstring.')
    ap.add_argument('--mask-from-label', default=None,
                     help='Flatline relative mode only: masks/<label>/ (from build_mask.py) whose '
                          '`excluded` bins are dropped from the per-channel baseline pass before '
                          "thresholding -- e.g. a mask combining gross_artifact+saturation+square_wave "
                          "so those types' known-bad windows don't skew a channel's own flatline "
                          'baseline. Does not affect the final flatline excluded verdict itself.')
    ap.add_argument('--subjects', default=None,
                     help='Comma-separated subject IDs to restrict to (default: all present). '
                          'Use to skip subjects whose metrics are still being written.')
    args = ap.parse_args()

    overrides = {'var_thresh': args.var_thresh, 'frac_thresh': args.frac_thresh,
                 'min_range': args.min_range, 'sat_frac_thresh': args.sat_frac_thresh,
                 'rail_margin_frac': args.rail_margin_frac, 'std_thresh': args.std_thresh,
                 'linear': args.linear or None, 'mask_label': args.mask_from_label}
    subjects = [s.strip() for s in args.subjects.split(',')] if args.subjects else None

    types = config.ARTIFACT_TYPES if args.artifact_type == 'all' else [args.artifact_type]
    for artifact_type in types:
        params = default_params(artifact_type)
        for k in params:
            if overrides.get(k) is not None:
                params[k] = overrides[k]
        label = args.label or label_for(artifact_type, params)
        print(f"=== build_exclusions: {artifact_type} (label={label}) params={params} ===", flush=True)
        run_type(args.level_root, artifact_type, label, params, subjects=subjects)


if __name__ == '__main__':
    main()
