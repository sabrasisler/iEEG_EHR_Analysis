#!/usr/bin/env python3
"""
Step B of the metric/threshold split: OR a chosen per-artifact-type exclusion
(one <label> per type, from build_exclusions.py) into a single combined 60s
exclusion MASK — the artifact the next stage (bipolar re-referencing) consumes.

Cheap, CSV-only. Because it just picks one already-computed per-type exclusion
each, a threshold sweep on one type = re-run that type's build_exclusions + this,
never the others and never the raw pass.

Output per subject: masks/<mask_label>/sub-XXX.csv with the join key
(subject_id, session_id, run_id, channel, bin_start, bin_end), one boolean column
per type (excluded_<type>) for transparency, and `excluded` = OR across types.
A params.json records which <type>/<label> fed each + git provenance.

Usage:
  python -m ieeg_ehr.qc.build_mask --level-root <path> --label maskA \
      --saturation default --flatline default --square_wave default --gross_artifact std4
"""

import argparse
import json
import re

import pandas as pd

from ieeg_ehr import config
from ieeg_ehr.qc import build_exclusions

BIN_SEC = 60.0
# subject_id/session_id are NOT columns in the per-type exclusion CSVs (one file
# already covers exactly one subject/session -- see build_exclusions.py), so the
# join key is just run/channel/bin; subject/session are parsed from the filename.
KEY = ['run_id', 'channel', 'bin_start']

_EXCL_CSV_RE = re.compile(r'^sub-(?P<subject>[^_]+)_ses-(?P<session>[^_]+)$')


def _parse_subject_session(path):
    """Recover ('sub-039', 'ses-01') from an exclusion/mask filename like sub-039_ses-01.csv."""
    m = _EXCL_CSV_RE.match(path.stem)
    if not m:
        raise ValueError(f"Unexpected filename: {path.name}")
    return f"sub-{m.group('subject')}", f"ses-{m.group('session')}"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--level-root', default=str(config.DEFAULT_LEVEL_ROOT))
    ap.add_argument('--label', required=True, help='Mask label (output folder name)')
    ap.add_argument('--types', default=None,
                     help='Comma-separated subset of artifact types to combine (default: all '
                          f'{config.ARTIFACT_TYPES}) -- e.g. a mask meant to feed flatline\'s own '
                          '--mask-from-label should typically be gross_artifact,saturation,'
                          'square_wave (i.e. everything EXCEPT flatline, to avoid circularity).')
    for t in config.ARTIFACT_TYPES:
        auto = build_exclusions.label_for(t, build_exclusions.default_params(t))
        ap.add_argument(f'--{t}', default=None,
                         help=f'Which {t} exclusion label to combine (default: config-default {auto})')
    args = ap.parse_args()

    types = [t.strip() for t in args.types.split(',')] if args.types else list(config.ARTIFACT_TYPES)
    bad = [t for t in types if t not in config.ARTIFACT_TYPES]
    if bad:
        raise SystemExit(f"--types: unknown artifact type(s) {bad}, must be from {config.ARTIFACT_TYPES}")

    chosen = {t: (getattr(args, t) or build_exclusions.label_for(t, build_exclusions.default_params(t)))
              for t in types}
    type_dirs = {t: config.exclusion_dir(args.level_root, t, lbl) for t, lbl in chosen.items()}
    for t, d in type_dirs.items():
        if not d.exists():
            raise SystemExit(f"Missing exclusion dir for {t}: {d} (run build_exclusions first)")

    # (subject_id, session_id) pairs present in every chosen type dir
    subj_sets = {t: {_parse_subject_session(p) for p in d.glob('sub-*_ses-*.csv')}
                 for t, d in type_dirs.items()}
    common = set.intersection(*subj_sets.values()) if subj_sets else set()
    for t, s in subj_sets.items():
        missing = s ^ common
        if missing:
            print(f"  NOTE: {t} has subject/sessions not shared by all types (skipped): "
                  f"{sorted(missing)}", flush=True)

    out_dir = config.mask_dir(args.level_root, args.label)
    out_dir.mkdir(parents=True, exist_ok=True)

    for subject_id, session_id in sorted(common):
        tag = f'{subject_id}_{session_id}'
        merged = None
        for t in types:
            df = pd.read_csv(type_dirs[t] / f'{tag}.csv',
                             usecols=KEY + ['bin_end', 'excluded'])
            df = df.rename(columns={'excluded': f'excluded_{t}'})
            if merged is None:
                merged = df
            else:
                merged = merged.merge(df.drop(columns=['bin_end']), on=KEY, how='outer')
        excl_cols = [f'excluded_{t}' for t in types]
        merged[excl_cols] = merged[excl_cols].fillna(False).astype(bool)
        merged['bin_end'] = merged['bin_start'] + BIN_SEC   # uniform grid; robust to NaN from outer join
        merged['excluded'] = merged[excl_cols].any(axis=1)
        merged = merged.sort_values(['run_id', 'channel', 'bin_start']).reset_index(drop=True)
        out_cols = KEY + ['bin_end'] + excl_cols + ['excluded']
        merged[out_cols].to_csv(out_dir / f'{tag}.csv', index=False)
        per_type = ', '.join('%s=%d' % (t, int(merged['excluded_%s' % t].sum()))
                             for t in types)
        print("  %s: %d bins, %d excluded (%s)"
              % (tag, len(merged), int(merged['excluded'].sum()), per_type), flush=True)

    # Pull each per-type exclusion's own params.json so the mask sidecar links all the way back
    # to the metrics that fed this specific rollup, without having to chase per_type_dirs by hand.
    source_metrics = {}
    for t, d in type_dirs.items():
        type_params_path = d / 'params.json'
        if type_params_path.exists():
            with open(type_params_path) as f:
                type_params = json.load(f)
            source_metrics[t] = {
                'exclusion_params_json': str(type_params_path),
                'metrics_per_window_dir': type_params.get('metrics_per_window_dir'),
                'thresholds': type_params.get('thresholds'),
                'git': type_params.get('git'),
            }
        else:
            source_metrics[t] = {'exclusion_params_json': str(type_params_path)}

    prov = config.warn_if_dirty()
    params_out = {
        'mask_label': args.label,
        'bin_sec': BIN_SEC,
        'types': types,
        'per_type_labels': chosen,
        'per_type_dirs': {t: str(d) for t, d in type_dirs.items()},
        'source_metrics': source_metrics,
        'n_subject_sessions': len(common),
        'run_timestamp': config.run_timestamp(),
        'git': prov,
    }
    with open(out_dir / 'params.json', 'w') as f:
        json.dump(params_out, f, indent=2, default=str)
    print(f"Wrote {out_dir} ({len(common)} subject/sessions) + params.json", flush=True)


if __name__ == '__main__':
    main()
