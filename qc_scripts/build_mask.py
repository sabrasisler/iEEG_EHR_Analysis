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
  python -m qc_scripts.build_mask --level-root <path> --label maskA \
      --saturation default --flatline default --square_wave default --gross_artifact std4
"""

import argparse
import json

import pandas as pd

from qc_scripts import config, build_exclusions

BIN_SEC = 60.0
KEY = ['subject_id', 'session_id', 'run_id', 'channel', 'bin_start']


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--level-root', default=str(config.DEFAULT_LEVEL_ROOT))
    ap.add_argument('--label', required=True, help='Mask label (output folder name)')
    for t in config.ARTIFACT_TYPES:
        auto = build_exclusions.label_for(t, build_exclusions.default_params(t))
        ap.add_argument(f'--{t}', default=None,
                         help=f'Which {t} exclusion label to combine (default: config-default {auto})')
    args = ap.parse_args()

    chosen = {t: (getattr(args, t) or build_exclusions.label_for(t, build_exclusions.default_params(t)))
              for t in config.ARTIFACT_TYPES}
    type_dirs = {t: config.exclusion_dir(args.level_root, t, lbl) for t, lbl in chosen.items()}
    for t, d in type_dirs.items():
        if not d.exists():
            raise SystemExit(f"Missing exclusion dir for {t}: {d} (run build_exclusions first)")

    # subjects present in every chosen type dir
    subj_sets = {t: {p.stem for p in d.glob('sub-*.csv')} for t, d in type_dirs.items()}
    common = set.intersection(*subj_sets.values()) if subj_sets else set()
    for t, s in subj_sets.items():
        missing = s ^ common
        if missing:
            print(f"  NOTE: {t} has subjects not shared by all types (skipped): {sorted(missing)}",
                  flush=True)

    out_dir = config.mask_dir(args.level_root, args.label)
    out_dir.mkdir(parents=True, exist_ok=True)

    for subject_id in sorted(common):
        merged = None
        for t in config.ARTIFACT_TYPES:
            df = pd.read_csv(type_dirs[t] / f'{subject_id}.csv',
                             usecols=KEY + ['bin_end', 'excluded'])
            df = df.rename(columns={'excluded': f'excluded_{t}'})
            if merged is None:
                merged = df
            else:
                merged = merged.merge(df.drop(columns=['bin_end']), on=KEY, how='outer')
        excl_cols = [f'excluded_{t}' for t in config.ARTIFACT_TYPES]
        merged[excl_cols] = merged[excl_cols].fillna(False).astype(bool)
        merged['bin_end'] = merged['bin_start'] + BIN_SEC   # uniform grid; robust to NaN from outer join
        merged['excluded'] = merged[excl_cols].any(axis=1)
        merged = merged.sort_values(['run_id', 'channel', 'bin_start']).reset_index(drop=True)
        out_cols = KEY + ['bin_end'] + excl_cols + ['excluded']
        merged[out_cols].to_csv(out_dir / f'{subject_id}.csv', index=False)
        per_type = ', '.join('%s=%d' % (t, int(merged['excluded_%s' % t].sum()))
                             for t in config.ARTIFACT_TYPES)
        print("  %s: %d bins, %d excluded (%s)"
              % (subject_id, len(merged), int(merged['excluded'].sum()), per_type), flush=True)

    prov = config.warn_if_dirty()
    params_out = {
        'mask_label': args.label,
        'bin_sec': BIN_SEC,
        'per_type_labels': chosen,
        'per_type_dirs': {t: str(d) for t, d in type_dirs.items()},
        'n_subjects': len(common),
        'git': prov,
    }
    with open(out_dir / 'params.json', 'w') as f:
        json.dump(params_out, f, indent=2, default=str)
    print(f"Wrote {out_dir} ({len(common)} subjects) + params.json", flush=True)


if __name__ == '__main__':
    main()
