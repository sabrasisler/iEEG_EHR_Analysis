#!/usr/bin/env python3
"""
Read-only status report: which subjects/sessions/runs have output at each
pipeline stage. Derives everything from what's actually on disk (against the
file registry's canonical subject/session/run list) rather than maintaining a
separate hand-updated ledger, so it can never drift out of sync with reality.

Stages checked:
  - raw_voltage metrics   (qc/raw_voltage/metrics/per_window/sub-XXX_ses-YY_<type>.csv)
  - raw_voltage mask      (qc/raw_voltage/masks/<label>/sub-XXX_ses-YY.csv)
  - bipolar variance      (qc/bipolar/metrics/per_window/sub-XXX_bipolar_variance.csv)
  - bipolar PSD NWB       (derivatives/preprocessed/bipolar_fft/sub-XXX/ses-XXX/..._bipolar_psd.nwb,
                            checked per RUN since that output is per-run, not per-subject)

Usage:
  python -m ieeg_ehr.qc.processing_status \
      --raw-voltage-root <qc/raw_voltage> --bipolar-root <qc/bipolar> \
      --bipolar-psd-root <derivatives/preprocessed/bipolar_fft> \
      --mask-label <label> \
      --out-subject-csv <path> --out-run-csv <path>
"""

import argparse
from pathlib import Path

import pandas as pd

from ieeg_ehr import config


def _subjects_sessions_runs():
    registry = pd.read_csv(config.FILE_REGISTRY_CSV)
    registry['subject_id'] = registry['sub_id']
    registry['session_id'] = registry['ses_id']
    registry['run_id'] = registry['run_id']
    return registry[['subject_id', 'session_id', 'run_id']].drop_duplicates()


def build_subject_status(registry, raw_voltage_root, bipolar_root, mask_label):
    rows = []
    subjects = sorted(registry['subject_id'].unique())
    for subject_id in subjects:
        row = {'subject_id': subject_id}

        # raw_voltage metrics and masks are written per subject-SESSION
        # (sub-XXX_ses-YY_<type>.csv), not per subject — the session was added
        # to these filenames by the migrate_add_session_to_filenames pass. Glob
        # rather than construct the name, since a subject can have >1 session
        # (sub-197/209/255). A subject counts as done if ANY session is present.
        per_window = config.metrics_per_window_dir(raw_voltage_root)
        for artifact_type in config.ARTIFACT_TYPES:
            hits = list(per_window.glob(f'{subject_id}_ses-*_{artifact_type}.csv'))
            row[f'raw_voltage_metric_{artifact_type}'] = bool(hits)

        mask_hits = list((Path(raw_voltage_root) / 'masks' / mask_label).glob(f'{subject_id}_ses-*.csv'))
        row[f'raw_voltage_mask_{mask_label}'] = bool(mask_hits)

        bip_p = config.metrics_per_window_dir(bipolar_root) / f'{subject_id}_bipolar_variance.csv'
        row['bipolar_variance_metric'] = bip_p.exists()

        rows.append(row)
    return pd.DataFrame(rows)


def build_run_status(registry, bipolar_psd_root):
    rows = []
    for _, r in registry.iterrows():
        nwb_path = (Path(bipolar_psd_root) / r['subject_id'] / r['session_id'] /
                    f"{r['subject_id']}_{r['session_id']}_{r['run_id']}_bipolar_psd.nwb")
        rows.append({
            'subject_id': r['subject_id'], 'session_id': r['session_id'], 'run_id': r['run_id'],
            'bipolar_psd_nwb': nwb_path.exists(),
        })
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--raw-voltage-root', default=str(config.DEFAULT_LEVEL_ROOT))
    ap.add_argument('--bipolar-root', default=str(config.BIPOLAR_LEVEL_ROOT))
    ap.add_argument('--bipolar-psd-root', default=str(config.BIPOLAR_PSD_DERIV_ROOT))
    ap.add_argument('--mask-label', default=config.CANONICAL_MASK_LABEL,
                     help='Which raw_voltage mask label to check for (default: the pinned mask)')
    ap.add_argument('--out-subject-csv', default=None,
                     help='Write the per-subject summary here (default: print only)')
    ap.add_argument('--out-run-csv', default=None,
                     help='Write the per-run bipolar-PSD coverage here (default: print summary only)')
    args = ap.parse_args()

    registry = _subjects_sessions_runs()
    print(f"Registry: {registry['subject_id'].nunique()} subjects, "
          f"{len(registry)} subject/session/run rows.", flush=True)

    subject_df = build_subject_status(registry, args.raw_voltage_root, args.bipolar_root, args.mask_label)
    run_df = build_run_status(registry, args.bipolar_psd_root)

    bool_cols = [c for c in subject_df.columns if c != 'subject_id']
    print("\n=== Per-subject stage completion (count of subjects with output present) ===")
    print(subject_df[bool_cols].sum().to_string())

    print(f"\n=== Bipolar PSD NWB coverage: {int(run_df['bipolar_psd_nwb'].sum())}/{len(run_df)} "
          f"runs done ===")
    missing_runs = run_df[~run_df['bipolar_psd_nwb']]
    if len(missing_runs):
        n_show = min(20, len(missing_runs))
        print(f"  {len(missing_runs)} runs missing bipolar PSD output; first {n_show}:")
        print(missing_runs.head(n_show).to_string(index=False))

    if args.out_subject_csv:
        subject_df.to_csv(args.out_subject_csv, index=False)
        print(f"\nWrote {args.out_subject_csv}")
    if args.out_run_csv:
        run_df.to_csv(args.out_run_csv, index=False)
        print(f"Wrote {args.out_run_csv}")


if __name__ == '__main__':
    main()
