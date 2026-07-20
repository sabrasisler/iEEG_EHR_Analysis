#!/usr/bin/env python3
"""
Cache the bipolar-referenced trace for a sample of runs so QC threshold/mask
experiments (qc_scripts/plot_bipolar_flagged_runs.py) don't need to re-read
raw NWB + re-reference every time a new --std-thresh or raw-voltage mask is
tried. bipolar_reref.rereference()'s output is normally transient (`del`eted
right after use in run_pipeline_bipolar.py's process_session) -- this script
is the one place that persists it, deliberately to $SCRATCH
(config.bipolar_trace_cache_dir()) since it's throwaway/reproducible-on-demand,
not a production derivative.

Reuses the exact same pure functions run_pipeline_bipolar.py calls (no
re-referencing logic duplicated): io_utils.load_all_channels_with_electrodes,
bipolar_reref.derive_pairs, bipolar_reref.rereference.

Output per run:
  $SCRATCH/bipolar_trace_cache/sub-XXX/ses-YY/sub-XXX_ses-YY_run-ZZZZ.npz   (bipolar_v)
  $SCRATCH/bipolar_trace_cache/sub-XXX/ses-YY/sub-XXX_ses-YY_run-ZZZZ.json  (sidecar)

Usage:
  python -m preprocessing.save_bipolar_sample_traces \
      --subjects 039,071,085,088,099,150,176,191,193,198,205,207,211,217,227,244,248 \
      --runs-per-subject 3 --max-runs 40
"""

import argparse
import json
from pathlib import Path

import numpy as np

from qc_scripts import config, io_utils
from preprocessing import bipolar_reref


def save_one_run(subject, session, run, nwb_path, out_root):
    print(f"  sub-{subject} ses-{session} run-{run}: loading...", flush=True)
    data_v, channel_names, sfreq, elec_df, elec_indices = \
        io_utils.load_all_channels_with_electrodes(nwb_path)

    pairs, filtered_elec_df = bipolar_reref.derive_pairs(elec_df)
    if not pairs:
        print(f"  WARNING: no bipolar pairs derived for run-{run}; skipping.", flush=True)
        return False

    bipolar_v = bipolar_reref.rereference(data_v, elec_indices, pairs)
    del data_v

    out_dir = out_root / f'sub-{subject}' / f'ses-{session}'
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f'sub-{subject}_ses-{session}_run-{run}'

    np.savez_compressed(out_dir / f'{stem}.npz', bipolar_v=bipolar_v)

    sidecar = {
        'subject': f'sub-{subject}', 'session': f'ses-{session}', 'run': f'run-{run}',
        'source_nwb': str(nwb_path),
        'sfreq': float(sfreq),
        'n_samples': int(bipolar_v.shape[0]),
        'n_pairs': int(bipolar_v.shape[1]),
        # Same naming convention as the bipolar_variance metric CSVs
        # (run_pipeline_bipolar.py's _variance_rows) so exclusion rows join
        # on 'channel' directly.
        'channel': [p['location'] for p in pairs],
        'anode_channel': [p['anode_location'] for p in pairs],
        'cathode_channel': [p['cathode_location'] for p in pairs],
        'git': config.git_provenance(),
        'run_timestamp': config.run_timestamp(),
    }
    with open(out_dir / f'{stem}.json', 'w') as f:
        json.dump(sidecar, f, indent=2, default=str)

    print(f"    wrote {out_dir / stem}.npz (+.json), {bipolar_v.shape[0]} samples x "
          f"{bipolar_v.shape[1]} pairs", flush=True)
    return True


def run(subjects, runs_per_subject, max_runs):
    out_root = config.bipolar_trace_cache_dir()
    out_root.mkdir(parents=True, exist_ok=True)

    n_saved = 0
    for subject in subjects:
        if n_saved >= max_runs:
            break
        session_runs = io_utils.get_session_runs(subject)[:runs_per_subject]
        for session, run, nwb_path in session_runs:
            if n_saved >= max_runs:
                break
            try:
                if save_one_run(subject, session, run, nwb_path, out_root):
                    n_saved += 1
            except Exception as e:
                print(f"  ERROR: sub-{subject} ses-{session} run-{run} failed: {e!r}; skipping.",
                      flush=True)

    print(f"=== saved {n_saved} runs to {out_root} ===", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--subjects', required=True, help='Comma-separated subject IDs')
    ap.add_argument('--runs-per-subject', type=int, default=3,
                     help='Max runs to cache per subject, in registry order (default: 3)')
    ap.add_argument('--max-runs', type=int, default=40, help='Total run cap across all subjects')
    args = ap.parse_args()

    subjects = [s.strip().replace('sub-', '') for s in args.subjects.split(',')]
    run(subjects, args.runs_per_subject, args.max_runs)


if __name__ == '__main__':
    main()
