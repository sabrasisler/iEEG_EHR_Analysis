#!/usr/bin/env python3
"""
Re-derive gross_artifact's excluded flag at a different GROSS_STD_THRESH,
without touching raw NWB data or re-running the other two detectors:
metric_value in the existing per-window gross_artifact CSVs is already the
signed z-score, so a different threshold is pure arithmetic on a column
that's already on disk.

Writes a new output root (per_window/summary/plots) containing:
  - saturation/flatline per-window CSVs: symlinked from --src-dir, unchanged
    (their thresholds are independent of gross_artifact's).
  - gross_artifact per-window CSVs: rewritten with excluded recomputed at
    --thresh, streamed in chunks (these files can be multiple GB).

After this, run summarize_exclusions.py / plot_distributions.py against
--dst-dir as usual to see the effect of the new threshold.

Usage:
  python -m qc_scripts.reclassify_gross_artifact_threshold \
      --src-dir /path/to/qc_variance \
      --dst-dir /path/to/qc_variance_gross_thresh4 \
      --thresh 4.0
"""

import argparse
import shutil
from pathlib import Path

import pandas as pd


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--src-dir', required=True)
    ap.add_argument('--dst-dir', required=True)
    ap.add_argument('--thresh', type=float, required=True)
    ap.add_argument('--chunksize', type=int, default=1_000_000)
    args = ap.parse_args()

    src_per_window = Path(args.src_dir) / 'per_window'
    dst_per_window = Path(args.dst_dir) / 'per_window'
    dst_per_window.mkdir(parents=True, exist_ok=True)

    for src_path in sorted(src_per_window.glob('sub-*.csv')):
        dst_path = dst_per_window / src_path.name

        if src_path.name.endswith('_gross_artifact.csv'):
            if dst_path.exists() or dst_path.is_symlink():
                dst_path.unlink()
            first = True
            for chunk in pd.read_csv(src_path, chunksize=args.chunksize):
                chunk['excluded'] = chunk['metric_value'] > args.thresh
                chunk.to_csv(dst_path, mode='a', header=first, index=False)
                first = False
            print(f"Reclassified {src_path.name} at thresh={args.thresh} -> {dst_path}")
        else:
            if dst_path.exists() or dst_path.is_symlink():
                dst_path.unlink()
            dst_path.symlink_to(src_path.resolve())
            print(f"Symlinked {src_path.name} -> {dst_path}")


if __name__ == '__main__':
    main()
