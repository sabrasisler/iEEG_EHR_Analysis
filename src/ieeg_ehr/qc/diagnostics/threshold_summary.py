"""
Generic threshold-sweep summary, usable for any artifact type. Compares
per-channel exclusion-rate stats across N already-built exclusion labels
(exclusions/<type>/<label>/, from build_exclusions) for the SAME artifact
type -- the "how much more data does this cost" table, complementing
threshold_diff.py's "what does it look like" trace plots.

Unlike summarize_exclusions.py (which reads a combined masks/<label>/ and
needs all 4 artifact types built), this reads a single-type exclusion dir
directly, so you can sweep one type's threshold without touching the others.

Streams each subject's CSV in chunks (never a giant all-subjects concat),
mirroring summarize_exclusions.py's accumulate() pattern.

Writes validation/threshold_sweeps/<artifact_type>_threshold_summary.csv with
one row per label: mean/median/std/max pct_windows_excluded (over 60s bins,
per subject/channel), n_channels_flagged (using config.FLAG_REVIEW_STD_THRESH,
same convention as summarize_exclusions), and total bins excluded across all
channels.

Usage:
  python -m ieeg_ehr.qc.diagnostics.threshold_summary --artifact-type flatline \
      --labels var5e-13,var1e-12,var1e-11
"""
import argparse
from pathlib import Path

import pandas as pd

from ieeg_ehr import config

CHUNK = 500_000


def accumulate(label_dir):
    """Stream sub-*_ses-*.csv in `label_dir`, return {(subject_id, channel): [n_excluded, n_total]}.
    subject_id is parsed from the filename (sub-XXX_ses-YY.csv) since it's no
    longer a column -- one file already covers exactly one subject/session."""
    counts = {}
    for csv in sorted(Path(label_dir).glob('sub-*_ses-*.csv')):
        subject_id = csv.stem.split('_ses-')[0]
        for chunk in pd.read_csv(csv, usecols=['channel', 'excluded'], chunksize=CHUNK):
            g = chunk.groupby('channel')['excluded'].agg(['sum', 'count'])
            for channel, row in g.iterrows():
                acc = counts.setdefault((subject_id, channel), [0, 0])
                acc[0] += int(row['sum'])
                acc[1] += int(row['count'])
    return counts


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--level-root', default=str(config.DEFAULT_LEVEL_ROOT))
    ap.add_argument('--artifact-type', required=True, choices=config.ARTIFACT_TYPES)
    ap.add_argument('--labels', required=True,
                     help='Comma-separated exclusions/<type>/<label>/ names to compare, '
                          'e.g. var5e-13,var1e-12,var1e-11')
    args = ap.parse_args()

    level = args.level_root
    atype = args.artifact_type
    labels = [l.strip() for l in args.labels.split(',') if l.strip()]

    rows = []
    for label in labels:
        label_dir = config.exclusion_dir(level, atype, label)
        if not label_dir.exists():
            print(f"  {label}: missing ({label_dir}), skipping")
            continue
        counts = accumulate(label_dir)
        if not counts:
            print(f"  {label}: no data, skipping")
            continue
        pct = pd.Series({k: 100.0 * n_excl / n_tot if n_tot else 0.0
                          for k, (n_excl, n_tot) in counts.items()})
        total_excluded = sum(n_excl for n_excl, _ in counts.values())
        total_bins = sum(n_tot for _, n_tot in counts.values())
        cutoff = pct.mean() + config.FLAG_REVIEW_STD_THRESH * pct.std()
        n_flagged = int((pct > cutoff).sum())
        row = {
            'label': label, 'mean_pct': pct.mean(), 'median_pct': pct.median(),
            'std_pct': pct.std(), 'max_pct': pct.max(),
            'n_channels': len(pct), 'n_channels_flagged': n_flagged,
            'total_bins_excluded': total_excluded, 'total_bins': total_bins,
        }
        rows.append(row)
        print(f"  {label}: mean={row['mean_pct']:.4f}% median={row['median_pct']:.4f}% "
              f"max={row['max_pct']:.4f}% flagged={n_flagged}/{row['n_channels']} "
              f"total_excluded_bins={total_excluded}")

    if not rows:
        print("No labels produced data.")
        return

    summary = pd.DataFrame(rows)
    out_dir = config.threshold_sweep_dir(level)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f'{atype}_threshold_summary.csv'
    summary.to_csv(out_path, index=False)
    print(f"\nWrote {out_path} ({len(summary)} labels)")


if __name__ == '__main__':
    main()
