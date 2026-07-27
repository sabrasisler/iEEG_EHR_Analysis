#!/usr/bin/env python3
"""
Population-level exclusion-rate summaries from a combined MASK
(masks/<label>/sub-*.csv, from build_mask.py). The mask carries one boolean
column per artifact type (excluded_<type>) plus the combined `excluded`, all on
the 60s bin grid, so this reads that one source for every type.

Memory-safe by construction: each subject's mask file is read in chunks and
reduced to per-(subject, channel, type) `(n_excluded, n_total)` counters — never
a giant all-subjects concat (which is what OOM'd the previous version at
64/150GB). Scales to the full ~250-subject cohort.

`pct_windows_excluded` is now over 60s bins (not 2s windows) — same meaning,
coarser denominator.

Usage:
  python -m ieeg_ehr.qc.summarize_exclusions --mask-dir /path/to/qc/raw_voltage/masks/<label>
"""

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

from ieeg_ehr import config

CHUNK = 500_000


_MASK_CSV_RE = re.compile(r'^sub-(?P<subject>[^_]+)_ses-(?P<session>[^_]+)$')


def _subject_id_from_mask_path(path):
    """Mask CSVs no longer carry a subject_id column (one file already covers
    exactly one subject/session -- see build_mask.py); recover 'sub-XXX' from
    the filename, e.g. sub-039_ses-01.csv -> 'sub-039'."""
    m = _MASK_CSV_RE.match(path.stem)
    if not m:
        raise ValueError(f"Unexpected mask filename: {path.name}")
    return f"sub-{m.group('subject')}"


def accumulate(mask_dir, artifact_types):
    """
    Stream every subject/session's mask CSV in chunks. Returns a dict
    {(subject_id, channel, type): [n_excluded, n_total]} where `type` ranges
    over the artifact types plus 'any' (the combined mask). Aggregates across
    all of a subject's sessions under one subject_id key, matching the
    pre-session-split granularity of this report.
    """
    counts = {}
    cols = ['channel', 'excluded'] + [f'excluded_{t}' for t in artifact_types]
    type_cols = {t: f'excluded_{t}' for t in artifact_types}
    type_cols['any'] = 'excluded'

    for csv in sorted(Path(mask_dir).glob('sub-*_ses-*.csv')):
        subject_id = _subject_id_from_mask_path(csv)
        for chunk in pd.read_csv(csv, usecols=lambda c: c in cols, chunksize=CHUNK):
            for typ, col in type_cols.items():
                if col not in chunk.columns:
                    continue
                g = chunk.groupby('channel')[col].agg(['sum', 'count'])
                for channel, row in g.iterrows():
                    key = (subject_id, channel, typ)
                    acc = counts.setdefault(key, [0, 0])
                    acc[0] += int(row['sum'])
                    acc[1] += int(row['count'])
    return counts


def to_summary(counts, typ):
    rows = [{'subject_id': s, 'channel': c, 'artifact_type': typ,
             'pct_windows_excluded': 100.0 * n_excl / n_tot if n_tot else 0.0}
            for (s, c, t), (n_excl, n_tot) in counts.items() if t == typ]
    return pd.DataFrame(rows)


def flag_for_review(summary, std_thresh):
    if summary.empty:
        return summary
    mean = summary['pct_windows_excluded'].mean()
    std = summary['pct_windows_excluded'].std()
    cutoff = mean + std_thresh * std
    return summary[summary['pct_windows_excluded'] > cutoff].copy()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--mask-dir', required=True, help='masks/<label>/ directory from build_mask.py')
    args = ap.parse_args()

    mask_dir = Path(args.mask_dir)
    summary_dir = mask_dir / 'summary'
    summary_dir.mkdir(parents=True, exist_ok=True)

    counts = accumulate(mask_dir, config.ARTIFACT_TYPES)
    all_flagged = []

    for typ in list(config.ARTIFACT_TYPES) + ['any']:
        summary = to_summary(counts, typ)
        if summary.empty:
            print(f"No data for '{typ}', skipping.")
            continue
        out_path = summary_dir / f'exclusion_rates_{typ}.csv'
        summary.to_csv(out_path, index=False)
        s = summary['pct_windows_excluded']
        print(f"\n[{typ}] pct stats: mean={s.mean():.4f} median={s.median():.4f} "
              f"std={s.std():.4f} max={s.max():.4f}  -> {out_path} ({len(summary)} rows)")
        top = summary.sort_values('pct_windows_excluded', ascending=False).head(10)
        print(top.to_string(index=False))
        if typ != 'any':   # flag per-type, not on the combined 'any'
            flagged = flag_for_review(summary, config.FLAG_REVIEW_STD_THRESH)
            if not flagged.empty:
                all_flagged.append(flagged)

    if all_flagged:
        flagged_df = pd.concat(all_flagged, ignore_index=True)
        out_path = summary_dir / 'flagged_for_review.csv'
        flagged_df.to_csv(out_path, index=False)
        print(f"\nWrote {out_path} ({len(flagged_df)} flagged subject/channel/artifact rows)")
    else:
        print("\nNothing flagged for review.")


if __name__ == '__main__':
    main()
