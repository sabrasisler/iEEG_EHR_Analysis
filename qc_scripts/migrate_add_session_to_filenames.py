#!/usr/bin/env python3
"""
One-time migration for the pre-session-split file layout: renames every
sub-XXX.* file under a qc/<level> tree to sub-XXX_ses-YY.*, and drops the
now-redundant subject_id/session_id columns from the CSVs (they're parseable
from the filename once every file is single-subject/single-session).

Only safe to run on data confirmed single-session-per-file (verified by hand
for the raw_voltage cohort before running this: every existing subject's
metrics only ever contain ses-01). Idempotent: skips a file if its destination
already exists, or if it's already missing subject_id/session_id columns.

Touches: metrics/per_window/*.csv, metrics/run_info/*.json,
exclusions/<type>/<label>/*.csv, masks/<label>/*.csv. Does NOT touch params.json
sidecars (those aren't per-subject) or the bipolar_fft derivatives tree (already
folder-per-session, correctly named).

Usage:
  python -m qc_scripts.migrate_add_session_to_filenames --level-root <path> --session 01
  python -m qc_scripts.migrate_add_session_to_filenames --level-root <path> --session 01 --dry-run

Per-subject files are fully independent, so metrics/per_window (the slow part --
some subjects' CSVs are multi-GB) can be sharded across parallel jobs with
--subjects, e.g. a Slurm array where each task takes a disjoint subject subset:
  python -m qc_scripts.migrate_add_session_to_filenames --stage per_window --subjects 150,176
Do not run overlapping --subjects sets concurrently -- two processes racing on
the same not-yet-migrated file can both pass the `dst.exists()` check before
either writes, and the second one's `src.unlink()` then fails since the first
already removed it.
"""

import argparse
import json
import os
import re
from pathlib import Path

import pandas as pd

from qc_scripts import config

_SUBJECT_ONLY_CSV_RE = re.compile(r'^sub-(?P<subject>[^_]+)$')
_SUBJECT_ONLY_JSON_RE = re.compile(r'^sub-(?P<subject>[^_]+)$')
_SUBJECT_TYPE_CSV_RE = re.compile(r'^sub-(?P<subject>[^_]+)_(?P<artifact_type>.+)$')


def _migrate_csv(src, dst, session, drop_cols=('subject_id', 'session_id'), dry_run=False):
    if dst.exists():
        print(f"  SKIP (dest exists): {dst}")
        return
    if dry_run:
        print(f"  [dry-run] {src} -> {dst}")
        return
    df = pd.read_csv(src)
    present = [c for c in drop_cols if c in df.columns]
    if present:
        # sanity check: every row must actually be the session we're tagging into the filename
        if 'session_id' in present:
            sessions = df['session_id'].unique()
            assert list(sessions) == [f'ses-{session}'], \
                f"{src} has session(s) {sessions}, expected only ses-{session} -- not a pure rename case"
        df = df.drop(columns=present)
    df.to_csv(dst, index=False)
    src.unlink()
    print(f"  {src} -> {dst} (dropped columns: {present})")


def _migrate_json(src, dst, session, dry_run=False):
    if dst.exists():
        print(f"  SKIP (dest exists): {dst}")
        return
    if dry_run:
        print(f"  [dry-run] {src} -> {dst}")
        return
    with open(src) as f:
        info = json.load(f)
    info.setdefault('session', f'ses-{session}')
    with open(dst, 'w') as f:
        json.dump(info, f, indent=2, default=str)
    src.unlink()
    print(f"  {src} -> {dst}")


def migrate_metrics_per_window(level_root, session, dry_run=False, subjects=None):
    d = config.metrics_per_window_dir(level_root)
    print(f"=== metrics/per_window ({d}) ===")
    for src in sorted(d.glob('sub-*.csv')):
        m = _SUBJECT_TYPE_CSV_RE.match(src.stem)
        if not m or '_ses-' in src.stem:
            continue   # already migrated or not a per-window file
        if subjects and f"sub-{m.group('subject')}" not in subjects:
            continue
        dst = d / f"sub-{m.group('subject')}_ses-{session}_{m.group('artifact_type')}.csv"
        _migrate_csv(src, dst, session, dry_run=dry_run)


def migrate_run_info(level_root, session, dry_run=False, subjects=None):
    d = config.metrics_run_info_dir(level_root)
    print(f"=== metrics/run_info ({d}) ===")
    for src in sorted(d.glob('sub-*.json')):
        m = _SUBJECT_ONLY_JSON_RE.match(src.stem)
        if not m:
            continue   # already migrated (has _ses- in the stem)
        if subjects and f"sub-{m.group('subject')}" not in subjects:
            continue
        dst = d / f"sub-{m.group('subject')}_ses-{session}.json"
        _migrate_json(src, dst, session, dry_run=dry_run)


def migrate_exclusions(level_root, session, dry_run=False, subjects=None):
    for artifact_type in config.ARTIFACT_TYPES:
        type_dir = Path(level_root) / 'exclusions' / artifact_type
        if not type_dir.exists():
            continue
        for label_dir in sorted(type_dir.iterdir()):
            if not label_dir.is_dir():
                continue
            print(f"=== exclusions/{artifact_type}/{label_dir.name} ===")
            for src in sorted(label_dir.glob('sub-*.csv')):
                m = _SUBJECT_ONLY_CSV_RE.match(src.stem)
                if not m:
                    continue
                if subjects and f"sub-{m.group('subject')}" not in subjects:
                    continue
                dst = label_dir / f"sub-{m.group('subject')}_ses-{session}.csv"
                _migrate_csv(src, dst, session, dry_run=dry_run)


def migrate_masks(level_root, session, dry_run=False, subjects=None):
    masks_dir = Path(level_root) / 'masks'
    if not masks_dir.exists():
        return
    for label_dir in sorted(masks_dir.iterdir()):
        if not label_dir.is_dir():
            continue
        print(f"=== masks/{label_dir.name} ===")
        for src in sorted(label_dir.glob('sub-*.csv')):
            m = _SUBJECT_ONLY_CSV_RE.match(src.stem)
            if not m:
                continue
            if subjects and f"sub-{m.group('subject')}" not in subjects:
                continue
            dst = label_dir / f"sub-{m.group('subject')}_ses-{session}.csv"
            _migrate_csv(src, dst, session, dry_run=dry_run)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--level-root', default=str(config.DEFAULT_LEVEL_ROOT))
    ap.add_argument('--session', default='01',
                     help='Session id (without ses- prefix) to tag every migrated file with; '
                          'only correct if every subject in this tree is truly single-session '
                          '(verified for raw_voltage before writing this script).')
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--stage', default='all',
                     choices=['all', 'per_window', 'run_info', 'exclusions', 'masks'],
                     help='Run just one stage (for parallel sharding) instead of all four.')
    ap.add_argument('--subjects', default=None,
                     help='Comma-separated subject IDs (e.g. 150,176) to restrict to -- shard '
                          'metrics/per_window (the slow, multi-GB-file stage) across parallel '
                          'jobs with disjoint --subjects sets. Default: all subjects.')
    args = ap.parse_args()
    subjects = ({f"sub-{s.strip().replace('sub-', '')}" for s in args.subjects.split(',')}
                if args.subjects else None)

    stages = {
        'per_window': migrate_metrics_per_window,
        'run_info': migrate_run_info,
        'exclusions': migrate_exclusions,
        'masks': migrate_masks,
    }
    to_run = stages if args.stage == 'all' else {args.stage: stages[args.stage]}
    for fn in to_run.values():
        fn(args.level_root, args.session, dry_run=args.dry_run, subjects=subjects)


if __name__ == '__main__':
    main()
