#!/usr/bin/env python3
"""
Which subject/sessions had FULL raw-voltage mask coverage when a QC level's
metrics were computed -- and which quietly did not.

WHY THIS EXISTS
---------------
A missing mask never stops anything in this pipeline; it silently degrades the
result. Three code paths fall back to an unmasked baseline with only a warning:
`feature_level/detect_power_outlier.py` (no mask file for a subject/session),
`build_exclusions.py` (flatline `--mask-from-label` with no mask file), and
`mask_projection.project_to_pairs` (an absent run, contact, or 60s bin reads as
"not excluded"). An unmasked baseline includes artifact windows, which inflates
the std, which DEFLATES z -- so the detector becomes LESS sensitive for exactly
the subjects whose QC inputs were incomplete, and a cross-subject comparison of
flag rates then conflates "clean" with "unmasked".

`detect_power_outlier` records `raw_voltage_mask_applied` per subject/session, so
tier 1 below is recoverable. Nothing records tier 2. This script reconstructs
both from what is already on disk, so it needs NO re-run of any metric pass.

THE TWO TIERS
-------------
  1. NO MASK FILE at all for a subject/session -> the whole session's baseline
     was unmasked. Recorded (where the producer wrote it) as
     `raw_voltage_mask_applied: false`.
  2. MASK FILE EXISTS but a RUN is absent from it -> that run's windows entered
     the baseline unmasked while the subject-level flag still said `true`.
     Because the baseline is pooled ACROSS runs, one uncovered run contaminates
     the whole session's mean/std. This tier is silent today.

Tier 2 is only decidable because the raw-voltage mask CSVs are DENSE -- they
carry `excluded=False` rows, not just exclusions -- so a run missing from the
file means missing coverage, NOT a clean run. If the mask tree ever becomes
sparse, this check becomes unsound and must be revisited.

LEVEL-AGNOSTIC, BY DESIGN
-------------------------
Driven by `--level-root`, like the rest of the QC tree, because the same question
applies at `feature_level/` and `bipolar/`. The two levels' run_info schemas
differ (feature_level is per subject/session and records `runs[]`; bipolar is per
subject and records neither runs nor any mask field), so the cohort and the
expected-run denominator are resolved defensively:

  expected runs  <- run_info `runs[]` when the producer recorded it (the truest
                    denominator: exactly what it read), else the file registry.
                    Which was used is reported per row in `runs_source`, because
                    a denominator you cannot see is a denominator you cannot
                    trust.

The registry fallback is the weaker denominator and is reported per row in
`runs_source` for that reason. It USED to over-count badly: the registry lists
runs whose NWB it could not parse (sub-236 has 109 rows but 107 readable), and a
level that skips those looked under-covered. `_load_registry` now drops rows with
no `n_channels`, which removed 100% of the false positives it produced (see that
function). It is still the weaker signal — it reports what a subject/session HAS
rather than what the level actually READ, so a level that skipped a readable run
for its own reasons will still show up here.

THIS IS A REPORT, NOT A COHORT FILE. Cohort membership lives only in
`cohorts/*.json` (DECISIONS.md 2026-07-27). Filter on `fully_covered` and build
a cohort file as a separate, deliberate step.

Run on a job, not the login node -- it reads one column from every mask CSV and
the largest is ~190 MB:
    python -m ieeg_ehr.qc.report_mask_coverage
    python -m ieeg_ehr.qc.report_mask_coverage --level-root $OAK/qc/bipolar
"""

import argparse
import json
import logging
import re
from pathlib import Path

import pandas as pd

from ieeg_ehr import config, io

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

# 'sub-039_ses-01' (feature_level) and 'sub-039' (bipolar) both parse; the
# session group is optional precisely because the two levels name files
# differently, and this script must not care which it is looking at.
_RUN_INFO_RE = re.compile(r'^sub-(?P<subject>[^_]+)(?:_ses-(?P<session>[^_]+))?$')
_RUN_ID_RE = re.compile(r'run-([A-Za-z0-9]+)')

SUMMARY_COLUMNS = [
    'subject', 'session', 'mask_label', 'mask_file_exists', 'mask_applied',
    'runs_source', 'n_runs_expected', 'n_runs_in_mask', 'n_runs_missing',
    'fully_covered',
]


def _load_registry():
    """(sub_id, ses_id, run_id) for every READABLE run — the level-independent
    source of which runs a subject/session has.

    Rows with a blank `n_channels` are dropped. Those are NWBs the registry
    builder itself could not parse (2136 of them cohort-wide as of 2026-07-28),
    and every pipeline stage skips them by design -- so counting them as
    "expected" makes a level look under-covered for runs nothing could ever have
    read. MEASURED: before this filter the bipolar level reported 37
    subject/sessions with missing runs, and every single missing run was one of
    these unparseable rows (sub-085's 2, sub-236's 2, sub-256's 5, ...) -- i.e.
    100% false positives.
    """
    reg = pd.read_csv(config.FILE_REGISTRY_CSV,
                      usecols=['sub_id', 'ses_id', 'run_id', 'n_channels'])
    n_all = len(reg)
    reg = reg[reg['n_channels'].notna()]
    n_dropped = n_all - len(reg)
    if n_dropped:
        logger.info('registry: ignoring %d/%d rows with no n_channels (unparseable NWBs '
                    'that every stage skips)', n_dropped, n_all)
    return reg.drop(columns=['n_channels'])


def _resolve_cohort(level_root, registry):
    """[(subject, session, run_info_path|None)] for every subject/session this
    level has a run_info record for.

    A per-subject run_info (bipolar) is expanded across that subject's sessions
    from the registry, so the output granularity is always subject/session — the
    granularity a mask file has.
    """
    run_info_dir = Path(config.metrics_run_info_dir(level_root))
    if not run_info_dir.is_dir():
        logger.warning('no run_info directory at %s — nothing to report', run_info_dir)
        return []

    cohort = []
    for path in sorted(run_info_dir.glob('sub-*.json')):
        m = _RUN_INFO_RE.match(path.stem)
        if not m:
            logger.warning('skipping unparseable run_info filename: %s', path.name)
            continue
        subject, session = m.group('subject'), m.group('session')
        if session:
            cohort.append((subject, session, path))
            continue
        sessions = sorted(
            str(s).replace('ses-', '')
            for s in registry.loc[registry.sub_id == f'sub-{subject}', 'ses_id'].dropna().unique()
        )
        if not sessions:
            logger.warning('sub-%s: run_info has no session and the registry has no '
                            'sessions either — reporting as session "?"', subject)
            sessions = ['?']
        for sess in sessions:
            cohort.append((subject, sess, path))
    return cohort


def _read_run_info(path):
    if path is None:
        return {}
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError) as exc:
        logger.warning('unreadable run_info %s: %s', path, exc)
        return {}


def _runs_from_run_info(info):
    """Run IDs the producing job actually read, or None if it did not record them.

    Returned as a set of 'run-XXXX' to match the mask CSV's `run_id` values.
    """
    runs = info.get('runs')
    if not runs:
        return None
    ids = set()
    for entry in runs:
        found = _RUN_ID_RE.search(str(entry))
        if found:
            ids.add(f'run-{found.group(1)}')
    return ids or None


def _mask_applied_flag(info):
    """The producer's own record of whether it found a mask. None when this level
    does not write the field (bipolar), which is NOT the same as False."""
    params = info.get('params') or {}
    if 'raw_voltage_mask_applied' in params:
        return bool(params['raw_voltage_mask_applied'])
    return None


def _mask_label_for(info, override):
    if override:
        return override
    params = info.get('params') or {}
    return params.get('raw_voltage_mask_label') or config.CANONICAL_MASK_LABEL


def _runs_in_mask(mask_path):
    """Distinct run_ids present in one mask CSV, or None if the file is absent.

    Only `run_id` is read: the mask CSVs run to ~190 MB and every other column is
    irrelevant here.
    """
    if not mask_path.exists():
        return None
    try:
        df = pd.read_csv(mask_path, usecols=['run_id'])
    except (OSError, ValueError) as exc:
        logger.warning('unreadable mask %s: %s', mask_path, exc)
        return None
    return set(df['run_id'].astype(str).unique())


def build_report(level_root, mask_label_override=None, subjects=None):
    """(summary_df, missing_runs_df) — one row per subject/session, and one row
    per uncovered run.

    subjects: restrict to these IDs (without the `sub-` prefix). Reading one
    column from every mask CSV is minutes of IO, so this is what makes a smoke
    test cheap.
    """
    registry = _load_registry()
    cohort = _resolve_cohort(level_root, registry)
    if subjects:
        wanted = set(subjects)
        cohort = [c for c in cohort if c[0] in wanted]
        missing_ids = wanted - {c[0] for c in cohort}
        if missing_ids:
            logger.warning('requested subject(s) absent from %s: %s',
                            config.metrics_run_info_dir(level_root), sorted(missing_ids))

    summary_rows, missing_rows = [], []
    for subject, session, run_info_path in cohort:
        info = _read_run_info(run_info_path)
        mask_label = _mask_label_for(info, mask_label_override)
        mask_path = Path(config.mask_csv(subject, session, mask_label))

        expected = _runs_from_run_info(info)
        runs_source = 'run_info'
        if expected is None:
            runs_source = 'registry'
            expected = set(
                str(r) for r in registry.loc[
                    (registry.sub_id == f'sub-{subject}')
                    & (registry.ses_id == f'ses-{session}'), 'run_id'
                ].dropna().unique()
            )

        in_mask = _runs_in_mask(mask_path)
        mask_file_exists = in_mask is not None
        missing = sorted(expected - (in_mask or set()))

        summary_rows.append({
            'subject': subject,
            'session': session,
            'mask_label': mask_label,
            'mask_file_exists': mask_file_exists,
            'mask_applied': _mask_applied_flag(info),
            'runs_source': runs_source,
            'n_runs_expected': len(expected),
            'n_runs_in_mask': len(in_mask) if mask_file_exists else 0,
            'n_runs_missing': len(missing),
            'fully_covered': bool(mask_file_exists and not missing),
        })
        for run_id in missing:
            missing_rows.append({
                'subject': subject,
                'session': session,
                'mask_label': mask_label,
                'run_id': run_id,
                'reason': 'no_mask_file' if not mask_file_exists else 'run_absent_from_mask',
            })

    summary_df = pd.DataFrame(summary_rows, columns=SUMMARY_COLUMNS)
    missing_df = pd.DataFrame(
        missing_rows,
        columns=['subject', 'session', 'mask_label', 'run_id', 'reason'],
    )
    return summary_df, missing_df


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--level-root', default=str(config.FEATURE_LEVEL_ROOT),
                    help=f'QC level to audit (default: {config.FEATURE_LEVEL_ROOT})')
    ap.add_argument('--mask-label', default=None,
                    help='Override the mask label to check against. Default: each '
                         "subject's own recorded raw_voltage_mask_label, falling back "
                         f'to {config.CANONICAL_MASK_LABEL}.')
    ap.add_argument('--subjects', default=None,
                    help='Comma-separated subject IDs to restrict to (default: every '
                         "subject with a run_info record at this level). Useful for a "
                         'cheap smoke test — the full sweep reads every mask CSV.')
    ap.add_argument('--out-dir', default=None,
                    help='Default: <level-root>/validation/mask_coverage/<timestamp>/')
    args = ap.parse_args()

    io.warn_if_dirty()

    subjects_filter = [s.strip() for s in args.subjects.split(',')] if args.subjects else None
    level_root = Path(args.level_root)
    summary_df, missing_df = build_report(level_root, args.mask_label, subjects_filter)

    if args.out_dir:
        out_dir = Path(args.out_dir)
    else:
        stamp = io.run_timestamp().replace(':', '').replace('-', '')[:15]
        out_dir = Path(config.validation_dir(level_root)) / 'mask_coverage' / stamp
    out_dir.mkdir(parents=True, exist_ok=True)

    params = {'level_root': str(level_root), 'mask_label_override': args.mask_label}
    parents = [str(config.metrics_run_info_dir(level_root)), str(config.FILE_REGISTRY_CSV)]
    subjects = sorted(summary_df['subject'].unique().tolist()) if len(summary_df) else []

    n_total = len(summary_df)
    n_covered = int(summary_df['fully_covered'].sum()) if n_total else 0
    counts = {
        'n_subject_sessions': n_total,
        'n_fully_covered': n_covered,
        'n_no_mask_file': int((~summary_df['mask_file_exists']).sum()) if n_total else 0,
        'n_with_missing_runs': int(((summary_df['n_runs_missing'] > 0)
                                    & summary_df['mask_file_exists']).sum()) if n_total else 0,
        'n_missing_run_rows': len(missing_df),
    }

    io.write_table(summary_df, out_dir / 'mask_coverage.csv',
                   params=params, parents=parents, subjects=subjects,
                   script='ieeg_ehr/qc/report_mask_coverage.py', extra=counts)
    io.write_table(missing_df, out_dir / 'mask_coverage_runs.csv',
                   params=params, parents=parents, subjects=subjects,
                   script='ieeg_ehr/qc/report_mask_coverage.py', extra=counts)

    io.log_analysis(
        f'raw-voltage mask coverage audit, {level_root.name} level: '
        f'{n_covered}/{n_total} subject-sessions fully covered, '
        f"{counts['n_no_mask_file']} with no mask file, "
        f"{counts['n_with_missing_runs']} with a mask but missing runs",
        out_dir,
    )

    logger.info('%d/%d subject-sessions fully covered', n_covered, n_total)
    if counts['n_no_mask_file']:
        logger.warning('%d subject-session(s) have NO mask file — their baseline was '
                       'UNMASKED: %s', counts['n_no_mask_file'],
                       summary_df.loc[~summary_df['mask_file_exists'],
                                      ['subject', 'session']].to_dict('records'))
    if counts['n_with_missing_runs']:
        logger.warning('%d subject-session(s) have a mask file but are MISSING RUNS from '
                       'it — the session-pooled baseline is contaminated for these: %s',
                       counts['n_with_missing_runs'],
                       summary_df.loc[(summary_df['n_runs_missing'] > 0)
                                      & summary_df['mask_file_exists'],
                                      ['subject', 'session', 'n_runs_missing']].to_dict('records'))
    logger.info('wrote %s', out_dir)


if __name__ == '__main__':
    main()
