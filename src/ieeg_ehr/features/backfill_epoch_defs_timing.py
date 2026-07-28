#!/usr/bin/env python3
"""
Backfill `epoch_start_sec` / `hop_sec` onto epoch_defs files written before those
columns existed -- and AUDIT the run-timing assumption while doing it.

WHY
---
The view layer must turn a cache row's `window_idx` into RUN-RELATIVE SECONDS to
join against the 60s QC mask grid. That needs the run's PSD `starting_time` and
`rate`, which live only in the NWB. Rather than re-open NWBs on every view call,
or bloat the cache with a per-row copy of a per-run constant, the two derived
values go in the tiny per-epoch index (see build_pain_epoch_power.py's defs
block for the reasoning).

The 83 subject-sessions already on disk predate those columns. This script adds
them WITHOUT touching the 34 GB cache -- it reads NWB *metadata only* (~0.3 s per
run) and rewrites just the defs Parquet.

THE AUDIT HALF
--------------
Every observed (starting_time, rate) is recorded and any deviation from
(0.0, 1.0) is reported LOUDLY rather than silently normalised. That pairing was
verified for exactly one subject (sub-019) before this script existed; the whole
point is to stop assuming it. If a run does deviate, the derived columns are
still correct for it -- they are computed per run, not from the assumption -- but
you want to know, because it means every place that hardcodes a 1 s hop is
suspect.

Run as a Slurm array (one task per subject) or directly:
    python -m ieeg_ehr.features.backfill_epoch_defs_timing --subjects 019 039
    python -m ieeg_ehr.features.backfill_epoch_defs_timing --all --audit-only
"""

import argparse
import json
import logging
import warnings

import numpy as np
import pandas as pd

from ieeg_ehr import config, io

logger = logging.getLogger(__name__)

NEW_COLUMNS = ['epoch_start_sec', 'hop_sec']
EXPECTED_STARTING_TIME = 0.0
EXPECTED_RATE = 1.0


def _run_timing(nwb_path):
    """(starting_time, rate, n_time) for one run -- METADATA ONLY.

    Deliberately does not touch `decomp.data`: the point of this script is that it
    is cheap enough to run over the whole cohort, and a PSD data array is hundreds
    of MB.
    """
    from pynwb import NWBHDF5IO
    with NWBHDF5IO(str(nwb_path), 'r') as handle:
        decomp = handle.read().processing['ecephys']['psd_log_bins']
        return float(decomp.starting_time), float(decomp.rate), int(decomp.data.shape[0])


def backfill_one(subject, session, epoch_minutes=None, audit_only=False, overwrite=False):
    """Returns (audit_rows, status). status is one of written/audited/skipped/absent."""
    defs_path = config.pain_epoch_defs_path(subject, session, epoch_minutes)
    if not defs_path.exists():
        return [], 'absent'

    defs = pd.read_parquet(defs_path)
    already = all(c in defs.columns for c in NEW_COLUMNS)
    if already and not overwrite and not audit_only:
        logger.info('sub-%s ses-%s: already has %s, skipping (use --overwrite)',
                    subject, session, NEW_COLUMNS)
        return [], 'skipped'

    audit, timing = [], {}
    for run_id in sorted(defs['run_id'].unique()):
        run = str(run_id).replace('run-', '')
        nwb = config.bipolar_psd_nwb_path(subject, session, run)
        if not nwb.exists():
            # A defs row whose NWB has since vanished cannot be backfilled. Loud,
            # and the row is left without timing rather than guessed at.
            logger.error('sub-%s ses-%s %s: NWB missing at %s -- cannot backfill this run',
                         subject, session, run_id, nwb)
            audit.append({'subject_id': f'sub-{subject}', 'session_id': f'ses-{session}',
                          'run_id': run_id, 'starting_time': None, 'rate': None,
                          'n_time': None, 'nwb_missing': True, 'deviates': True})
            continue
        st, rate, n_time = _run_timing(nwb)
        timing[run_id] = (st, rate)
        deviates = not (np.isclose(st, EXPECTED_STARTING_TIME) and np.isclose(rate, EXPECTED_RATE))
        audit.append({'subject_id': f'sub-{subject}', 'session_id': f'ses-{session}',
                      'run_id': run_id, 'starting_time': st, 'rate': rate,
                      'n_time': n_time, 'nwb_missing': False, 'deviates': deviates})
        if deviates:
            logger.warning('DEVIATION sub-%s ses-%s %s: starting_time=%r rate=%r '
                           '(expected %r / %r) -- derived columns are still correct for '
                           'this run, but anything assuming a 1s hop is suspect here',
                           subject, session, run_id, st, rate,
                           EXPECTED_STARTING_TIME, EXPECTED_RATE)

    if audit_only:
        return audit, 'audited'
    if not timing:
        logger.error('sub-%s ses-%s: no run timing recoverable, defs left unchanged',
                     subject, session)
        return audit, 'absent'

    st_col = defs['run_id'].map(lambda r: timing.get(r, (np.nan, np.nan))[0])
    rate_col = defs['run_id'].map(lambda r: timing.get(r, (np.nan, np.nan))[1])
    defs['epoch_start_sec'] = st_col + defs['row_start'] / rate_col
    defs['hop_sec'] = 1.0 / rate_col

    n_bad = int(defs['epoch_start_sec'].isna().sum())
    if n_bad:
        logger.error('sub-%s ses-%s: %d/%d epochs have no timing (missing NWB); they '
                     'will be UNUSABLE for masked views', subject, session, n_bad, len(defs))

    # Same params as the builder records, plus the schema bump, so the sidecar's
    # config_hash changes and a stale reader notices rather than silently mixing
    # backfilled and non-backfilled defs.
    io.write_table(defs, defs_path, kind='table',
                   script='ieeg_ehr/features/backfill_epoch_defs_timing.py',
                   params={'epoch_minutes_before': epoch_minutes or config.EPOCH_MINUTES_BEFORE,
                           'anchor': 'pain_score_time', 'masked': False,
                           'averaged': False, 'normalized': False,
                           'defs_schema': 'v2-run-timing'},
                   parents=[str(config.bipolar_psd_nwb_path(subject, session,
                                                            str(r).replace('run-', '')))
                            for r in sorted(timing)],
                   subjects=[f'sub-{subject}'])
    logger.info('sub-%s ses-%s: backfilled %d epochs across %d runs -> %s',
                subject, session, len(defs), len(timing), defs_path.name)
    return audit, 'written'


def _subject_sessions(epoch_minutes=None):
    """Every (subject, session) that has a defs file, read from the tree itself
    rather than a cohort list -- the question here is "what is on disk", not
    "what is in the cohort"."""
    unit = config.pain_epoch_unit_dir(epoch_minutes)
    out = []
    for p in sorted((unit / config.EPOCH_DEFS_SUBDIR).glob('sub-*_ses-*_defs.parquet')):
        stem = p.stem.replace('_defs', '')
        subject, session = stem.replace('sub-', '').split('_ses-')
        out.append((subject, session))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument('--subjects', nargs='+', help='Subject IDs, e.g. 019 039')
    g.add_argument('--all', action='store_true', help='Every subject/session with a defs file')
    ap.add_argument('--session', default=None, help='Restrict to one session (default: all present)')
    ap.add_argument('--epoch-minutes', type=float, default=None)
    ap.add_argument('--audit-only', action='store_true',
                    help='Report run timing without rewriting any defs file')
    ap.add_argument('--overwrite', action='store_true')
    ap.add_argument('--audit-out', default=None,
                    help='Write the per-run audit table here (Parquet, on Oak)')
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    warnings.filterwarnings('ignore', category=UserWarning, module='pynwb')
    io.warn_if_dirty()

    targets = _subject_sessions(args.epoch_minutes)
    if args.subjects:
        wanted = {s.replace('sub-', '') for s in args.subjects}
        targets = [(s, ses) for s, ses in targets if s in wanted]
    if args.session:
        targets = [(s, ses) for s, ses in targets if ses == args.session]
    if not targets:
        raise SystemExit('No matching defs files on disk.')

    audit, counts = [], {}
    for subject, session in targets:
        rows, status = backfill_one(subject, session, args.epoch_minutes,
                                    audit_only=args.audit_only, overwrite=args.overwrite)
        audit.extend(rows)
        counts[status] = counts.get(status, 0) + 1

    adf = pd.DataFrame(audit)
    logger.info('=== %s over %d subject-session(s): %s',
                'AUDIT' if args.audit_only else 'BACKFILL', len(targets), counts)
    if not adf.empty:
        n_dev = int(adf['deviates'].sum())
        logger.info('runs audited: %d | distinct (starting_time, rate): %s | deviations: %d',
                    len(adf),
                    sorted({(r.starting_time, r.rate) for r in adf.itertuples()
                            if not r.nwb_missing}),
                    n_dev)
        if n_dev:
            logger.warning('DEVIATING RUNS:\n%s',
                           adf[adf['deviates']].to_string(index=False))
        else:
            logger.info('No deviations: every run is starting_time=%r rate=%r, so the '
                        '1s-hop assumption holds across everything on disk.',
                        EXPECTED_STARTING_TIME, EXPECTED_RATE)
    if args.audit_out and not adf.empty:
        io.write_table(adf, args.audit_out, kind='table',
                       script='ieeg_ehr/features/backfill_epoch_defs_timing.py',
                       params={'expected_starting_time': EXPECTED_STARTING_TIME,
                               'expected_rate': EXPECTED_RATE},
                       subjects=sorted(adf['subject_id'].unique()))
        logger.info('audit table -> %s', args.audit_out)


if __name__ == '__main__':
    main()
