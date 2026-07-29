#!/usr/bin/env python3
"""
Sweep every stored PSD run, record which windowing design wrote it, and emit the
list of subjects whose PSD needs re-running.

WHY A FULL SWEEP
----------------
The only prior check read the runs that carry a pain epoch, inside the 83
subject-sessions that have `epoch_defs`. There are 97 subjects and ~6236 PSD files
in `preprocessed/bipolar_fft`, so subjects without a cache, and runs without a pain
event, had never been looked at. Whether anything beyond sub-247 and sub-257 is
affected is genuinely unknown until this runs.

Metadata-only: `psd_timing.read_run_timing` never touches `decomp.data`, so the
whole cohort costs ~0.3 s per run rather than reading hundreds of MB per file.

MAP / REDUCE
------------
`--subjects X` writes one shard per subject under `shards/`; `--reduce`
concatenates them into the three artifacts below. Sharding rather than appending to
a shared file because array tasks would otherwise race on it -- the same reason
`metrics_run_info_dir` is per-subject.

    qc/psd_timing/run_timing.parquet        one row per run (the full record)
    qc/psd_timing/psd_rerun_runs.csv        offending runs, with reasons
    qc/psd_timing/psd_rerun_subjects.txt    THE DELIVERABLE: subject IDs, one per
                                            line, ready for --subjects

The subject list is what is actionable: `run_pipeline_bipolar.py` accepts
`--subjects` only and has no `--runs` flag, so a whole subject is re-run regardless.
When re-running, pass `--skip-variance-metrics` -- the bipolar variance metric is
computed on the TIME-DOMAIN signal on a 2 s grid and is unaffected by the PSD hop,
so recomputing it is wasted work.

Usage:
    python -m ieeg_ehr.qc.audit_psd_timing --subjects 247      # one shard
    python -m ieeg_ehr.qc.audit_psd_timing --reduce            # combine + lists
"""

import argparse
import logging

import numpy as np
import pandas as pd

from ieeg_ehr import config, io
from ieeg_ehr.qc import psd_timing

logger = logging.getLogger(__name__)

SHARD_SUBDIR = 'shards'


def _shard_path(subject):
    return config.psd_timing_dir() / SHARD_SUBDIR / f'sub-{subject}_runs.parquet'


def psd_subjects():
    """Subjects with any PSD output on disk, from the tree itself.

    Read from the filesystem rather than a cohort list on purpose: the question is
    "what has been processed", and a cohort file would answer "what we intended".
    """
    root = config.BIPOLAR_PSD_DERIV_ROOT
    return sorted(p.name.replace('sub-', '') for p in root.glob('sub-*') if p.is_dir())


def subject_runs(subject):
    """(session, run) for every PSD file this subject has, from filenames."""
    root = config.BIPOLAR_PSD_DERIV_ROOT / f'sub-{subject}'
    out = []
    for path in sorted(root.glob('ses-*/sub-*_ses-*_run-*_bipolar_psd.nwb')):
        stem = path.stem.replace('_bipolar_psd', '')
        _, ses_part, run_part = stem.split('_', 2)
        out.append((ses_part.replace('ses-', ''), run_part.replace('run-', ''), path))
    return out


def audit_subject(subject):
    runs = subject_runs(subject)
    if not runs:
        logger.warning('sub-%s: no PSD files on disk', subject)
        return pd.DataFrame(columns=psd_timing.RUN_TIMING_COLUMNS)
    rows = [psd_timing.describe_run(subject, ses, run, nwb_path=path)
            for ses, run, path in runs]
    df = pd.DataFrame(rows)[psd_timing.RUN_TIMING_COLUMNS]

    n_bad = int((~df['ok']).sum())
    designs = df['design'].value_counts().to_dict()
    hops = sorted({round(h, 6) for h in df['hop_sec'] if np.isfinite(h)})
    level = logger.warning if n_bad else logger.info
    level('sub-%s: %d runs | designs %s | hops %s | %d NOT ok',
          subject, len(df), designs, hops, n_bad)
    if n_bad:
        for r in df[~df['ok']].head(5).itertuples():
            logger.warning('  sub-%s %s: %s', subject, r.run_id, r.reason)
    return df


def write_shard(subject, df):
    path = _shard_path(subject)
    io.write_table(df, path, kind='table',
                   script='ieeg_ehr/qc/audit_psd_timing.py',
                   params={'expected_hop_sec': psd_timing.EXPECTED_HOP_SEC,
                           'expected_starting_time': psd_timing.EXPECTED_STARTING_TIME},
                   subjects=[f'sub-{subject}'])
    return path


def reduce_shards():
    shard_dir = config.psd_timing_dir() / SHARD_SUBDIR
    shards = sorted(shard_dir.glob('sub-*_runs.parquet'))
    if not shards:
        raise SystemExit(f'no shards in {shard_dir}; run the per-subject audit first')

    frames = [pd.read_parquet(p) for p in shards]
    table = pd.concat([f for f in frames if not f.empty], ignore_index=True)
    table = table.sort_values(['subject_id', 'session_id', 'run_id']).reset_index(drop=True)

    params = {'expected_hop_sec': psd_timing.EXPECTED_HOP_SEC,
              'expected_starting_time': psd_timing.EXPECTED_STARTING_TIME,
              'current_design': psd_timing.DESIGN_SINGLE_LEVEL}
    io.write_table(table, psd_timing.run_timing_path(), kind='table',
                   script='ieeg_ehr/qc/audit_psd_timing.py', params=params,
                   subjects=sorted(table['subject_id'].unique()),
                   extra={'n_shards': len(shards)})

    bad = table[~table['ok'].astype(bool)]
    rerun = psd_timing.rerun_subjects(table)
    ok = psd_timing.ok_subjects(table)

    # These two stay plain text/CSV, not Parquet: they are inputs to a shell
    # command (`--subjects "$(paste -sd, list.txt)"`) and to human eyes. The
    # authoritative record is run_timing.parquet, which does have a sidecar.
    runs_path = psd_timing.rerun_runs_path()
    runs_path.parent.mkdir(parents=True, exist_ok=True)
    bad.to_csv(runs_path, index=False)

    subj_path = psd_timing.rerun_subjects_path()
    subj_path.write_text(''.join(f"{s.replace('sub-', '')}\n" for s in rerun))

    logger.info('')
    logger.info('=== PSD timing audit: %d runs across %d subjects ===',
                len(table), table['subject_id'].nunique())
    logger.info('designs: %s', table['design'].value_counts().to_dict())
    logger.info('hops (s): %s',
                table['hop_sec'].round(6).value_counts(dropna=False).to_dict())
    logger.info('runs OK: %d | runs NOT ok: %d', int(table['ok'].sum()), len(bad))
    logger.info('subjects fully OK: %d | subjects needing a PSD re-run: %d',
                len(ok), len(rerun))
    if rerun:
        per_subject = (bad.groupby('subject_id')
                       .agg(n_bad_runs=('run_id', 'size'),
                            designs=('design', lambda s: sorted(set(s))),
                            hops=('hop_sec', lambda s: sorted({round(x, 4) for x in s})))
                       .reset_index())
        total = table.groupby('subject_id')['run_id'].size().rename('n_runs_total')
        logger.warning('SUBJECTS NEEDING A PSD RE-RUN:\n%s',
                       per_subject.merge(total, on='subject_id').to_string(index=False))
    logger.info('')
    logger.info('re-run subject list -> %s', subj_path)
    logger.info('offending runs       -> %s', runs_path)
    logger.info('full record          -> %s', psd_timing.run_timing_path())
    if rerun:
        logger.info('')
        logger.info('To re-run (NOTE --skip-variance-metrics: the bipolar variance '
                    'metric is 2s-gridded from the time-domain signal and is NOT '
                    'affected by the PSD hop):')
        logger.info('  python -m ieeg_ehr.preprocessing.run_pipeline_bipolar '
                    '--subjects "$(paste -sd, %s)" --skip-variance-metrics',
                    subj_path)
    return table


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument('--subjects', nargs='+', help='Audit these subjects (writes shards)')
    g.add_argument('--all', action='store_true', help='Audit every subject serially')
    g.add_argument('--reduce', action='store_true',
                   help='Combine existing shards into the table + re-run lists')
    g.add_argument('--list-subjects', action='store_true',
                   help='Print the subjects with PSD output (for array sizing)')
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

    if args.list_subjects:
        for s in psd_subjects():
            print(s)
        return

    if args.reduce:
        io.warn_if_dirty()
        reduce_shards()
        return

    io.warn_if_dirty()
    subjects = psd_subjects() if args.all else [s.replace('sub-', '') for s in args.subjects]
    for subject in subjects:
        df = audit_subject(subject)
        if not df.empty:
            write_shard(subject, df)


if __name__ == '__main__':
    main()
