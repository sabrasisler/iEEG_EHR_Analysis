#!/usr/bin/env python3
"""Leave-one-subject-out refits of an lme4 domain run, for the consistency map.

    python -m ieeg_ehr.analysis.run_domain_lmer_loo --run-dir <domain_lmer run> --n-tasks
    python -m ieeg_ehr.analysis.run_domain_lmer_loo --run-dir <run> --task-id 7
    python -m ieeg_ehr.analysis.run_domain_lmer_loo --run-dir <run> --collect

WHY REFIT. `plot_domain_lmer_consistency` correlates each subject's own ROI
slope map with the GROUP's, and the group map must not contain that subject or
the correlation is partly a subject correlated with themselves. The region-level
profile (`plot_bandpower_consistency.loo_group_map`) removed the subject from an
inverse-variance mean of unpooled slopes; here the reference is the MODEL, so the
subject is removed from the model: the domain lmer is refitted once per excluded
subject with the identical formula (`domain_lmer_loo.R`), and the group's
(ROI, band) slope is read off as domain fixed slope + ROI random deviation.
Analyst's call, 2026-09-22.

ONLY THE BANDS THAT CARRY A SIGNIFICANT CELL are refitted -- the map is built
over the ROIs of the BH-significant (domain, band) cells of
`table_domain_lmer_heatmap_withcontrol_diagnostic.csv`, so no other band's
slopes are consumed. One Slurm array task per (band, excluded subject), plus one
`none` task per band (everyone in) for reference.

Outputs, all inside `<run>/consistency/`:
    loo_fits/<band>__<excluded>.csv    one per task, raw from R
    loo_roi_slopes.csv                 the collected table (with sidecar)

EXPLORATORY. Discovery cohort only. Nominations, not findings.
"""

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

from ieeg_ehr import io
from ieeg_ehr.analysis.run_domain_lmer import check_r, r_environment

logger = logging.getLogger(__name__)

SCRIPT = 'ieeg_ehr/analysis/run_domain_lmer_loo.py'
R_SCRIPT = Path(__file__).with_name('domain_lmer_loo.R')
OUT_SUBDIR = 'consistency'
LOO_DIR = 'loo_fits'
COLLECTED = 'loo_roi_slopes.csv'
#: The BH family the significant cells are read from -- the 30-cell with-Control
#: table, whose five outlined cells are the ones the analyst named. The 24-cell
#: table beside it (`table_domain_lmer_heatmap.csv`) has a different family and
#: SIX significant cells; it is deliberately not the source.
SIG_TABLE = 'table_domain_lmer_heatmap_withcontrol_diagnostic.csv'
BAND_ORDER = ('delta', 'theta', 'alpha', 'beta', 'gamma', 'high_gamma')
NONE = 'none'

DISCLAIMER = ('EXPLORATORY -- discovery cohort, NOMINATIONS NOT FINDINGS. '
              'Not confirmed out of sample.')


def run_provenance(run_dir):
    return json.loads((Path(run_dir) / 'provenance.json').read_text())


def significant_cells(run_dir):
    """[(domain, band), ...] BH-significant in the with-Control heatmap table."""
    t = pd.read_csv(Path(run_dir) / SIG_TABLE)
    t = t[t['p_bh_reject'].astype(str).str.lower() == 'true']
    order = {b: i for i, b in enumerate(BAND_ORDER)}
    t = t.assign(_o=t['band'].map(order)).sort_values(['_o', 'domain'])
    return list(zip(t['domain'], t['band']))


def tasks(run_dir):
    """Task list, index = array task id: (band, excluded subject)."""
    prov = run_provenance(run_dir)
    subjects = sorted(prov['subjects'])
    bands = [b for b in BAND_ORDER
             if b in {band for _, band in significant_cells(run_dir)}]
    return [(b, s) for b in bands for s in [NONE] + subjects]


def fit_one(run_dir, band, excluded, optimizer, force=False):
    prov = run_provenance(run_dir)
    frames_dir = Path(prov['params']['frames_dir'])
    out = Path(run_dir) / OUT_SUBDIR / LOO_DIR / f'{band}__{excluded}.csv'
    if out.exists() and not force:
        logger.info('%s exists; skipping (use --force to refit)', out.name)
        return 0

    df = io.read_table(frames_dir / f'{band}.parquet', on_stale='warn')
    keep = ['subject', 'channel_uid', 'log10_power', 'NRS_within',
            'NRS_submean', 'parcel', 'domain']
    work = (Path(os.environ.get('SCRATCH', '/tmp')) / 'domain_lmer_loo'
            / Path(run_dir).name)
    work.mkdir(parents=True, exist_ok=True)
    csv = work / f'{band}__{excluded}.csv'
    df[keep].to_csv(csv, index=False)

    cmd = ['Rscript', str(R_SCRIPT), '--in', str(csv), '--out', str(out),
           '--band', band, '--exclude', excluded, '--optimizer', optimizer]
    logger.info(' '.join(cmd))
    t0 = time.time()
    proc = subprocess.run(cmd, env=r_environment(), text=True,
                          capture_output=True)
    for line in (proc.stdout or '').splitlines():
        logger.info('[R] %s', line)
    for line in (proc.stderr or '').splitlines()[-40:]:
        (logger.error if proc.returncode else logger.warning)('[R] %s', line)
    logger.info('R rc=%d in %.1f s', proc.returncode, time.time() - t0)
    csv.unlink(missing_ok=True)
    return proc.returncode


def collect(run_dir):
    run_dir = Path(run_dir)
    todo = tasks(run_dir)
    loo = run_dir / OUT_SUBDIR / LOO_DIR
    missing = [f'{b}__{s}' for b, s in todo
               if not (loo / f'{b}__{s}.csv').exists()]
    if missing:
        raise SystemExit(f'{len(missing)} of {len(todo)} LOO fits missing, '
                         f'e.g. {missing[:5]}; rerun those array tasks.')
    tab = pd.concat([pd.read_csv(loo / f'{b}__{s}.csv') for b, s in todo],
                    ignore_index=True)
    prov = run_provenance(run_dir)
    params = {'formula': prov['params']['formula'],
              'group_slope': 'fixef(domain:NRS_within) + ranef(ROI)[NRS_within], '
                             'from a refit WITHOUT the excluded subject',
              'significant_cells': [list(c) for c in significant_cells(run_dir)],
              'sig_table': SIG_TABLE, 'n_tasks': len(todo)}
    io.write_table(tab, run_dir / OUT_SUBDIR / COLLECTED, params=params,
                   parents=[str(run_dir / 'provenance.json'),
                            prov['params']['frames_dir']],
                   subjects=sorted(prov['subjects']), script=SCRIPT,
                   extra={'status': DISCLAIMER})
    n_sing = int(tab.drop_duplicates(['band', 'excluded'])['singular'].sum())
    n_warn = int((tab.drop_duplicates(['band', 'excluded'])['n_warnings'] > 0).sum())
    logger.info('collected %d fits -> %s | singular %d | with warnings %d',
                len(todo), COLLECTED, n_sing, n_warn)
    return tab


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--run-dir', required=True)
    ap.add_argument('--task-id', type=int,
                    default=(int(os.environ['SLURM_ARRAY_TASK_ID'])
                             if 'SLURM_ARRAY_TASK_ID' in os.environ else None))
    ap.add_argument('--n-tasks', action='store_true',
                    help='print the number of array tasks and exit')
    ap.add_argument('--collect', action='store_true')
    ap.add_argument('--optimizer', default='bobyqa')
    ap.add_argument('--force', action='store_true')
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)s %(message)s')

    if args.n_tasks:
        print(len(tasks(args.run_dir)))
        return 0
    if args.collect:
        collect(args.run_dir)
        return 0
    if args.task_id is None:
        ap.error('--task-id (or SLURM_ARRAY_TASK_ID) is required')
    check_r()
    band, excluded = tasks(args.run_dir)[args.task_id]
    logger.info('task %d: band=%s exclude=%s', args.task_id, band, excluded)
    return fit_one(args.run_dir, band, excluded, args.optimizer, args.force)


if __name__ == '__main__':
    sys.exit(main())
