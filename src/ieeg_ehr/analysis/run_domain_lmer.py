#!/usr/bin/env python3
"""The PROCESSING DOMAIN model refitted in lme4, with crossed region effects.

    python -m ieeg_ehr.analysis.run_domain_lmer --frames-dir <run>/frames

    analysis/pain/bandpower/domain_lmer/<view_scheme>/<run>_<timestamp>/

WHAT THIS ADDS OVER `run_domain_model.py`. Three terms the statsmodels fit does
not have, and one it cannot have:

    (1 + NRS_within || ROI)        CROSSED -- statsmodels CANNOT fit this
    (1 + NRS_within || subj_roi)   subject's slope differs by region
    domain:NRS_submean             the between-subject term, per domain

The crossed one is the reason this script exists. `statsmodels.MixedLM` takes
ONE grouping variable and evaluates every `vc_formula` term within it, so an ROI
term crossing subjects comes back SILENTLY NESTED -- verified on synthetic data,
no error and no warning (`run_domain_model.py`, "WHY `subject:parcel` AND NOT
`domain:region`"). The ROI random slope is the guard that keeps one well-sampled
ROI from carrying its domain, and a nested version of it is not that guard. The
`--unit roi` runs dropped the region from the model ENTIRELY rather than nest it
dishonestly, which was the right call given the tool and is what this replaces.

WHAT IT DOES NOT CHANGE. The frame. Rows are still ONE CHANNEL x ONE EPOCH,
built by `mixed_model.build_cell_frame` exactly as the statsmodels run built
them -- this script reads that run's saved `frames/<band>.parquet` rather than
rebuilding, so a difference in results cannot come from a difference in data.
Run the source with `--save-frames` if the frames are not there.

READING THE OUTPUT. `domain_slopes.csv` is `emtrends(~ domain, var =
'NRS_within')` -- each domain's marginal pain slope, the same estimand as the
statsmodels run's `beta`, so the two are directly comparable and the comparison
is the point of running this. `domain_pairs.csv` is every pairwise domain
contrast (the statsmodels run gave only-vs-reference). `joint_tests.csv` is the
omnibus. `varcorr.csv` is where you look to see whether the new terms carried
anything: if the ROI and subj_roi variances come back at zero, the statsmodels
model was not missing much and its standard errors were closer to honest than
feared.

EXPLORATORY. Discovery cohort only. Nominations, not findings.
"""

import argparse
import logging
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

from ieeg_ehr import config, io

logger = logging.getLogger(__name__)

SCRIPT = 'ieeg_ehr/analysis/run_domain_lmer.py'
R_SCRIPT = Path(__file__).with_name('domain_lmer.R')

# Level 2 is the QUESTION, and the question has not changed -- "do the
# processing domains differ in how their power tracks pain" is the same one
# `run_domain_model.py` asks. Only the engine changed. Levels 1-2 are opened
# deliberately (CLAUDE.md), so this run lands in the EXISTING `bandpower`
# question beside the statsmodels run it is compared against, distinguished by
# the run name. A new level-2 folder here would claim a new question exists.
QUESTION = 'bandpower'
OUTPUT_TYPE = 'domain_model'
RUN_NAME = 'domain_lmer'

DISCLAIMER = ('EXPLORATORY -- discovery cohort, NOMINATIONS NOT FINDINGS. '
              'Not confirmed out of sample.')

#: The specification, recorded verbatim in provenance. Kept as a string beside
#: the R file so a provenance reader never has to open the R source to learn
#: what was fitted -- the one provenance error that cannot be caught later is a
#: sidecar that names a model the script did not fit.
FORMULA = ('log10_power ~ 0 + domain + domain:NRS_within + domain:NRS_submean '
           '+ (1 + NRS_within || ROI) + (1 + NRS_within || subject) '
           '+ (1 + NRS_within || subj_roi) + (1 | chan_id)')

#: The diagnosis-stratum variant. Full cell means over domain x dx: every
#: (domain, stratum) gets its own intercept and slope, so case-minus-control
#: within a domain is a contrast of two estimated slopes. `NRS_submean` is NOT
#: split by dx -- a between-subject term at n=51 (analyst's call). `dx` carries
#: no random effect: it is constant within subject and not estimable as one.
DX_FORMULA = ('log10_power ~ 0 + domain:dx + domain:dx:NRS_within '
              '+ domain:NRS_submean '
              '+ (1 + NRS_within || ROI) + (1 + NRS_within || subject) '
              '+ (1 + NRS_within || subj_roi) + (1 | chan_id)')

RANDOM_EFFECTS = {
    'ROI': '1 + NRS_within, CROSSED with subject',
    'subject': '1 + NRS_within',
    'subj_roi': '1 + NRS_within, subject x ROI',
    'chan_id': '1',
}


def r_environment():
    """Env for the R subprocess, with the group library on the path.

    `R_LIBS_USER` defaults under `$GROUP_HOME` because `$HOME` is 15 GB and NFS
    backed, and a compiled R library tree is neither small nor something worth
    backing up (org policy).
    """
    env = dict(os.environ)
    env.setdefault('R_LIBS_USER',
                   str(Path(os.environ.get('GROUP_HOME', Path.home()))
                       / 'R' / '4.4.2'))
    return env


def check_r():
    """Fail EARLY and by name, rather than at the first fit.

    A missing R package surfaces from `Rscript` as a traceback in a subprocess
    forty minutes into a job, which is the worst place to read it.
    """
    if shutil.which('Rscript') is None:
        raise SystemExit(
            'Rscript not found. `module load math R/4.4.2` first '
            '(see sbatch/domain_lmer.sbatch).')
    probe = ('for (p in c("data.table","lme4","lmerTest","emmeans")) '
             'if (!requireNamespace(p, quietly=TRUE)) '
             'cat("MISSING:", p, "\\n")')
    out = subprocess.run(['Rscript', '-e', probe], capture_output=True,
                         text=True, env=r_environment())
    missing = [ln.split(':', 1)[1].strip()
               for ln in out.stdout.splitlines() if ln.startswith('MISSING:')]
    if missing:
        raise SystemExit(
            f'R packages not installed: {", ".join(missing)}. Install into '
            f'$GROUP_HOME/R/4.4.2 (see sbatch/domain_lmer.sbatch).')


def fit_band(frame_path, band, run_dir, work_dir, args):
    """One lme4 fit. Returns the R subprocess's return code."""
    df = io.read_table(frame_path, on_stale='warn')

    keep = ['subject', 'channel_uid', 'log10_power', 'NRS_within',
            'NRS_submean', 'parcel', 'domain']
    missing = [c for c in keep if c not in df.columns]
    if missing:
        raise SystemExit(f'{frame_path} lacks {missing}. It was probably built '
                         'by a `--unit parcel` run; this model needs `--unit roi`.')

    csv = work_dir / f'{band}.csv'
    df[keep].to_csv(csv, index=False)
    logger.info('[%s] %d rows -> %s (%.0f MB)', band, len(df), csv,
                csv.stat().st_size / 1e6)

    cmd = ['Rscript', str(R_SCRIPT), '--in', str(csv),
           '--out', str(run_dir / 'bands'), '--band', band,
           '--df', args.df_method, '--optimizer', args.optimizer]
    if args.drop_domain:
        cmd += ['--drop-domain', ','.join(args.drop_domain)]

    logger.info('[%s] %s', band, ' '.join(cmd))
    t0 = time.time()
    proc = subprocess.run(cmd, env=r_environment(), text=True,
                          capture_output=True)
    for line in (proc.stdout or '').splitlines():
        logger.info('[R] %s', line)
    if proc.returncode != 0:
        for line in (proc.stderr or '').splitlines()[-40:]:
            logger.error('[R] %s', line)
    else:
        for line in (proc.stderr or '').splitlines():
            # lme4 sends convergence warnings to stderr on a SUCCESSFUL fit;
            # they are diagnostics, not failures, and belong in the log.
            logger.warning('[R] %s', line)
    logger.info('[%s] R finished rc=%d in %.1f s', band, proc.returncode,
                time.time() - t0)

    if not args.keep_csv:
        csv.unlink(missing_ok=True)
    return proc.returncode


def collect(run_dir, suffix):
    """Concatenate the per-band CSVs the R script wrote."""
    parts = sorted((run_dir / 'bands').glob(f'*_{suffix}.csv'))
    if not parts:
        return pd.DataFrame()
    return pd.concat([pd.read_csv(p) for p in parts], ignore_index=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--frames-dir', required=True,
                    help='<statsmodels run>/frames, holding <band>.parquet. '
                         'Re-run the source with --save-frames if absent.')
    ap.add_argument('--bands', nargs='*', default=None,
                    help='default: every frame in --frames-dir')
    ap.add_argument('--view-scheme', default=None,
                    help='level-4 folder; default taken from --frames-dir')
    ap.add_argument('--drop-domain', nargs='*', default=None,
                    help="e.g. Control. Default keeps every domain -- Control "
                         "is the quasi-control whose own slope says whether an "
                         "effect is global, so dropping it costs a diagnostic.")
    ap.add_argument('--df-method', default='satterthwaite',
                    choices=['satterthwaite', 'kenward-roger', 'asymptotic'],
                    help='Satterthwaite falls back to asymptotic on failure, '
                         'and the fallback is recorded per band in fitinfo.')
    ap.add_argument('--optimizer', default='bobyqa',
                    choices=['bobyqa', 'Nelder_Mead', 'nloptwrap'])
    ap.add_argument('--work-dir', default=None,
                    help='where the R input CSVs go; default $SCRATCH')
    ap.add_argument('--keep-csv', action='store_true',
                    help='keep the R input CSVs (debugging a fit by hand)')
    ap.add_argument('--question', default=QUESTION)
    ap.add_argument('--run-name', default=RUN_NAME)
    ap.add_argument('--run-dir', default=None,
                    help='PIN the run directory instead of timestamping a new '
                         'one. Required for a Slurm ARRAY: every task must '
                         'write into the SAME run, and each builds its own '
                         'timestamp otherwise -- six tasks, six run folders, '
                         'no collected result.')
    ap.add_argument('--collect-only', action='store_true',
                    help='Skip fitting; concatenate whatever bands/*.csv the '
                         'array left and write the run-level tables. Runs as '
                         'the dependent collect stage after the array.')
    args = ap.parse_args(argv)

    if args.collect_only and not args.run_dir:
        ap.error('--collect-only needs --run-dir: there is nothing to collect '
                 'without being told which run.')

    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)s %(message)s')
    check_r()

    frames_dir = Path(args.frames_dir)
    if not frames_dir.is_dir():
        raise SystemExit(f'no such frames directory: {frames_dir}')

    frames = {p.stem: p for p in sorted(frames_dir.glob('*.parquet'))}
    bands = args.bands or sorted(frames)
    unknown = [b for b in bands if b not in frames]
    if unknown:
        raise SystemExit(f'no frame for {unknown}; have {sorted(frames)}')

    # Level 4 mirrors the source run's view scheme, so the lme4 refit sits
    # beside the statsmodels run it is compared against.
    view_scheme = args.view_scheme or frames_dir.parent.parent.name
    run_dir = (Path(args.run_dir) if args.run_dir else
               config.analysis_run_dir(args.question, OUTPUT_TYPE,
                                       args.run_name, view_scheme=view_scheme))
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / 'bands').mkdir(exist_ok=True)

    work_dir = Path(args.work_dir or (Path(os.environ.get('SCRATCH', '/tmp'))
                                      / 'domain_lmer' / run_dir.name))
    work_dir.mkdir(parents=True, exist_ok=True)
    logger.info('run dir  %s', run_dir)
    logger.info('work dir %s', work_dir)

    failed = []
    if args.collect_only:
        logger.info('collect-only: gathering %s/bands', run_dir)
    else:
        for band in bands:
            if fit_band(frames[band], band, run_dir, work_dir, args) != 0:
                failed.append(band)
        if args.run_dir and not args.collect_only:
            # An ARRAY TASK stops here. Writing the run-level tables from each
            # task would race: six tasks concatenating a directory they are
            # still filling, each overwriting the last with a partial answer.
            # The dependent --collect-only stage writes them once, after all
            # tasks have landed.
            logger.info('array task done (%s); run-level tables are written '
                        'by the --collect-only stage', bands)
            return 1 if failed else 0

    slopes = collect(run_dir, 'slopes')
    pairs = collect(run_dir, 'pairs')
    varcorr = collect(run_dir, 'varcorr')
    fitinfo = collect(run_dir, 'fitinfo')

    if slopes.empty:
        raise SystemExit(f'every band failed; see {run_dir}/bands and the log')

    probe = io.read_table(frames[bands[0]], on_stale='warn')
    subjects = sorted(probe['subject'].unique())

    # The frame decides the model, so provenance must name the one that ran --
    # a sidecar claiming the pain-only formula on a dx run is the provenance
    # error that cannot be caught later, because the file looks self-consistent.
    has_dx = 'dx_state' in probe.columns
    n_case = int((probe.groupby('subject')['dx_state'].first() > 0.5).sum()) \
        if has_dx else None

    params = {
        'formula': DX_FORMULA if has_dx else FORMULA,
        'dx_model': bool(has_dx),
        'dx_n_case': n_case,
        'dx_n_control': (len(subjects) - n_case) if has_dx else None,
        'random_effects': RANDOM_EFFECTS,
        'engine': 'R lme4::lmer via lmerTest, REML=TRUE',
        'df_method_requested': args.df_method,
        'optimizer': args.optimizer,
        'dropped_domains': args.drop_domain or [],
        'frames_dir': str(frames_dir),
        'bands_fitted': [b for b in bands if b not in failed],
        'bands_failed': failed,
        'marginal_slopes': ("emtrends(~ dx | domain, var='NRS_within'); "
                            "pairs() gives case-minus-control WITHIN a domain"
                            if has_dx else
                            "emtrends(~ domain, var='NRS_within')"),
        'omnibus': ('NONE COMPUTED -- removed at the analyst\'s instruction. '
                    'joint_tests() is also the wrong test under 0 + domain '
                    'cell-means coding: it asks whether all domain slopes are '
                    'ZERO, not whether the domains DIFFER.'),
        'differs_from_statsmodels_run': (
            'adds (1 + NRS_within || ROI) CROSSED, which statsmodels cannot '
            'fit -- it evaluates every vc_formula within ONE grouping variable '
            'and would nest the term silently; adds (1 + NRS_within || '
            'subj_roi); makes NRS_submean domain-specific. Same frame, same '
            'rows: reads the statsmodels run\'s saved frames/<band>.parquet.'),
    }

    parents = [str(frames_dir)]
    common = dict(params=params, parents=parents, subjects=subjects,
                  script=SCRIPT, extra={'status': DISCLAIMER})

    # CSV under analysis/: small, terminal, read by eye (docs/io_conventions.md).
    io.write_table(slopes, run_dir / 'domain_slopes.csv', **common)
    io.write_table(pairs, run_dir / 'domain_pairs.csv', **common)
    io.write_table(varcorr, run_dir / 'varcorr.csv', **common)
    io.write_table(fitinfo, run_dir / 'fitinfo.csv', **common)
    io.write_run_provenance(run_dir, script=SCRIPT, params=params,
                            parents=parents, subjects=subjects,
                            extra={'status': DISCLAIMER})
    io.log_analysis('domain model refitted in lme4 with crossed ROI random '
                    'effects, discovery cohort', run_dir)

    logger.info('wrote %s', run_dir)
    if failed:
        logger.error('FAILED bands: %s', failed)
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
