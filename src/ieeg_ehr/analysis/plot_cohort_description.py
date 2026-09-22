#!/usr/bin/env python3
"""What the pain data looks like: one patient's timeline, then the whole cohort.

    <domain run>/cohort/<label>_<timestamp>/

Three figures, in the order a reader should meet them:

    fig_subject_grid.png     A panel per subject for the first N subjects --
                             the sanity check, and where the example is
                             CHOSEN from rather than assumed.
    fig_example_subject.png  One subject, large: hours since admission on x,
                             pain score on y, line with a dot per assessment.
    fig_cohort_violins.png   The three distributions the example cannot show:
                             assessments per patient, each patient's mean pain
                             score, and each patient's pain score range.

WHAT COUNTS AS AN ASSESSMENT, AND WHY IT IS NOT ALL OF THEM
------------------------------------------------------------
By default this describes the assessments THE MODEL ACTUALLY USED -- the pain
reports that became analysis epochs, i.e. those with a usable 5-minute window
of iEEG before them that survived QC. That is the right denominator for a
cohort description printed beside a result, and it is what the eligibility
criteria are stated on (>10 epochs, range >= 4, >= 5 non-modal scores).

It is NOT the same as the number of ratings the nurses recorded: a patient can
be rated through a gap in the recording. `--source ehr` describes every rating
in `pain-scores.csv` instead, and both counts are logged either way so the gap
between them is never invisible.

HOURS SINCE ADMISSION comes from the EHR CSVs' `session_start`, which
data_sop.md calls the authoritative session bound -- NOT from any NWB field,
and not from the first pain report. Using the first report would silently
redefine t=0 per patient as "whenever someone first asked", which is a
different quantity and would compress exactly the early-admission stretch a
timeline figure is for. The timestamps are offset-anchored, so the ELAPSED
hours are true and the absolute dates are fiction.

EXPLORATORY. Discovery cohort.
"""

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ieeg_ehr import config, io
from ieeg_ehr.analysis import fullres_cells, reference_run
from ieeg_ehr.analysis.run_fullres_grid import resolve_cohort
from ieeg_ehr.config import roi_schemes

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

SCRIPT = 'ieeg_ehr/analysis/plot_cohort_description.py'
SUBDIR = 'cohort'

DISCLAIMER = 'EXPLORATORY -- discovery cohort.'

#: One accent for the timeline, matching nothing else on purpose: this figure
#: is about the OUTCOME variable, not about a region or a domain, so borrowing
#: a domain hue would imply a link that is not there.
LINE_COLOUR = '#33608c'
VIOLIN_COLOUR = '#7aa6c2'


# ============================================================================
# DATA
# ============================================================================

def epoch_times(subjects, epoch_minutes=None):
    """(subject_id, session, epoch_id, pain_score, pain_time) for every epoch.

    `pain_time` lives only in `epoch_defs`, which the view does not carry, so
    the two are joined on (subject, session, epoch_id).
    """
    frames = []
    for sid in subjects:
        bare = str(sid).replace('sub-', '')
        for builder in (config.fullres_epoch_defs_path,
                        config.pain_epoch_defs_path):
            try:
                path = builder(bare, '01', epoch_minutes)
            except Exception:                                   # noqa: BLE001
                continue
            hits = sorted(path.parent.glob(f'sub-{bare}_ses-*_defs.parquet'))
            if hits:
                break
        else:
            hits = []
        if not hits:
            logger.warning('sub-%s: no epoch_defs found', bare)
            continue
        for h in hits:
            d = io.read_table(h, on_stale='ignore')
            keep = [c for c in ('epoch_id', 'pain_score', 'pain_time',
                                'session_id') if c in d.columns]
            d = d[keep].copy()
            d['subject_id'] = f'sub-{bare}'
            if 'session_id' in d:
                d['session'] = d['session_id'].astype(str).str.replace(
                    'ses-', '', regex=False)
            else:
                d['session'] = h.name.split('_ses-')[1][:2]
            frames.append(d)
    if not frames:
        raise SystemExit('no epoch_defs for any cohort subject')
    return pd.concat(frames, ignore_index=True)


def admission_starts(subjects):
    """{(subject, session): session_start} from the EHR pain-score exports.

    THE AUTHORITATIVE ADMISSION BOUND (data_sop.md 5.1). Not an NWB field --
    `session_start_time` there is a per-RUN quantity and comparing it across
    runs is a documented trap.
    """
    out = {}
    for sid in subjects:
        bare = str(sid).replace('sub-', '')
        for ses in ('01', '02', '03'):
            path = config.pain_scores_csv(bare, ses)
            if not path.exists():
                continue
            df = pd.read_csv(path, usecols=lambda c: c in (
                'session_start', 'session_end'))
            if 'session_start' not in df.columns or df.empty:
                continue
            out[(f'sub-{bare}', ses)] = pd.to_datetime(
                df['session_start'].iloc[0], errors='coerce')
    return out


def ehr_assessments(subjects):
    """Every rating in pain-scores.csv, for the `--source ehr` variant."""
    rows = []
    for sid in subjects:
        bare = str(sid).replace('sub-', '')
        for ses in ('01', '02', '03'):
            path = config.pain_scores_csv(bare, ses)
            if not path.exists():
                continue
            df = pd.read_csv(path)
            if not {'date', 'max_pain'} <= set(df.columns):
                continue
            df = df.assign(
                subject_id=f'sub-{bare}', session=ses,
                pain_time=pd.to_datetime(df['date'], errors='coerce'),
                pain_score=pd.to_numeric(df['max_pain'], errors='coerce'))
            rows.append(df[['subject_id', 'session', 'pain_time',
                            'pain_score']].dropna())
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def build(run_dir, args):
    """Long assessment frame with `hours` since admission, plus per-subject stats."""
    prov = json.loads((Path(run_dir) / 'provenance.json').read_text())
    params, subjects = prov.get('params', {}), sorted(prov.get('subjects', []))
    logger.info('cohort from the run: %d subject(s)', len(subjects))
    epoch_minutes = params.get('epoch_minutes')

    used = epoch_times(subjects, epoch_minutes)

    # The view is the authority on WHICH epochs the model saw -- epoch_defs can
    # hold reports the view later dropped to QC. Intersecting is what makes the
    # counts here match the counts the model was fitted on.
    if not args.no_view_filter:
        ref = reference_run.load(args.reference_run)
        view_dir = fullres_cells.resolve_view_dir(
            args.view_dir,
            mask_label=args.mask_label or ref.view_params.get('mask_label'),
            max_excluded_frac=ref.view_params.get('max_excluded_frac'),
            epoch_minutes=epoch_minutes)
        base = (roi_schemes.domain_scheme(params['roi_scheme'])['base']
                if params.get('unit') == 'roi' else params.get('roi_scheme'))
        paths, _, _, cohort, _, _ = resolve_cohort(
            ref, view_dir, cohort='reference', roi_scheme=base,
            insula_threshold=params.get('insula_threshold'))
        in_view = fullres_cells.load_epoch_scores(paths)[['subject_id',
                                                          'epoch_id']]
        n0 = len(used)
        used = used.merge(in_view.drop_duplicates(), on=['subject_id',
                                                         'epoch_id'])
        logger.info('epochs: %d in epoch_defs -> %d also in the view', n0,
                    len(used))

    ehr = ehr_assessments(subjects)
    logger.info('ASSESSMENT COUNTS: %d used by the model, %d recorded in the '
                'EHR across the same admissions (%.0f%% of ratings have a '
                'usable epoch)', len(used), len(ehr),
                100.0 * len(used) / max(len(ehr), 1))

    df = used if args.source == 'epochs' else ehr
    df = df[df['subject_id'].isin(subjects)].copy()

    starts = admission_starts(subjects)
    df['t0'] = [starts.get((s, ses)) for s, ses in zip(df['subject_id'],
                                                       df['session'])]
    missing = sorted(df.loc[df['t0'].isna(), 'subject_id'].unique())
    if missing:
        logger.warning('%d subject(s) have no EHR session_start; their '
                       'assessments are dropped from the TIMELINE but kept '
                       'in the distributions: %s', len(missing), missing)
    df['hours'] = (pd.to_datetime(df['pain_time'])
                   - pd.to_datetime(df['t0'])).dt.total_seconds() / 3600.0

    stats = (df.groupby('subject_id')
             .agg(n_assessments=('pain_score', 'size'),
                  mean_pain=('pain_score', 'mean'),
                  min_pain=('pain_score', 'min'),
                  max_pain=('pain_score', 'max'),
                  span_hours=('hours', lambda s: np.nan if s.isna().all()
                              else float(np.nanmax(s) - np.nanmin(s))))
             .reset_index())
    stats['pain_range'] = stats['max_pain'] - stats['min_pain']
    logger.info('\n%s', stats.describe().to_string())
    return df, stats, subjects, params


# ============================================================================
# FIGURES
# ============================================================================

def _style(ax, args):
    ax.spines[['top', 'right']].set_visible(False)
    ax.tick_params(labelsize=args.font_tick)


def draw_timeline(ax, sub, args, label=None, small=False):
    d = sub.dropna(subset=['hours']).sort_values('hours')
    ax.plot(d['hours'], d['pain_score'], '-', color=LINE_COLOUR,
            lw=1.1 if small else 1.8, alpha=0.75, zorder=1)
    ax.plot(d['hours'], d['pain_score'], 'o', color=LINE_COLOUR,
            ms=3.2 if small else 6.5, mec='white',
            mew=0.4 if small else 0.9, zorder=2)
    ax.set_ylim(-0.6, 10.6)
    ax.set_yticks([0, 5, 10])
    if label:
        ax.set_title(label, fontsize=args.font_tick if small
                     else args.font_label)
    _style(ax, args)


def fig_grid(df, stats, out_path, args):
    """A panel per subject. THE FIGURE THE EXAMPLE IS CHOSEN FROM.

    Drawing one patient large and calling them representative, without ever
    looking at the others, is how a figure ends up showing the prettiest
    timeline in the cohort. This is the contact sheet: same axes on every
    panel, sorted so the reader can see the range of shapes -- flat, spiky,
    sparse, long -- that the single example is standing in for.
    """
    subs = list(stats.sort_values('n_assessments', ascending=False)
                ['subject_id'])[:args.grid_n]
    ncol = args.grid_cols
    nrow = int(np.ceil(len(subs) / ncol))
    fig, axes = plt.subplots(nrow, ncol, squeeze=False, sharey=True,
                             figsize=(3.1 * ncol, 2.0 * nrow))
    for k, sid in enumerate(subs):
        ax = axes[k // ncol][k % ncol]
        s = stats[stats['subject_id'] == sid].iloc[0]
        draw_timeline(ax, df[df['subject_id'] == sid], args, small=True,
                      label=f'{sid.replace("sub-", "")}  '
                            f'n={int(s.n_assessments)}')
        if k % ncol == 0:
            ax.set_ylabel('Pain score', fontsize=args.font_tick)
        if k // ncol == nrow - 1:
            ax.set_xlabel('Hours since admission', fontsize=args.font_tick)
    for k in range(len(subs), nrow * ncol):
        axes[k // ncol][k % ncol].axis('off')
    fig.tight_layout()
    fig.savefig(out_path, dpi=args.dpi, bbox_inches='tight', pad_inches=0.05)
    plt.close(fig)
    logger.info('wrote %s (%d subjects)', out_path.name, len(subs))


def fig_example(df, stats, sid, out_path, args):
    fig, ax = plt.subplots(figsize=tuple(args.example_figsize))
    draw_timeline(ax, df[df['subject_id'] == sid], args)
    ax.set_xlabel('Hours since admission', fontsize=args.font_label)
    ax.set_ylabel('Pain score', fontsize=args.font_label)
    fig.tight_layout()
    fig.savefig(out_path, dpi=args.dpi, bbox_inches='tight', pad_inches=0.05)
    plt.close(fig)
    logger.info('wrote %s (%s)', out_path.name, sid)


def fig_violins(stats, out_path, args):
    """Three distributions, each with its own axis because the units differ.

    EVERY PATIENT IS DRAWN AS A POINT over the violin. At n=51 a kernel
    density is a smooth summary of a small sample and can imply structure the
    data does not have -- a bump where two patients sit. The strip makes the
    actual sample visible, and the violin is then just a readable envelope.
    """
    panels = [('n_assessments', 'Assessments per patient', '{:.0f}'),
              ('mean_pain', 'Mean pain score', '{:.1f}'),
              ('pain_range', 'Pain score range', '{:.0f}')]
    fig, axes = plt.subplots(1, 3, figsize=tuple(args.violin_figsize))
    rng = np.random.default_rng(0)

    for ax, (col, label, fmt) in zip(axes, panels):
        v = stats[col].to_numpy(dtype=float)
        v = v[np.isfinite(v)]
        parts = ax.violinplot([v], showextrema=False, widths=0.85)
        for b in parts['bodies']:
            b.set_facecolor(VIOLIN_COLOUR)
            b.set_alpha(0.55)
            b.set_edgecolor('0.35')
            b.set_linewidth(0.8)
        ax.scatter(1 + rng.uniform(-0.09, 0.09, len(v)), v, s=22,
                   color='0.22', alpha=0.65, edgecolor='none', zorder=3)
        med = float(np.median(v))
        ax.hlines(med, 0.62, 1.38, color='#b03a2e', lw=2.2, zorder=4)
        # Median and range in the corner rather than a legend: the numbers a
        # reader would otherwise have to estimate off the axis.
        ax.text(0.5, 1.02, f'median {fmt.format(med)}   '
                           f'({fmt.format(np.min(v))}–{fmt.format(np.max(v))})',
                transform=ax.transAxes, ha='center', va='bottom',
                fontsize=args.font_tick, color='0.3')
        ax.set_ylabel(label, fontsize=args.font_label)
        ax.set_xticks([])
        ax.set_xlim(0.45, 1.55)
        _style(ax, args)
        ax.spines['bottom'].set_visible(False)

    axes[1].set_ylim(0, 10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=args.dpi, bbox_inches='tight', pad_inches=0.05)
    plt.close(fig)
    logger.info('wrote %s', out_path.name)


def pick_example(stats, df):
    """The most ORDINARY subject, not the best-looking one.

    Ranked by summed absolute z-score across the three quantities the violins
    show, so the example is near the cohort median on all of them at once. A
    patient with a striking timeline is the wrong illustration of a typical
    one, and choosing by eye is how that happens.
    """
    cols = ['n_assessments', 'mean_pain', 'pain_range']
    ok = stats.dropna(subset=cols).copy()
    if ok.empty:
        return stats['subject_id'].iloc[0], stats
    z = ok[cols].apply(lambda c: (c - c.mean()) / (c.std(ddof=0) or 1.0))
    ok['typicality'] = z.abs().sum(axis=1)
    # Needs enough points to look like a timeline at all.
    enough = ok[ok['n_assessments'] >= ok['n_assessments'].median()]
    pool = enough if len(enough) else ok
    pool = pool.sort_values('typicality')
    return pool['subject_id'].iloc[0], pool


# ============================================================================

def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--run-dir', required=True)
    ap.add_argument('--source', choices=['epochs', 'ehr'], default='epochs',
                    help="'epochs' (default) describes the assessments the "
                         "MODEL used; 'ehr' describes every rating recorded. "
                         'Both counts are logged either way.')
    ap.add_argument('--example-subject', default=None,
                    help='sub-XXX. Default: the most typical subject by the '
                         'three violin quantities, which is chosen rather '
                         'than eyeballed for a reason -- see pick_example.')
    ap.add_argument('--grid-n', type=int, default=16)
    ap.add_argument('--grid-cols', type=int, default=4)
    ap.add_argument('--example-figsize', nargs=2, type=float,
                    default=[9.0, 4.0], metavar=('W', 'H'))
    ap.add_argument('--violin-figsize', nargs=2, type=float,
                    default=[11.0, 4.2], metavar=('W', 'H'))
    ap.add_argument('--font-label', type=float, default=15)
    ap.add_argument('--font-tick', type=float, default=13)
    ap.add_argument('--dpi', type=int, default=400)
    ap.add_argument('--reference-run', default=str(reference_run.CONTPAIN_HEATMAP))
    ap.add_argument('--view-dir', default=None)
    ap.add_argument('--mask-label', default=None)
    ap.add_argument('--no-view-filter', action='store_true',
                    help='Skip intersecting epoch_defs with the view. Faster, '
                         'and counts then include epochs the model dropped.')
    ap.add_argument('--label', default='cohort')
    args = ap.parse_args()

    io.warn_if_dirty()
    run_dir = Path(args.run_dir)
    df, stats, subjects, params = build(run_dir, args)

    sid = args.example_subject
    if sid is None:
        sid, pool = pick_example(stats, df)
        logger.info('example subject %s, the most typical by the three '
                    'violin quantities. Next candidates: %s', sid,
                    list(pool['subject_id'][1:6]))
    if sid not in set(stats['subject_id']):
        raise SystemExit(f'{sid} is not in this run\'s cohort')

    stamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    out_dir = run_dir / SUBDIR / f'{args.label}_{stamp}'
    out_dir.mkdir(parents=True, exist_ok=True)
    logger.info('versioned output dir: %s', out_dir)

    # The grid FIRST, because it is what the example is chosen from.
    fig_grid(df, stats, out_dir / 'fig_subject_grid.png', args)
    fig_example(df, stats, sid, out_dir / 'fig_example_subject.png', args)
    fig_violins(stats, out_dir / 'fig_cohort_violins.png', args)

    io.write_table(stats, out_dir / 'cohort_subjects.csv', script=SCRIPT,
                   parents=[str(run_dir / 'provenance.json')],
                   subjects=sorted(stats['subject_id']),
                   params={'source': args.source,
                           'view_filtered': not args.no_view_filter},
                   extra={'reading': 'one row per subject: how many pain '
                                     'assessments entered the analysis, and '
                                     'the mean and range of those scores'})
    io.write_table(df.drop(columns=['t0'], errors='ignore'),
                   out_dir / 'cohort_assessments.csv', script=SCRIPT,
                   parents=[str(run_dir / 'provenance.json')],
                   subjects=sorted(stats['subject_id']),
                   params={'source': args.source},
                   extra={'hours': 'since the EHR session_start, the '
                                   'authoritative admission bound; timestamps '
                                   'are offset-anchored so elapsed hours are '
                                   'true and absolute dates are fiction'})

    io.write_run_provenance(
        out_dir, script=SCRIPT,
        params={**vars(args), 'source_run': str(run_dir),
                'n_subjects': int(len(stats)),
                'n_assessments': int(len(df)),
                'example_subject': sid},
        parents=[str(run_dir / 'provenance.json')],
        subjects=sorted(stats['subject_id']),
        extra={'status': DISCLAIMER,
               'assessment_definition':
                   'a pain report that became an analysis epoch (a usable '
                   '5-minute pre-report iEEG window surviving QC), NOT every '
                   'rating the nurses recorded -- see the log for both counts',
               'time_origin':
                   "the EHR CSVs' session_start (data_sop.md 5.1), not an NWB "
                   'field and not the first pain report',
               'example_choice':
                   'the subject nearest the cohort median on all three violin '
                   'quantities at once, chosen by rank rather than by eye'})

    io.log_analysis(
        f'cohort description: pain assessment timeline and distributions for '
        f'{len(stats)} discovery subjects ({args.source})', out_dir)
    logger.info('done -> %s', out_dir)


if __name__ == '__main__':
    main()
