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
from ieeg_ehr.med_analysis import style

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
MEDIAN_COLOUR = '#b03a2e'

#: ONE STYLE FOR ALL THREE FIGURES. They are printed together, so a reader
#: reads them as one object and any difference in tick size, spine weight or
#: hue reads as meaning. Applied through `_style` rather than rcParams so the
#: two font sizes stay command-line arguments.
SPINE_WIDTH = 1.0

#: The combined figure (example subject over violins), in the med-figure house
#: style. ONE hue for everything -- the domain-model Modulatory blue, Sabra's
#: pick 2026-09-22 -- with the parts told apart by alpha rather than by colour:
#: a faint violin envelope, near-opaque patient dots, and a darker shade of the
#: same hue for the median so it reads as a summary of the dots, not a new
#: series. Saved at EXACTLY the declared size (no bbox_inches='tight').
COMBINED_COLOUR = '#447eae'
COMBINED_MEDIAN_COLOUR = '#1f3f5b'
COMBINED_WIDTH = 8.2
COMBINED_DPI = 300
GRID_MARKER, GRID_LINE = 3.2, 1.1
BIG_MARKER, BIG_LINE = 6.0, 1.7


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

def _style(ax, args, xlabel=None, ylabel=None):
    """The shared look: same spines, same tick size, same label size."""
    ax.spines[['top', 'right']].set_visible(False)
    for side in ('left', 'bottom'):
        ax.spines[side].set_linewidth(SPINE_WIDTH)
        ax.spines[side].set_color('0.25')
    ax.tick_params(labelsize=args.font_tick, width=SPINE_WIDTH, color='0.25')
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=args.font_label)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=args.font_label)


def draw_timeline(ax, sub, args, label=None, small=False, colour=LINE_COLOUR,
                  ms=None, lw=None):
    d = sub.dropna(subset=['hours']).sort_values('hours')
    lw = lw or (GRID_LINE if small else BIG_LINE)
    ms = ms or (GRID_MARKER if small else BIG_MARKER)
    ax.plot(d['hours'], d['pain_score'], '-', color=colour,
            lw=lw, alpha=0.75, zorder=1)
    ax.plot(d['hours'], d['pain_score'], 'o', color=colour,
            ms=ms, mec='white', mew=0.4 if small else 0.9, zorder=2)
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
            ax.set_ylabel('Pain score', fontsize=args.font_label)
        if k // ncol == nrow - 1:
            ax.set_xlabel('Hours since admission', fontsize=args.font_label)
    for k in range(len(subs), nrow * ncol):
        axes[k // ncol][k % ncol].axis('off')
    fig.tight_layout()
    fig.savefig(out_path, dpi=args.dpi, bbox_inches='tight', pad_inches=0.05)
    plt.close(fig)
    logger.info('wrote %s (%d subjects)', out_path.name, len(subs))


def fig_example(df, stats, sid, out_path, args):
    fig, ax = plt.subplots(figsize=tuple(args.example_figsize))
    draw_timeline(ax, df[df['subject_id'] == sid], args)
    _style(ax, args, xlabel='Hours since admission', ylabel='Pain score')
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
    panels = VIOLIN_PANELS
    fig, axes = plt.subplots(1, 3, figsize=tuple(args.violin_figsize))
    rng = np.random.default_rng(0)

    for ax, (col, label, fmt) in zip(axes, panels):
        draw_violin(ax, stats[col], label, fmt, args, rng)
        _style(ax, args, ylabel=label)
        # THE BOTTOM SPINE STAYS, the tick labels do not. Hiding the spine
        # leaves the violin floating with nothing to sit on; the labels would
        # only ever read "1", which is an artefact of violinplot's positional
        # x and says nothing.
        ax.tick_params(axis='x', length=0)

    axes[1].set_ylim(0, 10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=args.dpi, bbox_inches='tight', pad_inches=0.05)
    plt.close(fig)
    logger.info('wrote %s', out_path.name)


VIOLIN_PANELS = [('n_assessments', 'Assessments per patient', '{:.0f}'),
                 ('mean_pain', 'Mean pain score', '{:.1f}'),
                 ('pain_range', 'Pain score range', '{:.0f}')]


def draw_violin(ax, values, label, fmt, args, rng, colours=None,
                annot_size=None, dot_size=20):
    """One violin, every patient as a dot over it, the median as a bar.

    `colours` = (body, body_alpha, edge, dot, dot_alpha, median); the default
    is this script's original look.
    """
    body, body_a, edge, dot, dot_a, median = colours or (
        VIOLIN_COLOUR, 0.55, '0.35', '0.22', 0.65, MEDIAN_COLOUR)
    v = values.to_numpy(dtype=float)
    v = v[np.isfinite(v)]
    parts = ax.violinplot([v], showextrema=False, widths=args.violin_width)
    for b in parts['bodies']:
        b.set_facecolor(body)
        b.set_alpha(body_a)
        b.set_edgecolor(edge)
        b.set_linewidth(0.8)
    jitter = args.violin_width * 0.22
    ax.scatter(1 + rng.uniform(-jitter, jitter, len(v)), v, s=dot_size,
               color=dot, alpha=dot_a, edgecolor='none', zorder=3)
    med = float(np.median(v))
    half = args.violin_width * 0.62
    ax.hlines(med, 1 - half, 1 + half, color=median, lw=2.2, zorder=4)
    # THE MEDIAN ALONE. The min-max in parentheses was redundant with the
    # figure it sat on -- every patient is already drawn as a dot, so the
    # extremes are the top and bottom dots -- and it made three short
    # headings into three long ones.
    txt = f'median {fmt.format(med)}'
    if args.violin_annot == 'median_range':
        txt += f'   {fmt.format(np.min(v))}–{fmt.format(np.max(v))}'
    if args.violin_annot != 'none':
        ax.text(0.5, 1.02, txt, transform=ax.transAxes, ha='center',
                va='bottom', fontsize=annot_size or args.font_tick,
                color='0.3')
    ax.set_xticks([])
    # Tight to the violin, or a narrow violin just sits in a wide empty
    # panel and looks smaller rather than neater.
    ax.set_xlim(1 - args.violin_width, 1 + args.violin_width)


def fig_combined(df, stats, sid, out_path, args):
    """The example subject on top, the three cohort violins beneath.

    HOUSE STYLE (med_analysis.style + GROUPED_SIZES from
    plot_poster_epoch_meds: labels 11, ticks 9.5, legend 8.5 pt), light-grey
    left/bottom spines, black tick labels, no grid, one hue throughout.

    `--combined-top-height` / `--combined-bottom-height` are the two ROW
    SLOTS in inches, each including its own labels, so the file is
    8.2 x (top + bottom) inches exactly. Laid out in inches rather than a
    gridspec so the slot heights are what they say.
    """
    from ieeg_ehr.med_analysis.plot_poster_epoch_meds import GROUPED_SIZES

    saved = {k: getattr(style, k) for k in GROUPED_SIZES}
    for k, v in GROUPED_SIZES.items():
        setattr(style, k, v)
    try:
        W = COMBINED_WIDTH
        top_h, bot_h = args.combined_top_height, args.combined_bottom_height
        H = top_h + bot_h
        fig = plt.figure(figsize=(W, H))
        col = args.combined_colour

        def house(ax):
            style.style_axes(ax, grid_axis=None, tick_color=style.TEXT_PRIMARY)

        # Margins, inches. `inner` is the gap between violin panels, which has
        # to hold the next panel's tick labels and rotated y label.
        left, right, inner = 0.62, 0.06, 0.72
        x_w = W - left - right

        # -- top: the example subject ----------------------------------------
        t_below, t_above = 0.44, 0.24       # x label below; legend above
        ax_t = fig.add_axes((left / W, (bot_h + t_below) / H, x_w / W,
                             (top_h - t_below - t_above) / H))
        draw_timeline(ax_t, df[df['subject_id'] == sid], args, colour=col,
                      ms=4.0, lw=1.3)
        house(ax_t)
        style.label_axes(ax_t, xlabel='Hours since admission',
                         ylabel='Pain score')
        handle = matplotlib.lines.Line2D(
            [], [], color=col, lw=1.3, marker='o', ms=4.0, mec='white',
            mew=0.9)
        ax_t.legend([handle], [sid.replace('sub-', 'subject-')],
                    loc='lower right', bbox_to_anchor=(1.0, 1.0),
                    frameon=False, fontsize=style.LEGEND_SIZE, borderaxespad=0.1,
                    handlelength=1.8)

        # -- bottom: the three violins ---------------------------------------
        b_below, b_above = 0.08, 0.26       # "median N" above each violin
        p_w = (x_w - 2 * inner) / 3
        p_h = bot_h - b_below - b_above
        rng = np.random.default_rng(0)
        colours = (col, 0.25, col, col, 0.85, COMBINED_MEDIAN_COLOUR)
        axes = []
        for i, (c, label, fmt) in enumerate(VIOLIN_PANELS):
            ax = fig.add_axes(((left + i * (p_w + inner)) / W, b_below / H,
                               p_w / W, p_h / H))
            draw_violin(ax, stats[c], label, fmt, args, rng, colours=colours,
                        annot_size=style.TICK_SIZE, dot_size=11)
            house(ax)
            style.label_axes(ax, ylabel=label)
            ax.tick_params(axis='x', length=0)
            axes.append(ax)
        axes[1].set_ylim(0, 10)
        fig.align_ylabels([ax_t, axes[0]])

        fig.canvas.draw()
        rend = fig.canvas.get_renderer()
        off = []
        for t in fig.findobj(matplotlib.text.Text):
            if not t.get_visible() or not t.get_text().strip():
                continue
            e = t.get_window_extent(rend)
            if (e.x0 < -0.5 or e.y0 < -0.5 or e.x1 > fig.bbox.width + 0.5
                    or e.y1 > fig.bbox.height + 0.5):
                off.append(t.get_text())
        if off:
            logger.warning('text off the canvas: %s', off)
        fig.savefig(out_path, dpi=COMBINED_DPI, facecolor='white')
        plt.close(fig)
        logger.info('wrote %s  (%.2f x %.2f in at %d dpi, %s)', out_path.name,
                    W, H, COMBINED_DPI, sid)
    finally:
        for k, v in saved.items():
            setattr(style, k, v)


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
                    default=[10.0, 2.6], metavar=('W', 'H'),
                    help='SHORT on purpose: a pain timeline is a 0-10 score, '
                         'so a tall panel spends its height on empty range '
                         'rather than on resolving the trace.')
    ap.add_argument('--violin-figsize', nargs=2, type=float,
                    default=[9.0, 3.6], metavar=('W', 'H'))
    ap.add_argument('--violin-width', type=float, default=0.45,
                    help='Violin width in axis units. The x limits track it, '
                         'so a narrower violin does not just sit in a wider '
                         'empty panel.')
    ap.add_argument('--violin-annot', default='median',
                    choices=['median', 'median_range', 'none'])
    ap.add_argument('--combined-top-height', type=float, default=2.0,
                    help='Inches for the example-subject row of '
                         'fig_cohort_combined.png, labels included.')
    ap.add_argument('--combined-bottom-height', type=float, default=2.5,
                    help='Inches for the violin row, labels included.')
    ap.add_argument('--combined-colour', default=COMBINED_COLOUR)
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
    fig_combined(df, stats, sid, out_dir / 'fig_cohort_combined.png', args)

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
