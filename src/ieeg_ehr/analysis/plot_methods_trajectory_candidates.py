"""Candidate 8-h windows for the pain_change methods figure: pain and opioid exposure.

    python -m ieeg_ehr.analysis.plot_methods_trajectory_candidates

A candidate window is [t_a - 5 min, t_a - 5 min + `--window-h`) for each
assessment t_a of a discovery session, and holds every assessment whose 5-min
pre-score epoch lies inside it. A window passes when

- it holds at least `--min-scores` assessments,
- at least one scores above `--high-pain`,
- at most `--max-zeros` score 0,
- at least one opioid dose falls inside it,
- every assessment in it has an epoch in the pain epoch cache (recorded iEEG)
  and a QC verdict for that epoch's run,
- at least `--min-clean-pair-frac` of the bipolar pairs are CLEAN: no 60 s bin
  of any of the window's epochs is masked on the pair (`--mask-label`). The
  check covers the epochs Δpower uses, not the whole window, and runs per pair:
  the mask flags ~2% of (pair, bin) cells, so a window with no masked cell on
  any pair is rare (6 of 589 on 2026-10-21). The neural panel shows clean pairs
  only,
- it starts at least `--exclude-first-h` after session_start, as in the
  `nofirst24h` pain_change runs: the MAR export lacks pre-session doses.

`pick` takes up to `--n-options` passing windows, ranked by pain range, then
number of assessments, then number of opioid doses: first the best window of
each subject, then further windows that overlap no window already taken from
the same session.

Each option is two stacked panels on one time axis, hours from window start:
the pain scores, and the active-MME curve (`mme_curve.exposure_curve`, every
dose of the session) with each opioid dose a point on it. No neural data yet.

Writes `options.png`, `options.csv` (the plotted windows, numbered as in the
figure), `candidates.csv` (every passing window) and `provenance.json` to
`analysis/methods_figures/pain_change/trajectory_candidates/<run>_<timestamp>/`.
"""

import argparse
import logging

import numpy as np
import pandas as pd

from ieeg_ehr import config, io
from ieeg_ehr.analysis.pain_change import load_exposure
from ieeg_ehr.med_analysis import mme_curve, pain_link, style
from ieeg_ehr.med_analysis.style import plt
from ieeg_ehr.qc import mask_projection
from ieeg_ehr.views import cache_reader

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

SCRIPT = 'ieeg_ehr/analysis/plot_methods_trajectory_candidates.py'
EVENT = 'methods_figures'
QUESTION = config.PAIN_CHANGE_QUESTION
OUTPUT_TYPE = 'trajectory_candidates'

PAIN_COLOR = style.PALETTE[0]
MME_COLOR = style.PALETTE[1]
ROUTE_MARKER = {'PO': 'o', 'IV': '^'}
NCOLS = 4
CELL_W, CELL_H = 4.8, 3.0


def epoch_qc(subject, session, mask_label):
    """One row per assessment with an epoch: pain_time, pairs, masked_pairs.

    `masked_pairs` holds the pairs with a masked bin anywhere in the epoch. An
    epoch whose run is absent from the mask has no QC verdict and gets no row.
    """
    defs = cache_reader.load_defs(subject, session)
    mask = io.read_table(config.bipolar_mask_path(subject, session, mask_label),
                         on_stale='warn')
    channels = {r: sorted(g['channel'].unique()) for r, g in mask.groupby('run_id')}
    rows = []
    for e in defs.itertuples():
        if e.run_id not in channels:
            logger.warning('sub-%s ses-%s %s: run absent from the mask', subject,
                           session, e.run_id)
            continue
        pairs = channels[e.run_id]
        seconds = e.epoch_start_sec + np.arange(e.n_windows) * e.hop_sec
        hit = mask_projection.project_pair_mask_to_windows(
            mask, e.run_id, pairs, seconds).any(axis=0)
        rows.append(dict(pain_time=e.pain_time, pairs=frozenset(pairs),
                         masked_pairs=frozenset(np.asarray(pairs)[hit])))
    return (pd.DataFrame(rows).groupby('pain_time', as_index=False)
            .agg(pairs=('pairs', lambda x: frozenset().union(*x)),
                 masked_pairs=('masked_pairs', lambda x: frozenset().union(*x))))


def session_windows(scores, doses, args):
    """Every window anchored on an assessment of one session, pass or fail."""
    epoch_h = config.EPOCH_MINUTES_BEFORE / 60.0
    t = scores['t_h'].to_numpy()
    rows = []
    for t_a in t:
        w0 = t_a - epoch_h
        w1 = w0 + args.window_h
        inside = window_scores(scores, w0, w1)
        d = doses[(doses['t_h'] >= w0) & (doses['t_h'] < w1)]
        pain = inside['pain_score']
        recorded = inside.dropna(subset=['pairs'])
        pairs = frozenset().union(*recorded['pairs'])
        masked = frozenset().union(*recorded['masked_pairs'])
        rows.append(dict(
            start_h=w0, end_h=w1, n_scores=len(inside),
            n_high=int((pain > args.high_pain).sum()),
            n_zero=int((pain == 0).sum()),
            pain_min=float(pain.min()), pain_max=float(pain.max()),
            pain_range=float(pain.max() - pain.min()),
            n_doses=len(d), dose_mme=float(d['mme'].sum()),
            n_unrecorded=len(inside) - len(recorded),
            n_pairs=len(pairs), n_clean_pairs=len(pairs - masked),
            clean_pair_frac=len(pairs - masked) / len(pairs) if pairs else 0.0,
            score_times_h=';'.join(f'{x:.3f}' for x in inside['t_h']),
            scores=';'.join(f'{x:g}' for x in pain)))
    return pd.DataFrame(rows)


def pick(candidates, n):
    """Best window per subject, then non-overlapping extras per session."""
    ranked = candidates.sort_values(
        ['pain_range', 'n_scores', 'n_doses', 'clean_pair_frac'], ascending=False)
    chosen = ranked.drop_duplicates('subject')
    for r in ranked.drop(index=chosen.index).itertuples():
        if len(chosen) >= n:
            break
        same = chosen[(chosen['subject'] == r.subject) & (chosen['session'] == r.session)]
        if ((same['start_h'] < r.end_h) & (same['end_h'] > r.start_h)).any():
            continue
        chosen = pd.concat([chosen, candidates.loc[[r.Index]]])
    return chosen.head(n).reset_index(drop=True)


def window_scores(scores, start_h, end_h):
    """The assessments whose 5-min pre-score epoch lies inside [start_h, end_h]."""
    epoch_h = config.EPOCH_MINUTES_BEFORE / 60.0
    return scores[(scores['t_h'] - epoch_h >= start_h) & (scores['t_h'] <= end_h)]


def draw_pain(ax, scores, start_h):
    """Pain scores against hours from `start_h`. Returns the score x positions."""
    x = scores['t_h'] - start_h
    ax.plot(x, scores['pain_score'], color=PAIN_COLOR, lw=2, zorder=3)
    ax.scatter(x, scores['pain_score'], s=40, color=PAIN_COLOR,
               edgecolor='white', linewidth=1.5, zorder=4)
    ax.set_ylim(-0.8, 10.8)
    ax.set_yticks([0, 5, 10])
    return x


def draw_mme(ax, doses, start_h, end_h):
    """Active-MME curve from every dose of the session, in-window doses as points."""
    t = np.arange(start_h, end_h, mme_curve.GRID_STEP_H)
    curve = mme_curve.exposure_curve(doses, t)
    ax.plot(t - start_h, curve, color=MME_COLOR, lw=2, zorder=3)
    inside = doses[(doses['t_h'] >= start_h) & (doses['t_h'] < end_h)]
    for route, g in inside.groupby('route_class'):
        ax.scatter(g['t_h'] - start_h,
                   mme_curve.exposure_curve(doses, g['t_h'].to_numpy()),
                   s=45, marker=ROUTE_MARKER[route], color=MME_COLOR,
                   edgecolor='white', linewidth=1.5, zorder=4)
    ax.set_ylim(0, max(float(curve.max()) * 1.15, 1.0))


def legend_handles():
    return [plt.Line2D([], [], color=PAIN_COLOR, lw=2, marker='o', label='Pain score'),
            plt.Line2D([], [], color=MME_COLOR, lw=2, label='Estimated active MME'),
            plt.Line2D([], [], color=MME_COLOR, lw=0, marker='o', label='Opioid dose, PO'),
            plt.Line2D([], [], color=MME_COLOR, lw=0, marker='^', label='Opioid dose, IV')]


def plot_options(options, scores, doses, out_path, args):
    nrows = int(np.ceil(len(options) / NCOLS))
    height = CELL_H * nrows + 1.6
    fig = plt.figure(figsize=(CELL_W * NCOLS, height))
    outer = fig.add_gridspec(nrows, NCOLS, hspace=0.5, wspace=0.35,
                             top=1 - 1.0 / height, bottom=0.8 / height)
    for i, o in options.iterrows():
        cell = outer[i // NCOLS, i % NCOLS].subgridspec(2, 1, hspace=0.12)
        ax_p = fig.add_subplot(cell[0])
        ax_c = fig.add_subplot(cell[1], sharex=ax_p)
        key = (o['subject'], o['session'])
        s = window_scores(scores[key], o['start_h'], o['end_h'])
        d = doses[key]

        x = draw_pain(ax_p, s, o['start_h'])
        ax_p.tick_params(labelbottom=False)
        draw_mme(ax_c, d, o['start_h'], o['end_h'])
        ax_c.set_xlim(0, args.window_h)
        ax_c.set_xticks(np.arange(0, args.window_h + 0.1, 2))

        for ax in (ax_p, ax_c):
            for tx in x:
                ax.axvline(tx, color=style.GRID_COLOR, lw=1, zorder=1)
            style.style_axes(ax, grid_axis=None)
        ax_p.set_title(
            f'{i + 1}.  sub-{o["subject"]} ses-{o["session"]}, from h {o["start_h"]:.1f}, '
            f'{o["n_clean_pairs"]}/{o["n_pairs"]} clean pairs\n'
            f'{o["n_scores"]} scores ({o["pain_min"]:g}-{o["pain_max"]:g}), '
            f'{o["n_doses"]} opioid dose{"s" if o["n_doses"] != 1 else ""} '
            f'({o["dose_mme"]:.0f} MME)',
            fontsize=style.TICK_SIZE, color=style.TEXT_PRIMARY, loc='left', pad=3)
        if i % NCOLS == 0:
            ax_p.set_ylabel('Pain (0-10)', fontsize=style.TICK_SIZE,
                            color=style.TEXT_PRIMARY)
            ax_c.set_ylabel('Active MME (mg)', fontsize=style.TICK_SIZE,
                            color=style.TEXT_PRIMARY)
        if i >= len(options) - NCOLS:
            ax_c.set_xlabel('Hours from window start', fontsize=style.TICK_SIZE,
                            color=style.TEXT_PRIMARY)

    handles = legend_handles()
    fig.legend(handles=handles, loc='upper center', ncol=len(handles),
               fontsize=style.LEGEND_SIZE, frameon=False,
               bbox_to_anchor=(0.5, 1 - 0.1 / height))
    return style.save(fig, out_path, footnote=(
        f'{args.window_h:g} h windows: >= {args.min_scores} scores, >= 1 above '
        f'{args.high_pain:g}, <= {args.max_zeros} at 0, >= 1 opioid dose, every '
        f'5-min pre-score epoch recorded, >= {args.min_clean_pair_frac:.0%} of '
        f'pairs with no masked bin in any epoch ({args.mask_label}), start >= {args.exclude_first_h:g} h after '
        f'session start. Gray lines: pain assessments.'))


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--window-h', type=float, default=8.0)
    parser.add_argument('--min-scores', type=int, default=3)
    parser.add_argument('--high-pain', type=float, default=5.0,
                        help='at least one score must be ABOVE this')
    parser.add_argument('--max-zeros', type=int, default=1)
    parser.add_argument('--exclude-first-h', type=float, default=24.0)
    parser.add_argument('--n-options', type=int, default=20)
    parser.add_argument('--mask-label', default=config.bipolar_mask_label('std10'))
    parser.add_argument('--min-clean-pair-frac', type=float, default=0.9,
                        help='fraction of pairs with no masked bin in any epoch')
    parser.add_argument('--run-name', default='pain_mme_8h')
    args = parser.parse_args()

    io.warn_if_dirty()

    discovery = set(config.discovery_subjects())
    starts, all_doses = load_exposure()
    all_scores = pain_link.load_pain_scores()
    defs_dir = config.pain_epoch_unit_dir() / config.EPOCH_DEFS_SUBDIR
    keys = sorted({tuple(p.name.split('_')[i].split('-')[1] for i in (0, 1))
                   for p in defs_dir.glob('sub-*_defs.parquet')})
    keys = [k for k in keys if k[0] in discovery and k in starts.index]
    logger.info('%d discovery sessions with epochs and a MAR export', len(keys))

    scores, doses, windows = {}, {}, []
    for subject, session in keys:
        start = starts[(subject, session)]
        s = all_scores[(all_scores['subject'] == subject)
                       & (all_scores['session'] == session)].copy()
        s['t_h'] = (s['score_dt'] - start).dt.total_seconds() / 3600.0
        qc = epoch_qc(subject, session, args.mask_label)
        s = s.merge(qc.rename(columns={'pain_time': 'score_dt'}), on='score_dt',
                    how='left')
        d = all_doses[(all_doses['subject'] == subject)
                      & (all_doses['session'] == session)]
        scores[(subject, session)], doses[(subject, session)] = s, d
        w = session_windows(s, d, args)
        w.insert(0, 'session', session)
        w.insert(0, 'subject', subject)
        windows.append(w)
    windows = pd.concat(windows, ignore_index=True)

    steps = {'min_scores': windows['n_scores'] >= args.min_scores,
             'high_pain': windows['n_high'] >= 1,
             'max_zeros': windows['n_zero'] <= args.max_zeros,
             'opioid_dose': windows['n_doses'] >= 1,
             'all_recorded': windows['n_unrecorded'] == 0,
             'clean_pairs': windows['clean_pair_frac'] >= args.min_clean_pair_frac,
             'after_first_h': windows['start_h'] >= args.exclude_first_h}
    keep = pd.Series(True, index=windows.index)
    funnel = {'all': int(len(windows))}
    for name, ok in steps.items():
        if name == 'clean_pairs':
            logger.info('clean-pair fraction before that step: %s', windows.loc[
                keep, 'clean_pair_frac'].quantile([0, .1, .25, .5, .75, .9, 1]).round(3)
                .to_dict())
        keep &= ok
        funnel[name] = int(keep.sum())
        logger.info('after %-14s %6d windows, %d subjects', name, int(keep.sum()),
                    windows.loc[keep, 'subject'].nunique())
    candidates = windows[keep]
    if candidates.empty:
        raise SystemExit('no window passes every criterion')
    options = pick(candidates, args.n_options)
    options.insert(0, 'option', np.arange(1, len(options) + 1))
    logger.info('%d candidate windows across %d subjects; plotting %d',
                len(candidates), candidates['subject'].nunique(), len(options))

    run_dir = config.analysis_run_dir(QUESTION, OUTPUT_TYPE, args.run_name, event=EVENT)
    run_dir.mkdir(parents=True, exist_ok=True)
    plot_options(options, scores, doses, run_dir / 'options.png', args)

    subjects = sorted(candidates['subject'].unique())
    parents = ([io.parent_ref(p, digest=False) for p in config.pain_score_files()]
               + [io.parent_ref(config.bipolar_mask_dir(args.mask_label), digest=False),
                  io.parent_ref(defs_dir, digest=False)])
    common = dict(kind='table', script=SCRIPT, params=vars(args), parents=parents,
                  subjects=subjects)
    io.write_table(options, run_dir / 'options.csv', **common)
    io.write_table(candidates, run_dir / 'candidates.csv', **common)
    io.write_run_provenance(run_dir, script=SCRIPT, params=vars(args), parents=parents,
                            subjects=sorted(options['subject'].unique()),
                            extra={'funnel': funnel,
                                   'n_candidates': int(len(candidates)),
                                   'status': 'EXPLORATORY methods-figure candidates'})
    io.log_analysis(f'Methods-figure candidates: {len(options)} 8-h pain + MME '
                    f'windows of {len(candidates)} passing', run_dir)
    logger.info('-> %s', run_dir)


if __name__ == '__main__':
    main()
