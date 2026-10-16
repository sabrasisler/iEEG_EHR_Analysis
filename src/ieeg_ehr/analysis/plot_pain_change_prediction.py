"""Figures of a pain_change prediction run (`pain_change_prediction score`).

    python -m ieeg_ehr.analysis.plot_pain_change_prediction --run-dir <run> \\
        --weights insample_painchange_20261006-211218 --weighting uniform

Every figure averages within subject first, then shows mean +/- SEM across
subjects, and drops any group with fewer than `--min-subjects` subjects. The x
axis is hours since the first report, and the shaded first 5 min is where the
sliding window still overlaps the baseline window. Descriptive: no significance
marks.

    validation_scatter   predicted at t2 vs actual delta pain, one point per pair
    trajectories         one line per delta-pain bin over 0-3 h
    dose_response        per integer delta pain, the mean prediction within
                         +/- DOSE_WINDOW_MIN of t2
    confound_*           (a) dose-response by starting-pain tercile, (b) pain1 4-6,
                         (c) decrease bins by analgesic in [t1, t2), (d) no other
                         report before t1 + 3 h
    coverage             pairs and subjects contributing per step and bin
    subject_detail       per subject: each pair, and the bin means
    dc_groups_hours      top: predicted Δpain, bottom: ΔC(t) = C(t) - C(t1), from t1 to
                         each pair's t2, one line per whole-interval ΔC bin (DC_BINS),
                         first-day pairs out
    dc_groups_fraction   the same on the fraction of the interval, t1 = 0, t2 = 1

Written to `<run>/figures/` as PNG, suffixed with the weighting.
"""

import argparse
import json
import logging
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from ieeg_ehr import io  # noqa: E402
from ieeg_ehr.analysis.pain_change import FIGURES_SUBDIR, load_exposure  # noqa: E402
from ieeg_ehr.analysis.pain_change_prediction import (DPAIN_BINS, assign_bins,  # noqa: E402
                                                      load_trajectory)
from ieeg_ehr.config import opioid_mme  # noqa: E402
from ieeg_ehr.features.build_pain_change_trajectory import WINDOW_MIN  # noqa: E402
from ieeg_ehr.med_analysis import mme_curve  # noqa: E402

logger = logging.getLogger(__name__)

#: Diverging blue <-> red with a gray midpoint, one colour per default bin.
BIN_COLOURS = ('#184f95', '#6da7ec', '#8c8b86', '#ec8f8c', '#a8322f')
#: Validated all-pairs for up to four subjects; more subjects plot in one ink.
SUBJECT_COLOURS = ('#2a78d6', '#eb6834', '#1baf7a', '#4a3aa7')
SUBJECT_MARKERS = ('o', 's', '^', 'D')
NEUTRAL = '#5f5e5a'
#: Ordinal blue ramp (steps 250, 450, 650) for the starting-pain terciles.
TERCILE_COLOURS = ('#86b6ef', '#2a78d6', '#104281')
INK, MUTED, GRID, SHADE = '#2b2b29', '#6b6a66', '#e4e3df', '#f0efec'
DOSE_WINDOW_MIN = 10
#: Whole-interval dC (C_2 - C_1, typical doses) bins: |dC| < DC_SMALL is no change,
#: DC_SMALL to DC_LARGE small, above DC_LARGE large. Set by the user 2026-10-15.
DC_SMALL, DC_LARGE = 0.10, 0.35
DC_BINS = ('large fall', 'small fall', 'no change', 'small rise', 'large rise')
#: Diverging brown <-> teal with a gray midpoint: colour encodes dC, apart from the
#: blue <-> red of the delta-pain bins.
DC_COLOURS = ('#8c510a', '#d8b365', '#8c8b86', '#5ab4ac', '#01665e')
#: Pairs whose first epoch starts this soon after session_start leave the dC figures:
#: mme1 misses the OR doses, so first-day dC reads washout as no change.
DC_FIRST_HOURS_OUT = 24
#: Steps of the t1 -> t2 fraction axis.
FRACTION_BINS = 20


# ============================================================================
# Aggregation
# ============================================================================

def _sem(x):
    x = x.dropna()
    return x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else np.nan


def subject_first(df, value, keys):
    """Mean within subject at each `keys`, then mean, SEM and n across subjects."""
    per_subject = df.groupby(keys + ['subject'], observed=True)[value].mean().reset_index()
    return (per_subject.groupby(keys, observed=True)[value]
            .agg(mean='mean', sem=_sem, n_subjects='count').reset_index())


def group_sizes(df, group, value='value'):
    """{group: (n_subjects, n_pairs)} over pairs with any finite `value`."""
    g = df.dropna(subset=[value]).groupby(group, observed=True)
    return {k: (v['subject'].nunique(), v['pair_id'].nunique()) for k, v in g}


# ============================================================================
# Drawing helpers
# ============================================================================

def _style(ax):
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    for side in ('left', 'bottom'):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelcolor=INK)
    ax.grid(color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def _time_axis(ax):
    ax.axvspan(0, WINDOW_MIN / 60, color=SHADE, zorder=0, lw=0)
    ax.axhline(0, color=MUTED, lw=0.8, zorder=1)
    ax.set_xlim(0, 3)
    ax.set_xticks([0, 0.5, 1, 1.5, 2, 2.5, 3])
    ax.set_xlabel('Hours since first report', color=INK)


def _trajectory_lines(ax, df, group, groups, colours, min_subjects, styles=None,
                      label_fmt='{g}', x='t_min', value='value'):
    """One mean +/- SEM line per group; returns the groups drawn.

    `x` = 't_min' plots in hours. Any other column plots as it is.
    """
    sizes = group_sizes(df, group, value)
    stats = subject_first(df, value, [group, x])
    drawn = []
    for g, colour in zip(groups, colours):
        n_subj, n_pairs = sizes.get(g, (0, 0))
        if n_subj < min_subjects:
            logger.info('%s: %d subject(s), below %d, dropped', g, n_subj, min_subjects)
            continue
        s = stats[(stats[group] == g) & (stats['n_subjects'] >= 1)]
        h = s[x] / 60 if x == 't_min' else s[x]
        ls = (styles or {}).get(g, '-')
        ax.plot(h, s['mean'], color=colour, lw=1.6, ls=ls,
                label=f'{label_fmt.format(g=g)} ({n_subj} subj, {n_pairs} pairs)')
        ax.fill_between(h, s['mean'] - s['sem'], s['mean'] + s['sem'], color=colour,
                        alpha=0.18, lw=0)
        drawn.append(g)
    return drawn


def _legend(ax, **kw):
    leg = ax.legend(frameon=False, fontsize=8, labelcolor=INK, **kw)
    return leg


def _footer(fig, text):
    fig.text(0.01, -0.01, text, fontsize=7, color=MUTED, ha='left', va='top')


def _save(fig, out_dir, name):
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f'{name}.png'
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    logger.info('wrote %s', path)


# ============================================================================
# Figures driven by the predictions
# ============================================================================

def fig_validation(df, out_dir, tag, title):
    at_t2 = df[df['t_min'] == df['gap_min'].round()].dropna(subset=['value'])
    subjects = sorted(at_t2['subject'].unique())
    rng = np.random.default_rng(0)
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    _style(ax)
    lim = max(at_t2['value'].abs().max(), at_t2['delta_pain'].abs().max()) * 1.05
    ax.plot([-lim, lim], [-lim, lim], color=MUTED, lw=0.8, ls='--', label='identity')
    for i, sid in enumerate(subjects):
        s = at_t2[at_t2['subject'] == sid]
        few = len(subjects) <= len(SUBJECT_COLOURS)
        ax.scatter(s['delta_pain'] + rng.uniform(-0.15, 0.15, len(s)), s['value'],
                   s=22, color=SUBJECT_COLOURS[i] if few else NEUTRAL,
                   marker=SUBJECT_MARKERS[i] if few else 'o', alpha=0.85 if few else 0.4,
                   edgecolor='white', linewidth=0.5, label=sid if few else None)
    ax.set_xlim(at_t2['delta_pain'].min() - 1, at_t2['delta_pain'].max() + 1)
    ax.set_ylim(-lim, lim)
    ax.set_xlabel('Actual Δpain at report 2 (NRS)', color=INK)
    ax.set_ylabel('Predicted Δpain at t2 (NRS)', color=INK)
    ax.set_title(title, fontsize=10, color=INK)
    _legend(ax, loc='upper left')
    _footer(fig, f'{len(subjects)} subjects, {len(at_t2)} pairs. x jittered ±0.15.')
    _save(fig, out_dir, f'validation_scatter_{tag}')


def fig_trajectories(df, out_dir, name, title, min_subjects, bins, footer=''):
    fig, ax = plt.subplots(figsize=(7, 4.4))
    _style(ax)
    _time_axis(ax)
    drawn = _trajectory_lines(ax, df, 'bin', list(bins), BIN_COLOURS, min_subjects)
    ax.set_ylabel('Predicted Δpain (NRS)', color=INK)
    ax.set_title(title, fontsize=10, color=INK)
    _legend(ax, loc='upper left', bbox_to_anchor=(1.01, 1))
    _footer(fig, f'{df["subject"].nunique()} subjects, {df["pair_id"].nunique()} pairs. '
                 f'Bins with fewer than {min_subjects} subjects dropped '
                 f'({len(bins) - len(drawn)}). {footer}')
    _save(fig, out_dir, name)


def dose_points(df, hue=None):
    """Per pair the mean prediction within DOSE_WINDOW_MIN of t2, then subject-first."""
    near = df[(df['t_min'] - df['gap_min']).abs() <= DOSE_WINDOW_MIN]
    keys = ['delta_pain'] + ([hue] if hue else [])
    per_pair = (near.groupby(['pair_id', 'subject'] + keys, observed=True)['value']
                .mean().reset_index().dropna(subset=['value']))
    stats = subject_first(per_pair, 'value', keys)
    n_pairs = per_pair.groupby(keys, observed=True)['pair_id'].nunique().rename('n_pairs')
    return stats.merge(n_pairs.reset_index(), on=keys)


def _dose_axes(ax, title):
    _style(ax)
    ax.axhline(0, color=MUTED, lw=0.8)
    lo, hi = ax.get_xlim()
    ax.plot([lo, hi], [lo, hi], color=MUTED, lw=0.8, ls='--', label='identity')
    ax.set_xlabel('Actual Δpain at report 2 (NRS)', color=INK)
    ax.set_ylabel(f'Predicted Δpain, ±{DOSE_WINDOW_MIN} min of t2 (NRS)', color=INK)
    ax.set_title(title, fontsize=10, color=INK)


def fig_dose(df, out_dir, name, title, min_subjects, hue=None, hue_levels=(),
             hue_colours=()):
    pts = dose_points(df, hue)
    pts = pts[pts['n_subjects'] >= min_subjects]
    fig, ax = plt.subplots(figsize=(6, 4.6))
    series = [(None, NEUTRAL, pts)] if hue is None else [
        (lvl, c, pts[pts[hue] == lvl]) for lvl, c in zip(hue_levels, hue_colours)]
    offsets = np.linspace(-0.2, 0.2, len(series)) if len(series) > 1 else [0.0]
    for (lvl, colour, s), dx in zip(series, offsets):
        ax.errorbar(s['delta_pain'] + dx, s['mean'], yerr=s['sem'], fmt='o', ms=5,
                    color=colour, ecolor=colour, elinewidth=1, capsize=0,
                    label=None if lvl is None else str(lvl))
        if hue is None:
            for r in s.itertuples():
                ax.annotate(f'{r.n_subjects}/{r.n_pairs}', (r.delta_pain, r.mean),
                            textcoords='offset points', xytext=(0, 8), ha='center',
                            fontsize=7, color=MUTED)
    ax.set_xlim(pts['delta_pain'].min() - 1, pts['delta_pain'].max() + 1)
    _dose_axes(ax, title)
    _legend(ax, loc='upper left')
    _footer(fig, 'Labels: subjects/pairs. Points with fewer than '
                 f'{min_subjects} subjects dropped.')
    _save(fig, out_dir, name)


def fig_analgesic(df, out_dir, name, title, min_subjects, bins):
    labels = list(bins)
    decrease = [b for b in labels if 'decrease' in b]
    reference = [b for b in labels if b == 'no change']
    split = df.assign(group=np.where(df['bin'].isin(decrease),
                                     df['bin'].astype(str) + np.where(
                                         df['analgesic_in_interval'], ', analgesic',
                                         ', no analgesic'),
                                     df['bin'].astype(str)))
    groups, colours, styles = [], [], {}
    colour_of = dict(zip(labels, BIN_COLOURS))
    for b in decrease:
        for suffix, ls in ((', analgesic', '-'), (', no analgesic', ':')):
            groups.append(b + suffix)
            colours.append(colour_of[b])
            styles[b + suffix] = ls
    groups += reference
    colours += [colour_of[b] for b in reference]
    fig, ax = plt.subplots(figsize=(7, 4.4))
    _style(ax)
    _time_axis(ax)
    drawn = _trajectory_lines(ax, split, 'group', groups, colours, min_subjects, styles)
    ax.set_ylabel('Predicted Δpain (NRS)', color=INK)
    ax.set_title(title, fontsize=10, color=INK)
    _legend(ax, loc='upper left', bbox_to_anchor=(1.01, 1))
    _footer(fig, 'Analgesic: any in [t1, t2). Groups with fewer than '
                 f'{min_subjects} subjects dropped ({len(groups) - len(drawn)}).')
    _save(fig, out_dir, name)


def fig_coverage(df, out_dir, name, bins):
    finite = df.dropna(subset=['value'])
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharex=True)
    for ax, (what, col) in zip(axes, (('pairs', 'pair_id'), ('subjects', 'subject'))):
        _style(ax)
        _time_axis(ax)
        counts = finite.groupby(['bin', 't_min'], observed=True)[col].nunique().reset_index()
        for b, colour in zip(bins, BIN_COLOURS):
            s = counts[counts['bin'] == b]
            ax.plot(s['t_min'] / 60, s[col], color=colour, lw=1.4, label=b)
        ax.set_ylabel(f'N {what} with a prediction', color=INK)
        ax.set_ylim(bottom=0)
    _legend(axes[1], loc='upper left', bbox_to_anchor=(1.01, 1))
    fig.suptitle('Coverage per step', fontsize=10, color=INK)
    _save(fig, out_dir, name)


def fig_subject_detail(df, out_dir, name, title, bins):
    subjects = sorted(df['subject'].unique())
    ncols = min(4, len(subjects))
    nrows = int(np.ceil(len(subjects) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.6 * ncols, 2.8 * nrows),
                             sharex=True, sharey=True, squeeze=False)
    colour_of = dict(zip(bins, BIN_COLOURS))
    # Single pairs reach far past the bin means, so the shared y axis is clipped.
    lim = max(np.nanpercentile(df['value'].abs(), 99), 5.0)
    n_out = int((df['value'].abs() > lim).sum())
    for ax, sid in zip(axes.flat, subjects):
        _style(ax)
        _time_axis(ax)
        s = df[df['subject'] == sid]
        for _, p in s.groupby('pair_id'):
            ax.plot(p['t_min'] / 60, p['value'], color=colour_of[p['bin'].iloc[0]],
                    lw=0.5, alpha=0.35)
        means = s.groupby(['bin', 't_min'], observed=True)['value'].mean().reset_index()
        for b in bins:
            m = means[means['bin'] == b]
            if len(m):
                ax.plot(m['t_min'] / 60, m['value'], color=colour_of[b], lw=1.8)
        ax.set_title(f'{sid} ({s["pair_id"].nunique()} pairs)', fontsize=9, color=INK)
        ax.set_xlabel('')
        ax.set_ylim(-lim, lim)
    for ax in axes.flat[len(subjects):]:
        ax.set_visible(False)
    fig.supxlabel('Hours since first report', color=INK, fontsize=9)
    fig.supylabel('Predicted Δpain (NRS)', color=INK, fontsize=9)
    fig.suptitle(f'{title}\nThin: single pairs. Thick: bin means. y clipped at ±{lim:.0f} '
                 f'({n_out} of {df["value"].notna().sum()} values beyond).',
                 fontsize=10, color=INK)
    fig.tight_layout()
    handles = [plt.Line2D([], [], color=c, lw=1.8) for c in BIN_COLOURS[:len(bins)]]
    fig.legend(handles, list(bins), loc='upper center', bbox_to_anchor=(0.5, 0.0),
               ncol=len(bins), frameon=False, fontsize=8, labelcolor=INK)
    _save(fig, out_dir, name)


# ============================================================================
# Figures by opioid exposure change
# ============================================================================

def dc_bin(dc):
    """The DC_BINS label of each whole-interval dC, an ordered categorical."""
    mag = dc.abs()
    label = np.select([mag < DC_SMALL, (mag <= DC_LARGE) & (dc < 0), mag <= DC_LARGE, dc < 0],
                      ['no change', 'small fall', 'small rise', 'large fall'], 'large rise')
    return pd.Categorical(label, categories=list(DC_BINS), ordered=True)


def exposure_steps(pairs):
    """(pair_id, t_min, dc) for t = 0 ... t2, and per pair `dc_interval` and `e1_start_h`.

    dc(t) = C(t) - C(t1) in typical doses, C(t) the mean mme1 curve over the 5-min
    window ending at t (`mme_curve.window_mean`), the rule of `pain_change.add_exposure`,
    so dc at t2 is that pair's model dC. Pairs of a session with no MAR export go.
    """
    starts, doses = load_exposure()
    subject = pairs['subject'].str.replace('sub-', '', regex=False)
    has_mar = pd.MultiIndex.from_arrays([subject, pairs['session']]).isin(starts.index)
    if (~has_mar).any():
        logger.warning('NO MAR EXPORT: %d pairs of %s dropped', int((~has_mar).sum()),
                       sorted(pairs.loc[~has_mar, 'subject'].unique()))
    pairs, subject = pairs[has_mar], subject[has_mar]
    start = starts.reindex(pd.MultiIndex.from_arrays([subject, pairs['session']])).to_numpy()
    t1_h = (pairs['t1'].to_numpy() - start) / np.timedelta64(1, 'h')
    steps = []
    for (s, session), idx in pairs.groupby([subject, pairs['session']]).indices.items():
        g = pairs.iloc[idx]
        d = doses[(doses['subject'] == s) & (doses['session'] == session)]
        n = g['gap_min'].round().astype(int).to_numpy() + 1
        t_min = np.concatenate([np.arange(k) for k in n])
        end_h = np.repeat(t1_h[idx], n) + t_min / 60
        c = mme_curve.window_mean(d, end_h - WINDOW_MIN / 60, end_h) / opioid_mme.TYPICAL_DOSE_MME
        c1 = np.repeat(c[np.r_[0, np.cumsum(n)[:-1]]], n)
        steps.append(pd.DataFrame({'pair_id': np.repeat(g['pair_id'].to_numpy(), n),
                                   't_min': t_min, 'dc': c - c1}))
    steps = pd.concat(steps, ignore_index=True)
    at_t2 = steps.groupby('pair_id')['dc'].last().rename('dc_interval')
    per_pair = pd.DataFrame({'pair_id': pairs['pair_id'].to_numpy(),
                             'e1_start_h': t1_h - WINDOW_MIN / 60}).join(at_t2, on='pair_id')
    return steps, per_pair


def fig_dc_groups(df, out_dir, name, title, min_subjects, axis):
    """Predicted Δpain (top) and ΔC(t) (bottom) from t1 to t2, one line per ΔC bin.

    `axis` 'hours' plots hours since t1. 'fraction' plots (t - t1) / (t2 - t1) in
    FRACTION_BINS steps, each pair averaged within a step first.
    """
    if axis == 'fraction':
        step = np.minimum(np.floor(df['t_min'] / df['gap_min'] * FRACTION_BINS),
                          FRACTION_BINS - 1)
        df = (df.assign(x=(step + 0.5) / FRACTION_BINS)
              .groupby(['pair_id', 'subject', 'dc_group', 'x'], observed=True)[['value', 'dc']]
              .mean().reset_index())
        x = 'x'
    else:
        x = 't_min'
    fig, (top, bottom) = plt.subplots(2, 1, figsize=(7, 6.6), sharex=True,
                                      gridspec_kw={'height_ratios': (3, 2)})
    for ax in (top, bottom):
        _style(ax)
        ax.axhline(0, color=MUTED, lw=0.8, zorder=1)
    drawn = _trajectory_lines(top, df, 'dc_group', list(DC_BINS), DC_COLOURS, min_subjects, x=x)
    _trajectory_lines(bottom, df, 'dc_group', list(DC_BINS), DC_COLOURS, min_subjects,
                      x=x, value='dc')
    if axis == 'fraction':
        bottom.set_xlim(0, 1)
        bottom.set_xticks([0, 0.25, 0.5, 0.75, 1])
        bottom.set_xticklabels(['t1', '25%', '50%', '75%', 't2'])
        bottom.set_xlabel('Fraction of the interval from report 1 to report 2', color=INK)
    else:
        for ax in (top, bottom):
            ax.axvspan(0, WINDOW_MIN / 60, color=SHADE, zorder=0, lw=0)
        bottom.set_xlim(0, 3)
        bottom.set_xticks([0, 0.5, 1, 1.5, 2, 2.5, 3])
        bottom.set_xlabel('Hours since first report (each pair ends at its t2)', color=INK)
    top.set_ylabel('Predicted Δpain (NRS)', color=INK)
    bottom.set_ylabel('ΔC since t1 (typical doses)', color=INK)
    top.set_title(title, fontsize=10, color=INK)
    _legend(top, loc='upper left', bbox_to_anchor=(1.01, 1), title='ΔC, t1 to t2')
    _footer(fig, f'{df["subject"].nunique()} subjects, {df["pair_id"].nunique()} pairs. '
                 f'ΔC bins: |ΔC| < {DC_SMALL:g} no change, {DC_SMALL:g}-{DC_LARGE:g} small, '
                 f'> {DC_LARGE:g} large (1 typical dose = {opioid_mme.TYPICAL_DOSE_MME:g} MME, '
                 f'{opioid_mme.EXPOSURE_VERSION}). Pairs starting within '
                 f'{DC_FIRST_HOURS_OUT} h of session start left out. Groups under '
                 f'{min_subjects} subjects dropped ({len(DC_BINS) - len(drawn)}).')
    _save(fig, out_dir, name)


# ============================================================================
# Entry points
# ============================================================================

def render_all(run_dir, weights_name, weighting='uniform', min_subjects=3, bins=None):
    """Every prediction figure for one weighting."""
    bins = bins or DPAIN_BINS
    run_dir = Path(run_dir)
    out_dir = run_dir / FIGURES_SUBDIR
    params = json.loads((run_dir / 'provenance.json').read_text())['params']
    pairs, _ = load_trajectory(params['trajectory_dir'])
    preds = io.read_table(run_dir / f'predictions_{weights_name}.csv', on_stale='warn')
    meta = pairs.assign(bin=assign_bins(pairs['delta_pain'], bins))[
        ['pair_id', 'subject', 'bin', 'delta_pain', 'gap_min', 'pain1',
         'analgesic_in_interval', 'next_report_in_window']]
    df = preds.merge(meta, on='pair_id').rename(columns={f'pred_{weighting}': 'value'})
    tag = weighting
    title = f'Predicted Δpain, raw, {weighting}, {weights_name}'

    fig_validation(df, out_dir, tag, title)
    fig_trajectories(df, out_dir, f'trajectories_{tag}', title, min_subjects, bins)
    fig_dose(df, out_dir, f'dose_response_{tag}', title, min_subjects)

    cuts = df.drop_duplicates('pair_id')['pain1'].quantile([1 / 3, 2 / 3]).to_numpy()
    terciles = ['low', 'mid', 'high']
    tercile = pd.cut(df['pain1'], [-np.inf, cuts[0], cuts[1], np.inf], labels=terciles)
    fig_dose(df.assign(start=tercile), out_dir, f'confound_a_start_tercile_{tag}',
             f'{title}\nby starting-pain tercile (cuts {cuts[0]:g}, {cuts[1]:g})',
             min_subjects, hue='start', hue_levels=terciles, hue_colours=TERCILE_COLOURS)
    fig_trajectories(df[df['pain1'].between(4, 6)], out_dir, f'confound_b_pain1_4to6_{tag}',
                     f'{title}\npain1 between 4 and 6', min_subjects, bins)
    fig_analgesic(df, out_dir, f'confound_c_analgesic_{tag}',
                  f'{title}\ndecrease bins by analgesic in [t1, t2)', min_subjects, bins)
    fig_trajectories(df[~df['next_report_in_window'].astype(bool)], out_dir,
                     f'confound_d_no_next_report_{tag}',
                     f'{title}\nno other report in (t2, t1 + 3 h]', min_subjects, bins)
    fig_coverage(df, out_dir, f'coverage_{tag}', list(bins))
    fig_subject_detail(df, out_dir, f'subject_detail_{tag}', title, list(bins))

    steps, per_pair = exposure_steps(pairs)
    late = per_pair[per_pair['e1_start_h'] >= DC_FIRST_HOURS_OUT]
    groups = late.assign(dc_group=dc_bin(late['dc_interval']))[['pair_id', 'dc_group']]
    dc_df = df.merge(steps, on=['pair_id', 't_min']).merge(groups, on='pair_id')
    logger.info('dC groups (pairs): %s', groups['dc_group'].value_counts().to_dict())
    for axis in ('hours', 'fraction'):
        fig_dc_groups(dc_df, out_dir, f'dc_groups_{axis}_{tag}', title, min_subjects, axis)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--run-dir', required=True)
    ap.add_argument('--weights', required=True)
    ap.add_argument('--weighting', choices=['uniform', 'ivw'], default='uniform')
    ap.add_argument('--min-subjects', type=int, default=3)
    ap.add_argument('--bins', type=json.loads, default=None,
                    help='JSON {label: [lo, hi]}, inclusive, in display order')
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)s %(message)s')
    bins = {k: tuple(v) for k, v in args.bins.items()} if args.bins else None
    render_all(args.run_dir, args.weights, args.weighting, args.min_subjects, bins)
    io.log_analysis(f'pain_change prediction figures, {args.weighting}', Path(args.run_dir))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
