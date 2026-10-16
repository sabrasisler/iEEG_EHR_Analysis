"""Figures of a pain_change prediction run (`pain_change_prediction score`).

    python -m ieeg_ehr.analysis.plot_pain_change_prediction --run-dir <run> \\
        --weights <weights name> --prediction raw --weighting uniform

Every figure averages within subject first, then shows mean +/- SEM across
subjects, and drops any group, and any time step of a line, with fewer than
`--min-subjects` subjects. The time-axis figures show hours since the first report
out to PLOT_HORIZON_MIN, and shade the first 5 min, where the sliding window still
overlaps the baseline window. Descriptive: no significance marks.

Written to `<run>/figures/<kind>_dpain_<prediction>-<weighting>.png`, the figure
tokens of docs/run_tokens.md. Kinds:

    validation-scatter   predicted at t2 vs actual delta pain, one point per pair
    trajectories         one line per delta-pain bin over the first PLOT_HORIZON_MIN
    trajectories-by-delta  the same, one line per integer delta pain, -10 to +10
    trajectories-t2[-by-delta]  both, aligned to the second report over the
                         T2_HORIZON_MIN before it
    dose-response        per integer delta pain, the mean prediction within
                         +/- DOSE_WINDOW_MIN of t2
    confound-start-tercile  dose-response by starting-pain tercile
    confound-pain1-4to6  trajectories of pairs with pain1 between 4 and 6
    confound-no-next-report  trajectories of pairs with no other report before t1 + 3 h
    coverage             pairs and subjects contributing per step and bin
    subject-detail       per subject: each pair, and the bin means
    dc-groups-hours      top: predicted Δpain, bottom: ΔC(t) = C(t) - C(t1), from t1 to
                         each pair's t2, one line per whole-interval ΔC bin (DC_BINS),
                         first-day pairs out
    dc-groups-fraction   the same on the fraction of the interval, t1 = 0, t2 = 1

For the adjusted prediction the first EXTRAPOLATE_BELOW_MIN minutes are hatched: the
fit saw no gap that short.
"""

import argparse
import json
import logging
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.cm import ScalarMappable  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, Normalize  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from ieeg_ehr import io  # noqa: E402
from ieeg_ehr.analysis.pain_change import FIGURES_SUBDIR, load_exposure  # noqa: E402
from ieeg_ehr.analysis.pain_change_prediction import (  # noqa: E402
    DPAIN_BINS, EXTRAPOLATE_BELOW_MIN, assign_bins, load_trajectory)
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
#: The time-axis figures stop here, though the extraction runs to 3 h.
PLOT_HORIZON_MIN = 120
#: The t2-aligned figures start this long before the second report.
T2_HORIZON_MIN = 120
#: One line per integer change in pain, on a diverging blue <-> red ramp around gray 0.
DELTA_LINES = tuple(range(-10, 11))
DELTA_CMAP = LinearSegmentedColormap.from_list(
    'delta_pain', ['#184f95', '#86b6ef', '#8c8b86', '#ec8f8c', '#a8322f'])
DELTA_NORM = Normalize(min(DELTA_LINES), max(DELTA_LINES))
DELTA_COLOURS = tuple(DELTA_CMAP(DELTA_NORM(d)) for d in DELTA_LINES)
T2_NOTE = (' Each pair enters at its first report, so pairs closer than 2 h cover only the'
           ' right of the axis.')
EXTRAPOLATED_NOTE = " Hatched: below the fit's gap floor, so extrapolated."
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


def _time_axis(ax, grey_h=None):
    """Hours on x, the baseline overlap shaded, and hatched up to `grey_h` (extrapolated)."""
    if grey_h:
        ax.axvspan(0, grey_h, facecolor='none', edgecolor=GRID, hatch='///', lw=0, zorder=0)
    ax.axvspan(0, WINDOW_MIN / 60, color=SHADE, zorder=0, lw=0)
    ax.axhline(0, color=MUTED, lw=0.8, zorder=1)
    ax.set_xlim(0, PLOT_HORIZON_MIN / 60)
    ax.set_xticks(np.arange(0, PLOT_HORIZON_MIN / 60 + 0.25, 0.5))
    ax.set_xlabel('Hours since first report', color=INK)


def _t2_axis(ax):
    """Hours before the second report on x, ending at the report."""
    ax.axhline(0, color=MUTED, lw=0.8, zorder=1)
    ax.axvline(0, color=MUTED, lw=0.8, ls='--', zorder=1)
    ax.set_xlim(-T2_HORIZON_MIN / 60, 0)
    ax.set_xticks(np.arange(-T2_HORIZON_MIN / 60, 0.25, 0.5))
    ax.set_xlabel('Hours before second report', color=INK)


def _trajectory_lines(ax, df, group, groups, colours, min_subjects, styles=None,
                      label_fmt='{g}', value='value'):
    """One mean +/- SEM line per group of `value` over x_h; returns the groups drawn.

    x_h is hours for the time axes, or any x the caller puts there.
    """
    sizes = group_sizes(df, group, value)
    stats = subject_first(df, value, [group, 'x_h'])
    drawn = []
    for g, colour in zip(groups, colours):
        n_subj, n_pairs = sizes.get(g, (0, 0))
        if n_subj < min_subjects:
            logger.info('%s: %d subject(s), below %d, dropped', g, n_subj, min_subjects)
            continue
        s = stats[(stats[group] == g) & (stats['n_subjects'] >= min_subjects)]
        h = s['x_h']
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

def fig_validation(df, out_dir, name, title):
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
    _save(fig, out_dir, name)


def fig_trajectories(df, out_dir, name, title, min_subjects, groups, colours, grey_h=None,
                     group='bin', label_fmt='{g}', t2=False, footer='', colorbar=None):
    """One line per group over x_h: hours after t1, or before t2 when `t2` is set.

    `colorbar` = (cmap, norm, label) replaces the legend, for many ordered groups.
    """
    fig, ax = plt.subplots(figsize=(7, 4.4))
    _style(ax)
    if t2:
        _t2_axis(ax)
    else:
        _time_axis(ax, grey_h)
    drawn = _trajectory_lines(ax, df, group, groups, colours, min_subjects,
                              label_fmt=label_fmt)
    ax.set_ylabel('Predicted Δpain (NRS)', color=INK)
    ax.set_title(title, fontsize=10, color=INK)
    if colorbar:
        cmap, norm, label = colorbar
        cb = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), ax=ax, pad=0.02)
        cb.set_label(label, color=INK)
        cb.outline.set_visible(False)
    else:
        _legend(ax, loc='upper left', bbox_to_anchor=(1.01, 1))
    dropped = [label_fmt.format(g=g) for g in groups if g not in drawn]
    _footer(fig, f'{df["subject"].nunique()} subjects, {df["pair_id"].nunique()} pairs. '
                 f'Fewer than {min_subjects} subjects, not drawn: '
                 f'{", ".join(dropped) or "none"}.' + EXTRAPOLATED_NOTE * bool(grey_h) + footer)
    _save(fig, out_dir, name)


def near_t2(df, columns=('value',), keys=()):
    """Per pair, the mean of `columns` over the steps within DOSE_WINDOW_MIN of t2."""
    near = df[(df['t_min'] - df['gap_min']).abs() <= DOSE_WINDOW_MIN]
    return (near.groupby(['pair_id', 'subject', 'delta_pain', *keys], observed=True)
            [list(columns)].mean().reset_index())


def dose_points(df, hue=None):
    """Per pair the mean prediction within DOSE_WINDOW_MIN of t2, then subject-first."""
    keys = ['delta_pain'] + ([hue] if hue else [])
    per_pair = near_t2(df, keys=[hue] if hue else []).dropna(subset=['value'])
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


def fig_coverage(df, out_dir, name, bins, grey_h=None):
    finite = df.dropna(subset=['value'])
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharex=True)
    for ax, (what, col) in zip(axes, (('pairs', 'pair_id'), ('subjects', 'subject'))):
        _style(ax)
        _time_axis(ax, grey_h)
        counts = finite.groupby(['bin', 't_min'], observed=True)[col].nunique().reset_index()
        for b, colour in zip(bins, BIN_COLOURS):
            s = counts[counts['bin'] == b]
            ax.plot(s['t_min'] / 60, s[col], color=colour, lw=1.4, label=b)
        ax.set_ylabel(f'N {what} with a prediction', color=INK)
        ax.set_ylim(bottom=0)
    _legend(axes[1], loc='upper left', bbox_to_anchor=(1.01, 1))
    fig.suptitle('Coverage per step', fontsize=10, color=INK)
    _save(fig, out_dir, name)


def fig_subject_detail(df, out_dir, name, title, bins, grey_h=None):
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
        _time_axis(ax, grey_h)
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

    `axis` 'hours' plots hours since t1 out to PLOT_HORIZON_MIN. 'fraction' plots
    (t - t1) / (t2 - t1) in FRACTION_BINS steps, each pair averaged within a step first.
    """
    if axis == 'fraction':
        step = np.minimum(np.floor(df['t_min'] / df['gap_min'] * FRACTION_BINS),
                          FRACTION_BINS - 1)
        df = (df.assign(x_h=(step + 0.5) / FRACTION_BINS)
              .groupby(['pair_id', 'subject', 'dc_group', 'x_h'], observed=True)[['value', 'dc']]
              .mean().reset_index())
    else:
        df = df[df['t_min'] <= PLOT_HORIZON_MIN].assign(x_h=lambda d: d['t_min'] / 60)
    fig, (top, bottom) = plt.subplots(2, 1, figsize=(7, 6.6), sharex=True,
                                      gridspec_kw={'height_ratios': (3, 2)})
    for ax in (top, bottom):
        _style(ax)
        if axis == 'fraction':
            ax.axhline(0, color=MUTED, lw=0.8, zorder=1)
        else:
            _time_axis(ax)
    drawn = _trajectory_lines(top, df, 'dc_group', list(DC_BINS), DC_COLOURS, min_subjects)
    _trajectory_lines(bottom, df, 'dc_group', list(DC_BINS), DC_COLOURS, min_subjects,
                      value='dc')
    top.set_xlabel('')
    if axis == 'fraction':
        bottom.set_xlim(0, 1)
        bottom.set_xticks([0, 0.25, 0.5, 0.75, 1])
        bottom.set_xticklabels(['t1', '25%', '50%', '75%', 't2'])
        bottom.set_xlabel('Fraction of the interval from report 1 to report 2', color=INK)
    else:
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

def prediction_frame(run_dir, weights_name, prediction='raw', weighting='uniform',
                     bins=None):
    """One scoring run's predictions joined to pair metadata, the chosen one as `value`."""
    params = json.loads((Path(run_dir) / 'provenance.json').read_text())['params']
    pairs, _ = load_trajectory(params['trajectory_dir'], with_dz=False)
    preds = io.read_table(Path(run_dir) / f'predictions_{weights_name}.csv', on_stale='warn')
    meta = pairs.assign(bin=assign_bins(pairs['delta_pain'], bins or DPAIN_BINS))[
        ['pair_id', 'subject', 'session', 't1', 'bin', 'delta_pain', 'gap_min',
         'pain1', 'next_report_in_window']]
    return preds.merge(meta, on='pair_id').rename(
        columns={f'pred_{prediction}_{weighting}': 'value'})


def render_all(run_dir, weights_name, prediction='raw', weighting='uniform', min_subjects=3,
               bins=None):
    """Every figure for one prediction (raw | adj) and weighting (uniform | ivw)."""
    bins = bins or DPAIN_BINS
    out_dir = Path(run_dir) / FIGURES_SUBDIR
    df = prediction_frame(run_dir, weights_name, prediction, weighting, bins)
    tag = f'{prediction}-{weighting}'

    def name(kind):
        return f'{kind}_dpain_{tag}'

    title = f'Predicted Δpain, {prediction}, {weighting}, {weights_name}'
    grey_h = EXTRAPOLATE_BELOW_MIN / 60 if prediction == 'adj' else None
    # Validation and dose-response read each pair at its own t2, out to 3 h.
    early = df[df['t_min'] <= PLOT_HORIZON_MIN].assign(x_h=lambda d: d['t_min'] / 60)
    # Before t2 only, from each pair's t1. An adjusted step under the gap floor is
    # extrapolated, and with no common t1 on this axis it is dropped, not hatched.
    before_t2 = df.assign(x_h=(df['t_min'] - df['gap_min']) / 60)
    before_t2 = before_t2[before_t2['x_h'].between(-T2_HORIZON_MIN / 60, 0)]
    if prediction == 'adj':
        before_t2 = before_t2[~before_t2['extrapolated'].astype(bool)]
    t2_note = T2_NOTE + (' Extrapolated adjusted steps dropped.' if prediction == 'adj' else '')
    delta_lines = dict(group='delta_pain', label_fmt='{g:+d}',
                       colorbar=(DELTA_CMAP, DELTA_NORM, 'Δpain at report 2 (NRS)'))

    fig_validation(df, out_dir, name('validation-scatter'), title)
    fig_trajectories(early, out_dir, name('trajectories'), title, min_subjects, list(bins),
                     BIN_COLOURS, grey_h)
    fig_trajectories(early, out_dir, name('trajectories-by-delta'), title, min_subjects,
                     DELTA_LINES, DELTA_COLOURS, grey_h, **delta_lines)
    fig_trajectories(before_t2, out_dir, name('trajectories-t2'), title, min_subjects,
                     list(bins), BIN_COLOURS, t2=True, footer=t2_note)
    fig_trajectories(before_t2, out_dir, name('trajectories-t2-by-delta'), title,
                     min_subjects, DELTA_LINES, DELTA_COLOURS, t2=True, footer=t2_note,
                     **delta_lines)
    fig_dose(df, out_dir, name('dose-response'), title, min_subjects)

    cuts = df.drop_duplicates('pair_id')['pain1'].quantile([1 / 3, 2 / 3]).to_numpy()
    terciles = ['low', 'mid', 'high']
    tercile = pd.cut(df['pain1'], [-np.inf, cuts[0], cuts[1], np.inf], labels=terciles)
    fig_dose(df.assign(start=tercile), out_dir, name('confound-start-tercile'),
             f'{title}\nby starting-pain tercile (cuts {cuts[0]:g}, {cuts[1]:g})',
             min_subjects, hue='start', hue_levels=terciles, hue_colours=TERCILE_COLOURS)
    fig_trajectories(early[early['pain1'].between(4, 6)], out_dir,
                     name('confound-pain1-4to6'), f'{title}\npain1 between 4 and 6',
                     min_subjects, list(bins), BIN_COLOURS, grey_h)
    fig_trajectories(early[~early['next_report_in_window'].astype(bool)], out_dir,
                     name('confound-no-next-report'),
                     f'{title}\nno other report in (t2, t1 + 3 h]', min_subjects, list(bins),
                     BIN_COLOURS, grey_h)
    fig_coverage(early, out_dir, name('coverage'), list(bins), grey_h)
    fig_subject_detail(early, out_dir, name('subject-detail'), title, list(bins), grey_h)

    steps, per_pair = exposure_steps(df.drop_duplicates('pair_id'))
    late = per_pair[per_pair['e1_start_h'] >= DC_FIRST_HOURS_OUT]
    groups = late.assign(dc_group=dc_bin(late['dc_interval']))[['pair_id', 'dc_group']]
    dc_df = df.merge(steps, on=['pair_id', 't_min']).merge(groups, on='pair_id')
    logger.info('dC groups (pairs): %s', groups['dc_group'].value_counts().to_dict())
    for axis in ('hours', 'fraction'):
        fig_dc_groups(dc_df, out_dir, name(f'dc-groups-{axis}'), title, min_subjects, axis)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--run-dir', required=True)
    ap.add_argument('--weights', required=True)
    ap.add_argument('--prediction', choices=['raw', 'adj'], default='raw')
    ap.add_argument('--weighting', choices=['uniform', 'ivw'], default='uniform')
    ap.add_argument('--min-subjects', type=int, default=3)
    ap.add_argument('--bins', type=json.loads, default=None,
                    help='JSON {label: [lo, hi]}, inclusive, in display order')
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)s %(message)s')
    bins = {k: tuple(v) for k, v in args.bins.items()} if args.bins else None
    render_all(args.run_dir, args.weights, args.prediction, args.weighting,
               args.min_subjects, bins)
    io.log_analysis(f'pain_change prediction figures, {args.prediction}, {args.weighting}', Path(args.run_dir))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
