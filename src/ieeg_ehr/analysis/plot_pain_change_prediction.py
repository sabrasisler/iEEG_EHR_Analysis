"""Figures of a pain_change prediction run (`pain_change_prediction score`).

    python -m ieeg_ehr.analysis.plot_pain_change_prediction --run-dir <run> \\
        --weights <weights name> --prediction raw --weighting unit

Every figure averages within subject first, then shows mean +/- SEM across
subjects, and drops any group, and any time step of a line, with fewer than
`--min-subjects` subjects. The time-axis figures show hours since the first report
out to PLOT_HORIZON_MIN, and shade the first 5 min, where the sliding window still
overlaps the baseline window. Descriptive: no significance marks.

Written to `<run>/figures/<weighting>/<kind>_dpain_<prediction>-<weighting>.png`, one
folder per weighting: `unit` (equal weights, the default), `uniform` (beta-weighted
least squares), `ivw`. Kinds:

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
    dc-groups-actual     per ΔC bin, actual Δpain at report 2 beside the prediction
                         at t2, and pain at report 1 and report 2
    dc-pain-fell         pain fell vs no pain change (DC_PAIN_PANELS), one line per ΔC
                         bin: the prediction over the interval, with the fit
                         of prediction(t2) ~ Δpain + ΔC + (1 | subject) over all
                         dc-groups pairs (written to dc-pain-model_dpain_<tag>.csv)
    dose-response-by-dc  dose-response, one panel per ΔC bin, each with the slope of
                         prediction on actual Δpain (also in dc-pain-model)
    dc-shift-contributions  the ΔC shift of the prediction split into ROI x band
                         cells: decoder weight, the fit's dC effect on power, and their
                         product, summed by ROI and band (dc-shift-contributions_dpain_
                         <tag>.csv)
    dc-components        per ROI and region family x band group, the Δpain against the ΔC
                         coefficient of score_k(t2) ~ Δpain + ΔC + (1 | subject), its
                         part of the prediction (dc-components_dpain_<tag>.csv)

For the adjusted prediction the first EXTRAPOLATE_BELOW_MIN minutes are hatched: the
fit saw no gap that short.
"""

import argparse
import json
import logging
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.cm import ScalarMappable  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, Normalize  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import statsmodels.formula.api as smf  # noqa: E402

from ieeg_ehr import config, io  # noqa: E402
from ieeg_ehr.analysis.pain_change import FIGURES_SUBDIR, load_exposure  # noqa: E402
from ieeg_ehr.analysis.pain_change_prediction import (  # noqa: E402
    DPAIN_BINS, EXTRAPOLATE_BELOW_MIN, assign_bins, load_trajectory)
from ieeg_ehr.analysis.plot_band_map_domains import BAND_SYMBOL  # noqa: E402
from ieeg_ehr.analysis.plot_domain_glass import BAND_COLOURS  # noqa: E402
from ieeg_ehr.analysis.run_bandpower_mixed import BAND_SETS  # noqa: E402
from ieeg_ehr.config import opioid_mme, roi_schemes  # noqa: E402
from ieeg_ehr.features.build_pain_change_trajectory import WINDOW_MIN, cell_column  # noqa: E402
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
#: Both axes of dose-response-by-dc, in NRS.
DOSE_BY_DC_LIM = 10
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
#: Pain outcomes of dc-pain-fell, Δpain inclusive. Set by the user 2026-10-20: the
#: increases are too sparse to draw, and Δpain -2 is left out as a gap between the two.
DC_PAIN_PANELS = {'Pain fell': (-10, -3), 'No pain change': (-1, 1)}
#: y axis of dc-pain-fell, in NRS: the SEM of thin lines reaches far past the means.
DC_PAIN_YLIM = 15
#: DC_BINS as ranges of ΔC, for dc-pain-fell.
DC_BIN_LABELS = {'large fall': f'ΔC < −{DC_LARGE:g}',
                 'small fall': f'−{DC_LARGE:g} ≤ ΔC ≤ −{DC_SMALL:g}',
                 'no change': f'|ΔC| < {DC_SMALL:g}',
                 'small rise': f'{DC_SMALL:g} ≤ ΔC ≤ {DC_LARGE:g}',
                 'large rise': f'ΔC > {DC_LARGE:g}'}
#: Region families and band groups of the dc-components scatter. Set by the user
#: 2026-10-20, basal ganglia placed with the thalamus.
REGION_FAMILIES = {
    'Subcortical': ('Ventral thalamus', 'Dorsal/medial thalamus', 'Basal ganglia'),
    'Cingulate': ('ACC', 'MCC'),
    'Limbic/MTL': ('Amygdala', 'Hippocampus', 'Medial temporal lobe'),
    'Prefrontal': ('dPFC', 'vlPFC', 'vmPFC', 'OFC', 'Frontal operculum'),
    'Sensorimotor/insula': ('S1', 'SII', 'M1', 'Premotor', 'aIns', 'pIns'),
    'Posterior/temporal': ('PCC', 'Precuneus', 'SPL + IPS', 'Supramarginal', 'Angular',
                           'STG', 'MTG', 'ITG/fusiform', 'Occipital')}
#: matplotlib Dark2, the set POSTER_COLOURS of the heatmaps draws from.
FAMILY_COLOURS = ('#1b9e77', '#d95f02', '#7570b3', '#e7298a', '#66a61e', '#e6ab02')
BAND_GROUPS = {'low': ('delta', 'theta', 'alpha'), 'beta': ('beta',),
               'high': ('gamma', 'high_gamma')}
BAND_GROUP_MARKERS = {'low': 'o', 'beta': 's', 'high': '^'}
#: prediction(t2) ~ ... + (1 | subject) over all dc-groups pairs, dC continuous.
DC_PAIN_FORMULAS = ('value ~ delta_pain + dc', 'value ~ delta_pain + dc + pain1_within',
                    'value ~ delta_pain * dc')


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


def _dose_errorbar(ax, s, colour, dx=0.0, label=None, annotate=False):
    """dose_points rows as mean +/- SEM, labelled subjects/pairs when `annotate`."""
    ax.errorbar(s['delta_pain'] + dx, s['mean'], yerr=s['sem'], fmt='o', ms=5, color=colour,
                ecolor=colour, elinewidth=1, capsize=0, label=label)
    if annotate:
        for r in s.itertuples():
            ax.annotate(f'{r.n_subjects}/{r.n_pairs}', (r.delta_pain, r.mean),
                        textcoords='offset points', xytext=(0, 8), ha='center',
                        fontsize=7, color=MUTED)


def fig_dose(df, out_dir, name, title, min_subjects, hue=None, hue_levels=(),
             hue_colours=()):
    pts = dose_points(df, hue)
    pts = pts[pts['n_subjects'] >= min_subjects]
    fig, ax = plt.subplots(figsize=(6, 4.6))
    series = [(None, NEUTRAL, pts)] if hue is None else [
        (lvl, c, pts[pts[hue] == lvl]) for lvl, c in zip(hue_levels, hue_colours)]
    offsets = np.linspace(-0.2, 0.2, len(series)) if len(series) > 1 else [0.0]
    for (lvl, colour, s), dx in zip(series, offsets):
        _dose_errorbar(ax, s, colour, dx, None if lvl is None else str(lvl),
                       annotate=hue is None)
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


def to_fraction(df, group):
    """Each pair's steps averaged within FRACTION_BINS steps of (t - t1) / (t2 - t1)."""
    step = np.minimum(np.floor(df['t_min'] / df['gap_min'] * FRACTION_BINS), FRACTION_BINS - 1)
    return (df.assign(x_h=(step + 0.5) / FRACTION_BINS)
            .groupby(['pair_id', 'subject', group, 'x_h'], observed=True)[['value', 'dc']]
            .mean().reset_index())


def _fraction_axis(ax):
    ax.set_xlim(0, 1)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1])
    ax.set_xticklabels(['t1', '25%', '50%', '75%', 't2'])
    ax.set_xlabel('Fraction of the interval from report 1 to report 2', color=INK)


def fig_dc_groups(df, out_dir, name, title, min_subjects, axis):
    """Predicted Δpain (top) and ΔC(t) (bottom) from t1 to t2, one line per ΔC bin.

    `axis` 'hours' plots hours since t1 out to PLOT_HORIZON_MIN. 'fraction' plots
    (t - t1) / (t2 - t1) in FRACTION_BINS steps, each pair averaged within a step first.
    """
    if axis == 'fraction':
        df = to_fraction(df, 'dc_group')
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
        _fraction_axis(bottom)
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


def fig_dc_actual(df, out_dir, name, title, min_subjects):
    """Per ΔC bin, actual Δpain at report 2 beside the prediction at t2 (left), and
    pain at report 1 and report 2 (right). Subject means first, then mean +/- SEM.
    """
    at_t2 = (df[df['t_min'] == df['gap_min'].round()].dropna(subset=['value'])
             .assign(pain2=lambda d: d['pain1'] + d['delta_pain']))
    per_subject = (at_t2.groupby(['dc_group', 'subject'], observed=True)
                   [['delta_pain', 'value', 'pain1', 'pain2']].mean().reset_index())
    sizes = group_sizes(at_t2, 'dc_group')
    drawn = [g for g in DC_BINS if sizes.get(g, (0, 0))[0] >= min_subjects]
    rng = np.random.default_rng(0)
    fig, (left, right) = plt.subplots(1, 2, figsize=(11, 4.6),
                                      gridspec_kw={'width_ratios': (3, 2)})
    for ax in (left, right):
        _style(ax)
    left.axhline(0, color=MUTED, lw=0.8, zorder=1)
    for i, g in enumerate(drawn):
        colour = DC_COLOURS[DC_BINS.index(g)]
        s = per_subject[per_subject['dc_group'] == g]
        left.scatter(i - 0.15 + rng.uniform(-0.07, 0.07, len(s)), s['delta_pain'], s=12,
                     color=colour, alpha=0.35, lw=0, zorder=2)
        for col, dx, face in (('delta_pain', -0.15, colour), ('value', 0.15, 'white')):
            left.errorbar(i + dx, s[col].mean(), yerr=_sem(s[col]), fmt='o', ms=7,
                          color=colour, mfc=face, mew=1.5, elinewidth=1.4, capsize=0,
                          zorder=3)
        right.errorbar([i - 0.12, i + 0.12], s[['pain1', 'pain2']].mean(),
                       yerr=[_sem(s['pain1']), _sem(s['pain2'])], fmt='-', color=colour,
                       elinewidth=1.4, capsize=0, lw=1.4)
        right.plot(i - 0.12, s['pain1'].mean(), 'o', ms=6, color=colour, mfc='white', mew=1.5)
        right.plot(i + 0.12, s['pain2'].mean(), 'o', ms=6, color=colour)
    labels = [f'{g}\n{sizes[g][0]}/{sizes[g][1]}' for g in drawn]
    for ax in (left, right):
        ax.set_xticks(range(len(drawn)))
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_xlim(-0.6, len(drawn) - 0.4)
        ax.set_xlabel('ΔC, t1 to t2', color=INK)
    left.set_ylabel('Δpain, report 2 - report 1 (NRS)', color=INK)
    right.set_ylabel('Pain (NRS)', color=INK)
    right.set_title('Open: report 1. Filled: report 2.', fontsize=9, color=MUTED)
    handles = [plt.Line2D([], [], ls='', marker='o', ms=6, color=NEUTRAL, mfc=face, mew=1.5)
               for face in (NEUTRAL, 'white')]
    left.legend(handles, ['actual, report 2', 'predicted, t2'], loc='upper left',
                frameon=False, fontsize=8, labelcolor=INK)
    fig.suptitle(title, fontsize=10, color=INK)
    _footer(fig, f'{at_t2["subject"].nunique()} subjects, {at_t2["pair_id"].nunique()} pairs, '
                 f'the dc-groups pairs with a prediction at t2. Tick labels: subjects/pairs. '
                 f'Small dots: subject means of actual Δpain, x jittered. '
                 f'Large: mean ± SEM across subjects. Groups under {min_subjects} subjects '
                 f'dropped ({len(DC_BINS) - len(drawn)}).')
    _save(fig, out_dir, name)


def dc_pain_model(at_t2):
    """Fixed effects of each DC_PAIN_FORMULAS fit, statsmodels MixedLM, REML,
    random intercept per subject, one row per term.
    """
    df = at_t2.assign(pain1_within=at_t2['pain1']
                      - at_t2.groupby('subject')['pain1'].transform('mean'))
    return pd.DataFrame([row for formula in DC_PAIN_FORMULAS
                         for row in _fixed_effects(df, formula, f'{formula} + (1 | subject)')])


def _fixed_effects(df, formula, label):
    """One row per fixed effect of `formula` + (1 | subject), statsmodels MixedLM, REML."""
    fit = smf.mixedlm(formula, df, groups=df['subject']).fit(reml=True)
    ci = fit.conf_int()
    return [{'formula': label, 'term': term, 'estimate': fit.fe_params[term],
             'se': fit.bse_fe[term], 'ci_low': ci.loc[term, 0], 'ci_high': ci.loc[term, 1],
             'p': fit.pvalues[term], 'n_pairs': len(df), 'n_subjects': df['subject'].nunique(),
             'converged': fit.converged, 'subject_var': fit.cov_re.iloc[0, 0]}
            for term in fit.fe_params.index]


def dc_dose_slopes(per_pair):
    """Per ΔC bin, value ~ delta_pain + (1 | subject) on near_t2 pairs, as _fixed_effects rows."""
    return pd.DataFrame([
        row for g, d in per_pair.groupby('dc_group', observed=True)
        for row in _fixed_effects(d, 'value ~ delta_pain',
                                  f'value(t2 ± {DOSE_WINDOW_MIN} min) ~ delta_pain '
                                  f'+ (1 | subject), {g}')])


def fig_dose_by_dc(df, out_dir, name, title, min_subjects, slopes):
    """fig_dose, one panel per ΔC bin, each titled with its dc_dose_slopes slope."""
    pts = dose_points(df, hue='dc_group')
    pts = pts[pts['n_subjects'] >= min_subjects]
    fig, axes = plt.subplots(1, len(DC_BINS), figsize=(3.2 * len(DC_BINS), 4),
                             sharex=True, sharey=True)
    lim = DOSE_BY_DC_LIM
    n_out = int((pts[['delta_pain', 'mean']].abs() > lim).any(axis=1).sum())
    for ax, g, colour in zip(axes, DC_BINS, DC_COLOURS):
        _style(ax)
        ax.axhline(0, color=MUTED, lw=0.8)
        ax.plot([-lim, lim], [-lim, lim], color=MUTED, lw=0.8, ls='--')
        s = pts[pts['dc_group'] == g]
        _dose_errorbar(ax, s, colour, annotate=True)
        fe = slopes[slopes['formula'].str.endswith(f', {g}')].set_index('term')
        r = fe.loc['delta_pain']
        x = np.array([s['delta_pain'].min(), s['delta_pain'].max()])
        ax.plot(x, fe.loc['Intercept', 'estimate'] + r.estimate * x, color=colour, lw=1.2)
        ax.set_title(f'{g} ({r.n_subjects} subj, {r.n_pairs} pairs)\n'
                     f'slope {r.estimate:+.2f} [{r.ci_low:+.2f}, {r.ci_high:+.2f}]',
                     fontsize=9, color=INK, pad=14)
    axes[0].set_xlim(-lim, lim)
    axes[0].set_ylim(-lim, lim)
    axes[0].set_ylabel(f'Predicted Δpain, ±{DOSE_WINDOW_MIN} min of t2 (NRS)', color=INK)
    fig.supxlabel('Actual Δpain at report 2 (NRS)', color=INK, fontsize=10)
    fig.suptitle(f'{title}\nby ΔC, t1 to t2', fontsize=10, color=INK)
    fig.tight_layout()
    _footer(fig, 'Labels: subjects/pairs. Points with fewer than '
                 f'{min_subjects} subjects dropped. Axes clipped at ±{lim} ({n_out} of '
                 f'{len(pts)} points beyond). Dashed: identity. Solid: the fit. Slope: fixed '
                 'effect [95% CI] of value ~ Δpain + (1 | subject) over the pairs of the '
                 'panel, statsmodels MixedLM, REML. ΔC bins and first-day exclusion as '
                 'dc-groups.')
    _save(fig, out_dir, name)


def fig_dc_pain(df, out_dir, name, title, min_subjects, model):
    """One panel per DC_PAIN_PANELS pain outcome, one line per ΔC bin: the prediction
    over the fraction of the interval, with the `model` (dc_pain_model) estimates of
    the first formula in the title.
    """
    lines = [DC_BIN_LABELS[g] for g in DC_BINS]
    fig, axes = plt.subplots(2, len(DC_PAIN_PANELS), figsize=(5.6 * len(DC_PAIN_PANELS), 6),
                             sharex=True, sharey='row', gridspec_kw={'height_ratios': (3, 1)})
    dropped = []
    for (ax, legend_ax), (panel, (lo, hi)) in zip(axes.T, DC_PAIN_PANELS.items()):
        d = df[df['delta_pain'].between(lo, hi)].assign(
            dc_label=lambda x: x['dc_group'].map(DC_BIN_LABELS))
        _style(ax)
        ax.axhline(0, color=MUTED, lw=0.8, zorder=1)
        drawn = _trajectory_lines(ax, to_fraction(d, 'dc_label'), 'dc_label', lines,
                                  DC_COLOURS, min_subjects)
        dropped += [f'{panel}: {g}' for g in lines if g not in drawn]
        ax.set_title(f'{panel} (Δpain {lo:+d} to {hi:+d})', fontsize=10, color=INK)
        _fraction_axis(ax)
        ax.tick_params(labelbottom=True)
        ax.set_ylim(-DC_PAIN_YLIM, DC_PAIN_YLIM)
        legend_ax.axis('off')
        legend_ax.legend(*ax.get_legend_handles_labels(), frameon=False, fontsize=8,
                         labelcolor=INK, loc='upper center',
                         title='ΔC, t1 to t2 (typical doses)', title_fontsize=8)
    axes[0, 0].set_ylabel('Predicted Δpain (NRS)', color=INK)
    fe = model[model['formula'] == model['formula'].iloc[0]].set_index('term')
    fig.suptitle(f'{title}\nprediction(t2) ~ Δpain + ΔC + (1 | subject), all '
                 f'{fe["n_pairs"].iloc[0]} dc-groups pairs, ΔC continuous: ' + ',   '.join(
                     f'{label} {fe.loc[t, "estimate"]:+.2f} [{fe.loc[t, "ci_low"]:+.2f}, '
                     f'{fe.loc[t, "ci_high"]:+.2f}]'
                     for t, label in (('delta_pain', 'Δpain'), ('dc', 'ΔC'))),
                 fontsize=10, color=INK)
    fig.tight_layout()
    shown = df[np.logical_or.reduce([df['delta_pain'].between(lo, hi)
                                     for lo, hi in DC_PAIN_PANELS.values()])]
    _footer(fig, textwrap.fill(
        f'{shown["subject"].nunique()} subjects, {shown["pair_id"].nunique()} dc-groups pairs; '
        'pairs with Δpain of -2 or of +2 or more left out. Mean ± SEM across subjects of '
        f'subject means, y clipped at ±{DC_PAIN_YLIM}. '
        f'Lines under {min_subjects} subjects not drawn: {", ".join(dropped) or "none"}.', 160))
    _save(fig, out_dir, name)


def pair_cell_terms(at_t2, trajectory_dir, weights_name, source_run, weighting):
    """Per pair at t2 and cell of its subject's weights fold: the prediction weight,
    dz, and the fold's dC effect (`dc_beta` of the source `cells_loo.csv`).

    The prediction is sum_c w_c dz_c with w_c = beta_c iv_c / sum beta^2 iv over the
    cells finite at t2 (iv = 1 / var_resid for ivw, else 1; sign(beta_c) / sum |beta|
    for unit). A cell the pair lacks has weight 0 and dz NaN.
    """
    weights = io.read_table(config.pain_change_prediction_weights_path(weights_name),
                            on_stale='warn')
    dc_beta = (io.read_table(Path(source_run) / 'cells_loo.csv', on_stale='warn')
               .rename(columns={'region': 'roi', 'freq': 'band'})
               [['heldout_subject', 'roi', 'band', 'dc_beta']])
    weights = weights.merge(dc_beta, on=['heldout_subject', 'roi', 'band'], how='left')
    if weights['dc_beta'].isna().any():
        raise ValueError('weights cells with no dc_beta in the source run')
    _, dz = load_trajectory(trajectory_dir)
    dz = dz.merge(at_t2[['pair_id', 'subject', 't_min']], on=['pair_id', 't_min'])
    parts = []
    for subject, rows in dz.groupby('subject'):
        w = weights[weights['heldout_subject'] == subject]
        x = rows[[cell_column(r, b) for r, b in zip(w['roi'], w['band'])]].to_numpy(
            dtype=config.CACHE_ACCUMULATE_DTYPE)
        finite = np.isfinite(x)
        beta = w['beta'].to_numpy()
        iv = 1.0 / w['var_resid'].to_numpy() if weighting == 'ivw' else 1.0
        num, den = ((np.sign(beta), np.abs(beta)) if weighting == 'unit'
                    else (beta * iv, beta ** 2 * iv))
        weight = (np.where(finite, num, 0.0)
                  / np.where(finite, den, 0.0).sum(axis=1, keepdims=True))
        n = len(rows)
        parts.append(pd.DataFrame({
            'pair_id': np.repeat(rows['pair_id'].to_numpy(), len(w)), 'subject': subject,
            'roi': np.tile(w['roi'].to_numpy(), n), 'band': np.tile(w['band'].to_numpy(), n),
            'weight': weight.ravel(), 'dz': x.ravel(),
            'dc_beta': np.tile(w['dc_beta'].to_numpy(), n)}))
    return pd.concat(parts, ignore_index=True)


def dc_shift_contributions(terms):
    """Per subject and cell, its share of the ΔC shift of the prediction.

    A dC effect b_c on dz_c shifts the prediction by sum_c w_c b_c per typical dose,
    each subject with its own fold, averaged over its pairs. A cell a pair lacks
    enters as weight 0, so the shares sum to the subject's shift. The d_pain:dC term
    is left out.
    """
    t = terms.assign(contribution=terms['weight'] * terms['dc_beta'],
                     has_cell=np.isfinite(terms['dz']))
    return (t.groupby(['subject', 'roi', 'band'], sort=False)
            .agg(weight=('weight', 'mean'), dc_beta=('dc_beta', 'first'),
                 has_cell=('has_cell', 'any'), contribution=('contribution', 'mean'))
            .reset_index())


def component_fits(terms, at_t2, min_subjects):
    """score_k(t2) ~ delta_pain + dc + (1 | subject) per component, as _fixed_effects rows.

    score_k = sum of w_c dz_c over the component's cells, its part of the prediction.
    Components are each ROI (`level` roi, `bands` all) and each REGION_FAMILIES x
    BAND_GROUPS group (`level` family), fitted over the pairs that have any of the
    component's cells, with at least `min_subjects`.
    """
    family = {r: f for f, rois in REGION_FAMILIES.items() for r in rois}
    group = {b: g for g, bands in BAND_GROUPS.items() for b in bands}
    unmapped = sorted(set(terms['roi']) - set(family)) + sorted(set(terms['band']) - set(group))
    if unmapped:
        raise ValueError(f'no region family or band group for {unmapped}')
    scores = terms.assign(value=terms['weight'] * terms['dz'], family=terms['roi'].map(family),
                          band_group=terms['band'].map(group), all_bands='all')
    meta = at_t2[['pair_id', 'delta_pain', 'dc']]
    rows = []
    for level, keys in (('roi', ['roi', 'all_bands']), ('family', ['family', 'band_group'])):
        s = (scores.groupby(['pair_id', 'subject', *keys], sort=False)['value']
             .sum(min_count=1).dropna().reset_index().merge(meta, on='pair_id'))
        for (region, bands), d in s.groupby(keys, sort=False):
            if d['subject'].nunique() < min_subjects:
                continue
            rows += [dict(r, level=level, region=region, bands=bands)
                     for r in _fixed_effects(d, 'value ~ delta_pain + dc',
                                             'score_k ~ delta_pain + dc + (1 | subject)')]
    return pd.DataFrame(rows)


def _crosshair(ax, w, **kw):
    """Points of a component_fits pivot with their 95% CIs on both axes; returns (x, y)."""
    x, y = w[('estimate', 'delta_pain')], w[('estimate', 'dc')]
    ax.errorbar(x, y, xerr=[x - w[('ci_low', 'delta_pain')], w[('ci_high', 'delta_pain')] - x],
                yerr=[y - w[('ci_low', 'dc')], w[('ci_high', 'dc')] - y], capsize=0, **kw)
    return x, y


def fig_components(fits, out_dir, name, title, min_subjects):
    """Each component's Δpain coefficient (x) against its ΔC coefficient (y), with 95%
    CIs: ROIs (left), REGION_FAMILIES x BAND_GROUPS (right, colour family, marker band).
    """
    wide = fits[fits['term'].isin(['delta_pain', 'dc'])].pivot_table(
        index=['level', 'region', 'bands'], columns='term',
        values=['estimate', 'ci_low', 'ci_high'], sort=False)
    fig, (left, right) = plt.subplots(1, 2, figsize=(14, 6.2))
    for ax in (left, right):
        _style(ax)
        ax.axhline(0, color=MUTED, lw=0.8)
        ax.axvline(0, color=MUTED, lw=0.8)
        ax.set_xlabel('Δpain coefficient (score NRS per NRS of reported Δpain)', color=INK)
    left.set_ylabel('ΔC coefficient at fixed Δpain (score NRS per typical dose)', color=INK)
    roi = wide.loc['roi']
    x, y = _crosshair(left, roi, fmt='o', ms=5, color=NEUTRAL, ecolor=GRID, elinewidth=1)
    for (r, _), xi, yi in zip(roi.index, x, y):
        left.annotate(roi_schemes.ROI_RENAMES.get(r, r), (xi, yi), textcoords='offset points',
                      xytext=(4, 3), fontsize=7, color=INK)
    left.set_title(f'ROIs, all bands ({len(roi)})', fontsize=9, color=INK)
    family = wide.loc['family']
    for f, colour in zip(REGION_FAMILIES, FAMILY_COLOURS):
        for g, marker in BAND_GROUP_MARKERS.items():
            if (f, g) in family.index:
                _crosshair(right, family.loc[[(f, g)]], fmt=marker, ms=8, color=colour,
                           mec='white', mew=0.8, ecolor=colour, elinewidth=1.2, alpha=0.9)
    families = [plt.Line2D([], [], ls='', marker='o', ms=7, color=c)
                for c in FAMILY_COLOURS[:len(REGION_FAMILIES)]]
    groups = [plt.Line2D([], [], ls='', marker=m, ms=7, color=NEUTRAL)
              for m in BAND_GROUP_MARKERS.values()]
    right.add_artist(right.legend(families, list(REGION_FAMILIES), frameon=False, fontsize=8,
                                  labelcolor=INK, loc='lower right'))
    right.legend(groups, [f'{g}: {"/".join(BAND_SYMBOL[b] for b in BAND_GROUPS[g])}'
                          for g in BAND_GROUP_MARKERS], frameon=False, fontsize=8,
                 labelcolor=INK, loc='upper right')
    right.set_title(f'Region family x band group ({len(family)})', fontsize=9, color=INK)
    fig.suptitle(f'{title}\nPer component: score_k(t2) ~ Δpain + ΔC + (1 | subject), all '
                 'dc-groups pairs, ΔC continuous', fontsize=10, color=INK)
    fig.tight_layout()
    _footer(fig, textwrap.fill(
        'score_k: the component\'s part of the prediction at t2, sum of its cells\' weight x '
        'dz, so the components add up to the prediction. Each fit uses the pairs that have '
        f'the component, components under {min_subjects} subjects left out. 95% CIs. On the '
        'x axis: tracks reported pain. On the y axis: moves with opioid exposure at fixed '
        'pain. statsmodels MixedLM, REML. Families: ' + '; '.join(
            f'{f}: {", ".join(r)}' for f, r in REGION_FAMILIES.items()) + '.', 200))
    _save(fig, out_dir, name)


def shift_by_cell(per_subject):
    """Per cell, weight and contribution averaged over ALL subjects (0 where absent),
    so contributions sum to the mean shift, and dc_beta over the subjects with it.
    """
    n = per_subject['subject'].nunique()
    g = per_subject.groupby(['roi', 'band'], sort=False)
    has = per_subject[per_subject['has_cell']].groupby(['roi', 'band'], sort=False)
    return pd.DataFrame({'weight': g['weight'].sum() / n,
                         'contribution': g['contribution'].sum() / n,
                         'dc_beta': has['dc_beta'].mean(),
                         'n_subjects': has['subject'].nunique()}).reset_index()


def fig_dc_shift(cells, out_dir, name, title, rois, bands, model_dc):
    """Grouped bars per ROI, one bar per band: decoder weight, dC effect on power,
    contribution (rows), and the contribution summed by ROI and by band (right).
    """
    nb = len(bands)
    cells = cells.set_index(['roi', 'band'])
    fig = plt.figure(figsize=(20, 10))
    grid = fig.add_gridspec(3, 2, width_ratios=(5, 1), hspace=0.35, wspace=0.18)
    rows = [fig.add_subplot(grid[i, 0]) for i in range(3)]
    for ax in rows[1:]:
        ax.sharex(rows[0])
    by_roi, by_band = fig.add_subplot(grid[0:2, 1]), fig.add_subplot(grid[2, 1])
    centres = np.arange(len(rois)) * (nb + 1) + (nb - 1) / 2
    for ax, col, label in zip(rows, ('weight', 'dc_beta', 'contribution'),
                              ('Decoder weight (NRS per z)', 'ΔC effect on power\n'
                               '(z per typical dose)', 'Contribution to the shift\n'
                               '(NRS per typical dose)')):
        _style(ax)
        ax.grid(axis='x', visible=False)
        ax.axhline(0, color=MUTED, lw=0.8)
        for i in range(0, len(rois), 2):
            ax.axvspan(i * (nb + 1) - 0.6, i * (nb + 1) + nb - 0.4, color=SHADE, lw=0,
                       zorder=0)
        for j, band in enumerate(bands):
            v = [cells[col].get((roi, band), np.nan) for roi in rois]
            ax.bar(np.arange(len(rois)) * (nb + 1) + j, v, width=0.9,
                   color=BAND_COLOURS[band], label=BAND_SYMBOL[band] if col == 'weight' else None)
        ax.set_ylabel(label, color=INK, fontsize=9)
        ax.tick_params(labelbottom=False)
    rows[-1].tick_params(labelbottom=True)
    rows[-1].set_xticks(centres)
    rows[-1].set_xticklabels([roi_schemes.ROI_RENAMES.get(r, r) for r in rois], rotation=55,
                             ha='right', rotation_mode='anchor', fontsize=8)
    rows[-1].set_xlim(-1, len(rois) * (nb + 1) - 1)
    rows[0].legend(frameon=False, fontsize=8, labelcolor=INK, ncol=nb, loc='upper left')
    total = cells['contribution'].sum()
    absent = [r for r in rois if cells.loc[r, 'n_subjects'].isna().all()]
    roi_sum = cells.groupby(level='roi', sort=False)['contribution'].sum().sort_values()
    _style(by_roi)
    by_roi.barh(range(len(roi_sum)), roi_sum.to_numpy(),
                color=np.where(roi_sum.to_numpy() < 0, DC_COLOURS[4], DC_COLOURS[0]))
    by_roi.set_yticks(range(len(roi_sum)))
    by_roi.set_yticklabels([roi_schemes.ROI_RENAMES.get(r, r) for r in roi_sum.index],
                           fontsize=7)
    by_roi.axvline(0, color=MUTED, lw=0.8)
    by_roi.set_title('By ROI, all bands', fontsize=9, color=INK)
    band_sum = cells.groupby(level='band', sort=False)['contribution'].sum().reindex(bands)
    _style(by_band)
    by_band.barh(range(nb), band_sum.to_numpy(), color=[BAND_COLOURS[b] for b in bands])
    by_band.set_yticks(range(nb))
    by_band.set_yticklabels([BAND_SYMBOL[b] for b in bands], fontsize=8)
    by_band.invert_yaxis()
    by_band.axvline(0, color=MUTED, lw=0.8)
    by_band.set_title('By band, all ROIs', fontsize=9, color=INK)
    by_band.set_xlabel('NRS per typical dose', color=INK, fontsize=8)
    fig.suptitle(f'{title}\nΔC shift of the prediction by cell: sum of contributions '
                 f'{total:+.2f} NRS per typical dose; fitted ΔC effect on the prediction '
                 f'at t2 {model_dc:+.2f}', fontsize=10, color=INK)
    _footer(fig, 'Contribution = decoder weight x ΔC effect, per subject with its own LOO '
                 'fold and the cells each pair has at t2, averaged over pairs, then over '
                 'subjects with 0 where a subject lacks the cell, so the contributions sum '
                 'to the mean shift. Weight and contribution average over all subjects, the '
                 'ΔC effect over the subjects with the cell. The d_pain:dC term is left out. '
                 f'No held-out pair has: {", ".join(absent) or "none"}. Descriptive.')
    _save(fig, out_dir, name)


# ============================================================================
# Entry points
# ============================================================================

def prediction_frame(run_dir, weights_name, prediction='raw', weighting='unit',
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


def render_all(run_dir, weights_name, prediction='raw', weighting='unit', min_subjects=3,
               bins=None):
    """Every figure for one prediction (raw | adj) and weighting (uniform | ivw | unit)."""
    bins = bins or DPAIN_BINS
    out_dir = Path(run_dir) / FIGURES_SUBDIR / weighting
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
    groups = late.assign(dc_group=dc_bin(late['dc_interval']))[
        ['pair_id', 'dc_group', 'dc_interval']]
    dc_df = df.merge(steps, on=['pair_id', 't_min']).merge(groups, on='pair_id')
    logger.info('dC groups (pairs): %s', groups['dc_group'].value_counts().to_dict())
    for axis in ('hours', 'fraction'):
        fig_dc_groups(dc_df, out_dir, name(f'dc-groups-{axis}'), title, min_subjects, axis)
    fig_dc_actual(dc_df, out_dir, name('dc-groups-actual'), title, min_subjects)
    # At t2 the step dc is the whole-interval dC.
    at_t2 = dc_df[dc_df['t_min'] == dc_df['gap_min'].round()].dropna(subset=['value'])
    dose_df = df.merge(groups[['pair_id', 'dc_group']], on='pair_id')
    per_pair = near_t2(dose_df, keys=['dc_group']).dropna(subset=['value'])
    slopes = dc_dose_slopes(per_pair)
    model = pd.concat([dc_pain_model(at_t2), slopes], ignore_index=True)
    io.write_table(model, Path(run_dir) / f'dc-pain-model_dpain_{tag}.csv',
                   params={'formulas': DC_PAIN_FORMULAS, 'prediction': prediction,
                           'weighting': weighting, 'first_hours_out': DC_FIRST_HOURS_OUT},
                   parents=[Path(run_dir) / f'predictions_{weights_name}.csv'],
                   subjects=sorted(at_t2['subject'].unique()),
                   script='ieeg_ehr/analysis/plot_pain_change_prediction.py')
    fig_dc_pain(dc_df, out_dir, name('dc-pain-fell'), title, min_subjects, model)
    fig_dose_by_dc(dose_df, out_dir, name('dose-response-by-dc'), title, min_subjects, slopes)
    params = json.loads((Path(run_dir) / 'provenance.json').read_text())['params']
    weights_path = config.pain_change_prediction_weights_path(weights_name)
    source_run = Path(io.read_sidecar(weights_path)['params']['source_run'])
    source_params = json.loads((source_run / 'provenance.json').read_text())['params']
    terms = pair_cell_terms(at_t2, params['trajectory_dir'], weights_name, source_run,
                            weighting)
    per_subject = dc_shift_contributions(terms)
    # The heatmap's ROI and band order: the in-sample fit the LOO run refits.
    rois = list(dict.fromkeys(io.read_table(Path(source_params['reference_run']) / 'cells.csv',
                                            on_stale='warn')['region']))
    shift = shift_by_cell(per_subject)
    io.write_table(shift, Path(run_dir) / f'dc-shift-contributions_dpain_{tag}.csv',
                   params={'weights': weights_name, 'weighting': weighting,
                           'first_hours_out': DC_FIRST_HOURS_OUT},
                   parents=[weights_path, source_run / 'cells_loo.csv'],
                   subjects=sorted(per_subject['subject'].unique()),
                   script='ieeg_ehr/analysis/plot_pain_change_prediction.py')
    fig_dc_shift(shift, out_dir, name('dc-shift-contributions'), title,
                 [r for r in rois if r in set(shift['roi'])],
                 list(BAND_SETS[source_params['band_set']]),
                 model.set_index(['formula', 'term']).loc[
                     (f'{DC_PAIN_FORMULAS[0]} + (1 | subject)', 'dc'), 'estimate'])
    # The source fit's subject floor for a cell.
    fits = component_fits(terms, at_t2, source_params['min_subjects'])
    io.write_table(fits, Path(run_dir) / f'dc-components_dpain_{tag}.csv',
                   params={'weights': weights_name, 'weighting': weighting,
                           'formula': 'score_k ~ delta_pain + dc + (1 | subject)',
                           'min_subjects': source_params['min_subjects'],
                           'first_hours_out': DC_FIRST_HOURS_OUT},
                   parents=[weights_path, Path(run_dir) / f'predictions_{weights_name}.csv'],
                   subjects=sorted(at_t2['subject'].unique()),
                   script='ieeg_ehr/analysis/plot_pain_change_prediction.py')
    fig_components(fits, out_dir, name('dc-components'), title, source_params['min_subjects'])


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--run-dir', required=True)
    ap.add_argument('--weights', required=True)
    ap.add_argument('--prediction', choices=['raw', 'adj'], default='raw')
    ap.add_argument('--weighting', choices=['uniform', 'ivw', 'unit'], default='unit')
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
