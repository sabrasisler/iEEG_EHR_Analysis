#!/usr/bin/env python3
"""Across-subject consistency for an lme4 domain run: four figures.

    python -m ieeg_ehr.analysis.plot_domain_lmer_consistency --run-dir <domain_lmer run>
    python -m ieeg_ehr.analysis.plot_domain_lmer_consistency --run-dir <run> --figures profile

Everything lands in `<run>/consistency/`.

    signmap   The 4 processing domains x 6 bands: how many subjects share the
              group's slope sign, coloured by the EXCESS over a sign-flip null
              (`plot_domain_consistency.cell_stats`, the same statistic as the
              region-level `fig_consistency_signmap`). Pain-heatmap colouring
              (RdBu_r, symmetric) as a placeholder -- the analyst has not settled
              the colour scheme. No significance outlines.
    sigcells  The five BH-significant (domain, band) cells of the with-Control
              heatmap: every subject's own domain slope, as a strip + box. A
              DESCRIPTION of how consistent those cells are, not a test --
              choosing cells by their p-value and then testing agreement in
              them would be circular.
    sigcells_v2  The same cells redrawn in the house style (med_analysis/style.py):
              vertical light-grey violins, no group-slope diamond, each dot
              coloured by its own slope. Written beside v1, not over it.
    profile   One row per subject: Spearman r between that subject's unshrunken
              (ROI, band) slope map and the leave-one-out GROUP map, over the ROI
              cells of those five significant cells, against the subject's OWN
              null (free shuffle of their cell values). Needs the LOO refits
              from `run_domain_lmer_loo` (--collect first).

PER-SUBJECT SLOPES are two-stage and UNPOOLED, never BLUPs: one OLS line per
(subject, unit, band) through that subject's epochs, channels averaged per epoch
within the unit first (`plot_mixed_model_subject_lines.epoch_level`), on the
model's own `NRS_within`. They come from the run's saved FRAMES, so the rows are
exactly the rows the lme4 model saw.

THE FRAMES HAVE NO SESSION COLUMN, and `epoch_id` is unique only within a
session: sub-209 has two sessions and 1,067 (channel, epoch_id) rows collide.
Averaging per epoch would silently pool two different epochs. The frames are
written session by session with epochs in increasing order, so the session is
recovered from row order (a reset of `epoch_id`), and uniqueness of (subject,
session, channel, epoch) is ASSERTED per band -- a frame built differently fails
here instead of pooling.

THE PROFILE FILTERS: a band where the subject has fewer than two ROI cells is
dropped (the region-level profile's rule), then a subject needs at least
`PROFILE_MIN_CELLS` = 4 cells to get an r (`--min-cells`). The region-level floor
of 12 excluded 23 of 51 subjects here, because the map holds only the
significant cells (13 delta ROIs + 6 beta ROIs = 19 at most); lowered to 4 at the
analyst's instruction (2026-09-22). NOTE: with 4 cells there are only 24 distinct
orderings, so the smallest attainable p is ~1/24 > 0.025 -- a 4-cell subject can
be plotted but can never exceed their null. 5 cells is the least that can.

THE PROFILE NULL IS A FREE SHUFFLE, at the analyst's instruction (2026-09-22):
the subject's values are permuted across ALL their cells, not within band. The
region-level docstring (`plot_bandpower_consistency`, point 3) measured that a
free shuffle lets a shared delta-down / beta-up tilt carry the correlation; here
the map holds two bands with opposite group signs, so the same leak applies.
`r_within_band_p` (within-band shuffle) is computed alongside for comparison and
written to the table, not plotted.

EXPLORATORY. Discovery cohort only. Nominations, not findings.
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from ieeg_ehr import io
from ieeg_ehr.analysis.plot_bandpower_consistency import N_PERM
from ieeg_ehr.analysis.plot_domain_consistency import cell_stats
from ieeg_ehr.analysis.plot_domain_lmer import SLOPE_COL
from ieeg_ehr.analysis.plot_mixed_model_subject_lines import (epoch_level,
                                                              subject_slopes)
from ieeg_ehr.analysis.run_domain_lmer_loo import (BAND_ORDER, COLLECTED, NONE,
                                                   OUT_SUBDIR, SIG_TABLE,
                                                   run_provenance,
                                                   significant_cells)
from ieeg_ehr.analysis.run_bandpower_mixed import BAND_SETS

logger = logging.getLogger(__name__)

SCRIPT = 'ieeg_ehr/analysis/plot_domain_lmer_consistency.py'
DISCLAIMER = ('EXPLORATORY -- discovery cohort, NOMINATIONS NOT FINDINGS. '
              'Not confirmed out of sample.')
FIGURES = ('signmap', 'sigcells', 'sigcells_v2', 'profile')

#: The four processing domains; Control (the quasi-controls) is left off both
#: the sign map and the significant-cell figures by the analyst's instruction.
DOMAINS = ('Sensory', 'Affective', 'Cognitive', 'Modulatory')

PROFILE_PERM = 5000
PROFILE_SEED = 0
PROFILE_MIN_CELLS = 4
ALPHA_ONE_SIDED = 0.025
BLUE, GREY, RED = '#2166AC', '#9E9E9E', '#B2182B'
#: Panel (b) of the region-level `fig_consistency_profile` (13.6 x 6.6 in, two
#: equal panels), so the figure drops into the existing layout.
PROFILE_SIZE = (6.8, 6.6)
BAND_LABEL = {'high_gamma': 'high gamma'}
SAVE_EXT = ('png', 'pdf', 'svg')


# ============================================================================
# SLOPES FROM THE FRAMES
# ============================================================================

def load_frame(frames_dir, band):
    """The band's model frame, with a session ordinal recovered from row order."""
    df = io.read_table(Path(frames_dir) / f'{band}.parquet', on_stale='warn')
    df = df.reset_index(drop=True)
    # Within a subject the frame runs session by session with epochs in
    # increasing order, so a DROP in epoch_id between consecutive rows is a new
    # session. Asserted below rather than trusted.
    prev = df.groupby('subject')['epoch_id'].shift()
    df['session_ord'] = ((df['epoch_id'] < prev).astype(int)
                         .groupby(df['subject']).cumsum())
    dup = df.duplicated(['subject', 'session_ord', 'channel_uid', 'epoch_id'])
    if dup.any():
        raise SystemExit(f'{band}: {int(dup.sum())} (subject, session, channel, '
                         'epoch) rows still collide after session recovery; '
                         'the frame is not ordered as assumed.')
    n_multi = df.groupby('subject')['session_ord'].max().gt(0).sum()
    logger.info('%s: %d rows, %d subject(s) with >1 recovered session', band,
                len(df), n_multi)
    df['epoch_id'] = df['session_ord'].astype(str) + ':' + df['epoch_id'].astype(str)
    return df


def unit_slopes(df, unit_col, band):
    """Unpooled OLS slope per (subject, unit) -- `unit_col` is 'domain' or 'parcel'."""
    rows = []
    for unit, sub in df.groupby(unit_col):
        s = subject_slopes(epoch_level(sub))
        s[unit_col] = unit
        s['band'] = band
        rows.append(s)
    return pd.concat(rows, ignore_index=True)


# ============================================================================
# THE PROFILE STATISTIC
# ============================================================================

def _ranks(x):
    from scipy.stats import rankdata
    return rankdata(x)


def _rowwise_pearson(x, Y):
    xc = x - x.mean()
    Yc = Y - Y.mean(axis=1, keepdims=True)
    den = np.sqrt((xc ** 2).sum()) * np.sqrt((Yc ** 2).sum(axis=1))
    with np.errstate(invalid='ignore', divide='ignore'):
        return (Yc @ xc) / den


def profile_table(roi_slopes, loo, cells, n_perm=PROFILE_PERM, seed=PROFILE_SEED,
                  min_cells=PROFILE_MIN_CELLS):
    """Per subject: n_cells, r_full (Spearman), free-shuffle p, exceeds.

    Coverage: the subject's cell slope must be finite and the LOO group slope
    must exist; a band with fewer than two such cells is dropped (the region-
    level profile's filter), then < `min_cells` cells means no r.

    Each subject gets its OWN seeded stream, (seed, position in the sorted
    cohort), so a subject's p does not move when the floor changes who else is
    scored.
    """
    cell_keys = pd.DataFrame(cells, columns=['parcel', 'band'])
    s = roi_slopes.merge(cell_keys, on=['parcel', 'band'])
    g = loo[loo['excluded'] != NONE].rename(
        columns={'ROI': 'parcel', 'excluded': 'subject', 'roi_slope': 'group'})
    s = s.merge(g[['subject', 'parcel', 'band', 'group']],
                on=['subject', 'parcel', 'band'], how='left')
    s = s[np.isfinite(s['slope']) & np.isfinite(s['group'])]
    per_band = s.groupby(['subject', 'band'])['parcel'].transform('size')
    s = s[per_band >= 2]

    rows = []
    for i, subj in enumerate(sorted(roi_slopes['subject'].unique())):
        rng = np.random.default_rng([seed, i])
        d = s[s['subject'] == subj]
        n = len(d)
        rec = {'subject': subj, 'n_cells': n, 'n_bands': d['band'].nunique(),
               'r_full': np.nan, 'p': np.nan, 'exceeds': False,
               'r_within_band_p': np.nan}
        if n >= min_cells:
            x = _ranks(d['slope'].to_numpy(float))
            y = _ranks(d['group'].to_numpy(float))
            r = float(np.corrcoef(x, y)[0, 1])
            if not np.isfinite(r):   # all-tied ranks: no correlation defined
                rows.append(rec)
                continue
            # FREE shuffle of the subject's values across all their cells.
            X = np.tile(x, (n_perm, 1))
            rng.permuted(X, axis=1, out=X)
            null = _rowwise_pearson(y, X)
            p = (1 + int((null >= r).sum())) / (1 + n_perm)
            # Comparison only: the same statistic, shuffled WITHIN band.
            Xb = np.empty_like(X)
            bands = d['band'].to_numpy()
            for b in np.unique(bands):
                m = bands == b
                blk = np.tile(x[m], (n_perm, 1))
                rng.permuted(blk, axis=1, out=blk)
                Xb[:, m] = blk
            nb = _rowwise_pearson(y, Xb)
            rec.update({'r_full': r, 'p': p, 'exceeds': p < ALPHA_ONE_SIDED,
                        'r_within_band_p': (1 + int((nb >= r).sum())) / (1 + n_perm)})
        rows.append(rec)
    return pd.DataFrame(rows), s


# ============================================================================
# FIGURES
# ============================================================================

def _save(fig, out_dir, stem):
    for ext in SAVE_EXT:
        fig.savefig(out_dir / f'{stem}.{ext}', dpi=300, bbox_inches='tight')
    logger.info('wrote %s.{%s}', stem, ','.join(SAVE_EXT))


def _nice_refs(lo, hi):
    """Two or three round reference sizes spanning [lo, hi]."""
    span = hi - lo
    step = next(s for s in (1, 2, 5, 10, 20, 25, 50) if span / s <= 4)
    refs = {int(lo), int(np.round((lo + hi) / 2 / step) * step), int(hi)}
    return sorted(refs)


def fig_profile(prof, out_dir):
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    d = prof[np.isfinite(prof['r_full'])].sort_values('r_full').reset_index(drop=True)
    N = len(d)
    n_pos = int((d['r_full'] > 0).sum())
    n_sig = int(d['exceeds'].sum())
    k_area = 150.0 / max(d['n_cells'].max(), 1)

    fig, ax = plt.subplots(figsize=PROFILE_SIZE)
    ax.axvline(0, color='0.35', lw=1, ls='--', zorder=1)
    ax.scatter(d['r_full'], np.arange(N), s=d['n_cells'] * k_area,
               c=np.where(d['exceeds'], BLUE, GREY), edgecolors='white',
               linewidths=0.6, zorder=3)
    lo = min(-0.35, d['r_full'].min() - 0.05)
    hi = max(0.9, d['r_full'].max() + 0.05)
    ax.set_xlim(lo, hi)
    ax.set_ylim(-1, N)
    ax.set_yticks([])
    ax.set_ylabel('Subjects (sorted by r)', fontsize=10)
    ax.set_xlabel('Spearman r, subject slope map vs. leave-one-out group map',
                  fontsize=10)
    ax.set_title(f'{n_pos}/{N} subjects correlate positively; {n_sig} exceed '
                 f'their own null (≈{N * ALPHA_ONE_SIDED:.1f} expected by '
                 'chance)', fontsize=9.5)

    colour_leg = ax.legend(
        handles=[Line2D([], [], ls='', marker='o', ms=8, mfc=BLUE, mec='white',
                        label='Exceeds own null (p < 0.025)'),
                 Line2D([], [], ls='', marker='o', ms=8, mfc=GREY, mec='white',
                        label='Does not exceed')],
        loc='lower right', fontsize=8, frameon=True)
    ax.add_artist(colour_leg)
    refs = _nice_refs(d['n_cells'].min(), d['n_cells'].max())
    ax.legend(handles=[plt.scatter([], [], s=n * k_area, c='0.6',
                                   edgecolors='white', label=f'{n} cells')
                       for n in refs],
              loc='upper left', fontsize=8, frameon=True, labelspacing=1.0,
              title='band × ROI cells', title_fontsize=8)

    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.text(0.01, 0.01,
             "Each subject is tested against their own permutation null (5,000 "
             "free shuffles of their band × ROI cells), so significance "
             "depends on the subject's coverage as well as r.",
             fontsize=7.5, color='0.3', ha='left', va='bottom', wrap=True)
    _save(fig, out_dir, 'fig_consistency_profile_sigcells')
    plt.close(fig)
    return {'N': N, 'n_pos': n_pos, 'n_sig': n_sig}


def fig_signmap(cells, bands, bands_def, out_dir):
    import matplotlib.pyplot as plt

    def grid(col):
        return (cells.pivot(index='domain', columns='band', values=col)
                .reindex(index=list(DOMAINS), columns=bands))

    ex = grid('excess_sign').to_numpy(float)
    k = grid('n_sign_match').to_numpy(float)
    n = grid('n_with_slope').to_numpy(float)
    cap = max(0.05, np.ceil(np.nanmax(np.abs(ex)) * 20) / 20)

    cm = plt.get_cmap('RdBu_r').copy()
    cm.set_bad('0.85')
    fig, ax = plt.subplots(figsize=(1.35 * len(bands) + 3.0,
                                    0.85 * len(DOMAINS) + 2.4))
    im = ax.imshow(np.ma.masked_invalid(ex), aspect='auto', cmap=cm,
                   vmin=-cap, vmax=cap, interpolation='nearest')
    for i in range(len(DOMAINS)):
        for j in range(len(bands)):
            if not np.isfinite(ex[i, j]):
                continue
            shade = 'white' if abs(ex[i, j]) > 0.62 * cap else '0.1'
            ax.text(j, i, f'{int(k[i, j])}/{int(n[i, j])}\n{k[i, j] / n[i, j]:.0%}',
                    ha='center', va='center', fontsize=9, color=shade)
    ax.set_xticks(range(len(bands)))
    ax.set_xticklabels([f'{BAND_LABEL.get(b, b)}\n{bands_def[b][0]:g}-'
                        f'{bands_def[b][1]:g} Hz' for b in bands], fontsize=9)
    ax.set_yticks(range(len(DOMAINS)))
    ax.set_yticklabels(DOMAINS, fontsize=10)
    ax.set_xticks(np.arange(-0.5, len(bands), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(DOMAINS), 1), minor=True)
    ax.grid(which='minor', color='white', linewidth=1.2)
    ax.tick_params(which='minor', length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title("Do individual subjects share the group's sign?", fontsize=12)
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.03)
    cb.set_label('observed − null fraction sharing group sign', fontsize=9)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.text(0.01, 0.01,
             "Text: subjects whose own unpooled OLS slope has the sign of the lme4 "
             "group slope, out of subjects with a fittable slope. Colour: that "
             "fraction minus the cell's sign-flip null (the group sign is estimated "
             "from the same subjects, so chance agreement is above 50%). "
             + DISCLAIMER, fontsize=6.5, color='0.35', ha='left', va='bottom',
             wrap=True)
    _save(fig, out_dir, 'fig_consistency_signmap_domains')
    plt.close(fig)


def fig_sigcells(dom_slopes, group, sig, bands_def, out_dir):
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(0)
    fig, ax = plt.subplots(figsize=(6.8, 0.75 * len(sig) + 1.8))
    labels = []
    for i, (dom, band) in enumerate(sig):
        y = len(sig) - 1 - i
        s = dom_slopes[(dom_slopes['domain'] == dom)
                       & (dom_slopes['band'] == band)]['slope'].dropna()
        g = float(group.loc[(dom, band), 'NRS_within.trend'])
        colour = BLUE if g < 0 else RED
        ax.boxplot(s, positions=[y], vert=False, widths=0.5, showfliers=False,
                   whis=(5, 95), medianprops={'color': 'black', 'lw': 1.4},
                   boxprops={'lw': 1}, whiskerprops={'lw': 1}, capprops={'lw': 1})
        ax.scatter(s, y + rng.uniform(-0.18, 0.18, len(s)), s=16, c=colour,
                   alpha=0.7, edgecolors='white', linewidths=0.4, zorder=3)
        ax.plot(g, y, marker='D', ms=6, color='black', zorder=4)
        k = int((np.sign(s) == np.sign(g)).sum())
        ax.text(0.01, y + 0.36, f'{k} of {len(s)} share the group sign '
                f'({"negative" if g < 0 else "positive"})',
                transform=ax.get_yaxis_transform(), fontsize=7.5, color='0.3')
        labels.append(f'{dom}\n{BAND_LABEL.get(band, band)}')
    ax.axvline(0, color='0.35', lw=1, ls='--', zorder=1)
    ax.set_yticks(range(len(sig)))
    ax.set_yticklabels(labels[::-1], fontsize=9)
    ax.set_ylim(-0.6, len(sig) - 0.3)
    ax.set_xlabel('subject slope: d log10 power per pain point', fontsize=10)
    ax.set_title('Subject slopes in the significant domain × band cells',
                 fontsize=11)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.text(0.01, 0.01,
             'One point per subject: unpooled OLS slope of the domain\'s per-epoch '
             'channel mean on NRS_within. Box = IQR, whiskers 5th-95th pct; '
             '◆ = lme4 group slope. Descriptive only. ' + DISCLAIMER,
             fontsize=6.5, color='0.35', ha='left', va='bottom', wrap=True)
    _save(fig, out_dir, 'fig_consistency_sigcells')
    plt.close(fig)


def fig_sigcells_v2(dom_slopes, group, sig, bands_def, out_dir):
    """The same five cells as `fig_sigcells`, redrawn in the house style.

    Written beside v1 rather than over it -- a redesign, not a correction, so
    the original stays for comparison. Three changes:

    * VERTICAL VIOLINS, light grey, one per cell. The box's five numbers become
      the whole distribution, which is what a question about consistency is
      actually asking to see.
    * NO GROUP-SLOPE DIAMOND. The lme4 estimate is a different statistic from
      the unpooled subject slopes it sat on top of (partially pooled, channel
      random intercepts), and drawn at the same scale it read as their summary.
      It stays in the sign-agreement count, which is where it belongs.
    * EACH DOT IS COLOURED BY ITS OWN SLOPE, on a diverging map built from the
      style guide's blue/orange pair and centred on zero. Sign used to be a
      property of the CELL (every dot in a delta cell was blue); now it is a
      property of the subject, so the subjects who disagree with their cell
      are visible at a glance instead of by reading y positions. The scale is
      symmetric and clipped at the 98th percentile of |slope| so two extreme
      subjects cannot wash the rest to near-white; clipped dots saturate and
      the colourbar says so.

    Per-subject slopes are unchanged from v1: unpooled OLS, never BLUPs.
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
    from ieeg_ehr.med_analysis import style      # read as style.X: see use_poster

    rng = np.random.default_rng(0)
    data = []
    for dom, band in sig:
        s = (dom_slopes[(dom_slopes['domain'] == dom)
                        & (dom_slopes['band'] == band)]['slope']
             .dropna().to_numpy(dtype=float))
        g = float(group.loc[(dom, band), 'NRS_within.trend'])
        data.append((dom, band, s, g))

    cap = float(np.percentile(np.abs(np.concatenate([d[2] for d in data])), 98))
    cmap = LinearSegmentedColormap.from_list(
        'house_diverging', [style.BAR_COLOR, '#f4f3ef', style.NORM_COLOR])
    norm = TwoSlopeNorm(vmin=-cap, vcenter=0.0, vmax=cap)

    fig, ax = plt.subplots(figsize=(1.35 * len(sig) + 2.2, 5.2))
    xs = np.arange(len(sig))
    parts = ax.violinplot([d[2] for d in data], positions=xs, widths=0.78,
                          showextrema=False, showmedians=False)
    for body in parts['bodies']:
        body.set_facecolor(style.GRID_COLOR)
        body.set_edgecolor('none')
        body.set_alpha(1.0)
        body.set_zorder(1)

    top = max(float(np.max(d[2])) for d in data)
    bot = min(float(np.min(d[2])) for d in data)
    pad = 0.06 * (top - bot)
    labels = []
    for x, (dom, band, s, g) in zip(xs, data):
        # Median as a short dark tick: the violin alone has no centre mark, and
        # the diamond that used to supply one was the wrong statistic.
        ax.hlines(np.median(s), x - 0.22, x + 0.22, color=style.TEXT_PRIMARY,
                  lw=1.4, zorder=2)
        # A thin dark edge so near-zero dots, which the diverging map renders
        # almost white, stay visible against the grey violin.
        ax.scatter(x + rng.uniform(-0.14, 0.14, len(s)), s, c=s, cmap=cmap,
                   norm=norm, s=22, edgecolors='0.35', linewidths=0.35,
                   zorder=3)
        k = int((np.sign(s) == np.sign(g)).sum())
        ax.text(x, top + pad, f'{k}/{len(s)}\nshare group\nsign',
                ha='center', va='bottom', fontsize=style.TICK_SIZE - 1,
                color=style.TEXT_MUTED, linespacing=1.1)
        labels.append(f'{dom}\n{BAND_LABEL.get(band, band)}')

    # Above the violins: the dots are coloured by sign, so zero is the one
    # reference a reader needs to see through the grey.
    ax.axhline(0, color=style.ZERO_LINE_COLOR, lw=1, ls='--', zorder=1.5)
    # House style FIRST -- it recolours every tick label, so the cell labels
    # have to be set after it to stay primary.
    style.style_axes(ax, grid_axis='y')
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, fontsize=style.TICK_SIZE + 1,
                       color=style.TEXT_PRIMARY)
    ax.set_xlim(-0.6, len(sig) - 0.4)
    ax.set_ylim(bot - pad, top + 5.2 * pad)
    ax.tick_params(axis='x', length=0)
    style.label_axes(ax, ylabel='subject slope\n$\\Delta$ log10 power per NRS point',
                     title='Subject slopes in the significant domain × band cells')

    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax,
                      fraction=0.035, pad=0.02, extend='both')
    cb.set_label('that subject\'s slope (clipped at 98th pct |slope|)',
                 fontsize=style.TICK_SIZE, color=style.TEXT_PRIMARY)
    cb.ax.tick_params(labelsize=style.TICK_SIZE - 1, colors=style.TEXT_MUTED)
    cb.outline.set_visible(False)

    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.text(0.005, 0.03,
             'One point per subject: unpooled OLS slope of the domain\'s per-epoch '
             'channel mean on NRS_within, coloured by its own value (blue = power '
             'falls with pain, orange = rises). Violin = the distribution of those '
             'slopes; dark tick = median. Counts use the sign of the lme4 group '
             'slope. Descriptive only -- the cells were chosen by p-value, so this '
             'is not a test.',
             fontsize=style.FOOTNOTE_SIZE, color='0.25', ha='left', va='bottom',
             wrap=True)
    style.add_footnote(fig)
    for ext in SAVE_EXT:
        fig.savefig(out_dir / f'fig_consistency_sigcells_v2.{ext}', dpi=style.DPI,
                    facecolor='white', bbox_inches='tight')
    logger.info('wrote fig_consistency_sigcells_v2.{%s}', ','.join(SAVE_EXT))
    plt.close(fig)


# ============================================================================
# REPLOT: presentation versions from the saved tables (2026-09-23)
# ============================================================================

#: Where `--replot` writes, under `<run>/consistency/`. PNG only, no sidecars,
#: no tables -- at the analyst's instruction; the data behind each PNG is the
#: CSV (with its sidecar) one level up.
REPLOT_SUBDIR = 'revised_20260923'
#: The six cells BH-significant in the 24-cell (no-Control) table. v2 drew the
#: five of the 30-cell with-Control family; the analyst asked for all six.
REPLOT_SIG_TABLE = 'table_domain_lmer_heatmap.csv'
BAND_SYMBOL = {'delta': r'$\delta$', 'theta': r'$\theta$', 'alpha': r'$\alpha$',
               'beta': r'$\beta$', 'gamma': r'$\gamma$',
               'high_gamma': r'high-$\gamma$'}
TITLE_SIGCELLS = 'Individual pain slopes in the six significant domain × band cells'
TITLE_SIGNMAP = 'Patient-level sign agreement across domains and bands'
VIOLIN_GREY = '#E7E7E7'
#: Canvas (in) for both replot figures. Saved WITHOUT bbox_inches='tight',
#: which would crop or grow the canvas; constrained layout fits inside instead.
FIG_SIZE = (6.5, 5.0)
#: The two panels side by side in one figure.
COMBINED_SIZE = (13.0, 5.0)
#: Type sizes of the replot figures (pt). REPLOT_FS is what they have always
#: used; POSTER_FS matches the poster cuts (`plot_domain_lmer_med.POSTER`,
#: `plot_dx_cells.draw_poster`): headings 12 bold as their titles, row names
#: 11, axis labels 10, ticks 8.5-9.5.
REPLOT_FS = dict(title=10, title_weight='normal', xtick=8, ytick=8.5,
                 vtick=7.5, cell=8, ylabel=8.5, cb_label=7.5, cb_tick=7)
POSTER_FS = dict(title=12, title_weight='bold', xtick=9.5, ytick=11,
                 vtick=8.5, cell=9, ylabel=10, cb_label=9, cb_tick=8.5,
                 # heatmap band + Hz labels one step down, or "15-25 Hz" and
                 # "25-70 Hz" touch; violin labels on one line, angled, or
                 # "Cognitive" / "Modulatory" overrun their neighbours.
                 mtick=8.5, vrot=35, hz_axis=True,
                 # both headings centred; high gamma as "h gamma", as on the other posters
                 title_loc='center',
                 bands={**BAND_SYMBOL, 'high_gamma': r'h$\gamma$'})
#: The two headings wrapped: at 12 pt bold neither fits its panel on one line.
POSTER_TITLE_SIGNMAP = 'Patient-level sign agreement\nacross domains and bands'
POSTER_TITLE_SIGCELLS = ('Individual pain slopes in the six\n'
                         'significant domain × band cells')
#: `--poster`: the combined figure at 10.5 x 5 in, into <run>/poster/<ts>/.
POSTER_COMBINED_SIZE = (10.5, 5.0)
#: Gap between the two panels (constrained-layout wspace, figure fraction);
#: the default is 0.02.
POSTER_WSPACE = 0.05


def draw_sigcells_v3(fig, ax, dom_slopes, sig, cap, title=TITLE_SIGCELLS,
                     fs=REPLOT_FS):
    """`fig_sigcells_v2` restyled: band symbols, no sign counts, no captions,
    no grid except the zero line, and a colourbar in the sign map's format."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm
    from ieeg_ehr.med_analysis import style

    rng = np.random.default_rng(0)
    data = [dom_slopes[(dom_slopes['domain'] == d) & (dom_slopes['band'] == b)]
            ['slope'].dropna().to_numpy(dtype=float) for d, b in sig]
    # `cap` is the domain heatmap's (plot_domain_lmer: max |group slope|), so a
    # dot and a heatmap cell of the same colour have the same slope. Subject
    # slopes spread wider than group slopes, so many dots saturate.
    cmap = plt.get_cmap('RdBu_r')
    norm = TwoSlopeNorm(vmin=-cap, vcenter=0.0, vmax=cap)

    xs = np.arange(len(sig))
    parts = ax.violinplot(data, positions=xs, widths=0.78, showextrema=False,
                          showmedians=False)
    for body in parts['bodies']:
        body.set_facecolor(VIOLIN_GREY)
        body.set_edgecolor('none')
        body.set_alpha(1.0)
        body.set_zorder(1)
    for x, s in zip(xs, data):
        ax.hlines(np.median(s), x - 0.22, x + 0.22, color=style.TEXT_PRIMARY,
                  lw=1.2, zorder=2)
        ax.scatter(x + rng.uniform(-0.14, 0.14, len(s)), s, c=s, cmap=cmap,
                   norm=norm, s=12, edgecolors='0.35', linewidths=0.3,
                   zorder=3)
    ax.axhline(0, color='black', lw=0.8, ls='--', zorder=1.5)
    style.style_axes(ax, grid_axis=None)
    ax.tick_params(axis='y', colors='black', labelsize=fs['vtick'])
    ax.set_xticks(xs)
    rot = fs.get('vrot')
    ax.set_xticklabels([f'{d}{" " if rot else "\n"}{fs.get("bands", BAND_SYMBOL)[b]}'
                        for d, b in sig],
                       fontsize=fs['xtick'], color=style.TEXT_PRIMARY,
                       rotation=rot or 0, ha='right' if rot else 'center',
                       rotation_mode='anchor')
    ax.set_xlim(-0.6, len(sig) - 0.4)
    top = max(float(s.max()) for s in data)
    bot = min(float(s.min()) for s in data)
    pad = 0.06 * (top - bot)
    ax.set_ylim(bot - pad, top + pad)
    ax.tick_params(axis='x', length=0)
    ax.set_ylabel('subject slope\n$\\Delta$ log10 power per NRS point',
                  fontsize=fs['ylabel'], color='black')
    ax.set_title(title, fontsize=fs['title'], fontweight=fs['title_weight'],
                 color=style.TEXT_PRIMARY, loc=fs.get('title_loc', 'left'))

    # The sign map's colourbar: default outline, black ticks, no extend arrows.
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax,
                      fraction=0.04, pad=0.02)
    cb.set_label(f'subject slope (saturates at ±{cap:.3f})',
                 fontsize=fs['cb_label'])
    cb.ax.tick_params(labelsize=fs['cb_tick'])


def draw_signmap_v2(fig, ax, cells, bands, bands_def, title=TITLE_SIGNMAP,
                    fs=REPLOT_FS):
    """`fig_signmap` with band symbols and no caption."""
    import matplotlib.pyplot as plt

    def grid(col):
        return (cells.pivot(index='domain', columns='band', values=col)
                .reindex(index=list(DOMAINS), columns=bands))

    ex = grid('excess_sign').to_numpy(float)
    k = grid('n_sign_match').to_numpy(float)
    n = grid('n_with_slope').to_numpy(float)
    cap = max(0.05, np.ceil(np.nanmax(np.abs(ex)) * 20) / 20)
    cm = plt.get_cmap('RdBu_r').copy()
    cm.set_bad('0.85')
    im = ax.imshow(np.ma.masked_invalid(ex), aspect='auto', cmap=cm,
                   vmin=-cap, vmax=cap, interpolation='nearest')
    for i in range(len(DOMAINS)):
        for j in range(len(bands)):
            if not np.isfinite(ex[i, j]):
                continue
            shade = 'white' if abs(ex[i, j]) > 0.62 * cap else '0.1'
            ax.text(j, i, f'{int(k[i, j])}/{int(n[i, j])}\n{k[i, j] / n[i, j]:.0%}',
                    ha='center', va='center', fontsize=fs['cell'],
                    color=shade)
    ax.set_xticks(range(len(bands)))
    # `hz_axis`: the unit once, as an axis label, instead of on every tick.
    hz_axis = fs.get('hz_axis', False)
    ax.set_xticklabels([f'{fs.get("bands", BAND_SYMBOL)[b]}\n{bands_def[b][0]:g}–'
                        f'{bands_def[b][1]:g}' + ('' if hz_axis else ' Hz')
                        for b in bands],
                       fontsize=fs.get('mtick', fs['xtick']))
    if hz_axis:
        ax.set_xlabel('Frequency band (Hz)', fontsize=fs['ylabel'])
    ax.set_yticks(range(len(DOMAINS)))
    ax.set_yticklabels(DOMAINS, fontsize=fs['ytick'])
    ax.set_xticks(np.arange(-0.5, len(bands), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(DOMAINS), 1), minor=True)
    ax.grid(which='minor', color='white', linewidth=1.2)
    ax.tick_params(which='minor', length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title(title, fontsize=fs['title'], fontweight=fs['title_weight'])
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cb.set_label('observed − null fraction sharing group sign',
                 fontsize=fs['cb_label'])
    cb.ax.tick_params(labelsize=fs['cb_tick'])


#: `--roi-comparison`: the (domain, band) cells to break down, and where.
ROI_CMP_SUBDIR = 'domain_roi_comparison'
ROI_CMP_CELLS = (('Sensory', 'beta'), ('Cognitive', 'delta'),
                 ('Affective', 'beta'), ('Cognitive', 'alpha'))
#: The ROI grouping of `fig_band_map_domains` (plot_band_map_domains).
ROI_CMP_SCHEME = 'pain_domains_v4'


def _violins(ax, data, cmap, norm, rng):
    """Grey violins + median tick + slope-coloured dots, at x = 0..len-1."""
    from ieeg_ehr.med_analysis import style
    xs = np.arange(len(data))
    # A violin needs a density; below 3 subjects draw the dots alone.
    keep = [i for i, s in enumerate(data) if len(s) >= 3]
    if keep:
        parts = ax.violinplot([data[i] for i in keep], positions=xs[keep],
                              widths=0.78, showextrema=False, showmedians=False)
        for body in parts['bodies']:
            body.set_facecolor(VIOLIN_GREY)
            body.set_edgecolor('none')
            body.set_alpha(1.0)
            body.set_zorder(1)
    for x, s in zip(xs, data):
        if len(s) == 0:
            continue
        ax.hlines(np.median(s), x - 0.22, x + 0.22, color=style.TEXT_PRIMARY,
                  lw=1.2, zorder=2)
        ax.scatter(x + rng.uniform(-0.14, 0.14, len(s)), s, c=s, cmap=cmap,
                   norm=norm, s=12, edgecolors='0.35', linewidths=0.3, zorder=3)
    ax.axhline(0, color='black', lw=0.8, ls='--', zorder=1.5)
    style.style_axes(ax, grid_axis=None)
    ax.tick_params(axis='y', colors='black', labelsize=7.5)
    ax.tick_params(axis='x', length=0)
    ax.set_xticks(xs)
    ax.set_xlim(-0.6, len(data) - 0.4)


def draw_domain_roi(fig, dom_s, roi_s, domain, band, rois, cap):
    """Left: the domain's subject slopes. Right: the same band, one violin per
    ROI of that domain. Shared y and colour scale."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm

    rng = np.random.default_rng(0)
    cmap = plt.get_cmap('RdBu_r')
    norm = TwoSlopeNorm(vmin=-cap, vcenter=0.0, vmax=cap)
    d = [dom_s[(dom_s['domain'] == domain) & (dom_s['band'] == band)]['slope']
         .dropna().to_numpy(dtype=float)]
    r = [roi_s[(roi_s['region'] == roi) & (roi_s['band'] == band)]['slope']
         .dropna().to_numpy(dtype=float) for roi in rois]

    ax_d, ax_r = fig.subplots(1, 2, sharey=True,
                              gridspec_kw={'width_ratios': [1.3, len(rois)]})
    _violins(ax_d, d, cmap, norm, rng)
    _violins(ax_r, r, cmap, norm, rng)
    sym = BAND_SYMBOL[band]
    ax_d.set_xticklabels([f'{domain}\n{sym}\nn = {len(d[0])}'], fontsize=8,
                         color='black')
    ax_r.set_xticklabels([f'{roi}\n{sym}\nn = {len(s)}' for roi, s in zip(rois, r)],
                         fontsize=8, color='black')
    ax_r.tick_params(axis='y', labelleft=False)
    ax_d.set_ylabel('subject slope\n$\\Delta$ log10 power per NRS point',
                    fontsize=8.5, color='black')
    ax_d.set_title('Domain', fontsize=9.5, loc='left')
    ax_r.set_title(f'{domain} ROIs', fontsize=9.5, loc='left')
    fig.suptitle(f'{domain} {sym}: domain vs. ROI subject slopes', fontsize=10.5)
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap),
                      ax=[ax_d, ax_r], fraction=0.03, pad=0.015)
    cb.set_label(f'subject slope (saturates at ±{cap:.3f})', fontsize=7.5)
    cb.ax.tick_params(labelsize=7)


def roi_comparison(run_dir, roi_run, cells=ROI_CMP_CELLS):
    """One PNG per (domain, band): domain violin beside its ROIs' violins."""
    import matplotlib.pyplot as plt
    from ieeg_ehr.analysis.plot_band_map_domains import domain_rows
    from ieeg_ehr.config import roi_schemes

    out = run_dir / OUT_SUBDIR / ROI_CMP_SUBDIR
    out.mkdir(exist_ok=True)
    dom = io.read_table(run_dir / OUT_SUBDIR / 'domain_subject_slopes.csv',
                        on_stale='warn')
    roi = io.read_table(Path(roi_run) / 'subject_slopes.parquet', on_stale='warn')
    # The band map's labels: pure renames (dmPFC/SMA -> dmPFC), then its grouping.
    roi = roi.assign(region=roi['region'].replace(
        {k: v for k, v in roi_schemes.ROI_RENAMES.items()
         if k in set(roi['region'])}))
    groups = dict(domain_rows(list(dict.fromkeys(roi['region'])), ROI_CMP_SCHEME))
    # The domain heatmap's colour cap, as in the replot figures.
    cap = float(pd.read_csv(run_dir / REPLOT_SIG_TABLE)[SLOPE_COL].abs().max())
    for domain, band in cells:
        rois = groups[domain]
        logger.info('%s %s: ROIs %s', domain, band, rois)
        fig = plt.figure(figsize=COMBINED_SIZE, layout='constrained')
        draw_domain_roi(fig, dom, roi, domain, band, rois, cap)
        _save_png(fig, out / f'fig_domain_roi_{domain.lower()}_{band}.png')


def _save_png(fig, out_path):
    import matplotlib.pyplot as plt
    fig.savefig(out_path, dpi=300, facecolor='white')
    logger.info('wrote %s', out_path)
    plt.close(fig)


def replot(run_dir, bands_def, title_sigcells, title_signmap):
    """Redraw both figures from the saved tables; PNG only, into REPLOT_SUBDIR."""
    src = run_dir / OUT_SUBDIR
    out = src / REPLOT_SUBDIR
    out.mkdir(exist_ok=True)
    dom = io.read_table(src / 'domain_subject_slopes.csv', on_stale='warn')
    cells = io.read_table(src / 'signmap_domains_cells.csv', on_stale='warn')
    t_all = pd.read_csv(run_dir / REPLOT_SIG_TABLE)
    t = t_all
    t = t[t['p_bh_reject'].astype(str).str.lower() == 'true']
    sig = sorted([(d, b) for d, b in zip(t['domain'], t['band']) if d in DOMAINS],
                 key=lambda c: (BAND_ORDER.index(c[1]), DOMAINS.index(c[0])))
    logger.info('significant cells (%s): %s', REPLOT_SIG_TABLE, sig)
    bands = [b for b in BAND_ORDER if b in set(cells['band'])]
    # The heatmap's colour cap, from the same 24-cell table.
    cap = float(t_all[SLOPE_COL].abs().max())
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=FIG_SIZE, layout='constrained')
    draw_sigcells_v3(fig, ax, dom, sig, cap, title_sigcells)
    _save_png(fig, out / 'fig_consistency_sigcells_v3.png')

    fig, ax = plt.subplots(figsize=FIG_SIZE, layout='constrained')
    draw_signmap_v2(fig, ax, cells, bands, bands_def, title_signmap)
    _save_png(fig, out / 'fig_consistency_signmap_domains_v2.png')

    # Both panels in one figure: sign map left, violins right.
    fig, (ax_map, ax_vio) = plt.subplots(1, 2, figsize=COMBINED_SIZE,
                                         layout='constrained')
    draw_signmap_v2(fig, ax_map, cells, bands, bands_def, title_signmap)
    draw_sigcells_v3(fig, ax_vio, dom, sig, cap, title_sigcells)
    _save_png(fig, out / 'fig_consistency_signmap_sigcells_combined.png')


def poster_combined(run_dir, bands_def, title_sigcells, title_signmap):
    """The combined replot as a POSTER cut: 10.5 x 5 in, poster type sizes,
    bold headings, a wider gap between the panels. Same tables as `replot`;
    written to `<run>/poster/consistency_combined_<ts>/` with provenance."""
    import json
    from datetime import datetime
    import matplotlib.pyplot as plt

    src = run_dir / OUT_SUBDIR
    dom = io.read_table(src / 'domain_subject_slopes.csv', on_stale='warn')
    cells = io.read_table(src / 'signmap_domains_cells.csv', on_stale='warn')
    t_all = pd.read_csv(run_dir / REPLOT_SIG_TABLE)
    t = t_all[t_all['p_bh_reject'].astype(str).str.lower() == 'true']
    sig = sorted([(d, b) for d, b in zip(t['domain'], t['band']) if d in DOMAINS],
                 key=lambda c: (BAND_ORDER.index(c[1]), DOMAINS.index(c[0])))
    bands = [b for b in BAND_ORDER if b in set(cells['band'])]
    cap = float(t_all[SLOPE_COL].abs().max())

    stamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    out = run_dir / 'poster' / f'consistency_combined_{stamp}'
    out.mkdir(parents=True, exist_ok=True)
    fig, (ax_map, ax_vio) = plt.subplots(1, 2, figsize=POSTER_COMBINED_SIZE,
                                         layout='constrained')
    fig.get_layout_engine().set(wspace=POSTER_WSPACE)
    # The default headings are swapped for their wrapped forms; a heading
    # passed on the command line is used as given.
    if title_signmap == TITLE_SIGNMAP:
        title_signmap = POSTER_TITLE_SIGNMAP
    if title_sigcells == TITLE_SIGCELLS:
        title_sigcells = POSTER_TITLE_SIGCELLS
    draw_signmap_v2(fig, ax_map, cells, bands, bands_def, title_signmap,
                    fs=POSTER_FS)
    draw_sigcells_v3(fig, ax_vio, dom, sig, cap, title_sigcells, fs=POSTER_FS)
    _save_png(fig, out / 'fig_consistency_signmap_sigcells_combined.png')
    parent = json.loads((run_dir / 'provenance.json').read_text())
    io.write_run_provenance(
        out, script='ieeg_ehr/analysis/plot_domain_lmer_consistency.py',
        params={'size_in': POSTER_COMBINED_SIZE, 'fs': POSTER_FS,
                'wspace': POSTER_WSPACE, 'sig_cells': sig,
                'source_run': str(run_dir)},
        parents=[str(src / 'domain_subject_slopes.csv'),
                 str(src / 'signmap_domains_cells.csv'),
                 str(run_dir / REPLOT_SIG_TABLE)],
        subjects=parent.get('subjects'),
        extra={'status': 'EXPLORATORY -- nominations, not findings.'})
    io.log_analysis('domain lmer consistency combined figure -- poster', out)


# ============================================================================

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--run-dir', required=True)
    ap.add_argument('--figures', nargs='*', default=list(FIGURES),
                    choices=FIGURES)
    ap.add_argument('--n-perm-signflip', type=int, default=N_PERM)
    ap.add_argument('--min-cells', type=int, default=PROFILE_MIN_CELLS,
                    help='profile: fewest band x ROI cells a subject needs for an r')
    ap.add_argument('--replot', action='store_true',
                    help=f'redraw sigcells + signmap from the saved CSVs as PNGs '
                         f'in consistency/{REPLOT_SUBDIR}/; nothing is recomputed')
    ap.add_argument('--roi-comparison', metavar='ROI_RUN',
                    help=f'a run_bandpower_mixed ROI run: draw each of '
                         f'{ROI_CMP_CELLS} as domain vs. its ROIs into '
                         f'consistency/{ROI_CMP_SUBDIR}/ (PNG only)')
    ap.add_argument('--poster', action='store_true',
                    help='with --replot: draw only the combined figure as the '
                         '10.5 x 5 in POSTER cut into <run>/poster/<ts>/')
    ap.add_argument('--title-sigcells', default=TITLE_SIGCELLS)
    ap.add_argument('--title-signmap', default=TITLE_SIGNMAP)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)s %(message)s')
    io.warn_if_dirty()
    import matplotlib
    matplotlib.use('Agg')

    run_dir = Path(args.run_dir)
    prov = run_provenance(run_dir)
    frames_dir = Path(prov['params']['frames_dir'])
    subjects = sorted(prov['subjects'])
    bands = [b for b in BAND_ORDER if (frames_dir / f'{b}.parquet').exists()]
    # The lme4 run records no band set of its own; its frames run does.
    bands_def = BAND_SETS[run_provenance(frames_dir.parent)['params']['band_set']]
    if args.roi_comparison:
        roi_comparison(run_dir, args.roi_comparison)
        return
    if args.replot and args.poster:
        poster_combined(run_dir, bands_def, args.title_sigcells,
                        args.title_signmap)
        return
    if args.replot:
        replot(run_dir, bands_def, args.title_sigcells, args.title_signmap)
        return
    out_dir = run_dir / OUT_SUBDIR
    out_dir.mkdir(parents=True, exist_ok=True)

    sig = sorted([(d, b) for d, b in significant_cells(run_dir) if d in DOMAINS],
                 key=lambda c: (BAND_ORDER.index(c[1]), DOMAINS.index(c[0])))
    logger.info('significant cells (%s): %s', SIG_TABLE, sig)
    group = pd.read_csv(run_dir / SIG_TABLE).set_index(['domain', 'band'])

    need_roi = 'profile' in args.figures
    need_dom = bool({'signmap', 'sigcells', 'sigcells_v2'} & set(args.figures))
    dom_rows, roi_rows = [], []
    for band in bands:
        if not need_dom and band not in {b for _, b in sig}:
            continue
        df = load_frame(frames_dir, band)
        if need_dom:
            dom_rows.append(unit_slopes(df, 'domain', band))
        if need_roi and band in {b for _, b in sig}:
            roi_rows.append(unit_slopes(df, 'parcel', band))

    common = dict(parents=[str(run_dir / 'provenance.json'), str(frames_dir)],
                  script=SCRIPT, subjects=subjects, extra={'status': DISCLAIMER})
    slope_note = ('unpooled OLS per (subject, unit, band) on the unit\'s per-epoch '
                  'channel mean vs the model\'s NRS_within; rows = the lme4 frames')

    if need_dom:
        dom = pd.concat(dom_rows, ignore_index=True)
        io.write_table(dom, out_dir / 'domain_subject_slopes.csv',
                       params={'slope': slope_note, 'unit': 'domain'}, **common)
        if 'signmap' in args.figures:
            g = (group.reset_index()
                 .rename(columns={'NRS_within.trend': 'beta'})
                 [['domain', 'band', 'beta', 'p_bh_reject']])
            cells = cell_stats(dom, g, list(DOMAINS), bands,
                               n_perm=args.n_perm_signflip, seed=0)
            io.write_table(cells, out_dir / 'signmap_domains_cells.csv',
                           params={'slope': slope_note,
                                   'n_perm': args.n_perm_signflip,
                                   'group_sign': f'lme4 emtrends slope, {SIG_TABLE}'},
                           **common)
            fig_signmap(cells, bands, bands_def, out_dir)
        if 'sigcells' in args.figures:
            fig_sigcells(dom, group, sig, bands_def, out_dir)
        if 'sigcells_v2' in args.figures:
            fig_sigcells_v2(dom, group, sig, bands_def, out_dir)

    if need_roi:
        loo_path = out_dir / COLLECTED
        if not loo_path.exists():
            raise SystemExit(f'{loo_path} missing: run the LOO array and '
                             '`run_domain_lmer_loo --collect` first.')
        loo = io.read_table(loo_path, on_stale='warn')
        roi = pd.concat(roi_rows, ignore_index=True)
        members = loo.drop_duplicates('ROI').set_index('ROI')['domain']
        cells = [(r, b) for d, b in sig for r in members[members == d].index]
        logger.info('profile map: %d (ROI, band) cells', len(cells))
        prof, used = profile_table(roi, loo, cells, min_cells=args.min_cells)
        n_all = len(prof)
        n_excl = int((~np.isfinite(prof['r_full'])).sum())
        io.write_table(used, out_dir / 'profile_sigcells_cell_slopes.csv',
                       params={'slope': slope_note, 'cells': cells}, **common)
        io.write_table(prof, out_dir / 'profile_sigcells_subjects.csv',
                       params={'statistic': 'Spearman r, subject ROI slopes vs '
                                            'LOO lme4 ROI slopes (fixef + ranef)',
                               'null': 'free shuffle of subject values across all '
                                       'their cells',
                               'n_perm': PROFILE_PERM, 'seed': PROFILE_SEED,
                               'p': '(1 + #{r_null >= r}) / (1 + n_perm)',
                               'exceeds': f'p < {ALPHA_ONE_SIDED}',
                               'min_cells': args.min_cells,
                               'min_cells_per_band': 2, 'cells': cells,
                               'r_within_band_p': 'comparison only, not plotted'},
                       **common)
        counts = fig_profile(prof, out_dir)
        pd.set_option('display.width', 200)
        print(prof.sort_values('r_full', ascending=False).to_string(
            index=False, float_format=lambda v: f'{v:.4f}'))
        print(f"\nN={counts['N']}  n_pos={counts['n_pos']}  n_sig={counts['n_sig']}  "
              f"n_excluded={n_excl} (of {n_all}; < {args.min_cells} cells)  "
              f"within-band p<0.025: {int((prof['r_within_band_p'] < 0.025).sum())}")

    io.log_analysis('lme4 domain-run consistency: domain sign map, significant-'
                    'cell subject slopes, LOO-refit Spearman profile (EXPLORATORY)',
                    run_dir)


if __name__ == '__main__':
    main()
