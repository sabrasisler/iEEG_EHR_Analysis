#!/usr/bin/env python3
"""Across-subject consistency for an lme4 domain run: three figures.

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
FIGURES = ('signmap', 'sigcells', 'profile')

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
    out_dir = run_dir / OUT_SUBDIR
    out_dir.mkdir(parents=True, exist_ok=True)

    sig = sorted([(d, b) for d, b in significant_cells(run_dir) if d in DOMAINS],
                 key=lambda c: (BAND_ORDER.index(c[1]), DOMAINS.index(c[0])))
    logger.info('significant cells (%s): %s', SIG_TABLE, sig)
    group = pd.read_csv(run_dir / SIG_TABLE).set_index(['domain', 'band'])

    need_roi = 'profile' in args.figures
    need_dom = bool({'signmap', 'sigcells'} & set(args.figures))
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
