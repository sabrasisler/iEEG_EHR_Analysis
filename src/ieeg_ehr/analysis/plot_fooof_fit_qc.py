"""TIER 0: does the FOOOF fit hold up? Three diagnostics, before any physiology.

    python -m ieeg_ehr.analysis.plot_fooof_fit_qc

Runs BEFORE anything interprets an exponent, because `r_squared` is deliberately
stored and never thresholded (metric/threshold split, `config/psd_params.py`) --
so the decision of whether a cutoff is needed at all has to be made by looking.

1. FIT QUALITY (`01_fit_quality.png`). ECDFs of r2 and error per arm, plus a
   hexbin of r2 against the exponent for each arm.
   ASKS: is there a fit-quality region where the exponent goes unphysical? The
   shape of the answer matters more than the number: DIFFUSE noise at low r2 is
   tolerable, but a COHERENT low-r2 population sitting at a distinct exponent
   would drag every ROI mean it lands in, and no amount of averaging removes it.

2. WHERE IT FAILS (`02_per_session.png`). Per-subject-session r2, boxes ordered
   by median, over `mask_excluded_frac` and `n_windows_used` on a SHARED X AXIS.
   ASKS: is failure tracking DATA LOSS rather than physiology? A session with a
   high excluded fraction and few surviving windows produces a noisy exponent
   that is indistinguishable, downstream, from real between-subject variance.
   Two stacked panels rather than two y-scales on one: a dual axis would invite
   reading a correlation off the crossing of two arbitrarily-scaled lines.

3. WHAT A FIT LOOKS LIKE (`03_exemplar_gallery.png`). 12 channel-epochs sampled
   across r2 deciles -- observed spectrum, aperiodic component, full model --
   with the two arms side by side.
   ASKS: what does a MEDIAN fit actually look like here? Summary statistics hide
   systematic error. A fit range that clips the low-frequency bend, or a knee
   pinned at the range edge, shows up immediately in the residual shape and not
   at all in an r2 of 0.99.

The gallery re-fits those 12 spectra rather than storing model curves, which is
why this script needs the fit parameters: they come from each arm's view sidecar,
so the reconstruction uses what actually ran rather than the current defaults.
"""

import argparse
import glob
import json
import logging
import re
import sys

import numpy as np
import pandas as pd

from ieeg_ehr import config, io

logger = logging.getLogger(__name__)

SCRIPT = 'ieeg_ehr/analysis/plot_fooof_fit_qc.py'
QUESTION = 'psd_physiology'
OUTPUT_TYPE = 'fooof_fit_qc'

# --- palette: validated with the dataviz skill's validate_palette.js ----------
# Two categorical slots (the two arms), light mode, adjacent pairlist:
#   lightness band PASS · chroma floor PASS · CVD sep PASS (worst dE 24.7 protan)
#   normal-vision floor PASS (dE 33.6) · contrast vs surface PASS
# The observed spectrum is drawn in INK, not a third categorical slot -- it is the
# data the two model curves are fit to, not a competing identity. It also carries
# a different mark type (thin line + markers vs 2px solid), so identity never
# rests on colour alone.
ARM_COLOR = {'fixed': '#2a78d6', 'knee': '#eb6834'}
INK_PRIMARY = '#0b0b0b'
INK_SECONDARY = '#52514e'
INK_MUTED = '#8a8a86'
SURFACE = '#fcfcfb'
# Sequential ramp for the hexbin (magnitude = count): ONE hue, light->dark.
SEQ_BLUE = ['#eaf1fb', '#c5d9f4', '#8fb8e8', '#5a96dc', '#2a78d6', '#1b5399', '#0f325c']

ARMS = ('fixed', 'knee')


def _find_view_dirs(epoch_minutes=None):
    """{arm: (dir, params)} for each fooof arm present, newest dir per arm.

    Reads the params from a table sidecar rather than assuming the current
    defaults: the gallery reconstructs fits, and reconstructing with different
    parameters than produced the numbers would be a plot of something else.
    """
    base = config.fullres_epoch_unit_dir(epoch_minutes) / 'views'
    out = {}
    for arm in ARMS:
        cands = sorted(glob.glob(str(base / f'fooof-{arm}-*')))
        best = None
        for d in cands:
            n = len(glob.glob(f'{d}/aperiodic_*.parquet'))
            if best is None or n > best[0]:
                sc = sorted(glob.glob(f'{d}/aperiodic_*.parquet.provenance.json'))
                params = json.load(open(sc[0]))['params'] if sc else {}
                best = (n, d, params)
        if best and best[0]:
            out[arm] = (best[1], best[2])
            logger.info('arm %s: %d subject-sessions in %s',
                        arm, best[0], best[1].split('/')[-1])
        else:
            logger.warning('arm %s: no output found', arm)
    return out


def load_aperiodic(view_dirs):
    """Concatenated aperiodic tables, with `arm` and `unit` columns."""
    frames = []
    for arm, (d, _params) in view_dirs.items():
        for f in sorted(glob.glob(f'{d}/aperiodic_*.parquet')):
            df = pd.read_parquet(f)
            df['arm'] = arm
            m = re.search(r'(sub-\d+_ses-\d+)', f)
            df['unit'] = m.group(1) if m else 'unknown'
            frames.append(df)
    if not frames:
        raise FileNotFoundError('no FOOOF aperiodic tables found')
    df = pd.concat(frames, ignore_index=True)
    logger.info('loaded %d channel-epochs across %d units, arms %s',
                len(df), df.unit.nunique(), sorted(df.arm.unique()))
    return df


# ---------------------------------------------------------------------------
# 1. Fit quality
# ---------------------------------------------------------------------------

def plot_fit_quality(df, out_path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    cmap = LinearSegmentedColormap.from_list('seq_blue', SEQ_BLUE)
    fig, axes = plt.subplots(2, 2, figsize=(11, 8.6), facecolor=SURFACE)
    for ax in axes.ravel():
        ax.set_facecolor(SURFACE)
        for s in ('top', 'right'):
            ax.spines[s].set_visible(False)
        for s in ('left', 'bottom'):
            ax.spines[s].set_color(INK_MUTED)
        ax.tick_params(colors=INK_SECONDARY, labelsize=9)
        ax.grid(True, color=INK_MUTED, alpha=0.18, linewidth=0.6)
        ax.set_axisbelow(True)

    # --- (a) ECDF of r2, (b) ECDF of error -- one line per arm ---------------
    for ax, col, label, lo in ((axes[0, 0], 'r_squared', 'r$^2$', None),
                               (axes[0, 1], 'error', 'fit error', None)):
        for arm in ARMS:
            v = df.loc[df.arm == arm, col].dropna().to_numpy()
            if not len(v):
                continue
            v = np.sort(v)
            y = np.arange(1, len(v) + 1) / len(v)
            ax.plot(v, y, lw=2, color=ARM_COLOR[arm], label=arm, solid_capstyle='round')
            # Direct label at the median rather than a number on every point.
            med = np.median(v)
            ax.annotate(f'{arm}  median {med:.4g}',
                        xy=(med, 0.5), xytext=(6, -14 if arm == 'knee' else 8),
                        textcoords='offset points', fontsize=9,
                        color=INK_SECONDARY)
            ax.plot([med], [0.5], marker='o', ms=8, color=ARM_COLOR[arm],
                    mec=SURFACE, mew=2, zorder=5)
        ax.set_xlabel(label, color=INK_SECONDARY, fontsize=10)
        ax.set_ylabel('cumulative fraction of channel-epochs',
                      color=INK_SECONDARY, fontsize=10)
        ax.set_ylim(0, 1.02)
        leg = ax.legend(frameon=False, fontsize=9, loc='upper left')
        for t in leg.get_texts():
            t.set_color(INK_SECONDARY)
    axes[0, 0].set_title('Fit quality: r$^2$', color=INK_PRIMARY, fontsize=11,
                         loc='left', fontweight='bold')
    axes[0, 1].set_title('Fit quality: absolute error', color=INK_PRIMARY,
                         fontsize=11, loc='left', fontweight='bold')
    # r2 crowds at 1.0; a log axis on (1 - r2) would be clearer but harder to
    # read, so clip the left tail and say so instead of silently cropping.
    lo_r2 = float(df.r_squared.quantile(0.001))
    axes[0, 0].set_xlim(min(0.9, lo_r2), 1.0)
    axes[0, 0].annotate(f'x clipped at {min(0.9, lo_r2):.3f} '
                        f'(0.1st pct = {lo_r2:.4f})',
                        xy=(0.02, 0.02), xycoords='axes fraction',
                        fontsize=8, color=INK_MUTED)

    # --- (c,d) hexbin r2 vs exponent, per arm -------------------------------
    for ax, arm in ((axes[1, 0], 'fixed'), (axes[1, 1], 'knee')):
        sub = df[(df.arm == arm)].dropna(subset=['r_squared', 'aperiodic_exponent'])
        if sub.empty:
            ax.set_visible(False)
            continue
        # Trim the exponent axis to the 0.1-99.9 pct so a handful of runaway fits
        # cannot flatten the bulk into one row of cells -- the count of trimmed
        # points is annotated, never silently dropped.
        e_lo, e_hi = sub.aperiodic_exponent.quantile([0.001, 0.999])
        keep = sub.aperiodic_exponent.between(e_lo, e_hi)
        hb = ax.hexbin(sub.loc[keep, 'r_squared'], sub.loc[keep, 'aperiodic_exponent'],
                       gridsize=55, cmap=cmap, mincnt=1, linewidths=0,
                       bins='log', extent=(min(0.9, lo_r2), 1.0, e_lo, e_hi))
        cb = fig.colorbar(hb, ax=ax, pad=0.02)
        cb.set_label('channel-epochs (log)', color=INK_SECONDARY, fontsize=9)
        cb.ax.tick_params(colors=INK_SECONDARY, labelsize=8)
        cb.outline.set_visible(False)
        med = sub.aperiodic_exponent.median()
        ax.axhline(med, color=ARM_COLOR[arm], lw=2, ls='--', alpha=0.9)
        ax.annotate(f'median exponent {med:.3f}', xy=(0.02, 0.94),
                    xycoords='axes fraction', fontsize=9, color=ARM_COLOR[arm],
                    fontweight='bold')
        ax.annotate(f'{int((~keep).sum())} of {len(sub):,} outside '
                    f'[{e_lo:.2f}, {e_hi:.2f}] not shown',
                    xy=(0.02, 0.03), xycoords='axes fraction',
                    fontsize=8, color=INK_MUTED)
        ax.set_xlabel('r$^2$', color=INK_SECONDARY, fontsize=10)
        ax.set_ylabel('aperiodic exponent', color=INK_SECONDARY, fontsize=10)
        ax.set_title(f'{arm} arm: does a bad fit mean a bad exponent?',
                     color=INK_PRIMARY, fontsize=11, loc='left', fontweight='bold')

    fig.suptitle('Tier 0.1  FOOOF fit quality, and whether it contaminates the exponent',
                 color=INK_PRIMARY, fontsize=13, fontweight='bold', x=0.01, ha='left')
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    fig.savefig(out_path, dpi=170, facecolor=SURFACE)
    plt.close(fig)
    logger.info('wrote %s', out_path.name)


# ---------------------------------------------------------------------------
# 2. Per-session failure vs data loss
# ---------------------------------------------------------------------------

def plot_per_session(df, out_path, arm='fixed'):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    sub = df[df.arm == arm].copy()
    order = (sub.groupby('unit').r_squared.median().sort_values().index.tolist())
    pos = {u: i for i, u in enumerate(order)}

    fig, axes = plt.subplots(3, 1, figsize=(15, 10.5), facecolor=SURFACE,
                             sharex=True, height_ratios=[2.2, 1, 1])
    for ax in axes:
        ax.set_facecolor(SURFACE)
        for s in ('top', 'right'):
            ax.spines[s].set_visible(False)
        for s in ('left', 'bottom'):
            ax.spines[s].set_color(INK_MUTED)
        ax.tick_params(colors=INK_SECONDARY, labelsize=8)
        ax.grid(True, axis='y', color=INK_MUTED, alpha=0.18, linewidth=0.6)
        ax.set_axisbelow(True)

    # --- r2 boxes, ordered by median ----------------------------------------
    data = [sub.loc[sub.unit == u, 'r_squared'].dropna().to_numpy() for u in order]
    bp = axes[0].boxplot(data, positions=range(len(order)), widths=0.68,
                         patch_artist=True, showfliers=False,
                         medianprops=dict(color=INK_PRIMARY, lw=1.4),
                         whiskerprops=dict(color=INK_MUTED, lw=0.9),
                         capprops=dict(color=INK_MUTED, lw=0.9))
    for patch in bp['boxes']:
        patch.set_facecolor(ARM_COLOR[arm])
        patch.set_alpha(0.55)
        patch.set_edgecolor(SURFACE)       # 2px surface gap between adjacent fills
        patch.set_linewidth(2)
    axes[0].set_ylabel('r$^2$  (per channel-epoch)', color=INK_SECONDARY, fontsize=10)
    axes[0].set_title(f'Tier 0.2  Where the {arm} arm fails, and whether failure is '
                      f'data loss rather than physiology',
                      color=INK_PRIMARY, fontsize=13, loc='left', fontweight='bold')
    lo = float(np.nanpercentile(sub.r_squared, 0.5))
    axes[0].set_ylim(min(0.9, lo), 1.001)
    axes[0].annotate('sessions ordered by median r$^2$  ·  outliers not drawn  ·  '
                     f'y clipped at {min(0.9, lo):.3f}',
                     xy=(0.005, 0.03), xycoords='axes fraction',
                     fontsize=8, color=INK_MUTED)

    # --- mask_excluded_frac, SAME x order, its own panel --------------------
    # Separate panel rather than a second y-axis: a dual axis invites reading a
    # correlation off where two arbitrarily-scaled lines cross.
    # MEAN, not median: mask_excluded_frac is zero-inflated (most channel-epochs
    # lose nothing), so every session's median is exactly 0 and both the bar chart
    # and the Spearman below are degenerate -- measured as rho=nan on the first run.
    mean_excl = sub.groupby('unit').mask_excluded_frac.mean().reindex(order)
    any_excl = (sub.assign(any_excl=sub.mask_excluded_frac > 0)
                   .groupby('unit').any_excl.mean().reindex(order))
    axes[1].bar(range(len(order)), mean_excl.to_numpy(), width=0.68,
                color=INK_SECONDARY, alpha=0.85, edgecolor=SURFACE, linewidth=2,
                label='mean excluded fraction')
    axes[1].plot(range(len(order)), any_excl.to_numpy(), lw=2, color=ARM_COLOR[arm],
                 label='fraction of channel-epochs losing ANY window')
    axes[1].set_ylabel('mask exclusion', color=INK_SECONDARY, fontsize=10)
    leg1 = axes[1].legend(frameon=False, fontsize=8, loc='upper right')
    for t in leg1.get_texts():
        t.set_color(INK_SECONDARY)

    med_win = sub.groupby('unit').n_windows_used.mean().reindex(order)
    axes[2].bar(range(len(order)), med_win.to_numpy(), width=0.68,
                color=INK_SECONDARY, alpha=0.8, edgecolor=SURFACE, linewidth=2)
    axes[2].axhline(300, color=ARM_COLOR[arm], lw=2, ls='--', alpha=0.9)
    axes[2].annotate('300 = a complete 5-min epoch', xy=(0.995, 1.04),
                     xycoords='axes fraction', fontsize=8, ha='right',
                     color=ARM_COLOR[arm])
    axes[2].set_ylabel('mean\nn_windows_used', color=INK_SECONDARY, fontsize=10)
    axes[2].set_xticks(range(len(order)))
    axes[2].set_xticklabels([u.replace('sub-', '').replace('_ses-', '/')
                             for u in order], rotation=90, fontsize=6.5)
    axes[2].set_xlabel('subject-session  (ordered by median r$^2$, worst on the left)',
                       color=INK_SECONDARY, fontsize=10)

    # The number the plot exists to answer: does poor fit track data loss?
    per_unit = pd.DataFrame({'r2': sub.groupby('unit').r_squared.median(),
                             'excl': sub.groupby('unit').mask_excluded_frac.mean(),
                             'win': sub.groupby('unit').n_windows_used.mean()}).dropna()
    r_excl = r_win = float('nan')
    if len(per_unit) > 2 and per_unit.excl.nunique() > 1 and per_unit.win.nunique() > 1:
        r_excl = per_unit.r2.corr(per_unit.excl, method='spearman')
        r_win = per_unit.r2.corr(per_unit.win, method='spearman')
        axes[0].annotate(
            f'Spearman across sessions:  median r$^2$ vs excluded_frac  '
            f'$\\rho$ = {r_excl:+.3f}      vs n_windows_used  $\\rho$ = {r_win:+.3f}',
            xy=(0.005, 0.30), xycoords='axes fraction', fontsize=10,
            color=INK_PRIMARY, fontweight='bold')
        axes[0].annotate(
            'a POSITIVE rho means poor fit does NOT follow data loss',
            xy=(0.005, 0.24), xycoords='axes fraction', fontsize=9,
            color=INK_SECONDARY)

    fig.tight_layout()
    fig.savefig(out_path, dpi=170, facecolor=SURFACE)
    plt.close(fig)
    logger.info('wrote %s  (rho r2~excl %.3f, r2~nwin %.3f)',
                out_path.name, r_excl, r_win)
    return {'spearman_r2_vs_mean_excluded_frac':
                None if not np.isfinite(r_excl) else float(r_excl),
            'spearman_r2_vs_mean_n_windows':
                None if not np.isfinite(r_win) else float(r_win),
            'mask_excluded_frac_is_zero_inflated':
                bool((sub.mask_excluded_frac == 0).mean() > 0.5),
            'frac_channel_epochs_with_any_exclusion':
                float((sub.mask_excluded_frac > 0).mean()),
            'worst_5_sessions': order[:5], 'best_5_sessions': order[-5:]}


# ---------------------------------------------------------------------------
# 3. Exemplar gallery
# ---------------------------------------------------------------------------

def plot_gallery(df, view_dirs, out_path, n_exemplars=12, seed=0):
    """Re-fit n_exemplars spectra sampled across r2 deciles and draw them."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from fooof import FOOOF

    from ieeg_ehr.views import build_pain_epoch_fooof as fv
    from ieeg_ehr.views import fullres_reader as fr

    # Sample across r2 deciles of the FIXED arm, so the same channel-epochs can be
    # shown in both arms and the comparison is paired rather than two samples.
    base = df[df.arm == 'fixed'].dropna(subset=['r_squared']).copy()
    base['decile'] = pd.qcut(base.r_squared, 10, labels=False, duplicates='drop')
    rng = np.random.default_rng(seed)
    per = max(1, int(np.ceil(n_exemplars / max(1, base.decile.nunique()))))
    picks = (base.groupby('decile', group_keys=False)
                 .apply(lambda g: g.sample(min(per, len(g)), random_state=seed),
                        include_groups=True)
                 .sort_values('r_squared'))
    picks = picks.head(n_exemplars)

    freqs = fr.freqs_hz()
    fcols = fr.freq_columns()
    mean_dirs = sorted(glob.glob(str(config.fullres_epoch_unit_dir() / 'views'
                                     / 'fullresmean-*')))
    if not mean_dirs:
        raise FileNotFoundError('no epoch-mean view to read observed spectra from')
    mean_dir = mean_dirs[-1]

    n = len(picks)
    fig, axes = plt.subplots(n, 2, figsize=(11, 2.15 * n), facecolor=SURFACE,
                             squeeze=False)
    for i, (_, row) in enumerate(picks.iterrows()):
        unit = row['unit']
        f = f'{mean_dir}/mean_{unit}.parquet'
        if not glob.glob(f):
            for ax in axes[i]:
                ax.set_visible(False)
            continue
        mv = pd.read_parquet(f)
        sel = mv[(mv.epoch_id == row['epoch_id']) & (mv.channel == row['channel'])]
        if sel.empty:
            for ax in axes[i]:
                ax.set_visible(False)
            continue
        stored_log = sel[fcols].to_numpy()[0]

        for j, arm in enumerate(ARMS):
            ax = axes[i, j]
            ax.set_facecolor(SURFACE)
            for s in ('top', 'right'):
                ax.spines[s].set_visible(False)
            for s in ('left', 'bottom'):
                ax.spines[s].set_color(INK_MUTED)
            ax.tick_params(colors=INK_SECONDARY, labelsize=8)
            if arm not in view_dirs:
                ax.set_visible(False)
                continue
            params = view_dirs[arm][1]
            idx, ff = fv.select_freqs(freqs, arm)
            # FOOOF takes LINEAR power; the cache stores log10. Exponentiating in
            # float64 is the same step the view does -- passing stored values
            # would fit log-of-log and draw a plausible wrong curve.
            lin = np.power(10.0, stored_log[idx].astype(
                config.CACHE_LINEAR_DOMAIN_DTYPE))
            if not (np.isfinite(lin).all() and (lin > 0).all()):
                ax.annotate('non-finite spectrum (excluded from FOOOF)',
                            xy=(0.5, 0.5), xycoords='axes fraction', ha='center',
                            fontsize=9, color=INK_MUTED)
                continue
            fm = FOOOF(peak_width_limits=tuple(params.get(
                           'peak_width_limits', fv.DEFAULT_PEAK_WIDTH_LIMITS)),
                       max_n_peaks=params.get('max_n_peaks', fv.DEFAULT_MAX_N_PEAKS)
                                   + fv.ARMS[arm]['line_peak_slots'],
                       min_peak_height=params.get('min_peak_height',
                                                  fv.DEFAULT_MIN_PEAK_HEIGHT),
                       peak_threshold=params.get('peak_threshold',
                                                 fv.DEFAULT_PEAK_THRESHOLD),
                       aperiodic_mode=fv.ARMS[arm]['aperiodic_mode'], verbose=False)
            fm.fit(ff, lin)

            # Observed = data ink (thin, neutral). Model curves carry identity via
            # colour AND a heavier mark, so neither rests on colour alone.
            ax.plot(ff, np.log10(lin), lw=0.9, color=INK_SECONDARY, alpha=0.85,
                    label='observed', zorder=1)
            ax.plot(ff, fm._ap_fit, lw=2, color=ARM_COLOR[arm], ls='--',
                    label='aperiodic', zorder=3, solid_capstyle='round')
            ax.plot(ff, fm.fooofed_spectrum_, lw=2, color=ARM_COLOR[arm],
                    label='full model', zorder=2, solid_capstyle='round')
            ax.set_xscale('log')
            ap = np.atleast_1d(fm.aperiodic_params_)
            exp_ = ap[-1]
            ax.annotate(f'{arm}   r$^2$={fm.r_squared_:.4f}   exp={exp_:.2f}'
                        + (f'   knee={ap[1]:.1f}' if len(ap) == 3 else ''),
                        xy=(0.02, 0.06), xycoords='axes fraction', fontsize=8.5,
                        color=INK_PRIMARY, fontweight='bold')
            if j == 0:
                ax.set_ylabel(f'{unit}\nep{int(row["epoch_id"])} {row["channel"]}'
                              f'\nlog$_{{10}}$ power',
                              color=INK_SECONDARY, fontsize=7.5)
            if i == 0:
                ax.set_title(f'{arm} arm', color=INK_PRIMARY, fontsize=11,
                             loc='left', fontweight='bold')
                leg = ax.legend(frameon=False, fontsize=7.5, loc='upper right',
                                ncol=3, handlelength=1.4)
                for t in leg.get_texts():
                    t.set_color(INK_SECONDARY)
            if i == n - 1:
                ax.set_xlabel('frequency (Hz, log)', color=INK_SECONDARY, fontsize=9)

    fig.suptitle('Tier 0.3  Exemplar fits across r$^2$ deciles  '
                 '(worst at top, best at bottom; same channel-epoch in both arms)',
                 color=INK_PRIMARY, fontsize=12, fontweight='bold', x=0.01, ha='left')
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    fig.savefig(out_path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    logger.info('wrote %s (%d exemplars)', out_path.name, n)
    return picks[['unit', 'epoch_id', 'channel', 'r_squared',
                  'aperiodic_exponent']].to_dict('records')


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--run-name', default='tier0')
    ap.add_argument('--n-exemplars', type=int, default=12)
    ap.add_argument('--box-arm', default='fixed', choices=list(ARMS),
                    help='which arm the per-session panel shows (default fixed, '
                         'the primary arm)')
    ap.add_argument('--epoch-minutes', type=float, default=None)
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)s %(message)s')
    io.warn_if_dirty()

    view_dirs = _find_view_dirs(args.epoch_minutes)
    if not view_dirs:
        logger.error('no FOOOF output to plot')
        return 1
    df = load_aperiodic(view_dirs)

    run_dir = config.analysis_run_dir(question=QUESTION, output_type=OUTPUT_TYPE,
                                      run_name=args.run_name)
    run_dir.mkdir(parents=True, exist_ok=True)

    plot_fit_quality(df, run_dir / '01_fit_quality.png')
    session_stats = plot_per_session(df, run_dir / '02_per_session.png',
                                     arm=args.box_arm)
    exemplars = plot_gallery(df, view_dirs, run_dir / '03_exemplar_gallery.png',
                             n_exemplars=args.n_exemplars)

    # The backing table, so a number in the figure can be traced without re-running.
    summary = (df.groupby('arm')
                 .agg(n_channel_epochs=('r_squared', 'size'),
                      r2_median=('r_squared', 'median'),
                      r2_p05=('r_squared', lambda v: v.quantile(0.05)),
                      frac_r2_below_090=('r_squared', lambda v: (v < 0.90).mean()),
                      frac_r2_below_095=('r_squared', lambda v: (v < 0.95).mean()),
                      error_median=('error', 'median'),
                      exponent_median=('aperiodic_exponent', 'median'),
                      exponent_p01=('aperiodic_exponent', lambda v: v.quantile(0.01)),
                      exponent_p99=('aperiodic_exponent', lambda v: v.quantile(0.99)),
                      frac_exponent_le_0=('aperiodic_exponent', lambda v: (v <= 0).mean()),
                      n_peaks_median=('n_peaks', 'median'),
                      frac_at_peak_ceiling=('n_peaks', lambda v: (v >= v.max()).mean()),
                      n_units=('unit', 'nunique'))
                 .reset_index())
    io.write_table(summary, run_dir / 'fit_qc_summary.csv', kind='table',
                   script=SCRIPT, params={'arms': list(view_dirs),
                                          'box_arm': args.box_arm},
                   subjects=sorted(df.unit.unique().tolist()))
    print()
    print(summary.to_string(index=False))
    print()

    io.write_run_provenance(
        run_dir, script=SCRIPT,
        params={'arms': list(view_dirs), 'box_arm': args.box_arm,
                'n_exemplars': args.n_exemplars},
        parents=[str(d) for d, _ in view_dirs.values()],
        subjects=sorted(df.unit.unique().tolist()),
        extra={'session_stats': session_stats, 'exemplars': exemplars,
               'fooof_params_by_arm': {a: p for a, (_d, p) in view_dirs.items()}})
    io.log_analysis(f'Tier 0 FOOOF fit QC ({len(df):,} channel-epochs, '
                    f'{df.unit.nunique()} units)', run_dir)
    logger.info('run dir: %s', run_dir)
    return 0


if __name__ == '__main__':
    sys.exit(main())
