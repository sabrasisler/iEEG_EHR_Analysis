"""Per-subject pain->power lines for the pilot cells, with the group effect on top.

The pilot's other two figures describe the MODEL: `fig_pilot_variance` shows what
the variance components came out to, `fig_pilot_vs_twostage` shows that the fixed
effect agrees with the older two-stage map. Neither shows the DATA the model was
fitted to, so neither answers the question a reader actually starts with -- what
does a slope of -0.016 log10 units per pain point look like inside one patient?

This figure answers exactly that. One panel per pilot cell; inside a panel, one
thin line per subject (that subject's own least-squares fit through their own
epochs) and the population slope drawn over the top.

BOTH AXES ARE RELATIVE TO THE SUBJECT'S OWN AVERAGE. That is not a cosmetic
choice -- it is what the model estimates. `NRS_within` is already pain minus that
subject's mean pain, and subtracting each subject's mean power is the plotting
equivalent of the `subj_int` random intercept. Without the centring, subjects sit
at wildly different absolute power levels (var_subj_int dwarfs everything else)
and the panel is an unreadable vertical smear that hides the very slopes it is
supposed to show.

A subject line here is a plain per-subject OLS fit, NOT the model's BLUP. The
point of the figure is to show the evidence the model saw, un-shrunk, so that the
population line can be judged against it.

    python -m ieeg_ehr.analysis.plot_mixed_model_subject_lines --run-dir <run>

Writes `fig_pilot_subject_lines.png` and `subject_slopes.parquet` into the run
directory. Reads only artifacts the fit stage already wrote, so it can run while
the permutation array is still going.
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from ieeg_ehr import io

logger = logging.getLogger(__name__)

DISCLAIMER = ('EXPLORATORY pilot -- discovery cohort, NOMINATIONS NOT FINDINGS. '
              'Nothing here is corrected for multiple comparisons or confirmed '
              'out of sample.')

# Blue where power falls as pain rises, red where it climbs. Sign is the whole
# story in a spaghetti plot -- a reader counts colours long before they read an
# axis -- so the two are kept far apart and deliberately not a smooth ramp.
NEG_COLOR = '#2c6fad'
POS_COLOR = '#c1442f'


def epoch_level(frame):
    """One row per (subject, epoch): channels averaged, power centred per subject.

    The model's row is one CHANNEL x one epoch, and a channel random intercept
    absorbs each contact's level. A plot cannot draw that, so the panel shows the
    subject's ROI mean per epoch instead -- the same quantity the channel effect
    leaves behind once contact level is removed.

    Averaged in float64 (`config.CACHE_ACCUMULATE_DTYPE` territory): the frames
    are float32 and a bare mean over ~20 channels would accumulate in float32.
    """
    df = frame.copy()
    df['log10_power'] = df['log10_power'].astype('float64')
    per_epoch = (df.groupby(['subject', 'epoch_id'], as_index=False)
                   .agg(log10_power=('log10_power', 'mean'),
                        NRS_within=('NRS_within', 'first'),
                        NRS=('NRS', 'first'),
                        n_channels=('channel_uid', 'nunique')))
    # Centre power within subject -- the plotting counterpart of subj_int.
    per_epoch['power_centred'] = (
        per_epoch['log10_power']
        - per_epoch.groupby('subject')['log10_power'].transform('mean'))
    return per_epoch


def subject_slopes(per_epoch):
    """Per-subject OLS slope of centred power on within-subject pain.

    A subject needs at least two DISTINCT pain scores for a slope to exist at all;
    with one score the design matrix is rank-deficient and numpy would happily
    return a meaningless number. Those subjects keep their points and lose their
    line, and are counted in `n_no_slope` rather than dropped silently.
    """
    rows = []
    for subject, grp in per_epoch.groupby('subject'):
        x = grp['NRS_within'].to_numpy(dtype='float64')
        y = grp['power_centred'].to_numpy(dtype='float64')
        if len(np.unique(x)) < 2:
            rows.append({'subject': subject, 'slope': np.nan, 'intercept': np.nan,
                         'n_epochs': int(len(grp)), 'x_min': np.nan, 'x_max': np.nan})
            continue
        slope, intercept = np.polyfit(x, y, 1)
        rows.append({'subject': subject, 'slope': float(slope),
                     'intercept': float(intercept), 'n_epochs': int(len(grp)),
                     'x_min': float(x.min()), 'x_max': float(x.max())})
    return pd.DataFrame(rows)


def draw_panel(ax, per_epoch, slopes, cell):
    """One pilot cell: subject points, subject lines, then the population slope."""
    import matplotlib.pyplot as plt  # noqa: F401  (Agg already selected by caller)

    beta = float(cell['beta_nrs_within'])
    se = float(cell['se'])

    # Robust limits. One saturating epoch would otherwise set the y-scale for the
    # whole panel and flatten every line in it to a horizontal smear.
    y = per_epoch['power_centred'].to_numpy()
    x = per_epoch['NRS_within'].to_numpy()
    ylim = float(np.nanpercentile(np.abs(y), 99)) or 1.0
    xlim = float(np.nanmax(np.abs(x))) * 1.05 or 1.0

    ax.scatter(x, y, s=2, color='0.6', alpha=0.10, linewidths=0, zorder=1)

    fitted = slopes.dropna(subset=['slope'])
    for _, r in fitted.iterrows():
        xs = np.array([r['x_min'], r['x_max']])
        ax.plot(xs, r['slope'] * xs + r['intercept'],
                color=POS_COLOR if r['slope'] > 0 else NEG_COLOR,
                lw=0.8, alpha=0.45, zorder=2, solid_capstyle='round')

    # The population effect. Drawn through the origin because both axes are
    # subject-centred, which is precisely the constraint the fixed effect carries.
    xs = np.array([-xlim, xlim])
    ax.fill_between(xs, (beta - 1.96 * se) * xs, (beta + 1.96 * se) * xs,
                    color='0.25', alpha=0.20, lw=0, zorder=3)
    ax.plot(xs, beta * xs, color='black', lw=2.4, zorder=4)

    ax.axhline(0, color='0.85', lw=0.7, zorder=0)
    ax.axvline(0, color='0.85', lw=0.7, zorder=0)
    ax.set_xlim(-xlim, xlim)
    ax.set_ylim(-ylim, ylim)

    n_same = int((np.sign(fitted['slope']) == np.sign(beta)).sum()) if len(fitted) else 0
    frac = n_same / len(fitted) if len(fitted) else np.nan

    ax.set_title(f"{cell['region']}  {cell['freq_bin_low']:.1f}-"
                 f"{cell['freq_bin_high']:.1f} Hz\n{cell['group']}",
                 fontsize=8.5)
    ax.text(0.03, 0.97,
            f"beta {beta:+.4f}\nWald p {float(cell['p']):.2g}\n"
            f"{int(cell['n_subjects'])} subj, {int(cell['n_channels'])} chan\n"
            f"sign-consistent {n_same}/{len(fitted)} ({frac:.0%})",
            transform=ax.transAxes, fontsize=6.4, va='top', ha='left',
            bbox=dict(boxstyle='round,pad=0.28', fc='white', ec='0.8', alpha=0.85))
    ax.tick_params(labelsize=7)
    return {'region': cell['region'], 'freq_bin_index': int(cell['freq_bin_index']),
            'n_sign_consistent': n_same, 'n_with_slope': int(len(fitted)),
            'frac_sign_consistent': float(frac) if len(fitted) else np.nan,
            'n_no_slope': int(len(slopes) - len(fitted))}


def build(run_dir, out_path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    run_dir = Path(run_dir)
    cells = io.read_table(run_dir / 'pilot_cells.parquet', on_stale='warn')
    cells = cells.sort_values('cell_index').reset_index(drop=True)

    n = len(cells)
    ncol = 5
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.5 * ncol, 3.1 * nrow),
                             squeeze=False)

    summaries, all_slopes = [], []
    for i, cell in cells.iterrows():
        ax = axes[i // ncol][i % ncol]
        frame_path = run_dir / 'frames' / f"cell_{int(cell['cell_index']):03d}.parquet"
        if not frame_path.exists():
            ax.set_visible(False)
            logger.warning('cell %d: no saved frame at %s', cell['cell_index'],
                           frame_path)
            continue
        per_epoch = epoch_level(io.read_table(frame_path, on_stale='ignore'))
        slopes = subject_slopes(per_epoch)
        summaries.append(draw_panel(ax, per_epoch, slopes, cell))
        slopes = slopes.assign(region=cell['region'],
                               freq_bin_index=int(cell['freq_bin_index']),
                               cell_index=int(cell['cell_index']),
                               group=cell['group'],
                               beta_group=float(cell['beta_nrs_within']))
        all_slopes.append(slopes)
        logger.info('cell %2d %-14s bin %2d | %d subjects drawn',
                    int(cell['cell_index']), cell['region'],
                    int(cell['freq_bin_index']), len(slopes))

    for j in range(n, nrow * ncol):
        axes[j // ncol][j % ncol].set_visible(False)

    # y is set explicitly: the default sits supxlabel where the footnote goes, and
    # tight_layout will not arbitrate between two figure-level texts.
    fig.supxlabel('pain score relative to that subject\'s own mean  (NRS_within, points)',
                  fontsize=10, y=0.062)
    fig.supylabel('log10 power relative to that subject\'s own mean', fontsize=10)
    fig.suptitle('Per-subject pain-power relationship in each pilot cell\n'
                 'thin line = one subject\'s own OLS fit   thick line = population '
                 'fixed effect (+/- 1.96 SE)', fontsize=12)

    handles = [Line2D([], [], color=POS_COLOR, lw=1.6, label='subject slope > 0'),
               Line2D([], [], color=NEG_COLOR, lw=1.6, label='subject slope < 0'),
               Line2D([], [], color='black', lw=2.4, label='population fixed effect')]
    fig.legend(handles=handles, loc='lower right', fontsize=8.5, ncol=3,
               bbox_to_anchor=(0.995, 0.088))

    fig.tight_layout(rect=(0.02, 0.10, 1, 0.955))
    fig.text(0.01, 0.004,
             'Both axes are centred within subject, which is what makes the panels '
             'comparable: the model absorbs each subject\'s baseline power in a random '
             'intercept and each contact\'s level in a second one, so only the WITHIN-'
             'subject relationship is being estimated. A subject line is that subject\'s '
             'own unshrunk OLS fit, drawn only across the pain range they actually '
             'reported; subjects with a single distinct score have no line. The spread '
             'of line slopes around the black line IS the heterogeneity the LRT tests -- '
             'note it is visible even in cells whose population effect is ~0.\n'
             + DISCLAIMER,
             fontsize=6.5, va='bottom', ha='left', color='0.35', wrap=True)

    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return pd.DataFrame(summaries), pd.concat(all_slopes, ignore_index=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--run-dir', required=True,
                    help='Pilot run directory written by run_mixed_model_pilot.')
    ap.add_argument('--out-name', default='fig_pilot_subject_lines.png')
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)s %(message)s')
    io.warn_if_dirty()

    run_dir = Path(args.run_dir)
    out_path = run_dir / args.out_name
    summary, slopes = build(run_dir, out_path)

    io.write_table(slopes, run_dir / 'subject_slopes.parquet',
                   params={'source': 'per-subject OLS on subject-centred epoch means',
                           'note': 'unshrunk; not the model BLUPs'},
                   parents=[str(run_dir / 'pilot_cells.parquet')],
                   subjects=sorted(slopes['subject'].unique()),
                   script='ieeg_ehr/analysis/plot_mixed_model_subject_lines.py')

    logger.info('\n%s', summary.to_string(index=False))
    logger.info('wrote %s', out_path)
    io.log_analysis('mixed-model pilot: per-subject pain-power lines per cell '
                    '(EXPLORATORY pilot figure, not a finding)', run_dir)


if __name__ == '__main__':
    main()
