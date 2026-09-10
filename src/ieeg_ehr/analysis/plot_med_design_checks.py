"""Design-level checks on the medication contrast, before any cell is believed.

Three questions that no region x frequency map can answer, because they are about
the DESIGN rather than about any particular cell. Each one can invalidate the
whole analysis on its own.

1. IS MEDICATION INTERLEAVED IN TIME, OR DOES IT ARRIVE AS A BLOCK?
   If patients are largely unmedicated on days 1-2 and medicated on days 3-5,
   then `med_state` is partly a proxy for time since implant, and the contrast is
   measuring post-surgical recovery, electrode settling and sleep debt as much as
   pharmacology. The timeline also shows at a glance which patients contribute
   nothing to the within-subject contrast because they never switch state.

2. IS THERE POSITIVITY?
   The medicated and unmedicated pain distributions must OVERLAP. If medicated
   epochs only exist above a certain pain level, the "medicated slope" is fitted
   over a narrow range and then drawn across a region of the plot where there is
   no data. That is extrapolation wearing an interaction's clothing.

3. IS THE MEDICATION COEFFICIENT PARTLY A BETWEEN-SUBJECT PAIN EFFECT?
   One dot per patient: mean pain against proportion of epochs medicated. A
   strong positive relationship means patients who hurt more are medicated more,
   so an undecomposed `med_state` term carries a between-subject pain difference
   under a medication label.

    python -m ieeg_ehr.analysis.plot_med_design_checks --run-dir <med strata run>
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from ieeg_ehr import io
from ieeg_ehr.analysis import med_state as ms

logger = logging.getLogger(__name__)

DISCLAIMER = ('EXPLORATORY -- discovery cohort, NOMINATIONS NOT FINDINGS. '
              'Pain score is the 0-10 nurse-recorded rating (NRS), not a VAS.')

MED_COLOR = '#c1442f'
UNMED_COLOR = '#4a7ba7'


def load_design(run_dir):
    """Per epoch: subject, time in the EMU, pain score, medication state."""
    run_dir = Path(run_dir)
    state = io.read_table(run_dir / 'epoch_med_state.parquet', on_stale='warn')
    prov = {}
    try:
        import json
        prov = json.loads((run_dir / 'provenance.json').read_text())
    except (OSError, ValueError):
        pass
    cohort = {s.replace('sub-', '') for s in (prov.get('subjects') or [])}

    defs = ms.load_epoch_defs(subjects=cohort or None)
    d = defs.merge(state[['subject', 'session', 'epoch_id', 'med_state']],
                   on=['subject', 'session', 'epoch_id'], how='inner')
    if cohort:
        d = d[d['subject'].isin(cohort)]
    # Hours since that subject's FIRST assessment -- the closest proxy this data
    # has for time in the EMU, and the axis the block-vs-interleaved question
    # needs.
    d['hours_in_emu'] = (d['pain_time']
                         - d.groupby('subject')['pain_time'].transform('min')
                         ).dt.total_seconds() / 3600.0
    d['pain_within'] = (d['pain_score']
                        - d.groupby('subject')['pain_score'].transform('mean'))
    return d.sort_values(['subject', 'pain_time']).reset_index(drop=True)


def fig_timeline(d, out_path):
    """One row per patient: when each epoch happened and whether it was medicated."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    # Ordered by how INTERLEAVED they are, so the patients who contribute nothing
    # to the within-subject contrast collect at the ends rather than hiding in
    # the middle of an alphabetical list.
    frac = d.groupby('subject')['med_state'].mean()
    order = frac.sort_values().index.tolist()

    fig, axes = plt.subplots(1, 2, figsize=(19, 0.34 * len(order) + 3.4),
                             gridspec_kw={'width_ratios': [3, 1]}, sharey=True)

    ax = axes[0]
    for i, subject in enumerate(order):
        g = d[d['subject'] == subject]
        ax.plot(g['hours_in_emu'], np.full(len(g), i), '-', color='0.85', lw=0.6,
                zorder=1)
        for med, colour in ((True, MED_COLOR), (False, UNMED_COLOR)):
            s = g[g['med_state'] == med]
            ax.scatter(s['hours_in_emu'], np.full(len(s), i), s=14, c=colour,
                       linewidths=0, zorder=2)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(order, fontsize=6.5)
    ax.set_ylim(-1, len(order))
    ax.invert_yaxis()
    ax.set_xlabel('hours since that patient\'s first pain assessment', fontsize=9)
    ax.set_title('When was each epoch, and was it medicated?', fontsize=11)
    for day in range(1, int(np.ceil(d['hours_in_emu'].max() / 24)) + 1):
        ax.axvline(day * 24, color='0.8', lw=0.7, ls=':', zorder=0)
    ax.legend(handles=[Line2D([], [], marker='o', ls='none', color=MED_COLOR,
                              label='medicated'),
                       Line2D([], [], marker='o', ls='none', color=UNMED_COLOR,
                              label='not medicated')],
              fontsize=8, loc='lower right')

    # Right: the pain score as a parallel track, same row order.
    ax = axes[1]
    for i, subject in enumerate(order):
        g = d[d['subject'] == subject]
        if not len(g):
            continue
        # Scaled into the row's band so 0-10 reads without 36 separate axes.
        y = i + 0.45 - 0.9 * (g['pain_score'] / 10.0)
        ax.plot(g['hours_in_emu'], y, '-', color='0.55', lw=0.7, zorder=2)
    ax.set_xlabel('hours since first assessment', fontsize=9)
    ax.set_title('pain score over time\n(each row spans 0 at the bottom to 10 at '
                 'the top)', fontsize=10)

    n_switch = int((d.groupby('subject')['med_state'].nunique() > 1).sum())
    fig.suptitle('Per-patient design timeline: is medication interleaved, or a block?',
                 fontsize=13)
    fig.tight_layout(rect=(0, 0.06, 1, 0.95))
    fig.text(0.01, 0.005,
             f'{n_switch} of {d["subject"].nunique()} patients have epochs in BOTH '
             'medication states; only those contribute to the within-subject '
             'contrast. If red and blue separate into blocks along the x axis '
             'rather than interleaving, med_state is partly a proxy for time '
             'since implant, and the contrast carries post-surgical recovery, '
             'electrode settling and sleep debt alongside any drug effect. '
             'Dotted verticals are 24 h.\n' + DISCLAIMER,
             fontsize=7, va='bottom', ha='left', color='0.35', wrap=True)
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return n_switch


def fig_positivity(d, out_path):
    """Overlapping pain distributions by medication state -- the positivity check."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(17, 4.8))

    for ax, col, label in ((axes[0], 'pain_score', 'pain score (0-10)'),
                           (axes[1], 'pain_within',
                            'pain score relative to that patient\'s own mean')):
        lo, hi = d[col].min(), d[col].max()
        bins = np.linspace(lo, hi, 24)
        for med, colour, name in ((False, UNMED_COLOR, 'not medicated'),
                                  (True, MED_COLOR, 'medicated')):
            s = d.loc[d['med_state'] == med, col]
            ax.hist(s, bins=bins, density=True, alpha=0.55, color=colour,
                    label=f'{name} (n={len(s)})')
        ax.axvline(0 if col == 'pain_within' else d[col].mean(), color='0.3',
                   lw=1, ls='--')
        ax.set_xlabel(label, fontsize=9)
        ax.set_ylabel('density', fontsize=9)
        ax.legend(fontsize=8)
    axes[0].set_title('raw pain score', fontsize=11)
    axes[1].set_title('within-patient pain\n(the actual model predictor)',
                      fontsize=11)

    # Overlap of the within-patient predictor, which is what positivity is about.
    ax = axes[2]
    q = np.linspace(0, 100, 101)
    a = np.percentile(d.loc[d['med_state'], 'pain_within'], q)
    b = np.percentile(d.loc[~d['med_state'], 'pain_within'], q)
    ax.plot(b, a, color='0.3', lw=1.5)
    lim = float(np.nanmax(np.abs(np.concatenate([a, b])))) * 1.05
    ax.plot([-lim, lim], [-lim, lim], ls='--', color='0.6', lw=1)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect('equal')
    ax.set_xlabel('quantiles, not medicated', fontsize=9)
    ax.set_ylabel('quantiles, medicated', fontsize=9)
    ax.set_title('Q-Q of the model predictor\n(on the line = same distribution)',
                 fontsize=11)

    lo_ov = max(d.loc[d['med_state'], 'pain_within'].min(),
                d.loc[~d['med_state'], 'pain_within'].min())
    hi_ov = min(d.loc[d['med_state'], 'pain_within'].max(),
                d.loc[~d['med_state'], 'pain_within'].max())
    fig.suptitle('Positivity: do the two medication states cover the same pain range?',
                 fontsize=13)
    fig.tight_layout(rect=(0, 0.10, 1, 0.93))
    fig.text(0.01, 0.01,
             f'Common support of the within-patient predictor runs '
             f'[{lo_ov:+.2f}, {hi_ov:+.2f}]. Outside that range one state has no '
             'data and any difference between the two fitted slopes is '
             'EXTRAPOLATION rather than an interaction. Zero must sit inside both '
             'clouds for the intercept-level comparison to mean anything. The '
             'middle panel is the predictor the model actually uses; the left one '
             'is the raw scale and is shown because the shift between states is '
             'the confounding-by-indication problem in its most direct form.\n'
             + DISCLAIMER,
             fontsize=7, va='bottom', ha='left', color='0.35', wrap=True)
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return {'overlap_lo': float(lo_ov), 'overlap_hi': float(hi_ov)}


def fig_between_subject(d, out_path):
    """Subject mean pain against subject proportion medicated -- the decomposition."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from scipy import stats

    per = d.groupby('subject').agg(mean_pain=('pain_score', 'mean'),
                                   frac_med=('med_state', 'mean'),
                                   n_epochs=('epoch_id', 'size')).reset_index()
    fig, ax = plt.subplots(figsize=(7.2, 6.2))
    sc = ax.scatter(per['frac_med'], per['mean_pain'], s=per['n_epochs'] * 1.6,
                    c='#4a7ba7', alpha=0.75, edgecolors='white', linewidths=0.6)
    r, p = stats.spearmanr(per['frac_med'], per['mean_pain'])
    if len(per) > 2:
        b = np.polyfit(per['frac_med'], per['mean_pain'], 1)
        xs = np.linspace(per['frac_med'].min(), per['frac_med'].max(), 20)
        ax.plot(xs, np.polyval(b, xs), color='#c1442f', lw=1.6)
    ax.set_xlabel('proportion of that patient\'s epochs that were medicated',
                  fontsize=10)
    ax.set_ylabel('that patient\'s mean pain score', fontsize=10)
    ax.set_title('Is the medication term partly a between-patient pain effect?\n'
                 f'Spearman rho = {r:+.3f}  (p = {p:.3g}),  {len(per)} patients\n'
                 'marker area is that patient\'s epoch count', fontsize=11)
    fig.tight_layout(rect=(0, 0.12, 1, 1))
    fig.text(0.01, 0.01,
             'One dot per patient. A strong positive relationship means patients '
             'who hurt more are medicated more often, so an undecomposed '
             'med_state coefficient carries a BETWEEN-patient pain difference '
             'under a medication label. The within/between split that '
             'NRS_within + NRS_submean applies to pain has no counterpart for '
             'medication in the current models -- med_state enters raw. A flat or '
             'weak relationship means that gap costs little; a steep one means it '
             'matters and med_state should be decomposed the same way.\n'
             + DISCLAIMER,
             fontsize=7, va='bottom', ha='left', color='0.35', wrap=True)
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return per, {'rho': float(r), 'p': float(p)}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--run-dir', required=True)
    ap.add_argument('--out-subdir', default='design')
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)s %(message)s')
    io.warn_if_dirty()

    run_dir = Path(args.run_dir)
    out = run_dir / args.out_subdir
    out.mkdir(parents=True, exist_ok=True)

    d = load_design(run_dir)
    logger.info('%d epochs, %d patients', len(d), d['subject'].nunique())

    n_switch = fig_timeline(d, out / 'fig_design_timeline.png')
    logger.info('patients with BOTH medication states: %d / %d',
                n_switch, d['subject'].nunique())

    ov = fig_positivity(d, out / 'fig_positivity.png')
    logger.info('common support of within-patient pain: [%+.2f, %+.2f]',
                ov['overlap_lo'], ov['overlap_hi'])

    per, stat = fig_between_subject(d, out / 'fig_between_subject.png')
    logger.info('subject mean pain vs proportion medicated: rho=%+.3f p=%.3g',
                stat['rho'], stat['p'])

    io.write_table(per, out / 'per_subject_design.csv',
                   params={'source': 'epoch_med_state + epoch definitions'},
                   script='ieeg_ehr/analysis/plot_med_design_checks.py')
    io.log_analysis('medication design checks: timeline, positivity, '
                    'between-subject decomposition (EXPLORATORY)', run_dir)
    logger.info('wrote %s', out)


if __name__ == '__main__':
    main()
