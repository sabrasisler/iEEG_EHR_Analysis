"""Poster figures of a pain_change prediction run, styled per docs/figure_style.md.

    python -m ieeg_ehr.analysis.plot_pain_change_poster --tracking-dir <tracking run>

Panel 2, three axes in one row:

    left    predicted against actual Δpain at report 2, mean ± SEM per integer Δpain
            (`plot_pain_change_prediction.dose_panel`, the `dose-response` figure)
    middle  each subject's OLS slope of predicted on actual Δpain
            (`pain_change_tracking.slope_panel`, the `tracking_slope_hist` figure)
    right   predicted Δpain from report 1 to report 2 on the % of the interval, one line
            per Δpain bin (the `dc-groups-fraction` axis of `plot_pain_change_prediction`)

The tracking run (`pain_change_tracking`) names its scoring run, weights, prediction and
weighting, so all three axes come from one run. Written to
`<scoring run>/figures/<weighting>/poster-panel2_dpain_<prediction>-<weighting>.png` and
overwritten when only the look changes (docs/run_tokens.md).
"""

import argparse
import json
import logging
from pathlib import Path

from ieeg_ehr import io
from ieeg_ehr.analysis import plot_pain_change_prediction as pplot
from ieeg_ehr.analysis.pain_change import FIGURES_SUBDIR
from ieeg_ehr.analysis.pain_change_prediction import DPAIN_BINS
from ieeg_ehr.analysis.pain_change_tracking import slope_panel

plt = pplot.plt
logger = logging.getLogger(__name__)


def fig_panel2(df, table, out_path, prediction, min_subjects):
    # Steps up to t2 only. An adjusted step under the gap floor is extrapolated and
    # has no place on a fraction axis, so it goes.
    interval = df[df['t_min'] <= df['gap_min']]
    if prediction == 'adj':
        interval = interval[~interval['extrapolated'].astype(bool)]
    fig, (dose, slope, traj) = plt.subplots(1, 3, figsize=(15, 4.4))
    pplot.dose_panel(dose, df, min_subjects)
    pplot._title(dose, 'Predicted vs. reported pain change')
    slope_panel(slope, table)
    pplot._title(slope, 'Per-patient tracking slopes')
    pplot._style(traj)
    pplot._zero(traj)
    pplot._trajectory_lines(traj, pplot.to_fraction(interval, 'bin', ('value',)), 'bin',
                            list(DPAIN_BINS), pplot.BIN_COLOURS, min_subjects)
    pplot._fraction_axis(traj)
    traj.set_ylabel('Predicted Δpain', color=pplot.INK)
    pplot._title(traj, 'Predicted pain change between reports')
    # The counts go to the log line: inside the axes they would cover the lines.
    handles, labels = traj.get_legend_handles_labels()
    lo, hi = traj.get_ylim()
    traj.set_ylim(lo, hi + 0.4 * (hi - lo))
    traj.legend(handles, [label.split(' (')[0] for label in labels], frameon=False,
                fontsize=8, labelcolor=pplot.INK, loc='upper left', ncol=2,
                title='True Δpain at report 2', title_fontsize=8)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, facecolor='white', bbox_inches='tight')
    plt.close(fig)
    logger.info('wrote %s', out_path)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--tracking-dir', required=True, help='a pain_change_tracking run')
    ap.add_argument('--min-subjects', type=int, default=3)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    io.warn_if_dirty()

    tracking_dir = Path(args.tracking_dir)
    params = json.loads((tracking_dir / 'provenance.json').read_text())['params']
    run_dir = Path(params['run_dir'])
    prediction, weighting = params['prediction'], params['weighting']
    table = io.read_table(tracking_dir / 'per_subject_tracking.csv', on_stale='warn')
    df = pplot.prediction_frame(run_dir, params['weights'], prediction, weighting)
    tag = f'{prediction}-{weighting}'
    out_path = run_dir / FIGURES_SUBDIR / weighting / f'poster-panel2_dpain_{tag}.png'
    fig_panel2(df, table, out_path, prediction, args.min_subjects)
    io.log_analysis(f'pain_change poster panel 2, {tag}, {df["subject"].nunique()} subjects, '
                    f'{df["pair_id"].nunique()} pairs, {len(table)} subjects tracked', run_dir)
    print(out_path)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
