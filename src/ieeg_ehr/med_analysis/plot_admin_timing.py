"""
Fig 2 — when analgesics are given: time of day, and inter-dose intervals.

Two panels of small multiples:

- **Time of day**, one panel per drug. A full 24 h from midnight to midnight,
  30-minute bins, bars stacked by route. The panel title carries the SUBJECT
  count, the legend carries the ADMINISTRATION count per route. Both n's are
  reported because a drug with many administrations concentrated in few subjects
  is statistically fragile, and only showing one number hides which case you are
  looking at.

  This panel does not exist in the source analysis and is written fresh. It reads
  clinical practice directly off the clock: a drug on a fixed schedule shows sharp
  peaks at the medication passes, an as-needed drug is diffuse.

- **Inter-dose intervals**, one panel per drug-route formulation. Titles state the
  formulation, the interval count, and the number of subjects with >=2 doses.

  WHY-DIFFERENT: the axis is 0-24 h in 1 h bins, against the source's 0-30 h.
  As-needed opioid dosing concentrates at 4-6 h and is hard to read spread over
  30 hours. Intervals past the limit are counted and annotated on the panel, not
  dropped silently — the same convention as the source.

Intervals are computed WITHIN (subject, session) only. `taken_date` is on a
shared de-identified epoch, so a difference taken across sessions is arithmetic
on two unrelated clocks.

Clock time survives de-identification (confirmed by the person who performed it),
so this figure measures prescribing practice rather than the anonymizer.

There is no PRN / as-needed field in the MAR schema, so the scheduled-vs-PRN
split the source figure would have benefited from is not available and is not
inferred.

Run on Slurm, never the login node:
    python -m ieeg_ehr.med_analysis.plot_admin_timing
"""

import logging

import numpy as np
import pandas as pd

from ieeg_ehr import config, io
from ieeg_ehr.med_analysis import load, output, style
from ieeg_ehr.med_analysis.style import plt

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

OUTPUT_TYPE = 'timing_hist'
SCRIPT = 'ieeg_ehr/med_analysis/plot_admin_timing.py'

#: The spec allows 30-60 min. 60 is the default because the low-n panels
#: (ketorolac n=47, fentanyl n=94) are pure noise at 30-min resolution, and the
#: question this panel answers — scheduled versus as-needed — is about structure
#: on the scale of a medication pass, not a half hour. Override with --tod-bin.
TOD_BIN_MINUTES = 60

INTERVAL_MAX_H = 24
INTERVAL_EDGES = np.arange(0, INTERVAL_MAX_H + 1, 1)


def tod_edges(bin_minutes):
    """Bin edges spanning a full 24 h clock."""
    width = bin_minutes / 60.0
    return np.arange(0, 24 + width, width)


def interval_table(admin_df):
    """One row per inter-dose interval, within (subject, session, drug, route)."""
    rows = []
    keys = ['subject', 'session', 'drug', 'route']
    for (subject, session, drug, route), g in admin_df.groupby(keys):
        times = g['taken_dt'].sort_values().to_list()
        for prev, curr in zip(times, times[1:]):
            rows.append({
                'subject': subject, 'session': session, 'drug': drug,
                'route': route,
                'interval_h': (curr - prev).total_seconds() / 3600.0,
            })
    return pd.DataFrame(rows, columns=['subject', 'session', 'drug', 'route',
                                       'interval_h'])


def _panel_grid(n, ncols=3, panel_w=5.0, panel_h=3.4):
    nrows = int(np.ceil(n / ncols)) if n else 1
    fig, axes = plt.subplots(nrows, ncols, squeeze=False,
                             figsize=(panel_w * ncols, panel_h * nrows))
    flat = axes.ravel()
    for ax in flat[n:]:
        ax.set_visible(False)
    return fig, flat


def plot_time_of_day(admin_df, drugs, out_path, bin_minutes=TOD_BIN_MINUTES):
    edges = tod_edges(bin_minutes)
    fig, axes = _panel_grid(len(drugs))
    routes = sorted(admin_df['route'].unique())
    colors = style.categorical_colors(routes)

    for ax, drug in zip(axes, drugs):
        sub = admin_df[admin_df['drug'] == drug]
        present = [r for r in routes if (sub['route'] == r).any()]

        # Stacked rather than overlaid: overlaid bars occlude each other, and the
        # total per bin is the quantity of interest.
        bottoms = np.zeros(len(edges) - 1)
        for route in present:
            counts, _ = np.histogram(sub.loc[sub['route'] == route, 'hour_of_day'],
                                     bins=edges)
            ax.bar(edges[:-1], counts, width=bin_minutes / 60.0,
                   align='edge', bottom=bottoms, color=colors[route],
                   label=f'{route} (n={int(counts.sum())})', zorder=3)
            bottoms += counts

        ax.set_xlim(0, 24)
        ax.set_xticks(np.arange(0, 25, 3))
        ax.set_xticklabels(['12a', '3a', '6a', '9a', '12p', '3p', '6p', '9p', '12a'])
        style.style_axes(ax)
        style.label_axes(ax, 'Time of day', 'Administrations',
                         f'{drug.title()}\n{sub["subject"].nunique()} subjects')
        # 'best' + an opaque frame: a fixed corner collides with the tall bars in
        # whichever panel happens to peak there.
        ax.legend(fontsize=style.LEGEND_SIZE - 1, loc='best', frameon=True,
                  framealpha=0.9, facecolor='white', edgecolor='none')

    fig.suptitle('Analgesic administrations by time of day',
                 fontsize=style.TITLE_SIZE + 2, x=0.01, ha='left')
    fig.tight_layout(rect=(0, 0.01, 1, 0.98))
    return style.save(
        fig, out_path,
        footnote=f'{bin_minutes:g}-min bins; panel title = subjects, legend = '
                 'administrations; local clock time preserved by de-identification')


def plot_interdose(intervals, formulations, out_path):
    fig, axes = _panel_grid(len(formulations))

    for ax, (drug, route, label) in zip(axes, formulations):
        sub = intervals[(intervals['drug'] == drug) & (intervals['route'] == route)]
        within = sub[sub['interval_h'] <= INTERVAL_MAX_H]
        n_beyond = len(sub) - len(within)
        n_subj = sub['subject'].nunique()

        counts, _ = np.histogram(within['interval_h'], bins=INTERVAL_EDGES)
        ax.bar(INTERVAL_EDGES[:-1], counts, width=1.0, align='edge',
               color=style.BAR_COLOR, zorder=3)

        ax.set_xlim(0, INTERVAL_MAX_H)
        ax.set_xticks(np.arange(0, INTERVAL_MAX_H + 1, 4))
        style.style_axes(ax)
        style.label_axes(
            ax, 'Inter-dose interval (hours)', 'Number of intervals',
            f'{label}\nn={len(sub)} intervals from {n_subj} subjects with '
            f'≥2 doses')
        if n_beyond:
            ax.text(0.98, 0.95, f'{n_beyond} interval(s) beyond {INTERVAL_MAX_H}h '
                                'not shown',
                    transform=ax.transAxes, ha='right', va='top',
                    fontsize=style.FOOTNOTE_SIZE + 1, color=style.TEXT_MUTED)

    fig.suptitle('Inter-dose intervals by drug-route formulation',
                 fontsize=style.TITLE_SIZE + 2, x=0.01, ha='left')
    fig.tight_layout(rect=(0, 0.01, 1, 0.98))
    return style.save(
        fig, out_path,
        footnote='1-hour bins; intervals computed within (subject, session) only')


def main():
    parser = output.build_parser(__doc__)
    parser.add_argument('--max-panels', type=int, default=6,
                        help='drugs / formulations to show')
    parser.add_argument('--tod-bin', type=int, default=TOD_BIN_MINUTES,
                        help='time-of-day bin width in minutes (30-60)')
    args = parser.parse_args()

    io.warn_if_dirty()

    paths = config.med_admin_files()
    admin = load.load_administrations(paths=paths, subclasses=args.subclasses)

    counts = admin['drug'].value_counts()
    drugs = [d for d in counts.index if counts[d] >= args.min_admin][:args.max_panels]
    if not drugs:
        raise SystemExit(f'no drug reaches --min-admin={args.min_admin}')

    intervals = interval_table(admin)
    formulations = load.top_formulations(admin, n=args.max_panels,
                                         min_admin=args.min_admin)

    run_dir = output.resolve_run_dir(args, OUTPUT_TYPE)
    parents = output.source_parents(paths)
    subjects = sorted(admin['subject'].unique())

    plot_time_of_day(admin, drugs, run_dir / 'fig2a_time_of_day.png',
                     bin_minutes=args.tod_bin)
    plot_interdose(intervals, formulations, run_dir / 'fig2b_interdose_interval.png')

    # The numbers behind both panels, so the figure can be checked without
    # re-running it.
    tod = (admin.assign(hour_bin=pd.cut(admin['hour_of_day'],
                                        bins=tod_edges(args.tod_bin), right=False))
           .groupby(['drug', 'route', 'hour_bin'], observed=True)
           .size().reset_index(name='n_admin'))
    tod['hour_bin'] = tod['hour_bin'].astype(str)
    output.write_table(tod, run_dir, 'time_of_day_counts', SCRIPT,
                       params=vars(args), parents=parents, subjects=subjects)
    output.write_table(intervals, run_dir, 'interdose_intervals', SCRIPT,
                       params=vars(args), parents=parents, subjects=subjects)

    output.write_run(
        run_dir, SCRIPT, args, admin, paths,
        extra={
            'drugs_plotted': drugs,
            'formulations_plotted': [f'{d}|{r}' for d, r, _ in formulations],
            'time_of_day_bin_minutes': args.tod_bin,
            'interval_axis_max_h': INTERVAL_MAX_H,
            'n_intervals': int(len(intervals)),
            'n_intervals_beyond_axis': int((intervals['interval_h']
                                            > INTERVAL_MAX_H).sum()),
            'prn_note': ('No PRN/as-needed field exists in the MAR schema; the '
                         'scheduled-vs-as-needed split is not inferred.'),
        },
        description=(f'analgesic timing: time-of-day + inter-dose intervals, '
                     f'{len(drugs)} drugs, n={len(subjects)}'))


if __name__ == '__main__':
    main()
