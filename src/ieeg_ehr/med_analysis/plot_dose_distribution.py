"""
Fig 6 — what doses were actually given, per drug.

One panel per drug: x = the dose values that appear in the MAR (in that drug's
OWN unit), y = how many administrations were at each, segmented by route.

WHY ONE PANEL PER DRUG AND NEVER A SHARED AXIS. Doses are not comparable across
these four drugs and pooling them would produce a number with no unit
(DECISIONS 2026-09-03, call 2): acetaminophen, oxycodone and hydromorphone are
in mg, hydrocodone-acetaminophen is in TABLETS, and the three mg drugs span
0.2 mg to 1000 mg — three and a half orders of magnitude. `load.assert_single_unit`
is called per drug so a mixed-unit panel raises instead of rendering.

WHY ROUTE IS A SEGMENT AND NOT A FOOTNOTE. For hydromorphone the dose
distribution IS a route distribution: intravenous doses run 0.2-1 mg and oral
run 1-4 mg, so a single un-split histogram shows a spurious bimodal "dosing
choice" that is really two different drugs-as-given. Acetaminophen has a milder
version of the same thing — every 975 mg administration is by feeding tube,
every 325/500 mg one is oral. Colouring by route is what keeps the panel from
inventing structure.

WHAT A TABLET IS NOT. Hydrocodone-acetaminophen's dose column counts TABLETS,
and the product strength (mg hydrocodone - mg acetaminophen) lives in the drug
NAME rather than in any column, so "1" here is not comparable to "1" in a mg
panel and the underlying mg dose is not recoverable from this table. The panel
is labelled in tablets for that reason; converting would require parsing "5-325"
out of the product string, which is deliberately out of scope.

Run on Slurm, never the login node:
    python -m ieeg_ehr.med_analysis.plot_dose_distribution
"""

import argparse
import logging

import numpy as np
import pandas as pd

from ieeg_ehr import config, io
from ieeg_ehr.med_analysis import load, output, style
from ieeg_ehr.med_analysis.style import plt

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

OUTPUT_TYPE = 'dose_distribution'
SCRIPT = 'ieeg_ehr/med_analysis/plot_dose_distribution.py'

#: Same drugs, same order, as Fig 5 — descending administration count. Morphine
#: and ibuprofen are excluded there for having no estimable distribution, and
#: are kept out here too so the two figures describe one drug set.
DEFAULT_DRUGS = ('ACETAMINOPHEN', 'HYDROCODONE-ACETAMINOPHEN', 'OXYCODONE',
                 'HYDROMORPHONE', 'FENTANYL', 'KETOROLAC', 'TRAMADOL')


def dose_counts(admin_df, drugs):
    """One row per (drug, dose, route): administrations and distinct subjects.

    `assert_single_unit` runs per drug rather than per (drug, route): a drug
    whose oral and IV forms were charted in different units would make its
    panel's x axis meaningless, and that should stop the run, not be averaged
    over.
    """
    rows = []
    n_missing_dose = 0
    for drug in drugs:
        g = admin_df[admin_df['drug'] == drug]
        if g.empty:
            logger.warning('%s: no administrations, skipping', drug)
            continue
        unit = load.assert_single_unit(g, drug)

        missing = int(g['dose'].isna().sum())
        n_missing_dose += missing
        if missing:
            logger.warning('%s: %d administration(s) have no parseable dose and '
                           'are absent from the panel', drug, missing)

        for (dose, route), gg in g.dropna(subset=['dose']).groupby(
                ['dose', 'route']):
            rows.append({
                'drug': drug,
                'dose': float(dose),
                'dose_unit': unit,
                'route': route,
                'n_admin': len(gg),
                'n_subjects': gg['subject'].nunique(),
            })

    counts = pd.DataFrame(rows)
    if not counts.empty:
        counts = counts.sort_values(['drug', 'dose', 'route'])
    return counts.reset_index(drop=True), n_missing_dose


def plot_panels(counts, out_path, drugs, n_subjects_total, n_missing_dose):
    drugs = [d for d in drugs if d in set(counts['drug'])]

    # Routes shared across panels so one colour means one route everywhere,
    # ordered by overall volume so the legend reads most-common first.
    routes = (counts.groupby('route')['n_admin'].sum()
              .sort_values(ascending=False).index.tolist())
    colors = style.categorical_colors(routes)

    # Three across once there are more than four panels, so seven drugs are a
    # 3x3 block rather than a four-row column.
    ncols = 2 if len(drugs) <= 4 else 3
    nrows = int(np.ceil(len(drugs) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(11, 4.1 * nrows))
    axes = np.atleast_1d(axes).ravel()

    for ax, drug in zip(axes, drugs):
        sub = counts[counts['drug'] == drug]
        doses = sorted(sub['dose'].unique())
        unit = sub['dose_unit'].iloc[0]
        x = np.arange(len(doses), dtype=float)

        bottom = np.zeros(len(doses))
        for route in routes:
            vals = np.array([
                sub.loc[(sub['dose'] == d) & (sub['route'] == route),
                        'n_admin'].sum() for d in doses], dtype=float)
            if not vals.any():
                continue
            ax.bar(x, vals, bottom=bottom, width=0.62, color=colors[route],
                   label=route, zorder=3, edgecolor='white', linewidth=0.5)
            bottom += vals

        # The total above each bar, so a value is readable without the table.
        for xi, total in zip(x, bottom):
            ax.annotate(f'{int(total)}', (xi, total), textcoords='offset points',
                        xytext=(0, 3), ha='center', va='bottom',
                        fontsize=style.TICK_SIZE, color=style.TEXT_MUTED)

        ax.set_xticks(x)
        ax.set_xticklabels([f'{d:g}' for d in doses])
        ax.set_xlim(-0.6, len(doses) - 0.4)
        ax.set_ylim(0, bottom.max() * 1.16 if len(bottom) else 1)

        style.style_axes(ax, grid_axis='y')
        # Titles are left-aligned, so a long name runs into the next panel's
        # title rather than being clipped. Wrap at the hyphen instead.
        name = drug.title()
        if len(name) > 18 and '-' in name:
            name = name.replace('-', '-\n', 1)
        style.label_axes(ax, f'Dose ({unit})', 'Administrations',
                         f'{name}  (n={int(sub["n_admin"].sum())})')

    for ax in axes[len(drugs):]:
        ax.set_visible(False)

    handles = [plt.Rectangle((0, 0), 1, 1, color=colors[r]) for r in routes]
    fig.legend(handles, routes, title='Route', frameon=False,
               fontsize=style.LEGEND_SIZE, title_fontsize=style.LEGEND_SIZE,
               loc='lower right', bbox_to_anchor=(0.995, 0.005), ncols=len(routes))

    fig.suptitle('Doses given, per drug in its own unit\n'
                 f'{int(counts["n_admin"].sum())} administrations, '
                 f'{n_subjects_total} subjects with medication records',
                 fontsize=style.TITLE_SIZE, color=style.TEXT_PRIMARY,
                 x=0.008, ha='left')
    fig.tight_layout(rect=(0, 0.045, 1, 0.94))

    missing_note = (f'; {n_missing_dose} administration(s) had no parseable '
                    f'dose and are absent' if n_missing_dose else '')
    return style.save(
        fig, out_path,
        footnote=('units are NOT comparable across panels — hydrocodone-'
                  'acetaminophen counts TABLETS and its mg strength lives in '
                  'the product name, not in a column\n'
                  'route is a segment because for hydromorphone the dose '
                  'spread IS a route split (IV 0.2-1 mg, oral 1-4 mg)'
                  + missing_note))


def build_parser():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    output.add_output_arguments(parser)
    parser.add_argument('--drugs', nargs='+', default=list(DEFAULT_DRUGS),
                        help='medications to panel, as they appear in the MAR '
                             '(default: the four most-administered analgesics)')
    return parser


def main():
    args = build_parser().parse_args()

    io.warn_if_dirty()

    paths = config.med_admin_files()
    admin_all = load.load_administrations(paths=paths)
    # Validated against the whole corpus first, so a misspelling raises with
    # the available names rather than silently dropping a panel.
    drugs = load.select_drugs(admin_all, drugs=args.drugs)
    admin = admin_all[admin_all['drug'].isin(set(drugs))].copy()
    if admin.empty:
        raise SystemExit(f'no administrations for drugs {drugs}')
    counts, n_missing_dose = dose_counts(admin, drugs)
    if counts.empty:
        raise SystemExit('no administration had a parseable dose')

    logger.info('%d (drug, dose, route) combinations across %d drugs',
                len(counts), len(drugs))

    run_dir = output.resolve_run_dir(args, OUTPUT_TYPE)
    parents = output.source_parents(paths)
    params = vars(args)

    output.write_table(counts, run_dir, 'dose_counts', SCRIPT, params=params,
                       parents=parents,
                       subjects=sorted(admin['subject'].unique()))

    plot_panels(counts, run_dir / 'fig6_dose_distribution.png', drugs,
                n_subjects_total=admin['subject'].nunique(),
                n_missing_dose=n_missing_dose)

    output.write_run(
        run_dir, SCRIPT, args, admin, paths,
        extra={
            'drugs': drugs,
            'n_missing_dose': n_missing_dose,
            'units': counts.groupby('drug')['dose_unit'].first().to_dict(),
            'n_distinct_doses': counts.groupby('drug')['dose'].nunique().to_dict(),
            'unit_note': (
                'doses are never pooled across drugs; hydrocodone-acetaminophen '
                'is in tablets and its mg strength is only in the product name '
                '(DECISIONS 2026-09-03)'),
        },
        description=(f'dose distribution by drug and route, {len(drugs)} drugs, '
                     f'{int(counts["n_admin"].sum())} administrations, '
                     f'n={admin["subject"].nunique()}'))


if __name__ == '__main__':
    main()
