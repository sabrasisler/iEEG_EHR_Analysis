"""
Fig 1 — administration burden across analgesics.

A prevalence-vs-frequency scatter: x = total administrations, y = total subjects
who received the drug, one marker per drug, coloured by Level 2 subclass. Its
purpose is not to be interpreted on its own but to decide which drugs have enough
data to carry Figures 2-4, so the COMPANION TABLE is the real output — per drug:
administrations, distinct subjects, route breakdown, and median/IQR/modal dose in
that drug's own unit.

WHY-DIFFERENT from the source version, which plots one marker per drug-ROUTE
formulation with two legends (colour = drug, shape = route): route is a table
column here, not a plot encoding. Splitting by route at this stage triples the
marker count to answer a question the table answers better, and the drug-route
split is exactly what Figs 2 and 4 exist to show.

Dose is summarized within (drug, route) and never pooled across drugs — 516 of
1,754 analgesic administrations are dosed in `tablet` and fentanyl is in `mcg`,
so a pooled median dose would be a number with no unit. `load.assert_single_unit`
enforces it rather than trusting it.

Run on Slurm, never the login node:
    python -m ieeg_ehr.med_analysis.plot_admin_burden
"""

import logging

import pandas as pd

from ieeg_ehr import config, io
from ieeg_ehr.config import med_taxonomy
from ieeg_ehr.med_analysis import load, output, style
from ieeg_ehr.med_analysis.style import plt

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

OUTPUT_TYPE = 'burden_scatter'
SCRIPT = 'ieeg_ehr/med_analysis/plot_admin_burden.py'


def drug_summary(admin_df):
    """One row per drug: counts, route breakdown, dose summary in native units."""
    rows = []
    for drug, g in admin_df.groupby('drug'):
        route_counts = g['route'].value_counts()
        route_str = ', '.join(f'{r}: {n}' for r, n in route_counts.items())

        # Dose is summarized per (drug, route) because a drug given orally and IV
        # is often dosed in different units; the modal route's unit is what the
        # summary columns report, and the column says which.
        modal_route = route_counts.index[0]
        sub = g[g['route'] == modal_route]
        unit = load.assert_single_unit(sub, f'{drug} / {modal_route}')
        doses = sub['dose'].dropna()

        rows.append({
            'drug': drug,
            'level1': g['level1'].iloc[0],
            'level2': g['level2'].iloc[0],
            'is_combination': bool(g['is_combination'].iloc[0]),
            'n_admin': len(g),
            'n_subjects': g['subject'].nunique(),
            'n_sessions': g.groupby(['subject', 'session']).ngroups,
            'n_routes': route_counts.size,
            'route_breakdown': route_str,
            'dose_summary_route': modal_route,
            'dose_unit': unit,
            'dose_median': doses.median() if len(doses) else None,
            'dose_q1': doses.quantile(0.25) if len(doses) else None,
            'dose_q3': doses.quantile(0.75) if len(doses) else None,
            'dose_mode': doses.mode().iloc[0] if len(doses) else None,
            'admin_per_subject': len(g) / g['subject'].nunique(),
        })

    return (pd.DataFrame(rows).sort_values('n_admin', ascending=False)
            .reset_index(drop=True))


def subclass_summary(admin_df):
    """One row per Level 2 subclass. Subjects are a SET UNION, not a sum.

    Summing per-drug subject counts would double-count everyone who received two
    opioids, which is most of the opioid cohort.
    """
    rows = []
    for level2, g in admin_df.groupby('level2'):
        rows.append({'level2': level2,
                     'level1': g['level1'].iloc[0],
                     'n_admin': len(g),
                     'n_subjects': g['subject'].nunique(),
                     'n_drugs': g['drug'].nunique()})
    return (pd.DataFrame(rows).sort_values('n_admin', ascending=False)
            .reset_index(drop=True))


def plot_scatter(summary, out_path, n_subjects_total, label_top=6):
    subclasses = [s for s in med_taxonomy.ANALGESIC_SUBCLASS_ORDER
                  if s in set(summary['level2'])]
    subclasses += [s for s in summary['level2'].unique() if s not in subclasses]
    colors = style.categorical_colors(subclasses)

    fig, ax = plt.subplots(figsize=(9, 7))

    for level2 in subclasses:
        sub = summary[summary['level2'] == level2]
        ax.scatter(sub['n_admin'], sub['n_subjects'], s=170,
                   color=colors[level2], edgecolor='white', linewidth=0.8,
                   label=level2, zorder=3)

    # Label the drugs that carry the analysis; labelling all of them turns the
    # panel into a word cloud. Offsets alternate above/below because the top
    # drugs cluster in the same corner and a fixed offset stacks their labels on
    # top of each other.
    labelled = summary.head(label_top)
    for i, r in enumerate(labelled.itertuples()):
        above = i % 2 == 0
        ax.annotate(r.drug.title(), (r.n_admin, r.n_subjects),
                    textcoords='offset points',
                    xytext=(8, 7) if above else (8, -13),
                    va='bottom' if above else 'top',
                    fontsize=style.TICK_SIZE, color=style.TEXT_PRIMARY)

    # A drug on this line was given once to each subject who got it; drugs far
    # below it are repeat-dosed in a few people, which is the fragility Figs 2-4
    # have to survive.
    lim = max(summary['n_admin'].max(), summary['n_subjects'].max()) * 1.15
    ax.plot([0, lim], [0, lim], color=style.AXIS_COLOR, linewidth=1,
            linestyle='--', zorder=1)
    ax.annotate('1 administration per subject', (lim * 0.55, lim * 0.58),
                fontsize=style.FOOTNOTE_SIZE + 1, color=style.TEXT_MUTED,
                rotation=38, ha='center')

    ax.set_xscale('symlog', linthresh=10)
    ax.set_yscale('symlog', linthresh=10)
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)

    style.style_axes(ax, grid_axis='both')
    style.label_axes(ax, 'Total administrations', 'Total subjects',
                     'Analgesic administration burden\n'
                     f'{len(summary)} drugs, {n_subjects_total} subjects with '
                     'medication records')
    leg = ax.legend(title='Subclass', fontsize=style.LEGEND_SIZE,
                    title_fontsize=style.LEGEND_SIZE, frameon=False,
                    loc='lower right')
    leg._legend_box.align = 'left'

    fig.tight_layout()
    return style.save(fig, out_path,
                      footnote='log-log axes; dashed line = 1 administration per subject')


def main():
    parser = output.build_parser(__doc__)
    args = parser.parse_args()

    io.warn_if_dirty()

    paths = config.med_admin_files()
    admin = load.load_administrations(paths=paths, subclasses=args.subclasses)
    if admin.empty:
        raise SystemExit(f'no administrations for subclasses {args.subclasses}')

    summary = drug_summary(admin)
    by_subclass = subclass_summary(admin)
    logger.info('%d drugs, %d administrations, %d subjects',
                len(summary), len(admin), admin['subject'].nunique())

    run_dir = output.resolve_run_dir(args, OUTPUT_TYPE)
    parents = output.source_parents(paths)

    # This table, not the scatter, is what selects the drug set for Figs 2-4.
    output.write_table(summary, run_dir, 'drug_summary', SCRIPT,
                       params=vars(args), parents=parents,
                       subjects=sorted(admin['subject'].unique()))
    output.write_table(by_subclass, run_dir, 'subclass_summary', SCRIPT,
                       params=vars(args), parents=parents)
    output.write_table(load.drug_route_counts(admin), run_dir,
                       'drug_route_counts', SCRIPT, params=vars(args),
                       parents=parents)

    plot_scatter(summary, run_dir / 'fig1_admin_burden.png',
                 n_subjects_total=admin['subject'].nunique())

    output.write_run(
        run_dir, SCRIPT, args, admin, paths,
        extra={
            'drugs': summary['drug'].tolist(),
            'n_drugs': int(len(summary)),
            'subclass_counts': by_subclass.set_index('level2')['n_admin'].to_dict(),
            'anesthetics_note': (
                'Anesthetics are excluded: the MAR export does not capture '
                'procedural medication (1 propofol administration, 3 rocuronium, '
                '21 lidocaine, no ketamine/dexmedetomidine, and no row with an '
                'infusion_rate across all 98 sessions).'),
        },
        description=(f'analgesic administration burden, {len(summary)} drugs, '
                     f'{len(admin)} administrations, '
                     f'n={admin["subject"].nunique()}'))


if __name__ == '__main__':
    main()
