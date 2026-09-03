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
import math

import pandas as pd
from matplotlib import ticker

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


#: Marker area in pt^2. Shared with `style.label_points` so the label placer
#: knows how big an obstacle each marker is; the two must not drift apart.
MARKER_SIZE = 170


def plot_scatter(summary, out_path, n_subjects_total):
    subclasses = [s for s in med_taxonomy.ANALGESIC_SUBCLASS_ORDER
                  if s in set(summary['level2'])]
    subclasses += [s for s in summary['level2'].unique() if s not in subclasses]
    colors = style.categorical_colors(subclasses)

    fig, ax = plt.subplots(figsize=(9, 7))

    for level2 in subclasses:
        sub = summary[summary['level2'] == level2]
        ax.scatter(sub['n_admin'], sub['n_subjects'], s=MARKER_SIZE,
                   color=colors[level2], edgecolor='white', linewidth=0.8,
                   label=level2, zorder=3)

    # LINEAR axes. The previous symlog was unreadable for two reasons: with
    # `linthresh=10` position meant one thing below 10 administrations and
    # another above it, and a single shared limit driven by x's maximum (643)
    # left everything above y=73 as empty panel, because no drug can have more
    # subjects than administrations.
    #
    # The cost of linear is real and worth stating: 7 of the 12 drugs sit under
    # 50 administrations, so they crowd into the leftmost tenth of the x axis
    # and their positions stop being separable. Those drugs are carried by their
    # labels and by drug_summary.csv, not by their coordinates. What linear buys
    # is that a distance means the same thing everywhere on the axis, so ratios
    # between the drugs that DO carry Figs 2-4 can be read straight off it.
    x_hi = summary['n_admin'].max() * 1.12
    y_hi = summary['n_subjects'].max() * 1.12
    ax.set_xlim(0, x_hi)
    ax.set_ylim(0, y_hi)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(100))
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(25))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(10))
    ax.yaxis.set_minor_locator(ticker.MultipleLocator(5))

    # A drug on this line was given once to each subject who got it; drugs far
    # below it are repeat-dosed in a few people, which is the fragility Figs 2-4
    # have to survive. It stops at the top of the y axis rather than the corner
    # of the panel, since the two axes no longer share a range.
    diag_hi = min(x_hi, y_hi)
    ax.plot([0, diag_hi], [0, diag_hi], color=style.AXIS_COLOR, linewidth=1,
            linestyle='--', zorder=1)

    style.style_axes(ax, grid_axis='both')
    ax.grid(True, which='minor', color=style.GRID_COLOR, linewidth=0.4,
            alpha=0.55, zorder=0)
    style.label_axes(ax, 'Total administrations', 'Total subjects',
                     'Analgesic administration burden\n'
                     f'{len(summary)} drugs, {n_subjects_total} subjects with '
                     'medication records')
    leg = ax.legend(title='Subclass', fontsize=style.LEGEND_SIZE,
                    title_fontsize=style.LEGEND_SIZE, frameon=False,
                    loc='lower right')
    leg._legend_box.align = 'left'

    # Labels are placed by measuring text boxes in display space, so the layout
    # has to be final before they go on — hence tight_layout here rather than
    # after. The legend is passed in as an obstacle because the low-n drugs sit
    # in the same corner it occupies.
    fig.tight_layout()
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()

    # The on-screen angle of y=x depends on the axes box and on the two ranges,
    # which are no longer equal, so the rotation is measured from the transform
    # instead of hardcoded — and measured here, because tight_layout is what
    # settles that box.
    p0 = ax.transData.transform((0, 0))
    p1 = ax.transData.transform((diag_hi, diag_hi))
    diag_label = ax.annotate(
        '1 administration per subject', (diag_hi * 0.62, diag_hi * 0.62),
        textcoords='offset points', xytext=(0, 4),
        rotation=math.degrees(math.atan2(p1[1] - p0[1], p1[0] - p0[0])),
        rotation_mode='anchor', ha='center', va='bottom',
        fontsize=style.FOOTNOTE_SIZE + 1, color=style.TEXT_MUTED)

    obstacles = [leg.get_window_extent(renderer),
                 diag_label.get_window_extent(renderer)]

    # EVERY drug gets a name. This panel's whole job is to say which drugs have
    # enough data to carry Figs 2-4, and an unlabelled marker cannot answer
    # that — the reader has to join it to the companion table by eye. `summary`
    # is sorted by administrations, so the drugs that will carry those figures
    # are placed first and get the cleanest spots.
    style.label_points(ax, summary['n_admin'].tolist(),
                       summary['n_subjects'].tolist(),
                       [d.title() for d in summary['drug']],
                       marker_size=MARKER_SIZE, obstacles=obstacles)

    return style.save(fig, out_path,
                      footnote='linear axes; dashed line = 1 administration per subject')


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
