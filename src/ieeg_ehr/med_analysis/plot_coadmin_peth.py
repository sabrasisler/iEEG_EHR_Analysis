"""
Fig 4 — co-administration peri-event time histograms.

A grid of small PETH panels. Rows are the top analgesic drug-route formulations,
each labelled with the formulation and its administration count. Columns are
co-administered medication classes: opioids, acetaminophen, NSAIDs,
anticonvulsants, benzodiazepines, gabapentinoids, antiemetics.

Within each panel: x = time relative to the index administration, -5.5 to +5.5 h
in 1 h bins centred on integers; y = the FRACTION of index doses with at least
one co-administration of that class in that bin, on a fixed 0-1 axis.

The counting rule is the thing to preserve. Each anchor contributes a BINARY
indicator per class per bin: two anticonvulsants alongside one oxycodone dose
count once, not twice. The y-axis is therefore "what fraction of doses had a
co-administration", not "how many co-administrations" — those diverge sharply for
drugs given in clusters, and only the first is bounded and comparable across rows
with different administration counts. A tall bar in the zero bin means the two
drugs are handed out together at the same medication pass.

Time-locking happens within one (subject, session) at a time, because
`taken_date` is only comparable inside a session. The anchor row is excluded from
its own class count.

COMBINATION PRODUCTS. Hydrocodone-acetaminophen and oxycodone-acetaminophen would
co-occur with acetaminophen in the zero bin 100% of the time by definition, which
is a property of the tablet rather than a prescribing pattern. The taxonomy
already prevents it — those products classify as Opioids, not Acetaminophen — so
the acetaminophen column is single-ingredient by construction. The column is
labelled to say so, and `_assert_no_combination_leak` checks it rather than
trusting it.

Keeping an opioid column against opioid rows is deliberate: it is how
breakthrough dosing on top of a scheduled opioid shows up.

Run on Slurm, never the login node:
    python -m ieeg_ehr.med_analysis.plot_coadmin_peth
"""

import logging
import textwrap

import numpy as np
import pandas as pd

from ieeg_ehr import config, io
from ieeg_ehr.config import med_taxonomy
from ieeg_ehr.med_analysis import load, output, style
from ieeg_ehr.med_analysis.style import plt

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

OUTPUT_TYPE = 'coadmin_peth'
SCRIPT = 'ieeg_ehr/med_analysis/plot_coadmin_peth.py'

WINDOW_HOURS = 5.5
BIN_EDGES = np.arange(-WINDOW_HOURS, WINDOW_HOURS + 1, 1)
BIN_CENTERS = (BIN_EDGES[:-1] + BIN_EDGES[1:]) / 2.0


def _assert_no_combination_leak(all_admin, classes):
    """No combination product may contribute to the acetaminophen column.

    Cheap, and the failure it guards against is invisible in the figure: a 100%
    zero-bin bar looks like a striking result rather than a tautology.
    """
    if 'Acetaminophen' not in classes:
        return
    leaked = all_admin[(all_admin['coadmin_class'] == 'Acetaminophen')
                       & all_admin['is_combination']]
    if len(leaked):
        raise AssertionError(
            f'{len(leaked)} combination-product administration(s) '
            f'({sorted(set(leaked["drug"]))}) are classified into the '
            f'Acetaminophen column. They would co-occur with themselves in the '
            f'zero bin by definition. Fix the class assignment in '
            f'config/med_taxonomy.py.')


def aggregate(all_admin, formulations, classes):
    """(anchor counts, fraction-of-doses histograms).

    Returns `counts[label]` and `hist[label][cls] -> array over BIN_CENTERS` of
    the number of anchors with >=1 co-administration of that class in that bin.
    """
    anchor_counts = {label: 0 for _d, _r, label in formulations}
    hist = {label: {cls: np.zeros(len(BIN_CENTERS)) for cls in classes}
            for _d, _r, label in formulations}

    for _key, session_rows in all_admin.groupby(['subject', 'session'], sort=False):
        # Only tracked-class rows can be co-administrations. Anchors are matched
        # against the whole session, so an anchor whose own class is untracked
        # would still count — the asymmetry is deliberate.
        others = session_rows[session_rows['coadmin_class'].notna()]
        other_times = others['taken_dt'].to_numpy()
        other_class = others['coadmin_class'].to_numpy()
        other_idx = others.index.to_numpy()

        for drug, route, label in formulations:
            anchors = session_rows[(session_rows['drug'] == drug)
                                   & (session_rows['route'] == route)]
            for anchor in anchors.itertuples():
                anchor_counts[label] += 1
                delta_h = ((other_times - np.datetime64(anchor.taken_dt))
                           / np.timedelta64(1, 'h'))
                in_window = (np.abs(delta_h) <= WINDOW_HOURS) & (other_idx != anchor.Index)
                if not in_window.any():
                    continue
                d_sel = delta_h[in_window]
                c_sel = other_class[in_window]
                for cls in classes:
                    mask = c_sel == cls
                    if not mask.any():
                        continue
                    counts, _ = np.histogram(d_sel[mask], bins=BIN_EDGES)
                    # Binary per anchor per bin — see module docstring.
                    hist[label][cls] += (counts > 0).astype(float)

    return anchor_counts, hist


def _shared_ylim(anchor_counts, hist, formulations, classes):
    """One y-limit for every panel, taken from the data rather than fixed at 1.

    The source analysis pins the axis to 0-1 because a fraction cannot exceed 1.
    True, but co-administration fractions here top out near 0.4, so a fixed axis
    spends three quarters of every panel on empty space. A SHARED limit keeps
    panels comparable — which is the only thing the fixed axis was buying — while
    making the bars legible. Rounded up to a quarter so the ticks stay readable.
    """
    peak = max((hist[label][cls] / anchor_counts[label]).max()
               for _d, _r, label in formulations if anchor_counts[label]
               for cls in classes)
    return min(1.0, max(0.25, np.ceil(peak * 4) / 4))


def plot_peth(anchor_counts, hist, formulations, classes, out_path):
    nrows, ncols = len(formulations), len(classes)
    fig, axes = plt.subplots(nrows, ncols, squeeze=False, sharex=True, sharey=True,
                             figsize=(2.5 * ncols, 2.15 * nrows))
    ylim = _shared_ylim(anchor_counts, hist, formulations, classes)

    for row, (_drug, _route, label) in enumerate(formulations):
        n_anchors = anchor_counts[label]
        for col, cls in enumerate(classes):
            ax = axes[row][col]
            counts = hist[label][cls]
            frac = counts / n_anchors if n_anchors else counts
            ax.bar(BIN_CENTERS, frac, width=1.0, color=style.BAR_COLOR, zorder=3)
            ax.axvline(0, color=style.ZERO_LINE_COLOR, linewidth=1,
                       linestyle='--', zorder=2)
            style.style_axes(ax)
            if row == 0:
                ax.set_title(med_taxonomy.COADMIN_CLASS_LABELS.get(cls, cls),
                             fontsize=style.LEGEND_SIZE, color=style.TEXT_PRIMARY,
                             pad=6)
            if col == 0:
                # Wrapped: "PO Hydrocodone-Acetaminophen" on one line is taller
                # than the row and collides with its neighbours.
                wrapped = textwrap.fill(label, width=16)
                ax.set_ylabel(f'{wrapped}\n(n={n_anchors:,})',
                              fontsize=style.LEGEND_SIZE - 1,
                              color=style.TEXT_PRIMARY)
            if row == nrows - 1:
                ax.set_xlabel('Hours from dose', fontsize=style.LEGEND_SIZE,
                              color=style.TEXT_PRIMARY)

    axes[0][0].set_ylim(0, ylim)
    axes[0][0].set_xlim(-WINDOW_HOURS, WINDOW_HOURS)
    fig.suptitle('Co-administration around analgesic doses\n'
                 'fraction of index doses with ≥1 co-administration',
                 fontsize=style.TITLE_SIZE + 1, x=0.01, ha='left')
    fig.tight_layout(rect=(0, 0.02, 1, 0.96))
    return style.save(
        fig, out_path,
        footnote=(f'1-hour bins, ±5.5 h; shared y-axis 0-{ylim:g}. Each dose counts '
                  'once per class per bin. Index dose excluded from its own class '
                  '(hence the zero-bin dip in the opioid column for opioid rows). '
                  'Acetaminophen column is single-ingredient only.'))


def main():
    parser = output.build_parser(__doc__)
    parser.add_argument('--max-rows', type=int, default=5,
                        help='drug-route formulations to use as rows')
    args = parser.parse_args()

    io.warn_if_dirty()

    paths = config.med_admin_files()

    # The FULL table, not just analgesics: the columns are other drug classes, so
    # filtering to analgesics up front would empty most of the grid.
    all_admin = load.load_administrations(paths=paths)
    all_admin['coadmin_class'] = [
        med_taxonomy.coadmin_class(l1, l2)
        for l1, l2 in zip(all_admin['level1'], all_admin['level2'])]

    classes = [c for c in med_taxonomy.COADMIN_CLASS_ORDER
               if (all_admin['coadmin_class'] == c).any()]
    _assert_no_combination_leak(all_admin, classes)

    analgesics = all_admin[all_admin['level2'].isin(args.subclasses)]
    formulations = load.top_formulations(analgesics, n=args.max_rows,
                                         min_admin=args.min_admin)
    if not formulations:
        raise SystemExit(f'no formulation reaches --min-admin={args.min_admin}')

    logger.info('%d rows x %d classes over %d administrations',
                len(formulations), len(classes), len(all_admin))
    anchor_counts, hist = aggregate(all_admin, formulations, classes)

    run_dir = output.resolve_run_dir(args, OUTPUT_TYPE)
    parents = output.source_parents(paths)
    subjects = sorted(analgesics['subject'].unique())

    plot_peth(anchor_counts, hist, formulations, classes,
              run_dir / 'fig4_coadmin_peth.png')

    tidy = [{'formulation': label, 'coadmin_class': cls,
             'bin_center_h': float(center), 'n_anchors': anchor_counts[label],
             'n_anchors_with_coadmin': float(hist[label][cls][i]),
             'fraction': (float(hist[label][cls][i]) / anchor_counts[label]
                          if anchor_counts[label] else np.nan)}
            for _d, _r, label in formulations
            for cls in classes
            for i, center in enumerate(BIN_CENTERS)]
    output.write_table(pd.DataFrame(tidy), run_dir, 'coadmin_peth', SCRIPT,
                       params=vars(args), parents=parents, subjects=subjects)

    output.write_run(
        run_dir, SCRIPT, args, analgesics, paths,
        extra={
            'formulations': [f'{d}|{r}' for d, r, _ in formulations],
            'anchor_counts': {k: int(v) for k, v in anchor_counts.items()},
            'coadmin_classes': classes,
            'window_hours': WINDOW_HOURS,
            'counting_rule': ('binary indicator per anchor per class per bin; y is '
                              'the fraction of index doses with >=1 '
                              'co-administration, not a count'),
            'combination_note': ('Combination opioid-acetaminophen products '
                                 'classify as Opioids, so the Acetaminophen column '
                                 'is single-ingredient only; asserted, not assumed.'),
        },
        description=(f'analgesic co-administration PETHs, {len(formulations)} '
                     f'formulations x {len(classes)} classes, n={len(subjects)}'))


if __name__ == '__main__':
    main()
