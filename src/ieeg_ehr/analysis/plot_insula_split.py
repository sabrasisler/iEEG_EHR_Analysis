#!/usr/bin/env python3
"""Does the coarse anterior/posterior insula split look like anything? Two figures.

    analysis/pain/bandpower/insula_split/<scheme>/<run>_<timestamp>/

This is a LOOK-BEFORE-YOU-MODEL figure. `insula_ap` cuts the cohort's insular
contacts at the median MNI y and calls the anterior half aIns and the posterior
half pIns, which is a coarse enough operation that it deserves to be seen before
anything is fitted on top of it -- the point of the split is to let insula enter
the domain model (aIns to Affective, pIns to Sensory), and a split that puts a
third of the "anterior" dots behind the central sulcus would poison both domains
at once.

TWO FIGURES, because a glass brain alone cannot answer the question:

  fig_insula_split_glass.png   The anatomy: one dot per bipolar pair on the
        nilearn glass brain, coloured by part. This is what "do they look
        reasonable" means -- are the purple dots in front of the red ones, and
        do they both sit where an insula is.
  fig_insula_split_axes.png    The split itself: the y distribution with the
        threshold drawn on it, y-vs-z and y-vs-x scatters, and the per-subject
        breakdown. This is where the coarseness is visible -- how many contacts
        sit within a few mm of the line, and whether a subject contributes to
        one part only.

Nothing is fitted here and nothing downstream reads these PNGs; what downstream
reads is `insula_contacts.csv`, the per-contact assignment, and the THRESHOLD in
provenance.json -- which is what lets the model reproduce this exact split rather
than re-deriving a median from a possibly-different subject list.

Run on Slurm, never the login node:
    python -m ieeg_ehr.analysis.plot_insula_split --subjects-from <domain run dir>
"""

import argparse
import json
import logging
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ieeg_ehr import config, io
from ieeg_ehr.analysis import insula_ap, view_tables

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

SCRIPT = 'ieeg_ehr/analysis/plot_insula_split.py'
OUTPUT_TYPE = 'insula_split'
RUN_NAME = 'insula_ap'


def subjects_from_run(run_dir):
    """The `subjects[]` of an existing run's provenance -- never the folder name.

    Matching the cohort of the model this split is destined for matters more than
    it looks: the threshold is this cohort's median, so a figure drawn over a
    different subject list is a figure of a different split.
    """
    path = Path(run_dir)
    if path.is_dir():
        path = path / 'provenance.json'
    prov = json.loads(path.read_text())
    subjects = prov.get('subjects') or []
    if not subjects:
        raise SystemExit(f'{path} has no subjects[]')
    logger.info('cohort from %s: %d subject(s)', path, len(subjects))
    return sorted(subjects)


# ============================================================================
# FIGURES
# ============================================================================

def glass_figure(contacts, out_path, title, node_size=28):
    """Glass brain, one `plot_markers` call per part so each gets its own colour.

    `plot_markers` takes a colormap rather than per-point colours, so two colours
    means two calls into the same figure. Drawing posterior FIRST and anterior
    second is deliberate: where the two overlap near the threshold the anterior
    dots sit on top, which is the honest way round -- the ambiguity should be
    visible at the front of the insula, where the reader is looking for it.
    """
    from nilearn import plotting

    fig = plt.figure(figsize=(18, 5.5))
    display = None
    for part in (insula_ap.POSTERIOR, insula_ap.ANTERIOR):
        rows = contacts[contacts['ins_part'] == part]
        if rows.empty:
            continue
        coords = rows[['mni_x', 'mni_y', 'mni_z']].to_numpy(dtype=float)
        display = plotting.plot_markers(
            node_values=np.ones(len(coords)), node_coords=coords,
            node_size=node_size, display_mode='lyrz', colorbar=False, figure=fig,
            node_cmap=matplotlib.colors.ListedColormap([insula_ap.COLOURS[part]]),
            node_vmin=0, node_vmax=2, alpha=0.85)

    handles = []
    for part in (insula_ap.ANTERIOR, insula_ap.POSTERIOR):
        rows = contacts[contacts['ins_part'] == part]
        handles.append(mpatches.Patch(
            facecolor=insula_ap.COLOURS[part], edgecolor='black',
            label=f'{insula_ap.DISPLAY[part]} — {len(rows)} pairs, '
                  f'{rows["subject_id"].nunique()} subj'))
    fig.legend(handles=handles, loc='lower center', ncol=2, fontsize=11,
               frameon=True, bbox_to_anchor=(0.5, -0.04))
    fig.suptitle(title, fontsize=13)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    if display is not None:
        display.close()
    logger.info('Wrote %s', out_path)


def axes_figure(contacts, desc, out_path, title):
    """The split as numbers: the cut, the scatter it came from, and who is in it.

    The glass brain shows dots on a brain and is easy to over-read. These four
    panels show the thing the glass brain hides -- that the cut is a plane at one
    y value, that contacts pile up near it, and that a subject can land entirely
    in one part (which is what decides whether the model's insula effect is a
    within- or a between-subject contrast).
    """
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    thr = desc['thresholds']
    pooled = thr.get('all')

    def cut_for(hemi):
        return pooled if pooled is not None else thr.get(hemi)

    # -- (0,0) y histogram with the threshold -------------------------------
    ax = axes[0, 0]
    bins = np.histogram_bin_edges(contacts['mni_y'].to_numpy(dtype=float), bins=30)
    for part in (insula_ap.POSTERIOR, insula_ap.ANTERIOR):
        rows = contacts[contacts['ins_part'] == part]
        ax.hist(rows['mni_y'], bins=bins, color=insula_ap.COLOURS[part],
                alpha=0.8, label=insula_ap.DISPLAY[part])
    for hemi, cut in ([('all', pooled)] if pooled is not None else thr.items()):
        ax.axvline(cut, color='k', ls='--', lw=1.5,
                   label=f'cut {hemi}: y = {cut:+.1f} mm')
    ax.set_xlabel('MNI y (mm, + = anterior)')
    ax.set_ylabel('bipolar pairs')
    ax.set_title('The split, on the axis it was taken on')
    ax.legend(fontsize=8)

    # -- (0,1) sagittal y-z -------------------------------------------------
    ax = axes[0, 1]
    for part in (insula_ap.POSTERIOR, insula_ap.ANTERIOR):
        rows = contacts[contacts['ins_part'] == part]
        ax.scatter(rows['mni_y'], rows['mni_z'], s=14, alpha=0.7,
                   color=insula_ap.COLOURS[part], edgecolor='none',
                   label=insula_ap.DISPLAY[part])
    if pooled is not None:
        ax.axvline(pooled, color='k', ls='--', lw=1.5)
    ax.set_xlabel('MNI y (mm)')
    ax.set_ylabel('MNI z (mm)')
    ax.set_title('Sagittal view (both hemispheres overlaid)')
    ax.legend(fontsize=8)

    # -- (1,0) axial x-y, hemispheres visible -------------------------------
    ax = axes[1, 0]
    for part in (insula_ap.POSTERIOR, insula_ap.ANTERIOR):
        rows = contacts[contacts['ins_part'] == part]
        ax.scatter(rows['mni_x'], rows['mni_y'], s=14, alpha=0.7,
                   color=insula_ap.COLOURS[part], edgecolor='none')
    if pooled is not None:
        ax.axhline(pooled, color='k', ls='--', lw=1.5)
    else:
        for hemi, cut in thr.items():
            seg = [-100, 0] if hemi == 'L' else [0, 100]
            ax.plot(seg, [cut, cut], color='k', ls='--', lw=1.5)
    ax.axvline(0, color='0.7', lw=0.8)
    ax.set_xlabel('MNI x (mm, − = left)')
    ax.set_ylabel('MNI y (mm)')
    ax.set_title('Axial view — the cut is a plane in y')

    # -- (1,1) per-subject composition --------------------------------------
    # Sorted by anterior FRACTION, so the subjects that contribute to only one
    # part sit at the two ends and are countable at a glance. A subject entirely
    # in one part cannot contribute a within-subject aIns/pIns contrast.
    ax = axes[1, 1]
    comp = (contacts.pivot_table(index='subject_id', columns='ins_part',
                                 values='channel', aggfunc='size')
            .fillna(0))
    for part in (insula_ap.ANTERIOR, insula_ap.POSTERIOR):
        if part not in comp.columns:
            comp[part] = 0
    comp['frac_ant'] = comp[insula_ap.ANTERIOR] / comp.sum(axis=1)
    comp = comp.sort_values('frac_ant')
    y = np.arange(len(comp))
    ax.barh(y, comp[insula_ap.POSTERIOR], color=insula_ap.COLOURS[insula_ap.POSTERIOR],
            label=insula_ap.DISPLAY[insula_ap.POSTERIOR])
    ax.barh(y, comp[insula_ap.ANTERIOR], left=comp[insula_ap.POSTERIOR],
            color=insula_ap.COLOURS[insula_ap.ANTERIOR],
            label=insula_ap.DISPLAY[insula_ap.ANTERIOR])
    ax.set_yticks(y)
    ax.set_yticklabels([s.replace('sub-', '') for s in comp.index], fontsize=6)
    ax.set_xlabel('bipolar pairs')
    n_both = int(((comp[insula_ap.ANTERIOR] > 0)
                  & (comp[insula_ap.POSTERIOR] > 0)).sum())
    ax.set_title(f'Per subject — {n_both}/{len(comp)} have BOTH parts')
    ax.legend(fontsize=8, loc='lower right')

    fig.suptitle(title, fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    logger.info('Wrote %s', out_path)


# ============================================================================

def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--subjects-from', default=None,
                    help='Run directory (or provenance.json) whose subjects[] is '
                         'the cohort. Use the domain-model run this split is '
                         'destined for, so the median is taken over the same '
                         'electrodes the model will see.')
    ap.add_argument('--split', default='discovery',
                    help='Cohort split, used only when --subjects-from is absent.')
    ap.add_argument('--session', default=None,
                    help='Pin one session. Default: every session a subject '
                         'has, de-duplicated on channel -- three subjects have '
                         'a second one and the split has no session in its key.')
    ap.add_argument('--epoch-minutes', type=float, default=None)
    ap.add_argument('--axis', default=insula_ap.SPLIT_AXIS,
                    choices=['mni_x', 'mni_y', 'mni_z'])
    ap.add_argument('--per-hemisphere', action='store_true',
                    help='One median per hemisphere instead of one pooled '
                         'median. Balances the two parts within each side at the '
                         'cost of two different anatomical cuts.')
    ap.add_argument('--threshold', type=float, default=None,
                    help='Pin the cut instead of computing the median, to '
                         'reproduce an earlier run exactly.')
    ap.add_argument('--node-size', type=float, default=28)
    ap.add_argument('--run-name', default=RUN_NAME)
    view_tables.add_output_arguments(ap, question='bandpower')
    args = ap.parse_args()

    io.warn_if_dirty()

    if args.subjects_from:
        subjects = subjects_from_run(args.subjects_from)
    else:
        from ieeg_ehr.config import cohorts
        subjects = ['sub-' + s.replace('sub-', '') for s in
                    cohorts.viewable_subjects(args.split,
                                              minutes_before=args.epoch_minutes)]
        logger.info('split=%s: %d subject(s)', args.split, len(subjects))

    contacts, counts = insula_ap.load_insula_contacts(
        subjects, session=args.session, epoch_minutes=args.epoch_minutes)
    contacts, desc = insula_ap.median_split(
        contacts, axis=args.axis, per_hemisphere=args.per_hemisphere,
        threshold=args.threshold)

    logger.info('by hemisphere:\n%s',
                contacts.groupby(['hemisphere', 'ins_part'])
                .agg(n_contacts=('channel', 'size'),
                     n_subjects=('subject_id', 'nunique')).to_string())

    if not args.view_scheme:
        args.view_scheme = ('insulamedian'
                            + ('hemi' if args.per_hemisphere else '')
                            + args.axis.replace('mni_', ''))
    run_dir = view_tables.resolve_run_dir(args, OUTPUT_TYPE, None,
                                          run_name=args.run_name)
    logger.info('run dir: %s', run_dir)

    # CSV, not parquet: it is small, terminal, and read by eye (io_conventions).
    io.write_table(contacts, run_dir / 'insula_contacts.csv', kind='table',
                   script=SCRIPT,
                   params={'split': desc, 'session': args.session,
                           'contact_counts': counts},
                   subjects=sorted(contacts['subject_id'].unique()))

    io.write_run_provenance(
        run_dir, script=SCRIPT,
        params={**vars(args), 'split': desc, 'contact_counts': counts},
        subjects=sorted(contacts['subject_id'].unique()),
        extra={
            'purpose':
                'COARSE anterior/posterior insula split, for review BEFORE the '
                'domain model uses it. aIns is destined for the Affective '
                'domain and pIns for the Sensory one.',
            'coordinate_basis':
                "MNI_coord_1/2/3 from the bipolar NWB electrodes table = the "
                "MIDPOINT of the pair's two contacts. The insula membership test "
                "is on the ANODE's DK label, matching every other region "
                'assignment in the project.',
            'caveat':
                'The threshold is this cohort\'s MEDIAN, not an anatomical '
                'landmark, so it is a 50/50 split of these electrodes and would '
                'move with different coverage. The insula is folded and slanted, '
                'so a plane normal to y is not its anterior-posterior axis and '
                'contacts near the line are near-arbitrary. This supports "on '
                'average more anterior", not "this contact is in aIns". The '
                'anatomically correct fix is the Destrieux/a2009s assignment.',
            'reproduce':
                f'pass --threshold {desc["thresholds"]} to pin this exact cut.',
        })

    n_sub = contacts['subject_id'].nunique()
    thr_txt = ', '.join(f'{k}: y={v:+.1f}' for k, v in desc['thresholds'].items())
    title = (f'Insula A/P median split — {len(contacts)} bipolar pairs, '
             f'{n_sub} subjects, cut at {thr_txt} mm')
    glass_figure(contacts, run_dir / 'fig_insula_split_glass.png', title,
                 node_size=args.node_size)
    axes_figure(contacts, desc, run_dir / 'fig_insula_split_axes.png', title)

    io.log_analysis(f'insula anterior/posterior median split ({thr_txt}), '
                    f'{len(contacts)} pairs, n={n_sub} — REVIEW FIGURE, nothing '
                    'fitted', run_dir)
    logger.info('figures + provenance -> %s', run_dir)


if __name__ == '__main__':
    main()
