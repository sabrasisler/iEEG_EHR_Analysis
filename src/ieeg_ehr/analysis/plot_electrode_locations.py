#!/usr/bin/env python3
"""
Where the electrodes are: a static glass brain, one dot per bipolar pair, coloured
by the ROI the analysis assigned it.

Answers "which regions is this cohort actually sampling, and how unevenly" -- the
coverage question that sits behind every region row of a heatmap. A region with 11
subjects and one with 55 look identical on a colour scale; here they do not.

WHAT A DOT IS
-------------
One BIPOLAR PAIR, at the MIDPOINT of its two contacts. That midpoint is what the
NWB's MNI_coord_1/2/3 already stores -- verified 2026-07-29 against sub-019, where
contacts LA1 and LA2 average exactly to the stored pair value -- and it is the
right position for a bipolar signal, which is a difference between two contacts and
belongs to neither. It is also what roi_schemes.py describes as the eventual right
basis for region assignment.

There is a known asymmetry worth stating: the dot is drawn at the MIDPOINT, but its
ROI comes from the ANODE's DK label, because that is what the analysis uses. For a
pair spanning a boundary those disagree, and the dot will sit slightly outside the
region whose colour it carries. Fixing that means changing region assignment, not
the figure (see roi_schemes.py's own note).

REGIONS COME FROM roi_schemes, NOT FROM A LOCAL COPY
----------------------------------------------------
`region_for_dk_label` is the single definition, so this figure and the heatmaps
cannot disagree about which region a contact is in. It returns None for anything
outside the scheme's display list -- white matter, CSF, unmatched parcels, and
under roi_v2 the two regions dropped for coverage -- so non-ROI contacts are
excluded by construction rather than by a hand-maintained EXCLUDE list.

NO INTERACTIVE VIEWER. A static PNG, on purpose: it goes in a slide or a paper, and
an HTML viewer is not an artifact that survives in a run directory with provenance.

Run on Slurm, never the login node:
    python -m ieeg_ehr.analysis.plot_electrode_locations --roi-scheme roi_v2
"""

import argparse
import logging
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ieeg_ehr import config, io
from ieeg_ehr.analysis import view_tables
from ieeg_ehr.config import roi_schemes

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

OUTPUT_TYPE = 'electrode_locations'

# A coordinate outside these bounds is a localisation failure, not an electrode:
# MNI152 does not extend this far. Kept from the original script.
MNI_BOUNDS = {'x': (-100, 100), 'y': (-130, 100), 'z': (-80, 120)}


def load_contacts(subjects, session, roi_scheme, epoch_minutes=None):
    """One row per (subject, channel) with its ROI and MNI midpoint.

    DE-DUPLICATED ACROSS RUNS, which is the trap here. channel_meta is keyed
    (run_id, pair_index) because pair ORDER is a per-run property, so the same
    physical electrode appears once PER RUN -- and subjects in this cohort have up
    to 100+ runs. Aggregating without de-duplicating would weight a subject's
    electrodes by how many recordings they happen to have and make the counts in
    the legend meaningless.
    """
    frames = []
    for subject in subjects:
        path = config.pain_epoch_channel_meta_path(subject, session, epoch_minutes)
        if not path.exists():
            logger.warning('sub-%s: no channel_meta at %s, skipping', subject, path)
            continue
        meta = io.read_table(path, on_stale='ignore')
        missing = [c for c in ('mni_x', 'mni_y', 'mni_z') if c not in meta.columns]
        if missing:
            logger.warning('sub-%s: channel_meta lacks %s -- rebuild it '
                           '(views.channel_meta bumped to SCHEMA_VERSION 2)',
                           subject, missing)
            continue
        meta = meta.drop_duplicates('channel').copy()
        meta['subject_id'] = f'sub-{subject}'
        frames.append(meta)

    if not frames:
        raise SystemExit('no channel_meta with coordinates found for any subject')

    contacts = pd.concat(frames, ignore_index=True)
    contacts['region'] = [roi_schemes.region_for_dk_label(lbl, roi_scheme)
                          for lbl in contacts['dk_anode']]
    return contacts


def filter_contacts(contacts, hemisphere='both'):
    """Drop non-ROI, unlocalised and out-of-bounds contacts, LOUDLY.

    Coverage is a confound in this dataset, so every contact removed is counted and
    logged rather than silently vanishing -- the same rule features/common.add_region
    follows.
    """
    n0 = len(contacts)
    counts = {}

    no_region = contacts['region'].isna()
    counts['outside the ROI scheme (white matter, CSF, unmatched)'] = int(no_region.sum())
    contacts = contacts[~no_region]

    coords = contacts[['mni_x', 'mni_y', 'mni_z']].to_numpy(dtype=float)
    unlocalised = ~np.isfinite(coords).all(axis=1)
    counts['no MNI coordinate'] = int(unlocalised.sum())
    contacts = contacts[~unlocalised]

    coords = contacts[['mni_x', 'mni_y', 'mni_z']].to_numpy(dtype=float)
    in_bounds = (
        (coords[:, 0] >= MNI_BOUNDS['x'][0]) & (coords[:, 0] <= MNI_BOUNDS['x'][1]) &
        (coords[:, 1] >= MNI_BOUNDS['y'][0]) & (coords[:, 1] <= MNI_BOUNDS['y'][1]) &
        (coords[:, 2] >= MNI_BOUNDS['z'][0]) & (coords[:, 2] <= MNI_BOUNDS['z'][1]))
    counts['MNI coordinate out of bounds'] = int((~in_bounds).sum())
    contacts = contacts[in_bounds]

    if hemisphere != 'both':
        x = contacts['mni_x'].to_numpy(dtype=float)
        keep = x < 0 if hemisphere == 'left' else x > 0
        counts[f'other hemisphere (kept {hemisphere})'] = int((~keep).sum())
        contacts = contacts[keep]

    for reason, n in counts.items():
        if n:
            logger.info('dropped %5d/%d contact(s): %s', n, n0, reason)
    logger.info('%d contact(s) from %d subject(s) will be plotted',
                len(contacts), contacts['subject_id'].nunique())
    return contacts.reset_index(drop=True), counts


def region_palette(regions):
    """One colour per region, in the scheme's display order.

    glasbey, because these are up to 21 NOMINAL categories and glasbey is built to
    maximise separation over a large set. It cannot be colourblind-safe at this
    count -- no 21-colour categorical palette is -- so this palette is for an
    ANATOMICAL REFERENCE figure with a labelled legend and must not be reused to
    encode data. The pain-level palette (config.PAIN_BIN_COLORS) is the validated
    one, and it has three entries for a reason.
    """
    import colorcet as cc
    import seaborn as sns
    colours = sns.color_palette(cc.glasbey, n_colors=len(regions))
    return {r: matplotlib.colors.to_hex(c) for r, c in zip(regions, colours)}


def plot_glass_brain(contacts, regions, colours, title, out_path, node_size=22):
    """Static glass brain, one plot_markers call per region so each gets its colour.

    plot_markers takes a colormap, not per-point colours, so a single call cannot
    give 21 regions 21 colours. Drawing region by region into the same figure is the
    way round it, and it also fixes the legend order to the scheme's display order
    rather than to whatever order the contacts happened to arrive in.
    """
    from nilearn import plotting

    fig = plt.figure(figsize=(18, 6))
    display = None
    for region in regions:
        rows = contacts[contacts['region'] == region]
        if rows.empty:
            continue
        coords = rows[['mni_x', 'mni_y', 'mni_z']].to_numpy(dtype=float)
        display = plotting.plot_markers(
            node_values=np.ones(len(coords)), node_coords=coords,
            node_size=node_size, display_mode='lyrz', colorbar=False, figure=fig,
            node_cmap=matplotlib.colors.ListedColormap([colours[region]]),
            node_vmin=0, node_vmax=2, alpha=0.85,
        )

    handles = []
    for region in regions:
        rows = contacts[contacts['region'] == region]
        if rows.empty:
            continue
        handles.append(mpatches.Patch(
            facecolor=colours[region], edgecolor='black',
            label=f'{region} ({len(rows)} elec, {rows["subject_id"].nunique()} subj)'))
    fig.legend(handles=handles, loc='center left', bbox_to_anchor=(1.0, 0.5),
               fontsize=9, title='ROI (bipolar pairs, midpoint)', title_fontsize=10,
               frameon=True)
    fig.suptitle(title, fontsize=13)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    if display is not None:
        display.close()
    logger.info('Wrote %s', out_path)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--view-dir', default=None,
                    help='Take the subject list AND the roi_scheme from this view '
                         "run, so the figure shows exactly the analysis's electrodes. "
                         'Overrides --split and --roi-scheme.')
    ap.add_argument('--split', default='discovery')
    ap.add_argument('--roi-scheme', default='roi_v2')
    ap.add_argument('--session', default='01')
    ap.add_argument('--epoch-minutes', type=float, default=None)
    ap.add_argument('--hemisphere', choices=['both', 'left', 'right'], default='both')
    ap.add_argument('--node-size', type=float, default=22)
    ap.add_argument('--run-name', default=None)
    view_tables.add_output_arguments(ap)
    args = ap.parse_args()

    io.warn_if_dirty()

    # Prefer the view run's own subjects and scheme: a coverage figure that does not
    # match the analysis it illustrates is worse than no coverage figure.
    parents, view_params = [], {}
    if args.view_dir:
        _, subject_paths = view_tables.load_view_tables(Path(args.view_dir), 'subject')
        view_params, _ = view_tables.view_params_from(subject_paths)
        subjects = sorted({p.name.split('sub-')[1][:3] for p in subject_paths})
        roi_scheme = view_params.get('roi_scheme') or args.roi_scheme
        parents = [io.parent_ref(p, digest=False) for p in subject_paths]
        logger.info('from view %s: %d subject(s), roi_scheme=%r',
                    args.view_dir, len(subjects), roi_scheme)
    else:
        from ieeg_ehr.config import cohorts
        subjects = cohorts.viewable_subjects(args.split,
                                             minutes_before=args.epoch_minutes)
        roi_scheme = args.roi_scheme
        logger.info('split=%s: %d subject(s), roi_scheme=%r',
                    args.split, len(subjects), roi_scheme)

    regions = roi_schemes.roi_regions(roi_scheme)
    logger.info('roi_scheme %r -> %d region(s): %s', roi_scheme, len(regions), regions)

    contacts = load_contacts(subjects, args.session, roi_scheme, args.epoch_minutes)
    contacts, dropped = filter_contacts(contacts, args.hemisphere)
    colours = region_palette(regions)

    per_region = (contacts.groupby('region')
                  .agg(n_electrodes=('channel', 'size'),
                       n_subjects=('subject_id', 'nunique')))
    logger.info('coverage by region:\n%s',
                per_region.reindex([r for r in regions if r in per_region.index])
                .to_string())

    # There is no ViewConfig here to take a scheme_code from, so the level-4 folder
    # is named after the ROI scheme -- which is the only view axis this figure
    # depends on at all. It shows WHERE electrodes are; normalization and pain
    # binning do not enter into it.
    from ieeg_ehr.views.view_config import ROI_SCHEME_CODES
    if not args.view_scheme:
        args.view_scheme = (ROI_SCHEME_CODES.get(roi_scheme)
                            or Path(str(roi_scheme)).stem.replace('_', '')
                            or 'roidefault')
    run_dir = view_tables.resolve_run_dir(
        args, OUTPUT_TYPE, None,
        run_name=args.run_name or args.split)
    logger.info('run dir: %s', run_dir)

    io.write_table(contacts, run_dir / 'electrode_locations.parquet', kind='table',
                   script='ieeg_ehr/analysis/plot_electrode_locations.py',
                   params={'roi_scheme': roi_scheme, 'session': args.session,
                           'hemisphere': args.hemisphere,
                           'mni_bounds': {k: list(v) for k, v in MNI_BOUNDS.items()}},
                   parents=parents,
                   subjects=sorted(contacts['subject_id'].unique()))

    io.write_run_provenance(
        run_dir, script='ieeg_ehr/analysis/plot_electrode_locations.py',
        params={**vars(args), 'roi_scheme_resolved': roi_scheme,
                'view_params': view_params},
        parents=parents, subjects=sorted(contacts['subject_id'].unique()),
        extra={'roi_regions': regions,
               'roi_scheme_contents': roi_schemes.scheme_provenance(roi_scheme),
               'n_electrodes': int(len(contacts)),
               'n_subjects': int(contacts['subject_id'].nunique()),
               'contacts_dropped': dropped,
               'coordinate_basis':
                   'MNI_coord_1/2/3 from the bipolar NWB electrodes table = the '
                   'MIDPOINT of the pair\'s two contacts (verified 2026-07-29). ROI '
                   'comes from the ANODE DK label, matching the analysis, so a pair '
                   'spanning a boundary is drawn at a midpoint slightly outside the '
                   'region whose colour it carries.',
               'palette_note':
                   'glasbey, for up to 21 nominal categories. NOT colourblind-safe '
                   'at this count and not to be reused for data encoding; it is an '
                   'anatomical reference with a labelled legend.'},
    )

    plot_glass_brain(
        contacts, regions, colours,
        f'Electrode coverage — {contacts["subject_id"].nunique()} subjects, '
        f'{len(contacts)} bipolar pairs, roi_scheme={roi_scheme}'
        + ('' if args.hemisphere == 'both' else f' ({args.hemisphere} hemisphere)'),
        run_dir / 'electrode_locations_glass_brain.png', node_size=args.node_size)

    io.log_analysis(f'electrode coverage glass brain ({roi_scheme}), '
                    f'{len(contacts)} pairs, n={contacts["subject_id"].nunique()}',
                    run_dir)
    logger.info('figure + provenance -> %s', run_dir)


if __name__ == '__main__':
    main()
