#!/usr/bin/env python3
"""
How many subjects and how many electrodes carry each atlas label.

    analysis/electrode_localization/<atlas>_coverage/label_counts/<run>_<timestamp>/

Two horizontal bar panels on one shared label axis: subjects per label (left)
and bipolar pairs per label (right). EVERY label in the cohort appears,
including white matter, ventricles and unlabeled pairs, because the point is to
see the raw atlas coverage before any ROI scheme decides what to keep.

One electrode = one BIPOLAR PAIR. channel_meta repeats each pair once per run,
so pairs are de-duplicated on (subject, session, channel) before counting.
Hemispheres are merged in every atlas.

`--atlas dk`
    The pair's ANODE Desikan-Killiany label (`dk_anode`), which is how the
    analysis assigns regions. Cortical DK labels in this dataset mostly carry no
    hemisphere (`insula`); subcortical and white-matter labels carry
    `Left-`/`Right-`, and a few cortical ones `ctx-lh-`/`ctx-rh-`. Stripping
    those four prefixes merges hemispheres. No other spelling is merged:
    `Unknown` and `undefined` stay two labels, as they are two strings.

`--atlas dk_midpoint`
    Desikan-Killiany again, but looked up at the VIRTUAL ELECTRODE (see
    `hcpex` below) in FreeSurfer 7.4.1's `cvs_avg35_inMNI152` aparc+aseg: the
    same DK cortical parcels plus the aseg subcortical and white-matter labels,
    as a 1 mm MNI152 volume. Hemispheres merge by the same prefix strip as `dk`.
    FreeSurfer 7 names the thalamus `Thalamus` where the NWB anode labels say
    `Thalamus-Proper`. A midpoint on the grid with value 0 is outside the brain
    mask and counts as `No parcel`.

`--atlas hcpex`
    HCPex v1.1 (Huang et al. 2022, doi 10.1007/s00429-021-02421-6), 1 mm, MNI
    ICBM152 2009c asymmetric, looked up at the VIRTUAL ELECTRODE: the midpoint of
    the anode and cathode contacts' MNI coordinates, read per contact from the
    raw NWB electrodes table. The midpoint is checked against the bipolar NWB's
    stored pair coordinate (channel_meta `mni_x/y/z`), and any disagreement
    stops the run, because it means a contact was paired wrongly. A midpoint
    outside every HCPex parcel (most often white matter) counts as `Unlabeled`.
    The electrode coordinates' MNI template is not documented in the NWBs, so a
    pair near a parcel border can shift by a template difference of ~1-2 mm.

`--atlas cabnp`
    The 12 networks of the Cole-Anticevic Brain-wide Network Partition v1.1
    (Ji et al. 2019, doi 10.1016/j.neuroimage.2018.10.006), through its parcel
    key of 718 parcels, at the same virtual electrode. CAB-NP ships as CIFTI:
    its 360 cortical parcels live on the fs_LR surface and only its 358
    subcortical parcels have MNI voxels. So the lookup has two steps.
    CORTEX: the HCPex voxel, whose values 1-360 are the same 360 Glasser
    (HCP-MMP1) areas, translated to a network through the key's
    GLASSERLABELNAME. Hemisphere stays in that lookup, because CAB-NP assigns
    some left/right twins to different networks. SUBCORTEX: otherwise, the
    CAB-NP subcortical parcel at the pair's 2 mm voxel of the dlabel. A pair in
    neither is `No parcel`, which also takes HCPex-subcortical points that miss
    every CAB-NP voxel. Unlike the other two atlases, the no-label rows stay in
    the bars (in grey, at the bottom), because how many pairs carry no network
    is part of what this figure reports.

THE COHORT is the reference run's `subjects[]` (reference_run.CONTPAIN_HEATMAP
by default), so the counts describe the subjects the models saw.

Run on Slurm, never the login node:
    ATLAS=cabnp sbatch sbatch/atlas_coverage.sbatch
"""

import argparse
import logging
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import nibabel as nib
import numpy as np
import pandas as pd

from ieeg_ehr import config, io
from ieeg_ehr.analysis import reference_run
from ieeg_ehr.med_analysis import style

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

SCRIPT = 'ieeg_ehr/analysis/plot_atlas_coverage.py'
EVENT = 'electrode_localization'
OUTPUT_TYPE = 'label_counts'

HEMISPHERE_PREFIXES = ('ctx-lh-', 'ctx-rh-', 'Left-', 'Right-')
BLANK_LABEL = '(blank)'
UNLABELED = 'Unlabeled'
NO_COORDINATE = 'No MNI coordinate'
NO_PARCEL = 'No parcel'
OUTSIDE_GRID = 'Outside atlas grid'
#: Rows that name the ABSENCE of an atlas label rather than a label.
NO_LABEL = (UNLABELED, NO_COORDINATE, NO_PARCEL, OUTSIDE_GRID)

HCPEX_DIR = config.DERIVATIVES_BASE / 'atlases' / 'HCPex_v1.1'
#: HCPex values 1-360 are the Glasser cortical areas (left 1-180, right
#: 181-360); 361-426 are its own subcortical additions.
HCPEX_N_CORTICAL = 360
#: HCPex renames Glasser's hippocampal area H. Every other name matches.
HCPEX_TO_GLASSER = {'Hipp': 'H'}
CABNP_DIR = config.DERIVATIVES_BASE / 'atlases' / 'CAB-NP_v1.1'
CABNP_STEM = 'CortexSubcortex_ColeAnticevic_NetPartition_wSubcorGSR_parcels_LR'
FS_MNI_DIR = config.DERIVATIVES_BASE / 'atlases' / 'FreeSurfer7.4.1_cvs_avg35_inMNI152'
MNI_COLUMNS = ['MNI_coord_1', 'MNI_coord_2', 'MNI_coord_3']
#: Largest allowed gap (mm) between the computed and the stored pair midpoint.
MIDPOINT_TOLERANCE_MM = 1e-3

TITLES = {
    'dk': 'Desikan-Killiany anode labels, hemispheres merged: {n} labels',
    'dk_midpoint': ('Desikan-Killiany labels at the pair midpoint, hemispheres merged: '
                    '{n} labels'),
    'hcpex': 'HCPex v1.1 labels at the pair midpoint, hemispheres merged: {n} labels',
    'cabnp': 'CAB-NP v1.1 networks at the pair midpoint: {n} networks',
}
FOOTNOTES = {
    'dk': 'one electrode = one bipolar pair, by anode label',
    'dk_midpoint': ('one electrode = one bipolar pair, at the anode-cathode MNI '
                    'midpoint, in FreeSurfer cvs_avg35_inMNI152 aparc+aseg'),
    'hcpex': 'one electrode = one bipolar pair, at the anode-cathode MNI midpoint',
    'cabnp': ('one electrode = one bipolar pair, at the anode-cathode MNI midpoint; '
              'cortex via HCPex Glasser areas, subcortex via CAB-NP 2 mm voxels'),
}


def load_pairs(subjects, epoch_minutes=None):
    """One row per (subject, session, bipolar pair), every session on disk."""
    frames, paths = [], []
    for sid in subjects:
        bare = sid.replace('sub-', '')
        meta_dir = config.pain_epoch_channel_meta_path(bare, '01', epoch_minutes).parent
        hits = sorted(meta_dir.glob(f'sub-{bare}_ses-*_channels.parquet'))
        if not hits:
            raise SystemExit(f'{sid}: no channel_meta under {meta_dir}')
        for path in hits:
            meta = io.read_table(path, on_stale='warn').drop_duplicates('channel')
            meta = meta[['run_id', 'channel', 'dk_anode', 'mni_x', 'mni_y', 'mni_z']].copy()
            meta['subject_id'] = sid
            meta['session'] = path.name.split('_ses-')[1][:2]
            frames.append(meta)
            paths.append(path)
    return pd.concat(frames, ignore_index=True), paths


# ============================================================================
# DK
# ============================================================================

def collapse_hemisphere(label):
    label = str(label).strip().strip("'\"")
    for prefix in HEMISPHERE_PREFIXES:
        if label.startswith(prefix):
            return label[len(prefix):]
    return label or BLANK_LABEL


def label_dk(pairs):
    pairs['label'] = pairs['dk_anode'].fillna('').map(collapse_hemisphere)
    return pairs, []


# ============================================================================
# HCPex
# ============================================================================

def contact_coords(raw_path):
    """{contact name: MNI xyz} for the neural contacts of one raw NWB."""
    from pynwb import NWBHDF5IO
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=UserWarning, module='pynwb')
        with NWBHDF5IO(str(raw_path), 'r') as handle:
            elec = handle.read().electrodes.to_dataframe()
    elec = elec[elec['group_name'].isin(['sEEG', 'ECoG'])]
    duplicated = elec['location'][elec['location'].duplicated()].tolist()
    if duplicated:
        raise SystemExit(f'{raw_path.name}: contact names repeat: {duplicated}')
    return elec.set_index('location')[MNI_COLUMNS].astype(float)


def virtual_electrodes(pairs):
    """Anode-cathode MNI midpoint per pair, from the raw NWB of a run that has it.

    The contact set can change between the runs of one session (sub-167's `RI`
    shaft is in one of its two runs), so each pair's contacts are read from the
    run channel_meta recorded it in. A contact missing from that run's table
    raises a KeyError rather than being skipped.
    """
    registry = pd.read_csv(config.FILE_REGISTRY_CSV,
                           usecols=['sub_id', 'ses_id', 'run_id', 'raw_file_path'])
    raw_path = {(r.sub_id, r.ses_id, r.run_id): r.raw_file_path
                for r in registry.itertuples(index=False)}
    midpoints, sources = [], []
    for (sid, session, run_id), rows in pairs.groupby(
            ['subject_id', 'session', 'run_id'], sort=False):
        path = raw_path[(sid, f'ses-{session}', run_id)]
        coords = contact_coords(Path(path))
        sources.append(path)
        for idx, channel in rows['channel'].items():
            anode, cathode = channel.split('-')
            midpoints.append((idx, *((coords.loc[anode].to_numpy()
                                      + coords.loc[cathode].to_numpy()) / 2)))
        logger.info('%s ses-%s: %d pair(s) from %s', sid, session, len(rows), run_id)
    mid = pd.DataFrame(midpoints, columns=['idx', 'vx', 'vy', 'vz']).set_index('idx')
    return pairs.join(mid), sources


def check_midpoints(pairs):
    """Refuse a computed midpoint that disagrees with the stored pair coordinate."""
    computed = pairs[['vx', 'vy', 'vz']].to_numpy()
    stored = pairs[['mni_x', 'mni_y', 'mni_z']].to_numpy(dtype=float)
    both = np.isfinite(computed).all(axis=1) & np.isfinite(stored).all(axis=1)
    gap = np.abs(computed[both] - stored[both]).max(axis=1)
    logger.info('midpoint vs stored pair coordinate: %d pair(s) compared, max gap '
                '%.2e mm', both.sum(), gap.max())
    bad = pairs[both][gap > MIDPOINT_TOLERANCE_MM]
    if len(bad):
        raise SystemExit(f'{len(bad)} computed midpoint(s) disagree with the stored '
                         f'pair coordinate by > {MIDPOINT_TOLERANCE_MM} mm:\n'
                         f'{bad[["subject_id", "session", "channel"]].to_string()}')


def voxel_values(vol, affine, xyz):
    """(atlas value, has a coordinate, inside the grid) per MNI point.

    The value is 0 for a point off the grid or without a coordinate.
    """
    has_xyz = np.isfinite(xyz).all(axis=1)
    ijk = np.full(xyz.shape, -1)
    ijk[has_xyz] = np.rint(nib.affines.apply_affine(
        np.linalg.inv(affine), xyz[has_xyz])).astype(int)
    inside = ((ijk >= 0) & (ijk < vol.shape)).all(axis=1)
    values = np.zeros(len(xyz), dtype=int)
    values[inside] = vol[tuple(ijk[inside].T)]
    return values, has_xyz, inside


def hcpex_values(xyz):
    img = nib.load(HCPEX_DIR / 'HCPex.nii.gz')
    return voxel_values(np.rint(np.asarray(img.dataobj)).astype(int), img.affine, xyz)


def hcpex_names():
    """({value: short name}, {short name: long name}, {value: hemisphere}).

    HCPex.nii.txt rows are `index hemisphere short_name value`; the lookup
    table's long names end in `_L`/`_R`.
    """
    short, hemisphere = {0: UNLABELED}, {}
    for line in (HCPEX_DIR / 'HCPex.nii.txt').read_text().splitlines():
        if line.strip():
            _, hemi, name, value = line.split()
            short[int(value)] = name
            hemisphere[int(value)] = hemi
    long = {}
    for line in (HCPEX_DIR / 'HCPex_LookUpTable.txt').read_text().splitlines()[1:]:
        value, name = line.split('\t')[:2] if line.strip() else ('0', '')
        if int(value):
            long[short[int(value)]] = name.removesuffix('_L').removesuffix('_R')
    return short, long, hemisphere


def label_hcpex(pairs):
    pairs, sources = virtual_electrodes(pairs)
    check_midpoints(pairs)

    values, has_xyz, inside = hcpex_values(pairs[['vx', 'vy', 'vz']].to_numpy())

    short, long, _ = hcpex_names()
    pairs['hcpex_value'] = values
    pairs['label'] = [short[v] for v in values]
    pairs.loc[~has_xyz, 'label'] = NO_COORDINATE
    pairs['label_name'] = pairs['label'].map(long)
    logger.info('%d pair(s) without an MNI coordinate, %d outside the atlas grid, '
                '%d in no parcel', (~has_xyz).sum(), (has_xyz & ~inside).sum(),
                (inside & (values == 0)).sum())
    return pairs, sources + [HCPEX_DIR / 'HCPex.nii.gz']


# ============================================================================
# CAB-NP
# ============================================================================

def cabnp_subcortex():
    """(volume of CAB-NP subcortical KEYVALUEs, its affine) from the dlabel."""
    img = nib.load(CABNP_DIR / f'{CABNP_STEM}.dlabel.nii')
    models = img.header.get_axis(1)
    data = np.rint(np.asarray(img.dataobj)[0]).astype(int)
    vol = np.zeros(models.volume_shape, dtype=int)
    mask = models.volume_mask
    vol[tuple(models.voxel[mask].T)] = data[mask]
    return vol, models.affine


def cortical_networks(key):
    """{HCPex cortical value: CAB-NP network}, through the key's Glasser names.

    Raises KeyError if any of the 360 HCPex cortical areas has no Glasser match.
    """
    cortex = key[key['GLASSERLABELNAME'].notna()]
    glasser = {(r.HEMISPHERE, r.GLASSERLABELNAME.removeprefix(f'{r.HEMISPHERE}_')
                .removesuffix('_ROI')): r.NETWORK for r in cortex.itertuples()}
    short, _, hemisphere = hcpex_names()
    return {v: glasser[(hemisphere[v], HCPEX_TO_GLASSER.get(short[v], short[v]))]
            for v in range(1, HCPEX_N_CORTICAL + 1)}


def label_cabnp(pairs):
    pairs, sources = virtual_electrodes(pairs)
    check_midpoints(pairs)
    xyz = pairs[['vx', 'vy', 'vz']].to_numpy()

    key = pd.read_csv(CABNP_DIR / f'{CABNP_STEM}_LabelKey.txt', sep='\t')
    to_network = cortical_networks(key)
    sub_network = dict(zip(key['KEYVALUE'], key['NETWORK']))

    ctx_values, has_xyz, ctx_inside = hcpex_values(xyz)
    sub_values, _, sub_inside = voxel_values(*cabnp_subcortex(), xyz)
    is_ctx = (ctx_values >= 1) & (ctx_values <= HCPEX_N_CORTICAL)
    is_sub = ~is_ctx & (sub_values > 0)

    label = np.full(len(pairs), NO_PARCEL, dtype=object)
    label[is_ctx] = [to_network[v] for v in ctx_values[is_ctx]]
    label[is_sub] = [sub_network[v] for v in sub_values[is_sub]]
    label[has_xyz & ~ctx_inside & ~sub_inside] = OUTSIDE_GRID
    label[~has_xyz] = NO_COORDINATE

    pairs['hcpex_value'] = ctx_values
    pairs['cabnp_subcortex_value'] = sub_values
    pairs['compartment'] = np.select([is_ctx, is_sub], ['cortex', 'subcortex'], None)
    pairs['label'] = label
    hcpex_sub = ctx_values > HCPEX_N_CORTICAL
    logger.info('%d cortical + %d subcortical pair(s); %d pair(s) HCPex puts in its '
                'subcortex but no CAB-NP voxel covers, %d the reverse',
                is_ctx.sum(), is_sub.sum(), (hcpex_sub & ~is_sub).sum(),
                (is_sub & ~hcpex_sub).sum())
    return pairs, sources + [HCPEX_DIR / 'HCPex.nii.gz',
                             CABNP_DIR / f'{CABNP_STEM}.dlabel.nii',
                             CABNP_DIR / f'{CABNP_STEM}_LabelKey.txt']


# ============================================================================
# DK at the midpoint
# ============================================================================

def freesurfer_names():
    """{aparc+aseg value: name} from FreeSurferColorLUT.txt."""
    names = {}
    for line in (FS_MNI_DIR / 'FreeSurferColorLUT.txt').read_text().splitlines():
        parts = line.split()
        if parts and parts[0].isdigit():
            names[int(parts[0])] = parts[1]
    return names


def label_dk_midpoint(pairs):
    pairs, sources = virtual_electrodes(pairs)
    check_midpoints(pairs)
    img = nib.load(FS_MNI_DIR / 'aparc+aseg.mgz')
    values, has_xyz, inside = voxel_values(
        np.rint(np.asarray(img.dataobj)).astype(int), img.affine,
        pairs[['vx', 'vy', 'vz']].to_numpy())

    names = freesurfer_names()
    label = np.array([collapse_hemisphere(names[v]) for v in values], dtype=object)
    label[values == 0] = NO_PARCEL
    label[has_xyz & ~inside] = OUTSIDE_GRID
    label[~has_xyz] = NO_COORDINATE
    pairs['aparc_aseg_value'] = values
    pairs['label'] = label

    anode = (pairs['dk_anode'].fillna('').map(collapse_hemisphere)
             .str.replace('Thalamus-Proper', 'Thalamus'))
    logger.info('midpoint label matches the anode label for %d of %d pair(s)',
                (anode == pairs['label']).sum(), len(pairs))
    return pairs, sources + [FS_MNI_DIR / 'aparc+aseg.mgz']


LABELERS = {'dk': label_dk, 'dk_midpoint': label_dk_midpoint,
            'hcpex': label_hcpex, 'cabnp': label_cabnp}
#: Atlases whose no-label rows stay in the bars (see plot_counts).
SHOW_NO_LABEL = ('dk_midpoint', 'cabnp')


# ============================================================================
# COUNT + PLOT
# ============================================================================

def label_counts(pairs):
    """Subjects and pairs per label, most subjects first, ties by pairs."""
    keys = ['label', 'label_name'] if 'label_name' in pairs else ['label']
    counts = (pairs.groupby(keys, dropna=False)
              .agg(n_subjects=('subject_id', 'nunique'),
                   n_electrodes=('channel', 'size'))
              .reset_index())
    if 'compartment' in pairs:
        split = (pd.crosstab(pairs['label'], pairs['compartment'])
                 .add_prefix('n_electrodes_'))
        counts = counts.join(split, on='label').fillna(
            {c: 0 for c in split.columns}).astype({c: int for c in split.columns})
    return counts.sort_values(['n_subjects', 'n_electrodes', 'label'],
                              ascending=[False, False, True], ignore_index=True)


def plot_counts(counts, n_subjects, n_pairs, title, footnote, out_path,
                show_no_label=False):
    """Two bar panels, one bar per atlas label.

    Rows with NO label (`NO_LABEL`) stay in the CSV either way. By default they
    leave the bars for a subtitle: under HCPex they are ~half of all pairs, and
    one bar that long flattens 196 real labels to slivers. `show_no_label`
    keeps them as grey bars at the bottom instead, which reads when there are
    few labels (CAB-NP's 12 networks).
    """
    no_label = counts['label'].isin(NO_LABEL)
    n_labels = int((~no_label).sum())
    if show_no_label:
        subtitle = ''
        counts = pd.concat([counts[~no_label], counts[no_label]], ignore_index=True)
        colours = np.where(counts['label'].isin(NO_LABEL), style.AXIS_COLOR,
                           style.BAR_COLOR)
    else:
        subtitle = '; '.join(f'{r.label}: {r.n_electrodes} pairs, {r.n_subjects} '
                             'subjects' for r in counts[no_label].itertuples())
        counts = counts[~no_label]
        colours = style.BAR_COLOR
    y = np.arange(len(counts))
    height = 0.19 * len(counts) + 1.6
    fig, axes = plt.subplots(1, 2, sharey=True, layout='constrained',
                             figsize=(10, height))
    # Constrained layout ignores fig.text, so reserve 0.25 in for the footnote.
    fig.get_layout_engine().set(rect=(0, 0.25 / height, 1, 1 - 0.25 / height))
    panels = [('n_subjects', f'Subjects (of {n_subjects})'),
              ('n_electrodes', f'Bipolar pairs (of {n_pairs})')]
    for ax, (column, xlabel) in zip(axes, panels):
        values = counts[column].to_numpy()
        ax.barh(y, values, height=0.72, color=colours, zorder=2)
        for yi, v in zip(y, values):
            ax.text(v, yi, f' {v}', va='center', ha='left',
                    fontsize=style.TICK_SIZE - 2, color=style.TEXT_MUTED)
        ax.set_xlim(0, values.max() * 1.12)
        style.style_axes(ax, grid_axis='x')
        style.label_axes(ax, xlabel=xlabel)
        ax.xaxis.set_label_position('top')
        ax.xaxis.tick_top()
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.tick_params(axis='y', length=0)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(counts['label'], fontsize=style.TICK_SIZE - 1,
                            color=style.TEXT_PRIMARY)
    axes[0].set_ylim(len(counts) - 0.5, -0.5)
    fig.suptitle(f'{title.format(n=n_labels)}, {n_subjects} subjects, '
                 f'{n_pairs} bipolar pairs'
                 + (f'\nNot shown, {subtitle}' if subtitle else ''),
                 fontsize=style.TITLE_SIZE, color=style.TEXT_PRIMARY,
                 x=0.01, ha='left')
    style.save(fig, out_path, footnote=footnote)
    logger.info('Wrote %s', out_path)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--atlas', choices=sorted(LABELERS), required=True)
    ap.add_argument('--reference-run', default=str(reference_run.CONTPAIN_HEATMAP))
    ap.add_argument('--epoch-minutes', type=float, default=None)
    ap.add_argument('--run-name', default='contpain_ref')
    ap.add_argument('--exclude-subjects', nargs='+', default=[],
                    help='Drop these subjects from the reference cohort, e.g. ones '
                         'whose MNI coordinates fall outside the brain.')
    args = ap.parse_args()

    io.warn_if_dirty()

    ref = reference_run.load(args.reference_run)
    unknown = sorted(set(args.exclude_subjects) - set(ref.subjects))
    if unknown:
        raise SystemExit(f'--exclude-subjects not in the reference cohort: {unknown}')
    subjects = sorted(set(ref.subjects) - set(args.exclude_subjects))
    logger.info('cohort from %s: %d subject(s), %d excluded: %s', ref.run_dir.name,
                len(subjects), len(args.exclude_subjects), args.exclude_subjects)

    pairs, meta_paths = load_pairs(subjects, args.epoch_minutes)
    pairs, label_sources = LABELERS[args.atlas](pairs)
    counts = label_counts(pairs)
    n_subjects, n_pairs = pairs['subject_id'].nunique(), len(pairs)
    logger.info('%d row(s) over %d pair(s), %d subject(s):\n%s',
                len(counts), n_pairs, n_subjects, counts.to_string())

    run_dir = config.analysis_run_dir(question=f'{args.atlas}_coverage',
                                      output_type=OUTPUT_TYPE,
                                      run_name=args.run_name, event=EVENT)
    parents = ([io.parent_ref(ref.run_dir / 'provenance.json', digest=False)]
               + [io.parent_ref(p, digest=False) for p in meta_paths]
               + [io.parent_ref(p, digest=False) for p in label_sources])
    params = {'atlas': args.atlas,
              'reference_run': str(ref.run_dir),
              'epoch_minutes': args.epoch_minutes,
              'excluded_subjects': sorted(args.exclude_subjects)}
    if args.atlas == 'dk':
        params['label_column'] = 'dk_anode'
    else:
        params['midpoint_tolerance_mm'] = MIDPOINT_TOLERANCE_MM
    if args.atlas in ('dk', 'dk_midpoint'):
        params['hemisphere_prefixes_stripped'] = list(HEMISPHERE_PREFIXES)
    if args.atlas == 'cabnp':
        params['hcpex_to_glasser'] = HCPEX_TO_GLASSER

    io.write_table(pairs, run_dir / 'pair_labels.csv', script=SCRIPT,
                   params=params, parents=parents, subjects=subjects)
    io.write_table(counts, run_dir / 'label_counts.csv', script=SCRIPT,
                   params=params, parents=parents, subjects=subjects)
    io.write_run_provenance(run_dir, script=SCRIPT, params={**vars(args), **params},
                            parents=parents, subjects=subjects,
                            extra={'n_labels': int(len(counts)),
                                   'n_electrodes': int(n_pairs),
                                   'n_subjects': int(n_subjects)})
    plot_counts(counts, n_subjects, n_pairs, TITLES[args.atlas],
                FOOTNOTES[args.atlas], run_dir / 'label_counts.png',
                show_no_label=args.atlas in SHOW_NO_LABEL)

    io.log_analysis(f'{args.atlas} label coverage, hemispheres merged, '
                    f'{len(counts)} labels, {n_pairs} pairs, n={n_subjects}', run_dir)
    logger.info('figure + table + provenance -> %s', run_dir)


if __name__ == '__main__':
    main()
