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
Hemispheres are merged in both atlases.

`--atlas dk`
    The pair's ANODE Desikan-Killiany label (`dk_anode`), which is how the
    analysis assigns regions. Cortical DK labels in this dataset mostly carry no
    hemisphere (`insula`); subcortical and white-matter labels carry
    `Left-`/`Right-`, and a few cortical ones `ctx-lh-`/`ctx-rh-`. Stripping
    those four prefixes merges hemispheres. No other spelling is merged:
    `Unknown` and `undefined` stay two labels, as they are two strings.

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

THE COHORT is the reference run's `subjects[]` (reference_run.CONTPAIN_HEATMAP
by default), so the counts describe the subjects the models saw.

Run on Slurm, never the login node:
    ATLAS=hcpex sbatch sbatch/atlas_coverage.sbatch
"""

import argparse
import logging
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
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

HCPEX_DIR = config.DERIVATIVES_BASE / 'atlases' / 'HCPex_v1.1'
MNI_COLUMNS = ['MNI_coord_1', 'MNI_coord_2', 'MNI_coord_3']
#: Largest allowed gap (mm) between the computed and the stored pair midpoint.
MIDPOINT_TOLERANCE_MM = 1e-3

TITLES = {
    'dk': 'Desikan-Killiany anode labels, hemispheres merged',
    'hcpex': 'HCPex v1.1 labels at the pair midpoint, hemispheres merged',
}
FOOTNOTES = {
    'dk': 'one electrode = one bipolar pair, by anode label',
    'hcpex': 'one electrode = one bipolar pair, at the anode-cathode MNI midpoint',
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
    """Anode-cathode MNI midpoint per pair, from ONE raw NWB per subject-session.

    Contact positions do not change between the runs of one session, so the
    first run channel_meta lists stands for all of them. A pair whose contact is
    missing from that run's table raises a KeyError rather than being skipped.
    """
    registry = pd.read_csv(config.FILE_REGISTRY_CSV,
                           usecols=['sub_id', 'ses_id', 'run_id', 'raw_file_path'])
    raw_path = {(r.sub_id, r.ses_id, r.run_id): r.raw_file_path
                for r in registry.itertuples(index=False)}
    midpoints, sources = [], []
    for (sid, session), rows in pairs.groupby(['subject_id', 'session'], sort=False):
        run_id = sorted(rows['run_id'])[0]
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


def hcpex_names():
    """({atlas value: short name}, {short name: long name}), hemispheres merged.

    HCPex.nii.txt rows are `index hemisphere short_name value`; the lookup
    table's long names end in `_L`/`_R`.
    """
    short = {0: UNLABELED}
    for line in (HCPEX_DIR / 'HCPex.nii.txt').read_text().splitlines():
        if line.strip():
            _, _, name, value = line.split()
            short[int(value)] = name
    long = {}
    for line in (HCPEX_DIR / 'HCPex_LookUpTable.txt').read_text().splitlines()[1:]:
        value, name = line.split('\t')[:2] if line.strip() else ('0', '')
        if int(value):
            long[short[int(value)]] = name.removesuffix('_L').removesuffix('_R')
    return short, long


def label_hcpex(pairs):
    pairs, sources = virtual_electrodes(pairs)
    check_midpoints(pairs)

    img = nib.load(HCPEX_DIR / 'HCPex.nii.gz')
    vol = np.rint(np.asarray(img.dataobj)).astype(int)
    xyz = pairs[['vx', 'vy', 'vz']].to_numpy()
    has_xyz = np.isfinite(xyz).all(axis=1)
    ijk = np.full(xyz.shape, -1)
    ijk[has_xyz] = np.rint(nib.affines.apply_affine(
        np.linalg.inv(img.affine), xyz[has_xyz])).astype(int)
    inside = ((ijk >= 0) & (ijk < vol.shape)).all(axis=1)
    values = np.zeros(len(pairs), dtype=int)
    values[inside] = vol[tuple(ijk[inside].T)]

    short, long = hcpex_names()
    pairs['hcpex_value'] = values
    pairs['label'] = [short[v] for v in values]
    pairs.loc[~has_xyz, 'label'] = NO_COORDINATE
    pairs['label_name'] = pairs['label'].map(long)
    logger.info('%d pair(s) without an MNI coordinate, %d outside the atlas grid, '
                '%d in no parcel', (~has_xyz).sum(), (has_xyz & ~inside).sum(),
                (inside & (values == 0)).sum())
    return pairs, sources


LABELERS = {'dk': label_dk, 'hcpex': label_hcpex}


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
    return counts.sort_values(['n_subjects', 'n_electrodes', 'label'],
                              ascending=[False, False, True], ignore_index=True)


def plot_counts(counts, n_subjects, n_pairs, title, footnote, out_path):
    y = np.arange(len(counts))
    fig, axes = plt.subplots(1, 2, sharey=True,
                             figsize=(10, 0.19 * len(counts) + 1.6))
    panels = [('n_subjects', f'Subjects (of {n_subjects})'),
              ('n_electrodes', f'Bipolar pairs (of {n_pairs})')]
    for ax, (column, xlabel) in zip(axes, panels):
        values = counts[column].to_numpy()
        ax.barh(y, values, height=0.72, color=style.BAR_COLOR, zorder=2)
        for yi, v in zip(y, values):
            ax.text(v, yi, f' {v}', va='center', ha='left',
                    fontsize=style.TICK_SIZE - 2, color=style.TEXT_MUTED)
        ax.set_xlim(0, values.max() * 1.12)
        style.style_axes(ax, grid_axis='x')
        style.label_axes(ax, xlabel=xlabel)
        ax.xaxis.set_label_position('top')
        ax.xaxis.tick_top()
        ax.tick_params(axis='y', length=0)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(counts['label'], fontsize=style.TICK_SIZE - 1,
                            color=style.TEXT_PRIMARY)
    axes[0].set_ylim(len(counts) - 0.5, -0.5)
    fig.suptitle(f'{title}: {len(counts)} labels, {n_subjects} subjects, '
                 f'{n_pairs} bipolar pairs',
                 fontsize=style.TITLE_SIZE, color=style.TEXT_PRIMARY,
                 x=0.01, ha='left')
    fig.tight_layout()
    style.save(fig, out_path, footnote=footnote)
    logger.info('Wrote %s', out_path)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--atlas', choices=sorted(LABELERS), required=True)
    ap.add_argument('--reference-run', default=str(reference_run.CONTPAIN_HEATMAP))
    ap.add_argument('--epoch-minutes', type=float, default=None)
    ap.add_argument('--run-name', default='contpain_ref')
    args = ap.parse_args()

    io.warn_if_dirty()

    ref = reference_run.load(args.reference_run)
    subjects = sorted(ref.subjects)
    logger.info('cohort from %s: %d subject(s)', ref.run_dir.name, len(subjects))

    pairs, meta_paths = load_pairs(subjects, args.epoch_minutes)
    pairs, label_sources = LABELERS[args.atlas](pairs)
    counts = label_counts(pairs)
    n_subjects, n_pairs = pairs['subject_id'].nunique(), len(pairs)
    logger.info('%d label(s) over %d pair(s), %d subject(s):\n%s',
                len(counts), n_pairs, n_subjects, counts.to_string())

    run_dir = config.analysis_run_dir(question=f'{args.atlas}_coverage',
                                      output_type=OUTPUT_TYPE,
                                      run_name=args.run_name, event=EVENT)
    parents = ([io.parent_ref(ref.run_dir / 'provenance.json', digest=False)]
               + [io.parent_ref(p, digest=False) for p in meta_paths]
               + [io.parent_ref(p, digest=False) for p in label_sources])
    params = {'atlas': args.atlas,
              'reference_run': str(ref.run_dir),
              'epoch_minutes': args.epoch_minutes}
    if args.atlas == 'dk':
        params.update(label_column='dk_anode',
                      hemisphere_prefixes_stripped=list(HEMISPHERE_PREFIXES))
    else:
        params.update(atlas_dir=str(HCPEX_DIR),
                      midpoint_tolerance_mm=MIDPOINT_TOLERANCE_MM)
        parents += [io.parent_ref(HCPEX_DIR / 'HCPex.nii.gz')]

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
                FOOTNOTES[args.atlas], run_dir / 'label_counts.png')

    io.log_analysis(f'{args.atlas} label coverage, hemispheres merged, '
                    f'{len(counts)} labels, {n_pairs} pairs, n={n_subjects}', run_dir)
    logger.info('figure + table + provenance -> %s', run_dir)


if __name__ == '__main__':
    main()
