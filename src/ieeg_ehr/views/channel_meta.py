"""
The per-channel metadata the cache does not carry: pair order, DK labels, and the
pair's MNI position.

WHY THIS EXISTS
---------------
The cache stores `channel` as a dictionary-encoded pair NAME and nothing else. The
region axis needs each pair's Desikan-Killiany parcel, which lives only in the
NWB electrodes table. Run timing went into epoch_defs instead (it is constant per
run, so it belongs in the per-epoch index); DK labels are per CHANNEL, so they
cannot.

Built once per subject/session and cached as a tiny Parquet, because NWB metadata
opens cost ~0.3 s per run and a subject can have 100 runs -- paying that on every
view call, and again on every plot rerun, would dominate the runtime of a view
that is otherwise a column read and a reshape.

PER RUN, NOT PER SESSION -- and that matters
--------------------------------------------
Pair ORDER is what makes the cache's C-order ravel decodable: pair index j in a
reshaped block is `channels[j]` in electrodes-table order. That order is a
property of the RUN's electrodes table, and runs within a session are not
guaranteed to share a montage (sub-085 has runs missing entirely; corrupt NWBs are
skipped by design upstream). Storing one flat per-session channel list would
silently mislabel every channel of any run whose montage differs. So rows are
keyed (run_id, pair_index).
"""

import logging
import warnings

import pandas as pd

from ieeg_ehr import config, io

logger = logging.getLogger(__name__)

COLUMNS = ['run_id', 'pair_index', 'channel', 'dk_anode', 'dk_cathode',
           'mni_x', 'mni_y', 'mni_z', 'wm_vs_gm_anode', 'seeg_ecog_anode']

# Bumped whenever COLUMNS grows, so a table cached before the new columns existed
# rebuilds itself instead of being returned silently without them. Without this the
# only staleness check is "does it have all my runs", which a column addition
# passes.
#   1 -> run_id, pair_index, channel, dk_anode, dk_cathode
#   2 -> + MNI coordinates, WMvsGM, sEEG/ECoG   (2026-07-29)
SCHEMA_VERSION = 2

# THE PAIR COORDINATE IS THE MIDPOINT OF ITS TWO CONTACTS -- verified 2026-07-29
# against sub-019: contacts LA1 (-15.16, -4.47, -23.00) and LA2 (-21.12, -3.92,
# -23.98) give exactly the stored pair value (-18.139, -4.192, -23.488). So this is
# the VIRTUAL ELECTRODE position, which is what a bipolar signal actually comes
# from, and it is what roi_schemes.py describes as the eventual right basis for
# region assignment (currently the anode label).
_MNI_COLUMNS = ('MNI_coord_1', 'MNI_coord_2', 'MNI_coord_3')


def _read_run_channels(subject, session, run):
    """Electrodes table for one run -- METADATA ONLY, never `decomp.data`."""
    from pynwb import NWBHDF5IO
    path = config.bipolar_psd_nwb_path(subject, session, run)
    if not path.exists():
        return None
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=UserWarning, module='pynwb')
        with NWBHDF5IO(str(path), 'r') as handle:
            elec = handle.read().electrodes.to_dataframe()
    def col(name, default=None):
        # Every optional column is absent for at least one subject in this cohort
        # (4 have no DK column at all), so a missing column is a normal path and
        # must produce NaN rather than a KeyError.
        return list(elec[name]) if name in elec.columns else default

    out = pd.DataFrame({
        'run_id': f'run-{run}',
        'pair_index': range(len(elec)),
        'channel': list(elec['location']),
        'dk_anode': col('Desikan_Killiany_anode'),
        'dk_cathode': col('Desikan_Killiany_cathode'),
        # Midpoint of the pair's two contacts -- see _MNI_COLUMNS above.
        'mni_x': col(_MNI_COLUMNS[0]),
        'mni_y': col(_MNI_COLUMNS[1]),
        'mni_z': col(_MNI_COLUMNS[2]),
        # A DIRECT grey/white-matter call, rather than inferring it from the DK
        # string, and the electrode type -- one of the cohort matching axes in
        # architecture.md PART 6. Both are free here: the electrodes table is
        # already open.
        'wm_vs_gm_anode': col('WMvsGM_anode'),
        'seeg_ecog_anode': col('sEEG_ECoG_anode'),
    })
    return out[COLUMNS]


def build(subject, session, run_ids, epoch_minutes=None, overwrite=False):
    """Build (or return the cached) channel metadata table for one subject/session.

    `run_ids` are the runs that actually carry epochs -- taken from epoch_defs, so
    runs with no pain epoch are never opened.
    """
    path = config.pain_epoch_channel_meta_path(subject, session, epoch_minutes)
    if path.exists() and not overwrite:
        cached = io.read_table(path, on_stale='ignore')
        missing = sorted(set(run_ids) - set(cached['run_id'].unique()))
        absent_columns = [c for c in COLUMNS if c not in cached.columns]
        if absent_columns:
            # A table written before COLUMNS grew. Rebuilding is the only correct
            # move: returning it would hand the caller a frame missing columns it
            # asked for, and back-filling NaN would assert the data is unavailable
            # when it is sitting in the NWB.
            logger.info('sub-%s ses-%s: channel_meta predates columns %s, rebuilding',
                        subject, session, absent_columns)
        elif not missing:
            return cached
        # A previously-built table that predates some epochs' runs: rebuild rather
        # than silently returning a table that cannot label them.
        logger.info('sub-%s ses-%s: channel_meta missing %d run(s) %s, rebuilding',
                    subject, session, len(missing), missing[:5])

    frames = []
    for run_id in sorted(set(run_ids)):
        run = str(run_id).replace('run-', '')
        rows = _read_run_channels(subject, session, run)
        if rows is None:
            logger.error('sub-%s ses-%s %s: NWB absent, channels unknown for this run',
                         subject, session, run_id)
            continue
        frames.append(rows)
    if not frames:
        raise FileNotFoundError(
            f'sub-{subject} ses-{session}: no NWB readable for runs {sorted(set(run_ids))}; '
            'cannot label channels.'
        )

    meta = pd.concat(frames, ignore_index=True)
    io.write_table(meta, path, kind='table',
                   script='ieeg_ehr/views/channel_meta.py',
                   params={'source': 'bipolar_psd nwb electrodes table',
                           'keys': ['run_id', 'pair_index']},
                   parents=[str(config.bipolar_psd_nwb_path(subject, session,
                                                            str(r).replace('run-', '')))
                            for r in sorted(set(run_ids))],
                   subjects=[f'sub-{subject}'])
    n_runs = meta['run_id'].nunique()
    n_shapes = meta.groupby('run_id').size().nunique()
    logger.info('sub-%s ses-%s: channel_meta for %d runs (%d distinct montage sizes) -> %s',
                subject, session, n_runs, n_shapes, path.name)
    if n_shapes > 1:
        # Not an error -- but it means a per-session channel list would have been
        # wrong, so it is worth seeing when it happens.
        logger.warning('sub-%s ses-%s: runs do NOT share one montage size (%s); '
                       'per-run channel order is load-bearing here',
                       subject, session,
                       sorted(meta.groupby('run_id').size().unique()))
    return meta


def channels_for_run(meta, run_id):
    """Ordered pair names for one run -- index j is the cache's pair index j."""
    rows = meta[meta['run_id'] == run_id].sort_values('pair_index')
    return rows['channel'].tolist(), rows


def region_map(meta, scheme=None):
    """{channel -> ROI or None}, deduplicated across runs.

    A pair name maps to one parcel regardless of run, so collapsing is safe; a
    conflict would mean the same electrode name meant two things, which is worth
    hearing about rather than silently resolving.
    """
    from ieeg_ehr.config import roi_schemes
    pairs = meta[['channel', 'dk_anode']].drop_duplicates()
    conflicting = pairs['channel'].duplicated(keep=False)
    if conflicting.any():
        logger.warning('%d channel name(s) carry more than one DK label across runs; '
                       'taking the first: %s', int(conflicting.sum()),
                       sorted(pairs.loc[conflicting, 'channel'].unique())[:5])
        pairs = pairs.drop_duplicates('channel')
    return {row.channel: roi_schemes.region_for_dk_label(row.dk_anode, scheme)
            for row in pairs.itertuples()}


def main():
    """Rebuild channel_meta for a cohort, e.g. after COLUMNS grows.

    Exists because `build` is normally called from inside a view run, and a schema
    bump needs the tables refreshed WITHOUT waiting for the next full view build --
    the plotting scripts read these tables directly and skip any subject whose table
    predates the columns they need.

    Run on Slurm, never the login node:
        python -m ieeg_ehr.views.channel_meta --split discovery --overwrite
    """
    import argparse

    from ieeg_ehr.config import cohorts
    from ieeg_ehr.views import cache_reader

    ap = argparse.ArgumentParser(description=main.__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--subjects', nargs='+', default=None)
    ap.add_argument('--split', default='discovery')
    ap.add_argument('--session', default='01')
    ap.add_argument('--epoch-minutes', type=float, default=None)
    ap.add_argument('--overwrite', action='store_true',
                    help='Rebuild even when the cached table already has every '
                         'column and run. Without it, only stale tables are rebuilt.')
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)s %(message)s')
    subjects = args.subjects or cohorts.subjects_for_split(
        args.split, available=cohorts.subjects_with_epoch_cache(args.epoch_minutes))
    logger.info('rebuilding channel_meta (schema v%d) for %d subject(s)',
                SCHEMA_VERSION, len(subjects))

    ok, failed = 0, []
    for s in subjects:
        subject = str(s).replace('sub-', '')
        try:
            # run_ids come from epoch_defs, so runs carrying no pain epoch are never
            # opened -- the same contract build() documents.
            defs = cache_reader.load_defs(subject, args.session, args.epoch_minutes,
                                          require_timing=False)
            meta = build(subject, args.session, defs['run_id'].unique(),
                         args.epoch_minutes, overwrite=args.overwrite)
            n_xyz = int(meta[['mni_x', 'mni_y', 'mni_z']].notna().all(axis=1).sum())
            logger.info('sub-%s: %d row(s), %d with MNI coordinates',
                        subject, len(meta), n_xyz)
            ok += 1
        except Exception as exc:
            # One unreadable subject must not stop the rest; the summary at the end
            # is what says how complete the rebuild actually is.
            logger.error('sub-%s: %s', subject, exc)
            failed.append(subject)

    logger.info('done: %d rebuilt, %d failed%s', ok, len(failed),
                f' ({failed})' if failed else '')


if __name__ == '__main__':
    main()
