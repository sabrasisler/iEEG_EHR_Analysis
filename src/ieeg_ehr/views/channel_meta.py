"""
The per-channel metadata the cache does not carry: pair order and DK labels.

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

COLUMNS = ['run_id', 'pair_index', 'channel', 'dk_anode', 'dk_cathode']


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
    out = pd.DataFrame({
        'run_id': f'run-{run}',
        'pair_index': range(len(elec)),
        'channel': list(elec['location']),
        'dk_anode': (list(elec['Desikan_Killiany_anode'])
                     if 'Desikan_Killiany_anode' in elec.columns else None),
        'dk_cathode': (list(elec['Desikan_Killiany_cathode'])
                       if 'Desikan_Killiany_cathode' in elec.columns else None),
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
        if not missing:
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
