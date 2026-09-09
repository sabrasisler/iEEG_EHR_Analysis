"""
NWB loading + subject/run discovery, following the conventions already used in
outdated/scripts/raw_voltage_qc.py / plot_exclusion_qc.py (series.unit == 'volts', series.conversion
scaling, electrodes.to_dataframe() for channel names).
"""

import numpy as np
import pandas as pd
from pynwb import NWBHDF5IO

from ieeg_ehr import config


def sample_subjects(n_subjects=None, subject_list=None, seed=None):
    """
    Return a list of subject IDs (without the 'sub-' prefix) to run the
    pipeline on. Uses the existing sherlock file registry rather than
    re-discovering files via glob.
    """
    if subject_list is not None:
        return [s.replace('sub-', '') for s in subject_list]

    n_subjects = n_subjects if n_subjects is not None else config.N_SUBJECTS
    seed = seed if seed is not None else config.RANDOM_SEED

    registry = pd.read_csv(config.FILE_REGISTRY_CSV)
    all_subjects = sorted(registry['sub_id'].str.replace('sub-', '', regex=False).unique())

    rng = np.random.default_rng(seed)
    n_pick = min(n_subjects, len(all_subjects))
    picked = rng.choice(all_subjects, size=n_pick, replace=False)
    return sorted(picked.tolist())


def get_session_runs(subject, session=None):
    """
    Return a list of (session, run, nwb_path) tuples for a subject, in
    registry order, optionally filtered to a single session.
    """
    registry = pd.read_csv(config.FILE_REGISTRY_CSV)
    sub_rows = registry[registry['sub_id'] == f'sub-{subject}']
    if session is not None:
        sub_rows = sub_rows[sub_rows['ses_id'] == f'ses-{session}']
    sub_rows = sub_rows.sort_values('run_id')
    return list(zip(
        sub_rows['ses_id'].str.replace('ses-', '', regex=False),
        sub_rows['run_id'].str.replace('run-', '', regex=False),
        sub_rows['raw_file_path'],
    ))


def load_all_channels(nwb_path):
    """
    Load every channel's raw voltage trace for one NWB run.
    Returns (data_v, channel_names, sfreq) where data_v is
    (n_samples, n_channels) in volts.
    """
    io = NWBHDF5IO(nwb_path, 'r')
    nwb = io.read()
    series = nwb.acquisition['ElectricalSeries_sEEG']

    if series.unit != 'volts':
        io.close()
        raise ValueError(f"Unexpected unit '{series.unit}' (expected 'volts') in {nwb_path}")

    sfreq = float(series.rate)
    elec_indices = series.electrodes.data[:]
    elec_df = nwb.electrodes.to_dataframe().iloc[elec_indices]
    channel_names = list(elec_df['location'].values)

    data_v = series.data[:].astype(np.float32) * np.float32(series.conversion)
    io.close()
    return data_v, channel_names, sfreq


def load_all_channels_with_electrodes(nwb_path):
    """
    Same as load_all_channels, but also returns (elec_df, elec_indices) so
    callers can do electrode-index-aware processing (e.g. bipolar pairing)
    without a second NWB read. Returns
    (data_v, channel_names, sfreq, elec_df, elec_indices).
    """
    io = NWBHDF5IO(nwb_path, 'r')
    nwb = io.read()
    series = nwb.acquisition['ElectricalSeries_sEEG']

    if series.unit != 'volts':
        io.close()
        raise ValueError(f"Unexpected unit '{series.unit}' (expected 'volts') in {nwb_path}")

    sfreq = float(series.rate)
    elec_indices = series.electrodes.data[:]
    elec_df = nwb.electrodes.to_dataframe().iloc[elec_indices]
    channel_names = list(elec_df['location'].values)

    data_v = series.data[:].astype(np.float32) * np.float32(series.conversion)
    io.close()
    return data_v, channel_names, sfreq, elec_df, elec_indices


def load_channels_subset(nwb_path, channel_names_wanted):
    """
    Load only specific channels (by name) from one NWB run, using column
    fancy-indexing so unwanted channels are never read off disk.
    Returns (data_v, channel_names, sfreq) — channel_names matches the
    requested subset that was actually found in this run.
    """
    io = NWBHDF5IO(nwb_path, 'r')
    nwb = io.read()
    series = nwb.acquisition['ElectricalSeries_sEEG']

    if series.unit != 'volts':
        io.close()
        raise ValueError(f"Unexpected unit '{series.unit}' (expected 'volts') in {nwb_path}")

    sfreq = float(series.rate)
    elec_indices = series.electrodes.data[:]
    elec_df = nwb.electrodes.to_dataframe().iloc[elec_indices]
    all_channel_names = list(elec_df['location'].values)

    wanted = [c for c in channel_names_wanted if c in all_channel_names]
    idx_sorted = sorted(all_channel_names.index(c) for c in wanted)
    channel_names = [all_channel_names[i] for i in idx_sorted]

    data_v = series.data[:, idx_sorted].astype(np.float32) * np.float32(series.conversion)
    io.close()
    return data_v, channel_names, sfreq


def load_window_with_electrodes(nwb_path, start_sec, dur_sec, series_name='ElectricalSeries_sEEG'):
    """One TIME WINDOW of a run, without reading the rest of it.

    The existing loaders do `series.data[:]`, which for a 5-min epoch out of a
    2-5 hour run is 20-60x more I/O than needed. V1 raw NWB `.data` is chunked
    (10000, n_channels) -- along TIME -- so an HDF5 slice touches only the ~15
    chunks it overlaps even though the file is gzip-compressed. Measured: 0.22 s
    for 40 ch @ 500 Hz, 2.1 s for 250 ch @ 1000 Hz, versus a 1.5 GB whole-run read.

    V2 CAVEAT (docs/dataset_v2.md §4). `iEEG_EHR_V2` re-chunked to
    (120 s x 1 CHANNEL), which inverts the tradeoff: whole-channel reads get much
    faster and short full-width windows become the worst case. A 5-min window is
    >=120 s so it stays reasonable there (~1.6x overhead, since it spans 3 chunks
    per channel and is not chunk-aligned), but anything shorter would not, and a
    caller wanting V2 should read chunk-aligned blocks or whole channels instead.
    This function reads whichever tree the path points at and does not check.

    Returns (data_v, channel_names, sfreq, elec_df, n_samples_total).
    `data_v` is (n_samples, n_channels) float32 in volts. The window is CLIPPED
    to the run, so a short tail returns fewer samples rather than raising --
    callers check the length they got.
    """
    io = NWBHDF5IO(nwb_path, 'r')
    nwb = io.read()
    series = nwb.acquisition[series_name]
    if series.unit != 'volts':
        io.close()
        raise ValueError(f"Unexpected unit '{series.unit}' (expected 'volts') in {nwb_path}")

    sfreq = float(series.rate)
    n_total = series.data.shape[0]
    lo = max(0, int(round(start_sec * sfreq)))
    hi = min(n_total, lo + int(round(dur_sec * sfreq)))

    elec_indices = series.electrodes.data[:]
    elec_df = nwb.electrodes.to_dataframe().iloc[elec_indices]
    channel_names = list(elec_df['location'].values)

    data_v = series.data[lo:hi, :].astype(np.float32) * np.float32(series.conversion)
    io.close()
    return data_v, channel_names, sfreq, elec_df, n_total
