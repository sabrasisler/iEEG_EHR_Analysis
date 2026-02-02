#!/usr/bin/env python3
"""
Complete iEEG processing pipeline: Raw → Bipolar Re-referencing → Band Power (PSD)

Usage:
    python process_ieeg_pipeline.py input.nwb output.nwb [--overwrite]
"""

import numpy as np
import pandas as pd
from scipy import signal
import pynwb
from pynwb import NWBFile
from pynwb.ecephys import ElectricalSeries
from pynwb.misc import DecompositionSeries

from datetime import datetime
import os
import re
import argparse

# Frequency bands
BANDS = {
    'delta': (1.0, 4.0),
    'theta': (4.0, 8.0),
    'alpha': (8.0, 12.0),
    'beta': (15.0, 25.0),
    'gamma': (25.0, 70.0),
    'high_gamma': (70.0, 170.0)
}

def parse_electrode_shaft(location):
    """Extract shaft name and contact number from location"""
    match = re.match(r'^([A-Za-z]+)(\d+)$', location)
    if match:
        shaft = match.group(1)
        number = int(match.group(2))
        return shaft, number
    else:
        return None, None

def create_bipolar_pairs(elec_df):
    """Create bipolar electrode pairs from electrode dataframe"""
    
    # Parse shaft and contact number for each electrode
    elec_df = elec_df.copy()
    elec_df['shaft'] = None
    elec_df['contact_num'] = None
    
    for idx, row in elec_df.iterrows():
        shaft, num = parse_electrode_shaft(row['location'])
        elec_df.at[idx, 'shaft'] = shaft
        elec_df.at[idx, 'contact_num'] = num
    
    # Remove electrodes with unparseable labels
    valid_parse = elec_df['shaft'].notna()
    if (~valid_parse).sum() > 0:
        print(f"  Warning: {(~valid_parse).sum()} electrodes have unparseable locations")
        print(f"    Unparseable: {elec_df[~valid_parse]['location'].tolist()}")
        elec_df = elec_df[valid_parse].copy()
    
    # Group by shaft and create pairs
    pairs = []
    grouped = elec_df.groupby('shaft')
    
    for shaft_name, shaft_df in grouped:
        shaft_df = shaft_df.sort_values('contact_num')
        contacts = [(idx, row) for idx, row in shaft_df.iterrows()]
        
        for i in range(len(contacts) - 1):
            anode_idx, anode = contacts[i]
            cathode_idx, cathode = contacts[i + 1]
            
            if cathode['contact_num'] - anode['contact_num'] == 1:
                pair = {
                    'anode_idx': anode_idx,
                    'cathode_idx': cathode_idx,
                    'anode_location': anode['location'],
                    'cathode_location': cathode['location'],
                    'location': f"{anode['location']}-{cathode['location']}",
                    'shaft': shaft_name
                }
                pairs.append(pair)
    
    print(f"  Created {len(pairs)} bipolar pairs from {len(elec_df)} electrodes")
    
    return pairs, elec_df

def create_bipolar_electrode_table(elec_df, pairs):
    """Create enhanced electrode table for bipolar pairs"""
    
    # Identify coordinate columns to average
    coord_systems = ['MNI', 'LEPTO', 'MGRID', 'subINF', 'fsaverageINF', 'ScannerNativeRAS']
    coord_columns = []
    for coord_sys in coord_systems:
        for axis in ['_coord_1', '_coord_2', '_coord_3']:
            col_name = f"{coord_sys}{axis}"
            if col_name in elec_df.columns:
                coord_columns.append(col_name)
    
    single_columns = ['group', 'group_name']
    
    bipolar_rows = []
    
    for pair in pairs:
        anode_idx = pair['anode_idx']
        cathode_idx = pair['cathode_idx']
        
        anode_row = elec_df.loc[anode_idx]
        cathode_row = elec_df.loc[cathode_idx]
        
        new_row = {}
        new_row['location'] = pair['location']
        
        # Single columns (no suffix)
        for col in single_columns:
            if col in elec_df.columns:
                new_row[col] = anode_row[col]
        
        # Averaged coordinates (no suffix)
        for col in coord_columns:
            anode_val = anode_row[col]
            cathode_val = cathode_row[col]
            
            if pd.notna(anode_val) and pd.notna(cathode_val):
                new_row[col] = (anode_val + cathode_val) / 2
            elif pd.notna(anode_val):
                new_row[col] = anode_val
            elif pd.notna(cathode_val):
                new_row[col] = cathode_val
            else:
                new_row[col] = np.nan
        
        # All other columns with _anode and _cathode suffixes
        for col in elec_df.columns:
            if col not in single_columns and col not in coord_columns and col != 'location':
                new_row[f"{col}_anode"] = anode_row[col]
                new_row[f"{col}_cathode"] = cathode_row[col]
        
        bipolar_rows.append(new_row)
    
    return pd.DataFrame(bipolar_rows)

def compute_time_varying_band_power(bipolar_data, sfreq, bands, window_sec=1.0, overlap_sec=0.5):
    """Compute time-varying band power using scipy spectrogram"""
    
    n_samples, n_channels = bipolar_data.shape
    
    nperseg = int(window_sec * sfreq)
    noverlap = int(overlap_sec * sfreq)
    
    # Compute for first channel to get dimensions
    freqs, times, Sxx = signal.spectrogram(
        bipolar_data[:, 0],
        fs=sfreq,
        nperseg=nperseg,
        noverlap=noverlap,
        window='hann'
    )
    
    n_windows = len(times)
    n_bands = len(bands)
    band_power_time = np.zeros((n_windows, n_channels, n_bands))
    
    print(f"  Computing spectrogram: {n_windows} time windows, {len(freqs)} frequency bins")
    
    # Compute for all channels
    for ch_idx in range(n_channels):
        freqs, times, Sxx = signal.spectrogram(
            bipolar_data[:, ch_idx],
            fs=sfreq,
            nperseg=nperseg,
            noverlap=noverlap,
            window='hann'
        )
        
        # Extract power in each band
        for band_idx, (band_name, (fmin, fmax)) in enumerate(bands.items()):
            freq_mask = (freqs >= fmin) & (freqs < fmax)
            band_power_time[:, ch_idx, band_idx] = Sxx[freq_mask, :].mean(axis=0)
    
    return band_power_time, times

def process_raw_to_bipolar_psd(input_nwb_path, output_nwb_path, bands=BANDS, 
                                window_sec=1.0, overlap_sec=0.5, overwrite=False):
    """Complete pipeline: Raw → Bipolar → PSD with enhanced electrode table"""
    
    if os.path.exists(output_nwb_path) and not overwrite:
        print(f"Skipping (already exists): {output_nwb_path}")
        return False
    
    print(f"\nProcessing: {input_nwb_path}")
    
    # Load raw NWB
    io_in = pynwb.NWBHDF5IO(input_nwb_path, 'r')
    nwb_in = io_in.read()
    
    # Get raw series
    series = nwb_in.acquisition['ElectricalSeries_sEEG']
    data = series.data[:]
    sfreq = series.rate
    
    print(f"  Raw data: {data.shape}, {sfreq} Hz, duration={data.shape[0]/sfreq/60:.1f} min")
    
    # Get electrode metadata
    elec_indices = series.electrodes.data[:]
    elec_df = nwb_in.electrodes.to_dataframe().iloc[elec_indices]
    
    # Filter out "out of brain" electrodes
    location_columns = ['location', 'FS_label', 'Desikan_Killiany', 'Destrieux', 
                       'JP_label_original', 'JP_label_std']
    out_of_brain_mask = pd.Series([False] * len(elec_df), index=elec_df.index)
    
    for col in location_columns:
        if col in elec_df.columns:
            out_of_brain_mask |= elec_df[col].astype(str).str.contains(
                'out of brain', case=False, na=False
            )
    
    valid_elec_df = elec_df[~out_of_brain_mask].copy()
    print(f"  Using {len(valid_elec_df)}/{len(elec_df)} electrodes (excluded {out_of_brain_mask.sum()} out-of-brain)")
    
    # Create bipolar pairs
    pairs, filtered_elec_df = create_bipolar_pairs(valid_elec_df)
    
    if len(pairs) == 0:
        print("  ERROR: No valid bipolar pairs created!")
        io_in.close()
        return False
    
    # Compute bipolar data
    print(f"  Computing bipolar montage...")
    bipolar_data = np.zeros((data.shape[0], len(pairs)), dtype=data.dtype)
    
    for i, pair in enumerate(pairs):
        anode_col = np.where(elec_indices == pair['anode_idx'])[0][0]
        cathode_col = np.where(elec_indices == pair['cathode_idx'])[0][0]
        bipolar_data[:, i] = data[:, anode_col] - data[:, cathode_col]
    
    # Compute time-varying band power
    print(f"  Computing band power...")
    band_power_time, times = compute_time_varying_band_power(
        bipolar_data, sfreq, bands, window_sec, overlap_sec
    )
    
    print(f"  Band power shape: {band_power_time.shape}")
    
    # Create enhanced bipolar electrode table
    print(f"  Creating enhanced electrode table...")
    bipolar_elec_df = create_bipolar_electrode_table(elec_df, pairs)
    
    print(f"  Bipolar electrode table: {len(bipolar_elec_df)} rows, {len(bipolar_elec_df.columns)} columns")
    
    # Create output NWB
    print(f"  Creating output NWB...")
    nwb_out = NWBFile(
        session_description=nwb_in.session_description + " - bipolar referenced, band power computed",
        identifier=nwb_in.identifier + "_bipolar_psd",
        session_start_time=nwb_in.session_start_time,
        timestamps_reference_time=nwb_in.timestamps_reference_time,
        file_create_date=datetime.now().astimezone(),
        experimenter=nwb_in.experimenter,
        lab=nwb_in.lab,
        institution=nwb_in.institution,
        subject=nwb_in.subject
    )
    
    # Create device and electrode group
    device = nwb_out.create_device(
        name='NihonKohden',
        description='Nihon Kohden EEG-1200A'
    )
    
    elec_group = nwb_out.create_electrode_group(
        name='sEEG_bipolar',
        description='Bipolar referenced sEEG electrodes (anode-cathode pairs)',
        location='multiple',
        device=device
    )
    
    # Add electrodes with enhanced metadata
    for idx, row in bipolar_elec_df.iterrows():
        electrode_kwargs = {'group': elec_group}
        
        for col in bipolar_elec_df.columns:
            if col not in ['group', 'group_name']:
                electrode_kwargs[col] = row[col]
        
        nwb_out.add_electrode(**electrode_kwargs)
    
    # Create electrode table region
    electrode_region = nwb_out.create_electrode_table_region(
        region=list(range(len(pairs))),
        description='bipolar electrode pairs with averaged coordinates'
    )
    
    # Create processing module
    ecephys_module = nwb_out.create_processing_module(
        name='ecephys',
        description='Processed electrophysiology data including bipolar referencing and band power'
    )
    
    # Add DecompositionSeries
    sampling_rate = 1.0 / (times[1] - times[0])
    
    decomp_series = DecompositionSeries(
        name='band_power',
        description=f'Time-varying power in frequency bands (window={window_sec}s, overlap={overlap_sec}s). '
                    f'Source data was bipolar referenced (adjacent electrode pairs: anode-cathode).',
        data=band_power_time,
        metric='power',
        unit='V^2/Hz',
        rate=sampling_rate,
        source_channels=electrode_region
    )
    
    for band_name, band_limits in bands.items():
        decomp_series.add_band(band_name=band_name, band_limits=band_limits)
    
    ecephys_module.add(decomp_series)
    
    # Write output
    print(f"  Writing: {output_nwb_path}")
    with pynwb.NWBHDF5IO(output_nwb_path, 'w') as io_out:
        io_out.write(nwb_out)
    
    io_in.close()
    
    file_size_mb = os.path.getsize(output_nwb_path) / (1024**2)
    print(f"  Done! Output size: {file_size_mb:.1f} MB")
    
    return True

def main():
    parser = argparse.ArgumentParser(
        description='Process raw iEEG to bipolar-referenced band power'
    )
    parser.add_argument('input_nwb', help='Input NWB file (raw data)')
    parser.add_argument('output_nwb', help='Output NWB file (bipolar + band power)')
    parser.add_argument('--overwrite', action='store_true', 
                       help='Overwrite existing output file')
    
    args = parser.parse_args()
    
    success = process_raw_to_bipolar_psd(
        args.input_nwb, 
        args.output_nwb, 
        overwrite=args.overwrite
    )
    
    if success:
        print("\n✓ Processing completed successfully!")
    else:
        print("\n✗ Processing failed or skipped")
        exit(1)

if __name__ == '__main__':
    main()