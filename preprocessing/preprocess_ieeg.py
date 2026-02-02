#!/usr/bin/env python3
"""
iEEG Preprocessing: Bipolar re-referencing and band power extraction

Usage:
    # Discover files
    python preprocess_ieeg.py --discover [--subjects 259 260]
    
    # Process single file
    python preprocess_ieeg.py INPUT_NWB OUTPUT_NWB
    
    # Process from file list (for SLURM array jobs)
    python preprocess_ieeg.py --file-list file_list.txt --task-id $SLURM_ARRAY_TASK_ID
"""

import numpy as np
import pandas as pd
from scipy import signal
import pynwb
from pynwb import NWBFile
from pynwb.file import Subject
from pynwb.ecephys import ElectricalSeries
from pynwb.misc import DecompositionSeries
from datetime import datetime
import os
import re
import time
import argparse
import sys
import json
import glob
from pathlib import Path

# ============================================================================
# CONFIGURATION
# ============================================================================

DATA_DIR = '/oak/stanford/groups/ckeller1/data/iEEG_EHR/iEEG_NWB'

BANDS = {
    'delta': (1.0, 4.0),
    'theta': (4.0, 8.0),
    'alpha': (8.0, 12.0),
    'beta': (15.0, 25.0),
    'gamma': (25.0, 70.0),
    'high_gamma': (70.0, 170.0)
}

# ============================================================================
# CORE PROCESSING FUNCTIONS (FROM NOTEBOOK - UNCHANGED)
# ============================================================================

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
    elec_df = elec_df.copy()
    elec_df['shaft'] = None
    elec_df['contact_num'] = None
    
    for idx, row in elec_df.iterrows():
        shaft, num = parse_electrode_shaft(row['location'])
        elec_df.at[idx, 'shaft'] = shaft
        elec_df.at[idx, 'contact_num'] = num
    
    valid_parse = elec_df['shaft'].notna()
    if (~valid_parse).sum() > 0:
        print(f"  Warning: {(~valid_parse).sum()} electrodes have unparseable locations")
        elec_df = elec_df[valid_parse].copy()
    
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
        
        for col in single_columns:
            if col in elec_df.columns:
                new_row[col] = anode_row[col]
        
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
        
        for col in elec_df.columns:
            if col not in single_columns and col not in coord_columns and col != 'location':
                new_row[f"{col}_anode"] = anode_row[col]
                new_row[f"{col}_cathode"] = cathode_row[col]
        
        bipolar_rows.append(new_row)
    
    return pd.DataFrame(bipolar_rows)

def compute_time_varying_band_power(bipolar_data, sfreq, bands, nperseg=500, overlap_frac=0.5):
    """Compute time-varying band power using scipy spectrogram"""
    n_samples, n_channels = bipolar_data.shape
    noverlap = int(nperseg * overlap_frac)  
    
    freqs, times, Sxx = signal.spectrogram(
        bipolar_data[:, 0], fs=sfreq, noverlap=noverlap, nperseg=nperseg, window='hann'
    )
    
    n_windows = len(times)
    n_bands = len(bands)
    band_power_time = np.zeros((n_windows, n_channels, n_bands))
    
    print(f"  Computing spectrogram: {n_windows} time windows, {len(freqs)} frequency bins")
    
    for ch_idx in range(n_channels):
        freqs, times, Sxx = signal.spectrogram(
            bipolar_data[:, ch_idx], fs=sfreq, nperseg=nperseg, noverlap=noverlap, window='hann'
        )
        
        for band_idx, (band_name, (fmin, fmax)) in enumerate(bands.items()):
            freq_mask = (freqs >= fmin) & (freqs < fmax)
            band_power_time[:, ch_idx, band_idx] = Sxx[freq_mask, :].mean(axis=0)
        
        if (ch_idx + 1) % 10 == 0 or ch_idx == n_channels - 1:
            print(f"  Progress: {ch_idx+1}/{n_channels} channels", end='\r')
    
    print()
    return band_power_time, times

# ============================================================================
# FILE DISCOVERY
# ============================================================================

def get_output_path(input_path):
    """Convert input path to output path"""
    input_path = Path(input_path)
    filename = input_path.stem
    output_filename = f"{filename}_bipolar_psd.nwb"
    output_dir = input_path.parent.parent / 'preprocessed'
    return str(output_dir / output_filename)

def discover_files(subjects=None, force_overwrite=False):
    """Discover NWB files that need processing"""
    file_pairs = []
    
    if subjects is None:
        pattern = f"{DATA_DIR}/sub-*/ses-*/ieeg/sub-*_ses-*_run-*.nwb"
        all_files = glob.glob(pattern)
    else:
        all_files = []
        for sub_id in subjects:
            pattern = f"{DATA_DIR}/sub-{sub_id}/ses-*/ieeg/sub-{sub_id}_ses-*_run-*.nwb"
            all_files.extend(glob.glob(pattern))
    
    for input_path in sorted(all_files):
        output_path = get_output_path(input_path)
        if not force_overwrite and os.path.exists(output_path):
            continue
        file_pairs.append((input_path, output_path))
    
    return file_pairs

# ============================================================================
# MAIN PROCESSING
# ============================================================================

def process_file(input_path, output_path, nperseg=500, overlap_frac=0.5, force_overwrite=False):
    """Process a single NWB file"""
    start_time = time.time()
    
    # Create output directory first (before any processing)
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        try:
            os.makedirs(output_dir, exist_ok=True)
            print(f"✓ Created output directory: {output_dir}")
        except Exception as e:
            print(f"✗ ERROR: Could not create output directory: {e}")
            return False
    
    if os.path.exists(output_path) and not force_overwrite:
        print(f"⚠ Output exists (skipping): {output_path}")
        return True
    
    print("=" * 70)
    print(f"PROCESSING: {os.path.basename(input_path)}")
    print("=" * 70)
    
    # Load
    try:
        io_in = pynwb.NWBHDF5IO(input_path, 'r')
        nwb_in = io_in.read()
    except Exception as e:
        print(f"✗ ERROR loading: {e}")
        return False
    
    series = nwb_in.acquisition['ElectricalSeries_sEEG']
    sfreq = series.rate
    n_samples, n_channels = series.data.shape
    
    print(f"\nLoading data: {n_samples:,} samples, {n_channels} channels, {sfreq} Hz")
    data = series.data[:]
    
    elec_indices = series.electrodes.data[:]
    elec_df = nwb_in.electrodes.to_dataframe().iloc[elec_indices]
    
    # Bipolar pairs
    pairs, filtered_elec_df = create_bipolar_pairs(elec_df)
    
    # Bipolar re-referencing
    print(f"\nComputing bipolar re-referencing...")
    bipolar_data = np.zeros((data.shape[0], len(pairs)), dtype=data.dtype)
    for i, pair in enumerate(pairs):
        anode_col = np.where(elec_indices == pair['anode_idx'])[0][0]
        cathode_col = np.where(elec_indices == pair['cathode_idx'])[0][0]
        bipolar_data[:, i] = data[:, anode_col] - data[:, cathode_col]
    
    print(f"✓ Bipolar data: {bipolar_data.shape}")
    
    # Band power
    print(f"\nComputing band power...")
    band_power_time, times = compute_time_varying_band_power(
        bipolar_data, sfreq, BANDS, nperseg=nperseg, overlap_frac=overlap_frac
    )
    print(f"✓ Band power: {band_power_time.shape}")
    
    # Create output NWB
    print(f"\nCreating output NWB...")
    
    # Create new subject object (don't reuse from input)
    if nwb_in.subject is not None:
        subject_out = Subject(
            subject_id=nwb_in.subject.subject_id,
            age=nwb_in.subject.age if hasattr(nwb_in.subject, 'age') else None,
            sex=nwb_in.subject.sex if hasattr(nwb_in.subject, 'sex') else None,
            species=nwb_in.subject.species if hasattr(nwb_in.subject, 'species') else None,
            description=nwb_in.subject.description if hasattr(nwb_in.subject, 'description') else None
        )
    else:
        subject_out = None
    
    nwb_out = NWBFile(
        session_description=nwb_in.session_description + " - bipolar referenced, band power computed",
        identifier=nwb_in.identifier + "_bipolar_psd",
        session_start_time=nwb_in.session_start_time,
        timestamps_reference_time=nwb_in.timestamps_reference_time,
        file_create_date=datetime.now().astimezone(),
        experimenter=nwb_in.experimenter,
        lab=nwb_in.lab,
        institution=nwb_in.institution,
        subject=subject_out
    )
    
    # Create device
    device = nwb_out.create_device(
        name='NihonKohden',
        description='Nihon Kohden EEG-1200A'
    )
    
    # Create electrode group
    elec_group = nwb_out.create_electrode_group(
        name='sEEG_bipolar',
        description='Bipolar referenced sEEG electrodes (anode-cathode pairs)',
        location='multiple',
        device=device
    )
    
    bipolar_elec_df = create_bipolar_electrode_table(filtered_elec_df, pairs)
    
    # Add custom electrode columns (exclude standard NWB columns)
    standard_columns = ['location', 'group', 'group_name']
    custom_columns = [col for col in bipolar_elec_df.columns if col not in standard_columns]
    
    print(f"  Adding {len(custom_columns)} custom columns...")
    for col in custom_columns:
        nwb_out.add_electrode_column(name=col, description=f'Custom column: {col}')
    
    # Add electrode rows
    print(f"  Adding {len(bipolar_elec_df)} electrode rows...")
    for idx, row in bipolar_elec_df.iterrows():
        electrode_kwargs = {'group': elec_group}
        
        for col in bipolar_elec_df.columns:
            if col not in ['group', 'group_name']:
                electrode_kwargs[col] = row[col]
        
        nwb_out.add_electrode(**electrode_kwargs)
    
    electrode_region = nwb_out.create_electrode_table_region(
        region=list(range(len(pairs))),
        description='bipolar electrode pairs with averaged coordinates'
    )
    
    ecephys_module = nwb_out.create_processing_module(
        name='ecephys',
        description='Processed electrophysiology data including bipolar referencing and band power'
    )
    
    sampling_rate = 1.0 / (times[1] - times[0])
    processing_params = {
        'rereferencing_method': 'bipolar',
        'nperseg': nperseg,
        'overlap_fraction': overlap_frac,
        'window_function': 'hann',
        'frequency_bands': BANDS,
        'processing_date': datetime.now().isoformat()
    }
    
    decomp_series = DecompositionSeries(
        name='band_power',
        description=f'Time-varying power in frequency bands. Params: {json.dumps(processing_params)}',
        data=band_power_time,
        metric='power',
        unit='V^2/Hz',
        rate=sampling_rate,
        source_channels=electrode_region
    )
    
    for band_name, band_limits in BANDS.items():
        decomp_series.add_band(band_name=band_name, band_limits=band_limits)
    
    ecephys_module.add(decomp_series)
    
    # Write
    print(f"\nWriting to: {output_path}")
    try:
        with pynwb.NWBHDF5IO(output_path, 'w') as io_out:
            io_out.write(nwb_out)
        io_in.close()
        
        file_size_mb = os.path.getsize(output_path) / (1024**2)
        total_time = time.time() - start_time
        
        print(f"\n✓ COMPLETED in {total_time:.1f}s ({file_size_mb:.1f} MB)")
        return True
        
    except Exception as e:
        io_in.close()
        print(f"✗ ERROR writing: {e}")
        return False

# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='iEEG preprocessing pipeline')
    
    # Discovery mode
    parser.add_argument('--discover', action='store_true',
                        help='Discover files and create file list')
    parser.add_argument('--subjects', nargs='+', type=str,
                        help='Subject IDs to process (e.g., 259 260)')
    parser.add_argument('--force-overwrite', action='store_true',
                        help='Include files that already exist')
    
    # Array job mode
    parser.add_argument('--file-list', type=str,
                        help='File list for array jobs')
    parser.add_argument('--task-id', type=int,
                        help='Task ID for array jobs (1-indexed)')
    
    # Single file mode
    parser.add_argument('input_path', nargs='?', type=str,
                        help='Input NWB file')
    parser.add_argument('output_path', nargs='?', type=str,
                        help='Output NWB file')
    
    # Processing params
    parser.add_argument('--nperseg', type=int, default=500,
                        help='FFT segment size (default: 500)')
    parser.add_argument('--overlap', type=float, default=0.5,
                        help='Overlap fraction (default: 0.5)')
    
    args = parser.parse_args()
    
    # DISCOVERY MODE
    if args.discover:
        print("=" * 70)
        print("DISCOVERING FILES")
        print("=" * 70)
        
        file_pairs = discover_files(subjects=args.subjects, force_overwrite=args.force_overwrite)
        
        print(f"\nFound {len(file_pairs)} files to process")
        
        if len(file_pairs) == 0:
            print("✓ No files need processing")
            return
        
        # Show sample
        for i, (inp, out) in enumerate(file_pairs[:3]):
            print(f"  {i+1}. {Path(inp).name}")
        if len(file_pairs) > 3:
            print(f"  ... and {len(file_pairs)-3} more")
        
        # Write file list
        with open('file_list.txt', 'w') as f:
            for inp, out in file_pairs:
                f.write(f"{inp}\t{out}\n")
        
        print(f"\n✓ File list written to: file_list.txt")
        print(f"\nNext: Edit submit_preprocessing.sh to set --array=1-{len(file_pairs)}%8")
        print(f"Then: sbatch submit_preprocessing.sh")
        return
    
    # ARRAY JOB MODE
    if args.file_list and args.task_id:
        with open(args.file_list) as f:
            lines = f.readlines()
        
        if args.task_id < 1 or args.task_id > len(lines):
            print(f"✗ ERROR: Task ID {args.task_id} out of range (1-{len(lines)})")
            sys.exit(1)
        
        line = lines[args.task_id - 1].strip()
        input_path, output_path = line.split('\t')
        
        success = process_file(input_path, output_path, args.nperseg, args.overlap, args.force_overwrite)
        sys.exit(0 if success else 1)
    
    # SINGLE FILE MODE
    if args.input_path and args.output_path:
        if not os.path.exists(args.input_path):
            print(f"✗ ERROR: Input file not found: {args.input_path}")
            sys.exit(1)
        
        success = process_file(args.input_path, args.output_path, args.nperseg, args.overlap, args.force_overwrite)
        sys.exit(0 if success else 1)
    
    parser.print_help()

if __name__ == '__main__':
    main()