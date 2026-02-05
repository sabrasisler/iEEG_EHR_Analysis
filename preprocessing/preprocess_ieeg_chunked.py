#!/usr/bin/env python3
"""
iEEG Preprocessing: Bipolar re-referencing and band power extraction
MEMORY-EFFICIENT VERSION - processes data in chunks to avoid loading entire file

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
import pytz
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

# Memory management
CHUNK_DURATION_SEC = 60  # Process 60 seconds at a time (adjustable)

# ============================================================================
# CORE PROCESSING FUNCTIONS
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

def compute_band_power_chunk(data_chunk, sfreq, bands, nperseg=None, noverlap=None):
    """
    Compute band power for a single chunk of data using Welch's method.
    
    Parameters:
    -----------
    data_chunk : array, shape (n_samples, n_channels)
        Time series data chunk
    sfreq : float
        Sampling frequency
    bands : dict
        Frequency bands {name: (low, high)}
    nperseg : int, optional
        Length of each segment for Welch (default: 2*sfreq = 2 seconds)
    noverlap : int, optional
        Overlap between segments (default: nperseg//2 = 50%)
    
    Returns:
    --------
    band_power_chunk : array, shape (1, n_channels, n_bands)
        Band power for each channel and frequency band
    """
    # Default Welch parameters (2-second windows with 50% overlap)
    if nperseg is None:
        nperseg = int(2 * sfreq)  # 2 seconds
    if noverlap is None:
        noverlap = nperseg // 2  # 50% overlap
    
    n_samples, n_channels = data_chunk.shape
    n_bands = len(bands)
    
    # Initialize output (just 1 time point per chunk)
    band_power_chunk = np.zeros((1, n_channels, n_bands), dtype=np.float32)
    
    # Process each channel
    for ch_idx in range(n_channels):
        # Compute PSD using Welch's method
        freqs, psd = signal.welch(
            data_chunk[:, ch_idx],
            fs=sfreq,
            nperseg=nperseg,
            noverlap=noverlap,
            window='hann',
            scaling='density'
        )
        
        # Extract power for each band
        for band_idx, (band_name, (fmin, fmax)) in enumerate(bands.items()):
            freq_mask = (freqs >= fmin) & (freqs < fmax)
            
            if not freq_mask.any():
                # No frequencies in this band
                print(f"      Warning: No frequencies found for {band_name} "
                      f"({fmin}-{fmax} Hz) on channel {ch_idx}")
                band_power_chunk[0, ch_idx, band_idx] = np.nan
            else:
                # Average power across frequencies in the band
                band_power_chunk[0, ch_idx, band_idx] = psd[freq_mask].mean()
    
    return band_power_chunk

def compute_time_varying_band_power_chunked(series, pairs, electrode_indices, sfreq, bands, 
                                           nperseg=None, overlap_frac=0.5, 
                                           chunk_duration_sec=10):
    """
    Compute band power by processing data in chunks using Welch's method.
    Each chunk produces one band power estimate per channel.
    
    Parameters:
    -----------
    series : ElectricalSeries
        NWB ElectricalSeries object (NOT loaded into memory)
    pairs : list
        Bipolar electrode pairs
    electrode_indices : array
        Indices mapping pairs to original electrodes
    sfreq : float
        Sampling rate
    bands : dict
        Frequency bands
    nperseg : int, optional
        Welch segment size (default: 2*sfreq = 2 seconds)
    overlap_frac : float
        Overlap fraction for Welch segments (default: 0.5)
    chunk_duration_sec : float
        Duration of each time bin in seconds (default: 10s)
    """
    
    n_samples_total = series.data.shape[0]
    n_channels_total = series.data.shape[1]
    n_pairs = len(pairs)
    n_bands = len(bands)
    
    # Calculate chunk parameters
    chunk_size = int(chunk_duration_sec * sfreq)
    
    # Welch parameters
    if nperseg is None:
        nperseg = int(2 * sfreq)  # 2-second segments
    noverlap = int(nperseg * overlap_frac)
    
    # Calculate number of chunks (non-overlapping time bins)
    n_chunks = int(np.ceil(n_samples_total / chunk_size))
    
    print(f"\n  Time-varying band power with Welch's method:")
    print(f"    Time bin size: {chunk_duration_sec}s ({chunk_size:,} samples)")
    print(f"    Welch segment size: {nperseg/sfreq:.1f}s ({nperseg} samples)")
    print(f"    Welch overlap: {overlap_frac*100:.0f}% ({noverlap} samples)")
    print(f"    Total time bins: {n_chunks}")
    
    # Pre-allocate output array
    band_power_all = np.zeros((n_chunks, n_pairs, n_bands), dtype=np.float32)
    times = np.zeros(n_chunks, dtype=np.float32)
    
    for chunk_idx in range(n_chunks):
        # Calculate chunk boundaries
        start_idx = chunk_idx * chunk_size
        end_idx = min(start_idx + chunk_size, n_samples_total)
        
        # Load only this chunk from disk
        data_chunk = series.data[start_idx:end_idx, :]
        
        # Apply bipolar referencing to this chunk
        bipolar_chunk = np.zeros((data_chunk.shape[0], n_pairs), dtype=np.float32)
        for i, pair in enumerate(pairs):
            anode_col = np.where(electrode_indices == pair['anode_idx'])[0][0]
            cathode_col = np.where(electrode_indices == pair['cathode_idx'])[0][0]
            bipolar_chunk[:, i] = data_chunk[:, anode_col] - data_chunk[:, cathode_col]
        
        # Free memory
        del data_chunk
        
        # Compute band power for this chunk using Welch
        band_power_chunk = compute_band_power_chunk(
            bipolar_chunk, sfreq, bands, nperseg=nperseg, noverlap=noverlap
        )
        
        # Store results (squeeze out the time dimension since it's just 1)
        band_power_all[chunk_idx] = band_power_chunk[0]
        times[chunk_idx] = (start_idx + (end_idx - start_idx) / 2) / sfreq  # midpoint time
        
        # Free memory
        del bipolar_chunk, band_power_chunk
        
        # Progress
        progress_pct = 100 * (chunk_idx + 1) / n_chunks
        print(f"    Chunk {chunk_idx+1}/{n_chunks} ({progress_pct:.1f}%)", end='\r')
    
    print()
    
    print(f"  ✓ Final shape: {band_power_all.shape}")
    print(f"    ({band_power_all.shape[0]} time bins × {band_power_all.shape[1]} channels × "
          f"{band_power_all.shape[2]} bands)")
    
    return band_power_all, times

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
    """
    Discover NWB files that need processing
    
    Parameters:
    -----------
    subjects : list of str or str, optional
        Subject ID(s) to process. Can be:
        - None: process all subjects
        - Single string: '259' or 'sub-259'
        - List of strings: ['259', '260'] or ['sub-259', 'sub-260']
    force_overwrite : bool
        Include files even if output already exists
    """
    file_pairs = []
    
    # Normalize subjects input
    if subjects is not None:
        # Handle single subject string
        if isinstance(subjects, str):
            subjects = [subjects]
        
        # Normalize subject IDs (handle both '259' and 'sub-259' formats)
        normalized_subjects = []
        for sub in subjects:
            if sub.startswith('sub-'):
                normalized_subjects.append(sub.replace('sub-', ''))
            else:
                normalized_subjects.append(sub)
        subjects = normalized_subjects
    
    # Build file list
    if subjects is None:
        # Get all subjects
        pattern = f"{DATA_DIR}/sub-*/ses-*/ieeg/sub-*_ses-*_run-*.nwb"
        all_files = glob.glob(pattern)
    else:
        # Get specific subjects
        all_files = []
        for sub_id in subjects:
            pattern = f"{DATA_DIR}/sub-{sub_id}/ses-*/ieeg/sub-{sub_id}_ses-*_run-*.nwb"
            files = glob.glob(pattern)
            all_files.extend(files)
            print(f"  Found {len(files)} files for subject {sub_id}")
    
    # Filter by existing outputs
    for input_path in sorted(all_files):
        output_path = get_output_path(input_path)
        if not force_overwrite and os.path.exists(output_path):
            continue
        file_pairs.append((input_path, output_path))
    
    return file_pairs

# ============================================================================
# MAIN PROCESSING
# ============================================================================

def process_file(input_path, output_path, nperseg=500, overlap_frac=0.5, 
                chunk_duration_sec=60, force_overwrite=False):
    """Process a single NWB file with memory-efficient chunking"""
    start_time = time.time()

    # Capture timezone explicitly
    sf_timezone = pytz.timezone('America/Los_Angeles')
    processing_timestamp = datetime.now(sf_timezone)
    
    # Capture script metadata
    script_name = Path(__file__).name
    
    # Create output directory
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
    
    # Load NWB file
    try:
        io_in = pynwb.NWBHDF5IO(input_path, 'r')
        nwb_in = io_in.read()
    except Exception as e:
        print(f"✗ ERROR loading: {e}")
        return False
    
    series = nwb_in.acquisition['ElectricalSeries_sEEG']
    sfreq = series.rate
    n_samples, n_channels = series.data.shape
    duration_min = n_samples / sfreq / 60
    
    print(f"\nFile info:")
    print(f"  Samples: {n_samples:,} ({duration_min:.1f} minutes)")
    print(f"  Channels: {n_channels}")
    print(f"  Sampling rate: {sfreq} Hz")
    print(f"  Data size: {n_samples * n_channels * 4 / (1024**3):.2f} GB (float32)")
    
    # Get electrode info (small, can load into memory)
    elec_indices = series.electrodes.data[:]
    elec_df = nwb_in.electrodes.to_dataframe().iloc[elec_indices]
    
    # Create bipolar pairs
    print(f"\nCreating bipolar pairs...")
    pairs, filtered_elec_df = create_bipolar_pairs(elec_df)
    
    # Compute band power using chunked processing
    print(f"\nComputing band power (chunked processing)...")
    band_power_time, times = compute_time_varying_band_power_chunked(
        series=series,
        pairs=pairs,
        electrode_indices=elec_indices,
        sfreq=sfreq,
        bands=BANDS,
        nperseg=nperseg,
        overlap_frac=overlap_frac,
        chunk_duration_sec=chunk_duration_sec
    )
    
    # Create output NWB
    print(f"\nCreating output NWB...")
    
    # Create new subject object
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
    
    # Add custom electrode columns
    standard_columns = ['location', 'group', 'group_name']
    custom_columns = [col for col in bipolar_elec_df.columns if col not in standard_columns]
    
    for col in custom_columns:
        nwb_out.add_electrode_column(name=col, description=f'Custom column: {col}')
    
    # Add electrode rows
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
        # Script information
        'script_name': script_name,
        'processing_timestamp': processing_timestamp.isoformat(),
        'timezone': 'America/Los_Angeles',
        
        # Rereferencing
        'rereferencing_method': 'bipolar',
        
        # Welch's method parameters (COMPLETE)
        'method': 'welch',
        'nperseg': nperseg,
        'noverlap': int(nperseg * overlap_frac),
        'overlap_fraction': overlap_frac,
        'window_function': 'hann',
        'scaling': 'density',
        'detrend': False,  # Document default behavior
        
        # Time binning
        'chunk_duration_sec': chunk_duration_sec,
        'time_bin_method': 'non_overlapping',
        
        # Frequency bands
        'frequency_bands': BANDS,
        
        # Data properties
        'original_sampling_rate': float(sfreq),
        'output_sampling_rate': float(sampling_rate),
        'n_time_bins': len(times),
        'n_bipolar_pairs': len(pairs),
        
        # Input file
        'source_file': os.path.basename(input_path)
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
        
        print(f"\n✓ COMPLETED in {total_time:.1f}s ({total_time/60:.1f} min)")
        print(f"  Output size: {file_size_mb:.1f} MB")
        return True
        
    except Exception as e:
        io_in.close()
        print(f"✗ ERROR writing: {e}")
        import traceback
        traceback.print_exc()
        return False

# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='iEEG preprocessing pipeline (memory-efficient)')
    
    # Discovery mode
    parser.add_argument('--discover', action='store_true',
                        help='Discover files and create file list')

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
    # In the main() function, modify the --subjects help text:
    parser.add_argument('--subjects', nargs='+', type=str,
                        help='Subject ID(s) to process. Examples: --subjects 259, --subjects 259 260, --subjects sub-259')
    
    # Processing params
    parser.add_argument('--nperseg', type=int, default=500,
                        help='FFT segment size (default: 500)')
    parser.add_argument('--overlap', type=float, default=0.5,
                        help='Overlap fraction (default: 0.5)')
    parser.add_argument('--chunk-duration', type=float, default=60,
                        help='Chunk duration in seconds (default: 60)')
    
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
        
        for i, (inp, out) in enumerate(file_pairs[:3]):
            print(f"  {i+1}. {Path(inp).name}")
        if len(file_pairs) > 3:
            print(f"  ... and {len(file_pairs)-3} more")
        
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
        
        success = process_file(
            input_path, output_path, 
            args.nperseg, args.overlap, args.chunk_duration,
            args.force_overwrite
        )
        sys.exit(0 if success else 1)
    
    # SINGLE FILE MODE
    if args.input_path and args.output_path:
        if not os.path.exists(args.input_path):
            print(f"✗ ERROR: Input file not found: {args.input_path}")
            sys.exit(1)
        
        success = process_file(
            args.input_path, args.output_path,
            args.nperseg, args.overlap, args.chunk_duration,
            args.force_overwrite
        )
        sys.exit(0 if success else 1)
    
    parser.print_help()

if __name__ == '__main__':
    main()