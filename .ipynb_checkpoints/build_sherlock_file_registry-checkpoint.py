#!/usr/bin/env python3
"""
create_ieeg_file_registry.py
Create a registry of all iEEG files tracking raw and preprocessed files.
Supports incremental updates to avoid reprocessing unchanged files.
"""
import numpy as np
import pandas as pd
from pynwb import NWBHDF5IO
from pathlib import Path
from datetime import datetime
from typing import Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

BASE_PATH = Path('/oak/stanford/groups/ckeller1/data/iEEG_EHR/iEEG_NWB')

# ============================================================================
# TIMING EXTRACTION
# ============================================================================

def extract_timing_from_preprocessed_nwb(nwb_file: Path) -> Optional[Tuple]:
    """
    Extract start/end times from preprocessed NWB file (band_power data).
    Fast operation since preprocessed files are small.
    
    Returns:
        (start_datetime, end_datetime, n_channels, n_timepoints, sampling_rate) or None
    """
    try:
        with NWBHDF5IO(str(nwb_file), 'r') as io:
            nwb = io.read()
            
            # Get session start time
            session_start = nwb.session_start_time
            
            # Load band power data - use same access method as seizure analysis
            band_power_series = nwb.processing['ecephys']["band_power"]
            
            # Get data shape (don't load full data, just shape)
            band_power_data_shape = band_power_series.data.shape  # (timepoints, channels, bands)
            n_timepoints = band_power_data_shape[0]
            n_channels = band_power_data_shape[1]
            
            # Get timestamps efficiently - only load first and last
            if band_power_series.timestamps is not None:
                # Access first and last timestamp directly without loading full array
                start_offset = float(band_power_series.timestamps[0])
                end_offset = float(band_power_series.timestamps[-1])
                sampling_rate = None
            else:
                # Reconstruct from starting_time and rate (no array loading needed)
                starting_time = band_power_series.starting_time
                rate = band_power_series.rate
                start_offset = starting_time
                end_offset = starting_time + (n_timepoints - 1) / rate
                sampling_rate = rate
            
            # Convert timestamps to datetime (relative to session start)
            start_datetime = pd.to_datetime(session_start) + pd.Timedelta(seconds=float(start_offset))
            end_datetime = pd.to_datetime(session_start) + pd.Timedelta(seconds=float(end_offset))
            
            # Remove timezone info
            start_datetime = start_datetime.tz_localize(None)
            end_datetime = end_datetime.tz_localize(None)
            
            return (start_datetime, end_datetime, n_channels, n_timepoints, sampling_rate)
            
    except Exception as e:
        print(f"  Warning: Could not extract timing from {nwb_file.name}: {e}")
        return None

# ============================================================================
# FILE DISCOVERY
# ============================================================================

def parse_filename(filename: str) -> Optional[dict]:
    """
    Parse BIDS-style filename to extract sub_id, ses_id, run_id.
    
    Examples: 
    - sub-001_ses-01_run-EA63019M.nwb
    - sub-001_ses-01_run-EA63019M_bipolar_psd.nwb
    
    Returns: {'sub_id': 'sub-001', 'ses_id': 'ses-01', 'run_id': 'run-EA63019M'}
    """
    parts = filename.split('_')
    info = {}
    
    for part in parts:
        if part.startswith('sub-'):
            info['sub_id'] = part
        elif part.startswith('ses-'):
            info['ses_id'] = part
        elif part.startswith('run-'):
            # Remove extension if present
            run_part = part.split('.')[0]
            info['run_id'] = run_part
        elif part.startswith('file-'):
            # Some files use 'file-' instead of 'run-'
            file_part = part.split('.')[0]
            info['run_id'] = file_part
    
    # Check if we got all required parts
    if 'sub_id' in info and 'ses_id' in info:
        if 'run_id' not in info:
            info['run_id'] = 'none'
        return info
    else:
        return None


def find_preprocessed_for_run(base_path: Path, 
                              sub_id: str, 
                              ses_id: str, 
                              run_id: str) -> Optional[Path]:
    """
    Find preprocessed file corresponding to a raw file.
    
    Returns:
        Path to preprocessed file or None
    """
    preprocessed_dir = base_path / sub_id / ses_id / 'preprocessed'
    
    if not preprocessed_dir.exists():
        return None
    
    # Exact pattern match
    pattern = f"{sub_id}_{ses_id}_{run_id}_bipolar_psd.nwb"
    matching_files = list(preprocessed_dir.glob(pattern))
    
    if len(matching_files) > 0:
        return matching_files[0]
    
    return None


def load_existing_registry(output_path: Path) -> Optional[pd.DataFrame]:
    """Load existing registry if it exists."""
    if output_path.exists():
        print(f"Found existing registry: {output_path}")
        registry = pd.read_csv(output_path)
        
        # Convert datetime columns
        date_cols = ['raw_file_created', 'raw_file_modified', 
                     'preprocessed_file_created', 'preprocessed_file_modified',
                     'start_datetime', 'end_datetime']
        for col in date_cols:
            if col in registry.columns:
                registry[col] = pd.to_datetime(registry[col])
        
        print(f"  Loaded {len(registry)} existing entries")
        return registry
    return None


def create_file_key(sub_id: str, ses_id: str, run_id: str) -> str:
    """Create unique key for a file."""
    return f"{sub_id}|{ses_id}|{run_id}"


def needs_update(row: pd.Series, raw_file: Path, prep_file: Optional[Path]) -> bool:
    """
    Check if a registry entry needs updating.
    
    Update needed if:
    - Raw file modified timestamp changed
    - Preprocessed file status changed (new or deleted)
    - Preprocessed file modified timestamp changed
    """
    # Check raw file modification time
    raw_mtime = datetime.fromtimestamp(raw_file.stat().st_mtime)
    if pd.isna(row['raw_file_modified']) or row['raw_file_modified'] != raw_mtime:
        return True
    
    # Check preprocessed file status change
    has_prep = prep_file is not None
    if row['has_preprocessed'] != has_prep:
        return True
    
    # If preprocessed exists, check its modification time
    if has_prep and prep_file is not None:
        prep_mtime = datetime.fromtimestamp(prep_file.stat().st_mtime)
        if pd.isna(row['preprocessed_file_modified']) or row['preprocessed_file_modified'] != prep_mtime:
            return True
    
    return False


# ============================================================================
# REGISTRY CREATION
# ============================================================================

def create_file_registry(base_path: Path = BASE_PATH,
                        output_file: str = 'ieeg_file_registry.csv',
                        force_reset: bool = False) -> pd.DataFrame:
    """
    Create comprehensive file registry with incremental updates.
    
    Args:
        base_path: Root directory for iEEG files
        output_file: Output CSV filename
        force_reset: If True, rebuild entire registry from scratch
    
    Workflow:
    1. Load existing registry (unless force_reset)
    2. Scan all raw files
    3. For each file, check if it needs updating
    4. Only process new/changed files
    5. Merge with existing registry
    
    Returns:
        DataFrame with complete registry
    """
    output_path = base_path / output_file
    
    print(f"\n{'='*60}")
    print("Creating iEEG File Registry")
    if force_reset:
        print("(FORCE RESET - rebuilding from scratch)")
    else:
        print("(Incremental update mode)")
    print(f"{'='*60}\n")
    
    # Load existing registry
    existing_registry = None if force_reset else load_existing_registry(output_path)
    
    # Create lookup dictionary for existing entries
    existing_lookup = {}
    if existing_registry is not None:
        for idx, row in existing_registry.iterrows():
            key = create_file_key(row['sub_id'], row['ses_id'], row['run_id'])
            existing_lookup[key] = row
    
    # Step 1: Find all raw NWB files
    print("Step 1: Scanning for raw NWB files...")
    raw_files = list(base_path.glob('sub-*/ses-*/ieeg/*.nwb'))
    print(f"Found {len(raw_files)} raw NWB files")
    
    if len(raw_files) == 0:
        print("No raw files found!")
        return None
    
    # Count preprocessed files
    prep_files = list(base_path.glob('sub-*/ses-*/preprocessed/*_bipolar_psd.nwb'))
    print(f"Found {len(prep_files)} preprocessed files")
    
    # Step 2: Identify files to process
    print("\nStep 2: Identifying files needing updates...")
    files_to_process = []
    files_unchanged = []
    files_new = []
    
    for raw_file in raw_files:
        file_info = parse_filename(raw_file.name)
        
        if file_info is None:
            print(f"  Warning: Could not parse filename: {raw_file.name}")
            continue
        
        key = create_file_key(file_info['sub_id'], file_info['ses_id'], file_info['run_id'])
        prep_file = find_preprocessed_for_run(base_path, file_info['sub_id'], 
                                              file_info['ses_id'], file_info['run_id'])
        
        # Check if this is a new file or needs update
        if key in existing_lookup:
            existing_row = existing_lookup[key]
            if needs_update(existing_row, raw_file, prep_file):
                files_to_process.append((raw_file, file_info, prep_file))
            else:
                files_unchanged.append(key)
        else:
            files_to_process.append((raw_file, file_info, prep_file))
            files_new.append(key)
    
    print(f"\nFile status:")
    print(f"  New files: {len(files_new)}")
    print(f"  Changed files: {len(files_to_process) - len(files_new)}")
    print(f"  Unchanged files: {len(files_unchanged)}")
    print(f"  Total to process: {len(files_to_process)}")
    
    # If nothing to process, return existing registry
    if len(files_to_process) == 0:
        print("\nNo updates needed - registry is up to date!")
        return existing_registry
    
    # Step 3: Process files that need updating
    print(f"\nStep 3: Processing {len(files_to_process)} files...")
    registry_data = []
    
    for idx, (raw_file, file_info, prep_file) in enumerate(files_to_process):
        if (idx + 1) % 50 == 0 or (idx + 1) == len(files_to_process):
            print(f"  Processed {idx + 1}/{len(files_to_process)} files...")
        
        # Get raw file stats
        raw_stat = raw_file.stat()
        
        # Get preprocessed file stats if it exists
        if prep_file:
            prep_stat = prep_file.stat()
            prep_data = {
                'preprocessed_file_path': str(prep_file),
                'preprocessed_file_name': prep_file.name,
                'preprocessed_file_created': datetime.fromtimestamp(prep_stat.st_ctime),
                'preprocessed_file_modified': datetime.fromtimestamp(prep_stat.st_mtime),
                'preprocessed_file_size_mb': prep_stat.st_size / (1024 * 1024),
                'has_preprocessed': True
            }
        else:
            prep_data = {
                'preprocessed_file_path': None,
                'preprocessed_file_name': None,
                'preprocessed_file_created': None,
                'preprocessed_file_modified': None,
                'preprocessed_file_size_mb': None,
                'has_preprocessed': False
            }
        
        # Extract timing from preprocessed file
        if prep_file:
            timing = extract_timing_from_preprocessed_nwb(prep_file)
            if timing:
                start_dt, end_dt, n_ch, n_tp, sr = timing
                timing_data = {
                    'start_datetime': start_dt,
                    'end_datetime': end_dt,
                    'duration_minutes': (end_dt - start_dt).total_seconds() / 60,
                    'n_channels': n_ch,
                    'n_timepoints': n_tp,
                    'sampling_rate': sr
                }
            else:
                timing_data = {
                    'start_datetime': None,
                    'end_datetime': None,
                    'duration_minutes': None,
                    'n_channels': None,
                    'n_timepoints': None,
                    'sampling_rate': None
                }
        else:
            timing_data = {
                'start_datetime': None,
                'end_datetime': None,
                'duration_minutes': None,
                'n_channels': None,
                'n_timepoints': None,
                'sampling_rate': None
            }
        
        # Combine all data
        row_data = {
            'sub_id': file_info['sub_id'],
            'ses_id': file_info['ses_id'],
            'run_id': file_info['run_id'],
            'raw_file_path': str(raw_file),
            'raw_file_name': raw_file.name,
            'raw_file_created': datetime.fromtimestamp(raw_stat.st_ctime),
            'raw_file_modified': datetime.fromtimestamp(raw_stat.st_mtime),
            'raw_file_size_mb': raw_stat.st_size / (1024 * 1024),
            **prep_data,
            **timing_data
        }
        
        registry_data.append(row_data)
    
    # Step 4: Merge with existing registry
    new_registry = pd.DataFrame(registry_data)
    
    if existing_registry is not None:
        print("\nStep 4: Merging with existing registry...")
        
        # Remove old entries that were updated
        updated_keys = set([create_file_key(row['sub_id'], row['ses_id'], row['run_id']) 
                           for _, row in new_registry.iterrows()])
        
        existing_registry['_key'] = existing_registry.apply(
            lambda r: create_file_key(r['sub_id'], r['ses_id'], r['run_id']), axis=1
        )
        kept_rows = existing_registry[~existing_registry['_key'].isin(updated_keys)].drop('_key', axis=1)
        
        # Combine
        registry = pd.concat([kept_rows, new_registry], ignore_index=True)
        print(f"  Kept {len(kept_rows)} unchanged entries")
        print(f"  Added/updated {len(new_registry)} entries")
    else:
        registry = new_registry
    
    # Step 5: Sort and save
    registry = registry.sort_values(['sub_id', 'ses_id', 'run_id'])
    registry.to_csv(output_path, index=False)
    
    # Print summary
    print(f"\n{'='*60}")
    print("Registry Creation Complete")
    print(f"{'='*60}")
    print(f"Output: {output_path}")
    print(f"\nSummary:")
    print(f"  Total runs: {len(registry)}")
    print(f"  Total subjects: {registry['sub_id'].nunique()}")
    print(f"  Total sessions: {registry.groupby(['sub_id', 'ses_id']).ngroups}")
    print(f"  Preprocessed runs: {registry['has_preprocessed'].sum()}")
    print(f"  Unprocessed runs: {(~registry['has_preprocessed']).sum()}")
    print(f"  Runs with timing info: {registry['start_datetime'].notna().sum()}")
    
    print(f"\nFile sizes:")
    print(f"  Total raw data: {registry['raw_file_size_mb'].sum():.2f} MB ({registry['raw_file_size_mb'].sum()/1024:.2f} GB)")
    if registry['preprocessed_file_size_mb'].notna().any():
        print(f"  Total preprocessed data: {registry['preprocessed_file_size_mb'].sum():.2f} MB ({registry['preprocessed_file_size_mb'].sum()/1024:.2f} GB)")
    
    print(f"{'='*60}\n")
    
    return registry


# ============================================================================
# ANALYSIS FUNCTIONS
# ============================================================================

def print_processing_status(registry_df: pd.DataFrame):
    """Print summary of processing status."""
    print(f"\n{'='*60}")
    print("Processing Status Summary")
    print(f"{'='*60}\n")
    
    # Overall status
    total = len(registry_df)
    processed = registry_df['has_preprocessed'].sum()
    unprocessed = total - processed
    
    print(f"Total runs: {total}")
    print(f"Preprocessed: {processed} ({processed/total*100:.1f}%)")
    print(f"Unprocessed: {unprocessed} ({unprocessed/total*100:.1f}%)")
    
    # By session
    print(f"\nStatus by session:")
    session_status = registry_df.groupby(['sub_id', 'ses_id']).agg({
        'run_id': 'count',
        'has_preprocessed': 'sum'
    })
    session_status.columns = ['total_runs', 'preprocessed_runs']
    session_status['unprocessed_runs'] = session_status['total_runs'] - session_status['preprocessed_runs']
    session_status['percent_done'] = (session_status['preprocessed_runs'] / session_status['total_runs'] * 100).round(1)
    
    print(session_status)


def find_old_preprocessed_files(registry_df: pd.DataFrame, 
                                cutoff_date: str = '2025-02-08') -> pd.DataFrame:
    """Find preprocessed files created before a cutoff date."""
    cutoff = pd.to_datetime(cutoff_date)
    
    old_files = registry_df[
        (registry_df['has_preprocessed']) &
        (registry_df['preprocessed_file_created'] < cutoff)
    ].copy()
    
    print(f"\nPreprocessed files created before {cutoff_date}:")
    print(f"  Total: {len(old_files)} files")
    if len(old_files) > 0:
        print(f"  Total size: {old_files['preprocessed_file_size_mb'].sum():.2f} MB")
        print(f"\nAffected sessions:")
        print(old_files.groupby(['sub_id', 'ses_id']).size())
    
    return old_files


def find_files_for_time_window(registry_df: pd.DataFrame,
                               sub_id: str,
                               ses_id: str,
                               target_time: pd.Timestamp,
                               window_minutes: int = 7) -> pd.DataFrame:
    """
    Fast lookup: Find files that overlap with a time window.
    
    Args:
        registry_df: Registry DataFrame
        sub_id: Subject ID (e.g., 'sub-001')
        ses_id: Session ID (e.g., 'ses-01')
        target_time: Target timestamp
        window_minutes: Window around target time
    
    Returns:
        DataFrame of relevant files (only those with timing info)
    """
    # Filter to session with timing info
    session_files = registry_df[
        (registry_df['sub_id'] == sub_id) &
        (registry_df['ses_id'] == ses_id) &
        (registry_df['start_datetime'].notna())
    ].copy()
    
    if len(session_files) == 0:
        print(f"No files with timing info for {sub_id}/{ses_id}")
        return pd.DataFrame()
    
    # Find overlapping files
    window_start = target_time - pd.Timedelta(minutes=window_minutes)
    window_end = target_time + pd.Timedelta(minutes=window_minutes)
    
    relevant = session_files[
        (session_files['start_datetime'] <= window_end) &
        (session_files['end_datetime'] >= window_start)
    ]
    
    return relevant


def get_files_safe_to_delete_from_lambda(registry_df: pd.DataFrame) -> pd.DataFrame:
    """
    Get list of raw files that are safe to delete from lambda.
    Safe = has been preprocessed (so raw is backed up on Sherlock).
    
    Returns:
        DataFrame with raw files that have corresponding preprocessed files
    """
    safe_to_delete = registry_df[registry_df['has_preprocessed']].copy()
    
    print(f"\n{'='*60}")
    print("Files Safe to Delete from Lambda")
    print(f"{'='*60}\n")
    print(f"Total runs with preprocessed files: {len(safe_to_delete)}")
    print(f"Total raw data size: {safe_to_delete['raw_file_size_mb'].sum():.2f} MB")
    print(f"  ({safe_to_delete['raw_file_size_mb'].sum()/1024:.2f} GB)")
    
    return safe_to_delete[['sub_id', 'ses_id', 'run_id', 'raw_file_path', 
                           'raw_file_size_mb', 'preprocessed_file_path']]


# ============================================================================
# MAIN
# ============================================================================

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Create comprehensive iEEG file registry with incremental updates'
    )
    parser.add_argument('--output', type=str, default='sherlock_file_registry.csv',
                       help='Output CSV filename')
    parser.add_argument('--force-reset', action='store_true',
                       help='Force complete rebuild of registry (ignore existing)')
    parser.add_argument('--analyze', action='store_true',
                       help='Print analysis after creating registry')
    parser.add_argument('--cutoff-date', type=str, default='2025-02-08',
                       help='Cutoff date for old file analysis (YYYY-MM-DD)')
    
    args = parser.parse_args()
    
    # Create registry
    registry = create_file_registry(output_file=args.output, force_reset=args.force_reset)
    
    if registry is not None and args.analyze:
        print_processing_status(registry)
        find_old_preprocessed_files(registry, args.cutoff_date)
        get_files_safe_to_delete_from_lambda(registry)