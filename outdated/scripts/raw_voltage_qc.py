#!/usr/bin/env python3
"""
Stage 1: Raw Voltage QC
=======================
Detects flatline and saturation artifacts in raw monopolar iEEG voltage.
Operates on raw NWB files BEFORE bipolar re-referencing.

Outputs:
  - Per-run QC sidecar JSON  -> DERIV_DIR/qc/voltage/sub-XX/ses-XX/
  - Optional figures          -> FIGURES_DIR/sub-XX_ses-XX_run-XX_*.png

Usage:
  # Discover available runs for a subject
  python raw_voltage_qc.py --discover --subjects 019

  # Run QC on a subject (all sessions/runs)
  python raw_voltage_qc.py --subjects 019

  # Run QC with figures
  python raw_voltage_qc.py --subjects 019 --sessions 01 --figures

  # SLURM array job
  python raw_voltage_qc.py --file-list file_list.txt --task-id $SLURM_ARRAY_TASK_ID

  # Overwrite existing QC files
  python raw_voltage_qc.py --subjects 019 --force-overwrite
"""

import numpy as np
import json
import os
import sys
import glob
import base64
import argparse
import subprocess
import time
from pathlib import Path
from datetime import datetime

import pynwb

# ============================================================================
# CONFIGURATION — edit paths here, all other logic is path-agnostic
# ============================================================================

RAW_DIR     = '/oak/stanford/groups/ckeller1/data/iEEG_EHR/iEEG_NWB'
DERIV_DIR   = '/oak/stanford/groups/ckeller1/data/iEEG_EHR/derivatives/sisler'
FIGURES_DIR = '/home/groups/ckeller1/sisler/figures/qc/voltage'

# ============================================================================
# QC THRESHOLDS — named constants, all saved in output JSON
# ============================================================================

# NWB ElectricalSeries data is read raw (in volts, per series.unit); thresholds
# below are expressed in volts/volts² to match, rather than converting the data.
#
# Flatline: flag window if variance (V²) is below this value.
# 0.5 µV² (5e-13 V²) is conservative — real iEEG can be quiet but rarely this quiet.
# Tune by inspecting Figure 2 (variance distribution) to find the natural gap
# between dead and live channels in your data.
FLATLINE_VAR_THRESHOLD = 0.5e-12      # V²  (0.5 µV²)

# Saturation: flag window if max |voltage| exceeds this value.
# Nihon Kohden rails vary by gain setting; 3000 µV is a reasonable starting point.
# Inspect your raw traces to confirm.
SATURATION_ABS_THRESHOLD = 3000e-6    # V  (3000 µV)

# Saturation: flag window if this many consecutive samples hold identical values.
# Catches digital clipping even when below the absolute rail.
MIN_IDENTICAL_SAMPLES = 10           # samples

# Channel classification thresholds (fraction of windows flagged)
DEAD_CHANNEL_THRESHOLD       = 0.80  # >80% flagged → DEAD
UNRELIABLE_CHANNEL_THRESHOLD = 0.15  # >15% flagged → UNRELIABLE

# Heatmap display pooling resolution
HEATMAP_POOL_MIN = 2.0               # pool bins into N-minute blocks for display

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def get_git_commit():
    """Return short git commit hash, or 'unknown' if not in a repo."""
    try:
        return subprocess.check_output(
            ['git', 'rev-parse', '--short', 'HEAD'],
            stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return 'unknown'


def bool_array_to_b64(arr: np.ndarray) -> str:
    """Encode a boolean numpy array as a base64 string for compact JSON storage."""
    return base64.b64encode(np.packbits(arr)).decode('ascii')


def b64_to_bool_array(s: str, length: int) -> np.ndarray:
    """Decode a base64 string back to a boolean numpy array of given length."""
    packed = np.frombuffer(base64.b64decode(s), dtype=np.uint8)
    return np.unpackbits(packed)[:length].astype(bool)


def get_qc_output_path(sub: str, ses: str, run: str) -> Path:
    """Return path for the QC sidecar JSON."""
    return (Path(DERIV_DIR) / 'qc' / 'voltage' /
            f'sub-{sub}' / f'ses-{ses}' /
            f'sub-{sub}_ses-{ses}_run-{run}_voltage_qc.json')


def get_figure_prefix(sub: str, ses: str, run: str) -> str:
    """Return path prefix for figures (no extension)."""
    return str(Path(FIGURES_DIR) / f'sub-{sub}_ses-{ses}_run-{run}')


# ============================================================================
# FILE DISCOVERY
# ============================================================================

def discover_files(subjects=None, sessions=None, runs=None,
                   force_overwrite=False, write_list=False):
    """
    Find raw NWB files matching the subject/session/run filters.
    Returns list of (sub, ses, run, input_path) tuples.
    """
    if subjects is not None:
        if isinstance(subjects, str):
            subjects = [subjects]
        subjects = [s.replace('sub-', '') for s in subjects]

    if sessions is not None:
        if isinstance(sessions, str):
            sessions = [sessions]
        sessions = [s.replace('ses-', '') for s in sessions]

    if runs is not None:
        if isinstance(runs, str):
            runs = [runs]
        runs = [r.replace('run-', '') for r in runs]

    # Build glob pattern
    sub_pat = f"sub-{subjects[0]}" if subjects and len(subjects) == 1 else "sub-*"
    all_files = glob.glob(f"{RAW_DIR}/{sub_pat}/ses-*/ieeg/sub-*_ses-*_run-*.nwb")

    records = []
    for fpath in sorted(all_files):
        p = Path(fpath)
        # Parse sub/ses/run from filename
        parts = p.stem.split('_')
        sub = next((x.replace('sub-', '') for x in parts if x.startswith('sub-')), None)
        ses = next((x.replace('ses-', '') for x in parts if x.startswith('ses-')), None)
        run = next((x.replace('run-', '') for x in parts if x.startswith('run-')), None)

        if sub is None or ses is None or run is None:
            continue

        # Apply filters
        if subjects and sub not in subjects:
            continue
        if sessions and ses not in sessions:
            continue
        if runs and run not in runs:
            continue

        # Skip if output already exists and not overwriting
        qc_path = get_qc_output_path(sub, ses, run)
        if not force_overwrite and qc_path.exists():
            print(f"  Skipping (QC exists): sub-{sub} ses-{ses} run-{run}")
            continue

        records.append((sub, ses, run, str(fpath)))

    print(f"  Found {len(records)} run(s) to process")

    if write_list and records:
        with open('voltage_qc_file_list.txt', 'w') as f:
            for sub, ses, run, fpath in records:
                f.write(f"{sub}\t{ses}\t{run}\t{fpath}\n")
        print(f"  Written to: voltage_qc_file_list.txt")

    return records


# ============================================================================
# CORE QC LOGIC
# ============================================================================

def detect_identical_run(window: np.ndarray, min_run: int) -> bool:
    """
    Return True if the window contains a run of >= min_run consecutive
    identical sample values. Used to detect digital clipping.
    """
    if len(window) < min_run:
        return False
    # Count consecutive equal values using run-length logic
    diffs = np.diff(window)
    run_len = 1
    for d in diffs:
        if d == 0:
            run_len += 1
            if run_len >= min_run:
                return True
        else:
            run_len = 1
    return False


def run_voltage_qc(
    data: np.ndarray,          # shape: (n_samples, n_channels), raw voltage in volts
    sfreq: float,
    channel_names: list,
    window_sec: float = 2.0,
    flatline_var_threshold: float = FLATLINE_VAR_THRESHOLD,
    saturation_abs_threshold: float = SATURATION_ABS_THRESHOLD,
    min_identical_samples: int = MIN_IDENTICAL_SAMPLES,
    dead_channel_threshold: float = DEAD_CHANNEL_THRESHOLD,
    unreliable_channel_threshold: float = UNRELIABLE_CHANNEL_THRESHOLD,
) -> dict:
    """
    Core QC function. Operates on raw monopolar voltage.

    Returns a results dict containing:
      - per-window boolean arrays (flatline, saturated) per channel
      - per-channel summary stats and classification
      - overall summary counts
    """
    n_samples, n_channels = data.shape
    window_samples = int(window_sec * sfreq)

    # Discard last partial window if < 1 sec worth of samples
    min_window = int(sfreq)
    n_windows = n_samples // window_samples
    usable_samples = n_windows * window_samples

    print(f"  Windowing: {n_windows} windows × {window_sec}s "
          f"({window_samples} samples), discarding last "
          f"{n_samples - usable_samples} samples")

    # Pre-allocate flag arrays: shape (n_windows, n_channels)
    flatline_flags  = np.zeros((n_windows, n_channels), dtype=bool)
    saturated_flags = np.zeros((n_windows, n_channels), dtype=bool)

    # Per-window variance for output (used in figures)
    window_variance = np.zeros((n_windows, n_channels), dtype=np.float32)

    for w in range(n_windows):
        start = w * window_samples
        end   = start + window_samples
        chunk = data[start:end, :]  # (window_samples, n_channels)

        var = np.var(chunk, axis=0)
        window_variance[w] = var

        # Saturation checked FIRST — a hard-clipped window at the amplifier rail
        # has near-zero variance and would otherwise be mistyped as flatline.
        abs_max = np.max(np.abs(chunk), axis=0)
        sat_abs = abs_max > saturation_abs_threshold

        sat_clip = np.zeros(n_channels, dtype=bool)
        for ch in range(n_channels):
            # Identical-run check only applies at high amplitude — a low-amplitude
            # constant signal is a flatline, not a clipping artifact.
            # Require abs_max > 10% of the saturation threshold before checking runs.
            if not sat_abs[ch] and abs_max[ch] > (saturation_abs_threshold * 0.1):
                sat_clip[ch] = detect_identical_run(chunk[:, ch], min_identical_samples)

        saturated_flags[w] = sat_abs | sat_clip

        # Flatline: variance below threshold, only on channels not already
        # typed as saturated (avoids mistyping hard-clipped windows as flatlines).
        flatline_flags[w] = (var < flatline_var_threshold) & ~saturated_flags[w]

        if (w + 1) % 500 == 0 or (w + 1) == n_windows:
            print(f"    Window {w+1}/{n_windows} ({100*(w+1)/n_windows:.0f}%)", end='\r')

    print()

    # -------------------------------------------------------------------------
    # Per-channel summary
    # -------------------------------------------------------------------------
    channel_results = {}
    n_dead = 0
    n_unreliable = 0
    n_clean = 0

    for ch_idx, ch_name in enumerate(channel_names):
        fl = flatline_flags[:, ch_idx]
        sa = saturated_flags[:, ch_idx]
        combined = fl | sa

        frac_flat = float(fl.mean())
        frac_sat  = float(sa.mean())
        frac_any  = float(combined.mean())

        if frac_flat >= dead_channel_threshold:
            status = 'dead'
            n_dead += 1
        elif frac_any >= unreliable_channel_threshold:
            status = 'unreliable'
            n_unreliable += 1
        else:
            status = 'clean'
            n_clean += 1

        channel_results[ch_name] = {
            'status':         status,
            'frac_flatline':  round(frac_flat, 4),
            'frac_saturated': round(frac_sat, 4),
            'frac_flagged':   round(frac_any, 4),
            # Boolean arrays stored as compact base64
            'flatline_windows':  bool_array_to_b64(fl),
            'saturated_windows': bool_array_to_b64(sa),
            'n_windows':         int(n_windows),
        }

    summary = {
        'n_channels_total': n_channels,
        'n_dead':           n_dead,
        'n_unreliable':     n_unreliable,
        'n_clean':          n_clean,
        'n_windows':        n_windows,
    }

    return {
        'channel_results':  channel_results,
        'summary':          summary,
        'window_variance':  window_variance,   # (n_windows, n_channels) float32
        'flatline_flags':   flatline_flags,    # (n_windows, n_channels) bool
        'saturated_flags':  saturated_flags,   # (n_windows, n_channels) bool
        'n_windows':        n_windows,
        'window_sec':       window_sec,
        'sfreq':            sfreq,
    }


def save_qc_sidecar(
    qc_results: dict,
    sub: str, ses: str, run: str,
    source_file: str,
    duration_min: float,
    window_sec: float,
    flatline_var_threshold: float,
    saturation_abs_threshold: float,
    min_identical_samples: int,
    dead_channel_threshold: float,
    unreliable_channel_threshold: float,
) -> Path:
    """Write QC results to a JSON sidecar file. Returns the output path."""

    out_path = get_qc_output_path(sub, ses, run)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    output = {
        'metadata': {
            'script':      'raw_voltage_qc.py',
            'git_commit':  get_git_commit(),
            'timestamp':   datetime.now().isoformat(),
            'source_file': source_file,
            'duration_min': round(duration_min, 2),
            'sfreq':       qc_results['sfreq'],
            'parameters': {
                'window_sec':                  window_sec,
                'flatline_var_threshold_v2':   flatline_var_threshold,
                'saturation_abs_threshold_v':  saturation_abs_threshold,
                'min_identical_samples':       min_identical_samples,
                'dead_channel_threshold':      dead_channel_threshold,
                'unreliable_channel_threshold': unreliable_channel_threshold,
            },
        },
        'summary':         qc_results['summary'],
        'channel_results': qc_results['channel_results'],
    }

    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2)

    size_kb = out_path.stat().st_size / 1024
    print(f"  QC sidecar saved: {out_path}  ({size_kb:.1f} KB)")
    return out_path


# ============================================================================
# FIGURES
# ============================================================================

def make_figures(qc_results: dict, channel_names: list,
                 sub: str, ses: str, run: str,
                 heatmap_pool_min: float = HEATMAP_POOL_MIN):
    """Generate and save all QC figures."""
    try:
        import matplotlib
        matplotlib.use('Agg')  # non-interactive backend for Sherlock
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors
    except ImportError:
        print("  WARNING: matplotlib not available, skipping figures")
        return

    fig_prefix = get_figure_prefix(sub, ses, run)
    os.makedirs(os.path.dirname(fig_prefix), exist_ok=True)

    n_windows   = qc_results['n_windows']
    window_sec  = qc_results['window_sec']
    sfreq       = qc_results['sfreq']
    fl_flags    = qc_results['flatline_flags']    # (n_windows, n_channels)
    sat_flags   = qc_results['saturated_flags']   # (n_windows, n_channels)
    variance    = qc_results['window_variance']   # (n_windows, n_channels)
    ch_results  = qc_results['channel_results']

    n_channels = len(channel_names)
    title_base = f"sub-{sub}  ses-{ses}  run-{run}"

    # ------------------------------------------------------------------
    # Figure 1: Heatmap (pooled to N-minute blocks)
    # ------------------------------------------------------------------
    bins_per_min  = 60.0 / window_sec
    pool_bins     = max(1, int(heatmap_pool_min * bins_per_min))
    n_pools       = int(np.ceil(n_windows / pool_bins))

    fl_pooled  = np.zeros((n_channels, n_pools), dtype=np.float32)
    sat_pooled = np.zeros((n_channels, n_pools), dtype=np.float32)

    for p in range(n_pools):
        sl = slice(p * pool_bins, min((p + 1) * pool_bins, n_windows))
        fl_pooled[:, p]  = fl_flags[sl, :].mean(axis=0)
        sat_pooled[:, p] = sat_flags[sl, :].mean(axis=0)

    # Time axis in hours
    time_hours = np.arange(n_pools) * heatmap_pool_min / 60.0

    fig, axes = plt.subplots(2, 1, figsize=(max(14, n_pools // 8), max(6, n_channels // 5)),
                              sharex=True)
    fig.suptitle(f"Artifact Heatmap — {title_base}\n"
                 f"Pooled to {heatmap_pool_min}-min blocks  |  "
                 f"{n_channels} channels  |  {n_windows} total windows",
                 fontsize=11)

    # Flatline heatmap
    im0 = axes[0].imshow(fl_pooled, aspect='auto', origin='lower',
                          cmap='Blues', vmin=0, vmax=1,
                          extent=[time_hours[0], time_hours[-1], -0.5, n_channels - 0.5])
    axes[0].set_ylabel('Channel index')
    axes[0].set_title('Flatline fraction per block')
    plt.colorbar(im0, ax=axes[0], fraction=0.015, pad=0.02, label='Fraction flatlined')

    # Saturation heatmap
    im1 = axes[1].imshow(sat_pooled, aspect='auto', origin='lower',
                          cmap='Oranges', vmin=0, vmax=1,
                          extent=[time_hours[0], time_hours[-1], -0.5, n_channels - 0.5])
    axes[1].set_ylabel('Channel index')
    axes[1].set_xlabel('Time (hours)')
    axes[1].set_title('Saturation fraction per block')
    plt.colorbar(im1, ax=axes[1], fraction=0.015, pad=0.02, label='Fraction saturated')

    plt.tight_layout()
    heatmap_path = f"{fig_prefix}_heatmap.png"
    fig.savefig(heatmap_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Figure saved: {heatmap_path}")

    # ------------------------------------------------------------------
    # Figure 2: Variance ridge plot (log scale, sorted by median variance)
    # ------------------------------------------------------------------
    # Compute median variance per channel for sorting
    median_var = np.median(variance, axis=0)  # (n_channels,)
    sort_order = np.argsort(median_var)        # ascending: dead channels first

    status_colors = {'dead': '#d62728', 'unreliable': '#ff7f0e', 'clean': '#2ca02c'}

    fig2, ax = plt.subplots(figsize=(10, max(8, n_channels * 0.18)))
    fig2.suptitle(f"Per-Channel Variance Distribution (log scale) — {title_base}",
                  fontsize=11)

    # Violin plot for each channel, sorted
    positions = np.arange(n_channels)
    violin_data = []
    colors_sorted = []
    labels_sorted = []

    for rank, ch_idx in enumerate(sort_order):
        ch_name = channel_names[ch_idx]
        ch_var  = variance[:, ch_idx]
        # Log-transform (add small epsilon to avoid log(0) on flatlined windows)
        log_var = np.log10(ch_var + 1e-18)
        violin_data.append(log_var)
        status = ch_results[ch_name]['status']
        colors_sorted.append(status_colors[status])
        labels_sorted.append(ch_name)

    parts = ax.violinplot(violin_data, positions=positions,
                          vert=False, showmedians=True, widths=0.8)

    # Color each violin by channel status
    for i, (pc, col) in enumerate(zip(parts['bodies'], colors_sorted)):
        pc.set_facecolor(col)
        pc.set_alpha(0.6)
    parts['cmedians'].set_color('black')
    parts['cmedians'].set_linewidth(1.5)
    parts['cbars'].set_color('gray')
    parts['cmins'].set_color('gray')
    parts['cmaxes'].set_color('gray')

    # Flatline threshold line
    thresh_log = np.log10(FLATLINE_VAR_THRESHOLD + 1e-18)
    ax.axvline(thresh_log, color='steelblue', linestyle='--', linewidth=1.5,
               label=f'Flatline threshold ({FLATLINE_VAR_THRESHOLD:.2e} V²)')

    ax.set_yticks(positions)
    ax.set_yticklabels(labels_sorted, fontsize=max(5, 9 - n_channels // 30))
    ax.set_xlabel('log₁₀(Variance)  [V²]')
    ax.set_ylabel('Channel (sorted by median variance)')

    # Legend for status colors
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=c, alpha=0.7, label=s.capitalize())
                       for s, c in status_colors.items()]
    legend_elements.append(
        plt.Line2D([0], [0], color='steelblue', linestyle='--',
                   label=f'Flatline threshold ({FLATLINE_VAR_THRESHOLD:.2e} V²)')
    )
    ax.legend(handles=legend_elements, loc='lower right', fontsize=9)

    plt.tight_layout()
    variance_path = f"{fig_prefix}_variance_dist.png"
    fig2.savefig(variance_path, dpi=150, bbox_inches='tight')
    plt.close(fig2)
    print(f"  Figure saved: {variance_path}")

    # ------------------------------------------------------------------
    # Figure 3: Summary bar chart — fraction flagged per channel
    # ------------------------------------------------------------------
    # Sort channels by fraction flagged (descending)
    frac_flagged = np.array([ch_results[ch]['frac_flagged'] for ch in channel_names])
    frac_flat    = np.array([ch_results[ch]['frac_flatline'] for ch in channel_names])
    frac_sat     = np.array([ch_results[ch]['frac_saturated'] for ch in channel_names])
    statuses     = [ch_results[ch]['status'] for ch in channel_names]

    sort_desc = np.argsort(frac_flagged)[::-1]
    ch_sorted   = [channel_names[i] for i in sort_desc]
    fl_sorted   = frac_flat[sort_desc]
    sat_sorted  = frac_sat[sort_desc]
    stat_sorted = [statuses[i] for i in sort_desc]
    bar_colors  = [status_colors[s] for s in stat_sorted]

    fig3, ax3 = plt.subplots(figsize=(max(12, n_channels * 0.25), 5))
    fig3.suptitle(f"Per-Channel Flagged Fraction — {title_base}", fontsize=11)

    x = np.arange(n_channels)
    ax3.bar(x, fl_sorted,  color='steelblue', alpha=0.8, label='Flatline')
    ax3.bar(x, sat_sorted, bottom=fl_sorted, color='darkorange', alpha=0.8,
            label='Saturated')

    ax3.axhline(UNRELIABLE_CHANNEL_THRESHOLD, color='orange', linestyle='--',
                linewidth=1.5, label=f'Unreliable threshold ({UNRELIABLE_CHANNEL_THRESHOLD:.0%})')
    ax3.axhline(DEAD_CHANNEL_THRESHOLD, color='red', linestyle='--',
                linewidth=1.5, label=f'Dead threshold ({DEAD_CHANNEL_THRESHOLD:.0%})')

    ax3.set_xticks(x)
    ax3.set_xticklabels(ch_sorted, rotation=90,
                        fontsize=max(4, 8 - n_channels // 30))
    ax3.set_ylabel('Fraction of windows flagged')
    ax3.set_ylim(0, 1.05)
    ax3.set_xlabel('Channel (sorted by flagged fraction)')
    ax3.legend(fontsize=9)

    plt.tight_layout()
    bar_path = f"{fig_prefix}_channel_summary.png"
    fig3.savefig(bar_path, dpi=150, bbox_inches='tight')
    plt.close(fig3)
    print(f"  Figure saved: {bar_path}")


# ============================================================================
# PER-FILE PROCESSING
# ============================================================================

def process_file(sub: str, ses: str, run: str, input_path: str,
                 window_sec: float = 2.0,
                 flatline_var_threshold: float = FLATLINE_VAR_THRESHOLD,
                 saturation_abs_threshold: float = SATURATION_ABS_THRESHOLD,
                 min_identical_samples: int = MIN_IDENTICAL_SAMPLES,
                 dead_channel_threshold: float = DEAD_CHANNEL_THRESHOLD,
                 unreliable_channel_threshold: float = UNRELIABLE_CHANNEL_THRESHOLD,
                 make_figs: bool = False,
                 force_overwrite: bool = False) -> bool:
    """Run voltage QC on a single NWB file. Returns True on success."""

    t0 = time.time()
    print(f"\n{'='*70}")
    print(f"  sub-{sub}  ses-{ses}  run-{run}")
    print(f"  {input_path}")
    print(f"{'='*70}")

    # Check output
    qc_path = get_qc_output_path(sub, ses, run)
    if qc_path.exists() and not force_overwrite:
        print(f"  QC sidecar already exists (--force-overwrite to redo): {qc_path}")
        return True

    # Load NWB
    try:
        io = pynwb.NWBHDF5IO(input_path, 'r')
        nwb = io.read()
    except Exception as e:
        print(f"  ERROR loading NWB: {e}")
        return False

    try:
        series = nwb.acquisition['ElectricalSeries_sEEG']
    except KeyError:
        keys = list(nwb.acquisition.keys())
        print(f"  ERROR: 'ElectricalSeries_sEEG' not found. Available: {keys}")
        io.close()
        return False

    sfreq      = float(series.rate)
    n_samples  = series.data.shape[0]
    n_channels = series.data.shape[1]
    duration_min = n_samples / sfreq / 60.0

    print(f"  {n_samples:,} samples  |  {n_channels} channels  |  "
          f"{duration_min:.1f} min  |  {sfreq:.0f} Hz")

    # Get channel names from electrode table
    elec_indices = series.electrodes.data[:]
    elec_df = nwb.electrodes.to_dataframe().iloc[elec_indices]
    channel_names = list(elec_df['location'].values)

    # Load raw data into memory
    # For very long recordings this may be large; warn if >4 GB
    data_gb = n_samples * n_channels * 4 / (1024**3)  # float32
    print(f"  Loading raw data ({data_gb:.2f} GB float32)...")
    if data_gb > 4.0:
        print(f"  WARNING: large file ({data_gb:.1f} GB). Consider processing in chunks "
              f"if memory is limited on this node.")

    try:
        data = series.data[:].astype(np.float32)
    except Exception as e:
        print(f"  ERROR reading data: {e}")
        io.close()
        return False

    io.close()
    print(f"  Data loaded in {time.time()-t0:.1f}s")

    # Run QC
    print(f"\n  Running QC (window={window_sec}s)...")
    qc_results = run_voltage_qc(
        data               = data,
        sfreq              = sfreq,
        channel_names      = channel_names,
        window_sec         = window_sec,
        flatline_var_threshold    = flatline_var_threshold,
        saturation_abs_threshold  = saturation_abs_threshold,
        min_identical_samples     = min_identical_samples,
        dead_channel_threshold    = dead_channel_threshold,
        unreliable_channel_threshold = unreliable_channel_threshold,
    )

    # Print summary
    s = qc_results['summary']
    print(f"\n  Channel summary:")
    print(f"    Clean:      {s['n_clean']}")
    print(f"    Unreliable: {s['n_unreliable']}")
    print(f"    Dead:       {s['n_dead']}")
    print(f"    Total:      {s['n_channels_total']}")

    # Save sidecar
    save_qc_sidecar(
        qc_results               = qc_results,
        sub                      = sub,
        ses                      = ses,
        run                      = run,
        source_file              = os.path.basename(input_path),
        duration_min             = duration_min,
        window_sec               = window_sec,
        flatline_var_threshold   = flatline_var_threshold,
        saturation_abs_threshold = saturation_abs_threshold,
        min_identical_samples    = min_identical_samples,
        dead_channel_threshold   = dead_channel_threshold,
        unreliable_channel_threshold = unreliable_channel_threshold,
    )

    # Figures
    if make_figs:
        print(f"\n  Generating figures...")
        make_figures(qc_results, channel_names, sub, ses, run)

    elapsed = time.time() - t0
    print(f"\n  Done in {elapsed:.1f}s ({elapsed/60:.1f} min)")
    return True


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Stage 1: Raw voltage QC for iEEG NWB files',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    # Discovery / filtering
    parser.add_argument('--discover', action='store_true',
                        help='Discover files and write file list, then exit')
    parser.add_argument('--subjects',  nargs='+', help='Subject ID(s), e.g. 019 020')
    parser.add_argument('--sessions',  nargs='+', help='Session ID(s), e.g. 01 02')
    parser.add_argument('--runs',      nargs='+', help='Run ID(s), e.g. 01 02')
    parser.add_argument('--force-overwrite', action='store_true',
                        help='Reprocess even if QC sidecar already exists')

    # SLURM array job
    parser.add_argument('--file-list', type=str,
                        help='Tab-separated file list (sub, ses, run, path)')
    parser.add_argument('--task-id', type=int,
                        help='SLURM task ID (1-indexed)')
    parser.add_argument('--batch-size', type=int, default=1,
                        help='Files per SLURM task (default: 1)')

    # QC parameters
    parser.add_argument('--window-sec', type=float, default=2.0,
                        help='Window size in seconds (default: 2.0)')
    parser.add_argument('--flatline-threshold', type=float,
                        default=FLATLINE_VAR_THRESHOLD,
                        help=f'Flatline variance threshold V² (default: {FLATLINE_VAR_THRESHOLD:.2e})')
    parser.add_argument('--sat-threshold', type=float,
                        default=SATURATION_ABS_THRESHOLD,
                        help=f'Saturation |voltage| threshold V (default: {SATURATION_ABS_THRESHOLD:.2e})')
    parser.add_argument('--min-identical', type=int,
                        default=MIN_IDENTICAL_SAMPLES,
                        help=f'Min identical samples for clipping (default: {MIN_IDENTICAL_SAMPLES})')
    parser.add_argument('--dead-threshold', type=float,
                        default=DEAD_CHANNEL_THRESHOLD,
                        help=f'Fraction flagged to call channel DEAD (default: {DEAD_CHANNEL_THRESHOLD})')
    parser.add_argument('--unreliable-threshold', type=float,
                        default=UNRELIABLE_CHANNEL_THRESHOLD,
                        help=f'Fraction flagged to call UNRELIABLE (default: {UNRELIABLE_CHANNEL_THRESHOLD})')

    # Output
    parser.add_argument('--figures', action='store_true',
                        help='Generate and save QC figures')
    parser.add_argument('--heatmap-pool-min', type=float, default=HEATMAP_POOL_MIN,
                        help=f'Heatmap pooling resolution in minutes (default: {HEATMAP_POOL_MIN})')

    args = parser.parse_args()

    # Collect QC kwargs to pass through
    qc_kwargs = dict(
        window_sec               = args.window_sec,
        flatline_var_threshold   = args.flatline_threshold,
        saturation_abs_threshold = args.sat_threshold,
        min_identical_samples    = args.min_identical,
        dead_channel_threshold   = args.dead_threshold,
        unreliable_channel_threshold = args.unreliable_threshold,
        make_figs                = args.figures,
        force_overwrite          = args.force_overwrite,
    )

    # ------------------------------------------------------------------
    # DISCOVER MODE
    # ------------------------------------------------------------------
    if args.discover:
        print("Discovering files...")
        discover_files(
            subjects       = args.subjects,
            sessions       = args.sessions,
            runs           = args.runs,
            force_overwrite= args.force_overwrite,
            write_list     = True,
        )
        return

    # ------------------------------------------------------------------
    # SLURM ARRAY MODE
    # ------------------------------------------------------------------
    if args.file_list and args.task_id:
        with open(args.file_list) as f:
            lines = [l.strip() for l in f if l.strip()]

        batch_size = args.batch_size
        start = (args.task_id - 1) * batch_size
        end   = min(start + batch_size, len(lines))

        if start >= len(lines):
            print(f"ERROR: task-id {args.task_id} out of range (max "
                  f"{(len(lines) + batch_size - 1) // batch_size})")
            sys.exit(1)

        results = []
        for line in lines[start:end]:
            sub, ses, run, fpath = line.split('\t')
            ok = process_file(sub, ses, run, fpath, **qc_kwargs)
            results.append(ok)

        n_fail = sum(not r for r in results)
        print(f"\nBatch complete: {sum(results)}/{len(results)} succeeded")
        sys.exit(0 if n_fail == 0 else 1)

    # ------------------------------------------------------------------
    # STANDARD MODE (discover + process in one call)
    # ------------------------------------------------------------------
    records = discover_files(
        subjects        = args.subjects,
        sessions        = args.sessions,
        runs            = args.runs,
        force_overwrite = args.force_overwrite,
        write_list      = False,
    )

    if not records:
        print("No files to process.")
        return

    n_ok = 0
    n_fail = 0
    for sub, ses, run, fpath in records:
        ok = process_file(sub, ses, run, fpath, **qc_kwargs)
        if ok:
            n_ok += 1
        else:
            n_fail += 1

    print(f"\n{'='*70}")
    print(f"COMPLETE: {n_ok} succeeded, {n_fail} failed")
    print(f"{'='*70}")
    sys.exit(0 if n_fail == 0 else 1)


if __name__ == '__main__':
    main()