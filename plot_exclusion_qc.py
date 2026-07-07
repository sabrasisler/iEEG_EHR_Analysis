#!/usr/bin/env python3
"""
Two-stage exclusion QC: amplifier saturation (raw voltage) + variance outliers.

Stage 1 (saturation): flag individual samples with |voltage| > SAT_THRESHOLD_UV.
Any 2s bin containing a saturated sample is excluded as 'saturated'.

Stage 2 (variance): for bins that survive stage 1, compute per-bin variance.
Bins below FLATLINE_VAR_THRESHOLD (from raw_voltage_qc.py) are 'dead'. Remaining
bins more than VAR_STD_THRESH scaled-MADs from the channel's own median
bin-variance are 'variance_outlier' (median/MAD instead of mean/std, since a
handful of huge-variance bins from a shared artifact would otherwise inflate
the mean/std and mask smaller, real outliers elsewhere in the run). Everything
else is 'clean'.

Plots N random channels per run as black time series, with excluded bins
shaded as a colored background band (not a recolored trace — this stays
visible even for near-zero-amplitude 'dead' bins). No heatmaps/violin plots
(see raw_voltage_qc.py for those).

Usage:
  python plot_exclusion_qc.py --subject 222 --session 01 --n-runs 2
  python plot_exclusion_qc.py --subject 222 --session 01 --runs FA6150R5,FA6150R6
"""

import argparse
import glob
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pynwb import NWBHDF5IO

from raw_voltage_qc import RAW_DIR, FLATLINE_VAR_THRESHOLD

FIGURES_DIR = '/home/groups/ckeller1/sisler/figures/qc/voltage/exclusion_qc'

SAT_THRESHOLD_UV = 2500.0   # amplifier saturation, µV (subject-specific for now)

STATUS_COLORS = {
    'saturated':        'crimson',
    'variance_outlier': 'orange',
    'dead':             'dimgray',
}

MAD_SCALE = 1.4826  # scales MAD to be comparable to std under a normal distribution


def find_runs(subject, session, runs=None, n_runs=None):
    pattern = f"{RAW_DIR}/sub-{subject}/ses-{session}/ieeg/sub-{subject}_ses-{session}_run-*.nwb"
    all_files = sorted(glob.glob(pattern))
    if runs:
        wanted = set(runs)
        all_files = [f for f in all_files if Path(f).stem.split('_run-')[-1] in wanted]
    elif n_runs:
        all_files = all_files[:n_runs]
    if not all_files:
        raise FileNotFoundError(f"No NWB files matching {pattern}")
    return all_files


def classify_bins(trace_v, sfreq, window_sec, sat_threshold_v, var_std_thresh, baseline='mad'):
    """
    trace_v: 1D array, raw voltage in volts, for one channel.
    baseline: 'mad' (median + scaled-MAD, robust to a few huge-variance bins) or
              'std' (mean + std, the original approach).
    Returns: array of status strings, one per bin ('clean'/'saturated'/'variance_outlier'/'dead'),
             and samples_per_bin (int).
    """
    samples_per_bin = max(1, int(window_sec * sfreq))
    n_bins = len(trace_v) // samples_per_bin
    usable = n_bins * samples_per_bin
    bins = trace_v[:usable].reshape(n_bins, samples_per_bin)

    status = np.full(n_bins, 'clean', dtype=object)

    # Stage 1: saturation — any sample in the bin exceeds the absolute threshold.
    saturated = np.any(np.abs(bins) > sat_threshold_v, axis=1)
    status[saturated] = 'saturated'

    # Stage 2: variance, computed only over non-saturated bins.
    ok = ~saturated
    if ok.sum() >= 2:
        var = bins[ok].var(axis=1)
        if baseline == 'mad':
            center = np.median(var)
            spread = np.median(np.abs(var - center)) * MAD_SCALE
        elif baseline == 'std':
            center = var.mean()
            spread = var.std()
        else:
            raise ValueError(f"Unknown baseline '{baseline}' (expected 'mad' or 'std')")

        dead = var < FLATLINE_VAR_THRESHOLD
        outlier = (~dead) & (np.abs(var - center) > var_std_thresh * spread)

        ok_idx = np.where(ok)[0]
        status[ok_idx[dead]] = 'dead'
        status[ok_idx[outlier]] = 'variance_outlier'

    return status, samples_per_bin


def plot_run(nwb_path, subject, session, channel_names_wanted, window_sec,
             sat_threshold_uv, var_std_thresh, seed, baseline='mad'):
    run = Path(nwb_path).stem.split('_run-')[-1]
    print(f"\n{'='*70}\n  sub-{subject}  ses-{session}  run-{run}\n{'='*70}")

    io = NWBHDF5IO(nwb_path, 'r')
    nwb = io.read()
    series = nwb.acquisition['ElectricalSeries_sEEG']

    if series.unit != 'volts':
        io.close()
        raise ValueError(f"Unexpected unit '{series.unit}' (expected 'volts')")

    sfreq = float(series.rate)
    n_samples, n_channels = series.data.shape

    elec_indices = series.electrodes.data[:]
    elec_df = nwb.electrodes.to_dataframe().iloc[elec_indices]
    channel_names = list(elec_df['location'].values)

    wanted = [c for c in channel_names_wanted if c in channel_names]
    missing = set(channel_names_wanted) - set(wanted)
    if missing:
        print(f"  WARNING: channels not found in this run, skipping: {sorted(missing)}")

    idx_sorted = sorted(channel_names.index(c) for c in wanted)
    print(f"  Loading {len(idx_sorted)} channels ({n_samples:,} samples each)...")
    data_sel = series.data[:, idx_sorted].astype(np.float32) * np.float32(series.conversion)
    io.close()

    sat_threshold_v = sat_threshold_uv * 1e-6
    t = np.arange(n_samples) / sfreq

    fig, axes = plt.subplots(len(idx_sorted), 1, figsize=(14, 1.8 * len(idx_sorted)), sharex=True)
    if len(idx_sorted) == 1:
        axes = [axes]

    for ax, col, ch_idx in zip(axes, range(len(idx_sorted)), idx_sorted):
        name = channel_names[ch_idx]
        trace_v = data_sel[:, col]
        trace_uv = trace_v * 1e6

        status, spb = classify_bins(trace_v, sfreq, window_sec, sat_threshold_v, var_std_thresh,
                                     baseline=baseline)

        counts = {'saturated': 0, 'variance_outlier': 0, 'dead': 0}
        for b, s in enumerate(status):
            if s == 'clean':
                continue
            counts[s] += 1
            start, end = b * spb, (b + 1) * spb
            ax.axvspan(t[start], t[min(end, len(t) - 1)], color=STATUS_COLORS[s],
                       alpha=0.3, linewidth=0, zorder=1)

        ax.plot(t, trace_uv, linewidth=0.5, color='black', zorder=2)

        n_bins = len(status)
        ax.set_ylabel('µV', fontsize=8)
        ax.set_title(
            f"{name}  —  sat:{counts['saturated']}/{n_bins}  "
            f"var-outlier:{counts['variance_outlier']}/{n_bins}  "
            f"dead:{counts['dead']}/{n_bins}",
            fontsize=9, loc='left')
        ax.tick_params(labelsize=7)

    axes[-1].set_xlabel('Time (s)')

    from matplotlib.patches import Patch
    legend_elements = [plt.Line2D([0], [0], color='black', lw=1, label='clean')]
    legend_elements += [Patch(facecolor=c, alpha=0.3, label=s)
                         for s, c in STATUS_COLORS.items()]
    fig.legend(handles=legend_elements, loc='upper right', ncol=4, fontsize=8)

    unit_label = 'MADs' if baseline == 'mad' else 'stds'
    fig.suptitle(f"sub-{subject} ses-{session} run-{run}  "
                 f"(sat={sat_threshold_uv:.0f}µV, var-outlier={var_std_thresh:.0f} {unit_label} [{baseline}], "
                 f"dead<{FLATLINE_VAR_THRESHOLD:.1e}V²)", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    Path(FIGURES_DIR).mkdir(parents=True, exist_ok=True)
    suffix = '' if baseline == 'mad' else f'_{baseline}'
    out_path = Path(FIGURES_DIR) / f"sub-{subject}_ses-{session}_run-{run}_exclusion_qc{suffix}.png"
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Figure saved: {out_path}")

    return channel_names


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--subject', required=True)
    ap.add_argument('--session', required=True)
    ap.add_argument('--runs', default=None,
                     help='Comma-separated run IDs. If omitted, uses the first --n-runs runs found.')
    ap.add_argument('--n-runs', type=int, default=2,
                     help='Number of runs to process if --runs not given (default: 2)')
    ap.add_argument('--n-channels', type=int, default=15,
                     help='Number of random channels to plot (default: 15)')
    ap.add_argument('--window-sec', type=float, default=2.0,
                     help='Bin size in seconds (default: 2.0)')
    ap.add_argument('--sat-threshold-uv', type=float, default=SAT_THRESHOLD_UV,
                     help=f'Amplifier saturation threshold, µV (default: {SAT_THRESHOLD_UV})')
    ap.add_argument('--var-std-thresh', type=float, default=5.0,
                     help='Exclude bins whose variance is this many scaled-MADs (or stds, see '
                          "--baseline) from the channel's own baseline bin-variance (default: 5.0)")
    ap.add_argument('--baseline', choices=['mad', 'std'], default='mad',
                     help="Variance-outlier baseline statistic: 'mad' (median + scaled-MAD, "
                          "robust to a few huge-variance bins) or 'std' (mean + std) (default: mad)")
    ap.add_argument('--seed', type=int, default=42,
                     help='Random seed for channel selection (default: 42)')
    args = ap.parse_args()

    runs = args.runs.split(',') if args.runs else None
    nwb_paths = find_runs(args.subject, args.session, runs=runs, n_runs=args.n_runs)

    # Pick the random channel set once, from the first run, and reuse across runs.
    io = NWBHDF5IO(nwb_paths[0], 'r')
    nwb = io.read()
    series = nwb.acquisition['ElectricalSeries_sEEG']
    elec_indices = series.electrodes.data[:]
    elec_df = nwb.electrodes.to_dataframe().iloc[elec_indices]
    all_channel_names = list(elec_df['location'].values)
    io.close()

    rng = np.random.default_rng(args.seed)
    n_pick = min(args.n_channels, len(all_channel_names))
    channels_wanted = list(rng.choice(all_channel_names, size=n_pick, replace=False))
    print(f"Randomly selected {n_pick} channels (seed={args.seed}): {channels_wanted}")

    for nwb_path in nwb_paths:
        plot_run(nwb_path, args.subject, args.session, channels_wanted,
                  args.window_sec, args.sat_threshold_uv, args.var_std_thresh, args.seed,
                  baseline=args.baseline)


if __name__ == '__main__':
    main()
