#!/usr/bin/env python3
"""
Plot raw voltage traces for a few channels, for manual inspection.

By default picks the N lowest-variance channels (the ones QC would flag as
flatline/dead candidates) plus N highest-variance channels for comparison.

Usage:
  python plot_raw_voltage_channels.py --subject 222 --session 01 --run FA6150R5
  python plot_raw_voltage_channels.py --subject 222 --session 01 --run FA6150R5 \
      --channels LaIN1,LaIN2 --duration-sec 30
"""

import argparse
import glob
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pynwb
from pynwb import NWBHDF5IO

RAW_DIR = '/oak/stanford/groups/ckeller1/data/iEEG_EHR/iEEG_NWB'
OUT_DIR = '/home/groups/ckeller1/sisler/figures/qc/voltage/raw_traces'


def find_nwb(subject, session, run):
    pattern = f"{RAW_DIR}/sub-{subject}/ses-{session}/ieeg/sub-{subject}_ses-{session}_run-{run}.nwb"
    matches = glob.glob(pattern)
    if not matches:
        raise FileNotFoundError(f"No NWB file matching {pattern}")
    return matches[0]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--subject', required=True)
    ap.add_argument('--session', required=True)
    ap.add_argument('--run', required=True)
    ap.add_argument('--channels', default=None,
                     help='Comma-separated channel names to plot. If omitted, '
                          'auto-picks lowest/highest-variance channels.')
    ap.add_argument('--n-low', type=int, default=4,
                     help='Number of lowest-variance channels to plot (default: 4)')
    ap.add_argument('--n-high', type=int, default=2,
                     help='Number of highest-variance channels to plot for comparison (default: 2)')
    ap.add_argument('--start-sec', type=float, default=0.0,
                     help='Start offset into the recording, in seconds (default: 0)')
    ap.add_argument('--duration-sec', type=float, default=10.0,
                     help='Duration of the window to plot, in seconds (default: 10)')
    ap.add_argument('--window-sec', type=float, default=2.0,
                     help='Window size for per-window variance flagging, in seconds (default: 2.0)')
    ap.add_argument('--std-thresh', type=float, default=5.0,
                     help='Flag windows whose variance is this many stds from the '
                          "channel's own mean window-variance (default: 5.0)")
    args = ap.parse_args()

    nwb_path = find_nwb(args.subject, args.session, args.run)
    print(f"Reading {nwb_path}")

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

    start_idx = int(args.start_sec * sfreq)
    n_win = int(args.duration_sec * sfreq)
    end_idx = min(start_idx + n_win, n_samples)

    if args.channels:
        wanted = [c.strip() for c in args.channels.split(',')]
        chosen = [(name, channel_names.index(name)) for name in wanted]
    else:
        # Compute variance over the whole recording (in volts²) to rank channels,
        # but only load the plotting window itself into memory.
        print("Computing per-channel variance over full recording to rank channels...")
        data_full = series.data[:].astype(np.float32) * np.float32(series.conversion)
        variance = np.var(data_full, axis=0)
        order = np.argsort(variance)
        low_idx = order[:args.n_low]
        high_idx = order[-args.n_high:]
        chosen = [(channel_names[i], i) for i in list(low_idx) + list(high_idx)]
        del data_full

        print("Lowest-variance channels:")
        for i in low_idx:
            print(f"  {channel_names[i]:>10s}  var={variance[i]:.3e} V^2")
        print("Highest-variance channels:")
        for i in high_idx:
            print(f"  {channel_names[i]:>10s}  var={variance[i]:.3e} V^2")

    window_data = series.data[start_idx:end_idx, :].astype(np.float32) * np.float32(series.conversion)
    io.close()

    t = np.arange(start_idx, end_idx) / sfreq

    fig, axes = plt.subplots(len(chosen), 1, figsize=(12, 2.2 * len(chosen)), sharex=True)
    if len(chosen) == 1:
        axes = [axes]

    samples_per_window = max(1, int(args.window_sec * sfreq))

    for ax, (name, ch_idx) in zip(axes, chosen):
        trace_uv = window_data[:, ch_idx] * 1e6  # volts -> µV for readability

        n_full_windows = len(trace_uv) // samples_per_window
        n_flagged = 0
        if n_full_windows >= 2:
            usable = n_full_windows * samples_per_window
            win_view = trace_uv[:usable].reshape(n_full_windows, samples_per_window)
            win_var = win_view.var(axis=1)
            mean_var, std_var = win_var.mean(), win_var.std()
            flagged = np.abs(win_var - mean_var) > args.std_thresh * std_var
            n_flagged = int(flagged.sum())
        else:
            flagged = np.zeros(0, dtype=bool)

        ax.plot(t, trace_uv, linewidth=0.6, color='steelblue', zorder=1)
        for w in np.where(flagged)[0]:
            s = w * samples_per_window
            e = s + samples_per_window
            ax.plot(t[s:e], trace_uv[s:e], linewidth=0.6, color='crimson', zorder=2)

        ax.set_ylabel('µV', fontsize=9)
        ax.set_title(f"{name}  (idx {ch_idx})  —  {n_flagged}/{n_full_windows} windows "
                     f"flagged (>{args.std_thresh:.0f}σ)", fontsize=10, loc='left')
        ax.tick_params(labelsize=8)

    axes[-1].set_xlabel('Time (s)')
    fig.suptitle(f"sub-{args.subject} ses-{args.session} run-{args.run}  "
                 f"[{args.start_sec:.1f}-{args.start_sec + args.duration_sec:.1f}s]",
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    Path(OUT_DIR).mkdir(parents=True, exist_ok=True)
    out_path = (Path(OUT_DIR) /
                f"sub-{args.subject}_ses-{args.session}_run-{args.run}"
                f"_raw_traces_{args.start_sec:.0f}-{args.start_sec + args.duration_sec:.0f}s.png")
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"Figure saved: {out_path}")


if __name__ == '__main__':
    main()
