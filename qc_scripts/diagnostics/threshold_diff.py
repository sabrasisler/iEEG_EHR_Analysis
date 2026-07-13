"""
Generic threshold-sweep diagnostic, usable for any artifact type. Compares two
already-built exclusion labels (exclusions/<type>/<label>/, from build_exclusions)
for the SAME artifact type and shows which 60s bins the "compare" label newly
excludes that the "baseline" label does not -- i.e. what a looser/tighter
threshold adds. Direction-agnostic: works whether looser means a higher or
lower raw threshold value, since it just diffs the already-computed `excluded`
booleans.

Plots the top channels by number of newly-added bins, shading:
  green  = bins excluded at BOTH baseline and compare labels
  red    = bins excluded at compare but NOT baseline (what the sweep adds)
Writes plots + an info.json to
validation/threshold_sweeps/<artifact_type>_<compare_label>_vs_<baseline_label>/.

Usage:
  python -m qc_scripts.diagnostics.threshold_diff --artifact-type gross_artifact \
      --baseline-label std5 --compare-label std4
  python -m qc_scripts.diagnostics.threshold_diff --artifact-type flatline \
      --baseline-label var5e-13 --compare-label var1e-12 --n-examples 12
"""
import argparse
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from qc_scripts import config, io_utils

KEY = ['session_id', 'run_id', 'channel', 'bin_start']


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--level-root', default=str(config.DEFAULT_LEVEL_ROOT))
    ap.add_argument('--artifact-type', required=True, choices=config.ARTIFACT_TYPES)
    ap.add_argument('--baseline-label', required=True,
                     help='Existing exclusions/<type>/<label>/ to treat as baseline')
    ap.add_argument('--compare-label', required=True,
                     help='Existing exclusions/<type>/<label>/ whose newly-added bins get plotted')
    ap.add_argument('--n-examples', type=int, default=12)
    args = ap.parse_args()

    level = args.level_root
    atype = args.artifact_type
    base_label, cmp_label = args.baseline_label, args.compare_label
    base_dir = config.exclusion_dir(level, atype, base_label)
    cmp_dir = config.exclusion_dir(level, atype, cmp_label)
    out = config.threshold_sweep_dir(level) / f'{atype}_{cmp_label}_vs_{base_label}'
    out.mkdir(parents=True, exist_ok=True)

    # --- find compare-only bins across all subjects present in both dirs ---
    subjects = sorted({p.stem for p in base_dir.glob('sub-*.csv')} &
                      {p.stem for p in cmp_dir.glob('sub-*.csv')})
    per_channel = []   # (n_new, subject, session, run, channel)
    new_bins = {}      # (subject, run, channel) -> list of (bin_start, bin_end)
    total_new = 0
    for subject in subjects:
        base = pd.read_csv(base_dir / f'{subject}.csv')
        cmp = pd.read_csv(cmp_dir / f'{subject}.csv')
        m = cmp.merge(base[KEY + ['excluded']], on=KEY, how='left', suffixes=('_cmp', '_base'))
        m['excluded_base'] = m['excluded_base'].fillna(False)
        newmask = m['excluded_cmp'] & ~m['excluded_base']
        total_new += int(newmask.sum())
        for (sess, run, ch), grp in m[newmask].groupby(['session_id', 'run_id', 'channel']):
            per_channel.append((len(grp), subject, sess, run, ch))
            new_bins[(subject, run, ch)] = list(zip(grp['bin_start'], grp['bin_start'] + 60.0))

    per_channel.sort(reverse=True)
    chosen = per_channel[:args.n_examples]
    print(f"{total_new} {cmp_label}-only {atype} bins across {len(per_channel)} channels; "
          f"plotting top {len(chosen)}")

    info = {
        'description': f'{atype} bins excluded at {cmp_label} but NOT {base_label} '
                       f'(what switching the {atype} threshold from {base_label} to {cmp_label} adds).',
        'artifact_type': atype,
        'shading': {'green': f'excluded at BOTH {base_label} and {cmp_label} (baseline)',
                    'red': f'excluded at {cmp_label} but NOT {base_label} (newly added)'},
        'labels_compared': {'baseline': base_label, 'compare': cmp_label},
        'bin_seconds': 60,
        'total_compare_only_bins_all_channels': total_new,
        'n_channels_with_compare_only_bins': len(per_channel),
        'level_root': str(level),
        'run_timestamp': config.run_timestamp(),
        'git': config.git_provenance(),
        'plots': [],
    }

    for n_new, subject, sess, run, ch in chosen:
        subj = subject.replace('sub-', '')
        run_bare = run.replace('run-', '')
        runs = io_utils.get_session_runs(subj)
        nwb = next((p for s, r, p in runs if r == run_bare), None)
        if nwb is None:
            print(f"  {subject} {run} {ch}: nwb not found, skip"); continue
        data_v, found, sfreq = io_utils.load_channels_subset(nwb, [ch])
        if ch not in found:
            print(f"  {subject} {run} {ch}: channel not found, skip"); continue
        trace_uv = data_v[:, found.index(ch)] * 1e6
        t = np.arange(len(trace_uv)) / sfreq

        base = pd.read_csv(base_dir / f'{subject}.csv')
        baseb = base[(base['run_id'] == run) & (base['channel'] == ch) & base['excluded']]
        base_spans = list(zip(baseb['bin_start'], baseb['bin_start'] + 60.0))
        cmp_only_spans = new_bins[(subject, run, ch)]

        fig, ax = plt.subplots(figsize=(14, 3))
        for a, b in base_spans:
            ax.axvspan(a, b, color='#55a868', alpha=0.35, linewidth=0, zorder=1)
        for a, b in cmp_only_spans:
            ax.axvspan(a, b, color='#c44e52', alpha=0.40, linewidth=0, zorder=1)
        ax.plot(t, trace_uv, linewidth=0.4, color='black', zorder=2)
        ax.set_xlabel('Time (s)'); ax.set_ylabel(chr(181) + 'V')
        ax.set_title(f"{subject} {run} {ch}  {atype}  "
                     f"green={base_label}&{cmp_label} ({len(base_spans)})  "
                     f"red={cmp_label}-only ({len(cmp_only_spans)})", fontsize=10)
        fig.tight_layout()
        out_path = out / f"{subject}_{run}_{ch}.png"
        fig.savefig(out_path, dpi=150); plt.close(fig)
        print(f"  saved {out_path}  ({base_label}={len(base_spans)} {cmp_label}only={len(cmp_only_spans)})")
        info['plots'].append({
            'file': out_path.name, 'subject': subject, 'session': sess, 'run': run, 'channel': ch,
            f'n_bins_{base_label}': len(base_spans), f'n_bins_{cmp_label}_only': len(cmp_only_spans),
            'run_duration_s': round(len(trace_uv) / sfreq, 1),
        })

    with open(out / 'info.json', 'w') as f:
        json.dump(info, f, indent=2, default=str)
    print(f"Wrote {out / 'info.json'} ({len(info['plots'])} plots)")


if __name__ == '__main__':
    main()
