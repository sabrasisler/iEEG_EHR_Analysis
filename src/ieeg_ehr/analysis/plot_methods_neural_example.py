"""One methods-figure window with its z-scored band power: pain, MME, neural data.

    python -m ieeg_ehr.analysis.plot_methods_neural_example \\
        --options-run <trajectory_candidates run dir> --option 20 \\
        --reference-run <pain_change run dir>

The window is one row of a `plot_methods_trajectory_candidates` run's
`options.csv`. `--reference-run` supplies the pain_change view config (mask,
`max_excluded_frac`) and ROI scheme.

The z-score is BAND-LEVEL. Per channel and 2 s window, band power is the log10 of
the linear mean over the band's native bins after the line-noise notch
(`axes.aggregate_bands`). Its baseline is the mean and SD of that band power over
every masked window of every pain epoch in the session, read from the epoch cache
(`build_pain_epoch_fullres_zscore.open_session`, `axes.BaselineAccumulator`). The
8 h trace reads the same windows from the continuous `preprocessed/bipolar_psd`
NWBs and z-scores them against that baseline. Over the pain epochs the band z has
mean 0 and SD 1. pain_change instead averages per-frequency z-scores over a band,
whose spread is smaller.

An epoch value is the mean over the 2 s windows that start in the 5 min before a
score, NaN when more than `max_excluded_frac` of them are masked. The epoch band
power from the NWBs must reproduce the epoch band power from the cache, NaN
pattern included, or the script raises.

The channels are the reference run's ROI-labelled channels. A channel is CLEAN
when no window of the window's epochs is masked on it. `--channel` names the one
in the main figure. Its default is the clean channel whose epoch values vary most
from epoch to epoch (SD over epochs, mean over bands): a choice for legibility,
not for any relation to pain.

Writes to `analysis/methods_figures/pain_change/neural_example/<run>_<timestamp>/`:
    example.png         pain, MME, and one row per band for the chosen channel:
                        the 1-min mean z, each 5-min pre-score epoch shaded with
                        its mean as a bar
    channels.png        per-minute z of every clean channel, one heatmap per band,
                        channels grouped by ROI, to choose a different `--channel`
    epoch_band_z.csv    per epoch x ROI channel x band: z, excluded fraction
    minute_band_z.csv   per minute x clean channel x band: z
"""

import argparse
import json
import logging
import warnings
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from ieeg_ehr import config, io
from ieeg_ehr.analysis import fullres_cells
from ieeg_ehr.analysis import plot_methods_trajectory_candidates as candidates
from ieeg_ehr.analysis.pain_change import load_exposure
from ieeg_ehr.analysis.run_bandpower_mixed import BAND_SETS
from ieeg_ehr.analysis.run_mixed_model_pilot import roi_maps
from ieeg_ehr.features import build_pain_change_trajectory as traj
from ieeg_ehr.med_analysis import pain_link, style
from ieeg_ehr.med_analysis.style import plt
from ieeg_ehr.preprocessing.run_pipeline_bipolar import DATA_PATH
from ieeg_ehr.qc import mask_projection
from ieeg_ehr.views import axes, fullres_reader, view_config
from ieeg_ehr.views import build_pain_epoch_fullres_zscore as zview

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

SCRIPT = 'ieeg_ehr/analysis/plot_methods_neural_example.py'
OUTPUT_TYPE = 'neural_example'
BAND_SET = 'paper_bands_6_hg200'
BAND_COLOR = {'delta': style.PALETTE[2], 'beta': style.PALETTE[6],
              'high_gamma': style.PALETTE[7]}
BAND_LABEL = {'delta': 'Delta', 'theta': 'Theta', 'alpha': 'Alpha', 'beta': 'Beta',
              'gamma': 'Gamma', 'high_gamma': 'High gamma'}
EPOCH_SHADE = '#f6d5e5'
HEATMAP_CMAP = 'RdBu_r'


def band_power(log_power, bin_table, bands):
    """(..., F) log10 power -> (..., n_bands) log10 of the linear mean over each band."""
    lead = log_power.shape[:-1]
    out, names = axes.aggregate_bands(log_power.reshape(-1, log_power.shape[-1]),
                                      bin_table, bands=bands, is_difference=False,
                                      domain='log')
    if names != list(bands):
        raise ValueError(f'bands {names} are not {list(bands)}')
    return out.reshape(*lead, len(names))


def finite_or_nan(block):
    """The cache and the NWBs hold a few isolated non-finite values; they drop out."""
    block[~np.isfinite(block)] = np.nan
    return block


def cache_band_power(sess, bins, bin_table, bands, min_windows):
    """Band baseline (mean, sd) per session channel, and each epoch's mean band power.

    Pass 1 of `build_pain_epoch_fullres_zscore` on band power instead of bins. The
    epoch means are keyed on pain_time: (session-channel rows, (C_run, n_bands)).
    """
    acc = axes.BaselineAccumulator(len(sess.session_channels), len(bands))
    epochs = {}
    for ep, block, _kept, _frac in sess.epochs(axes.baseline_epoch_filter(sess.vc.baseline)):
        bp = band_power(finite_or_nan(block[:, :, bins]), bin_table, bands)
        rows = sess.rows_by_run[ep['run_id']]
        acc.update(bp, rows=rows)
        with np.errstate(invalid='ignore'), warnings.catch_warnings():
            warnings.filterwarnings('ignore', message='Mean of empty slice')
            epochs[ep['pain_time']] = (rows, np.nanmean(bp, axis=0))
    mean, sd = acc.finalize(min_windows=min_windows)
    return mean, sd, epochs


def nwb_band_power(runs, sess, channels, bins, bin_table, bands, m0, n_sec):
    """(n_sec, C, n_bands) band power of the 2 s window starting each second from m0.

    NaN where masked or unrecorded. Overlapping runs resolve as in
    `build_pain_change_trajectory.session_runs`: an earlier run keeps a shared second.
    """
    out = np.full((n_sec, len(channels), len(bands)), np.nan)
    slot = {c: i for i, c in enumerate(channels)}
    m1 = m0 + pd.Timedelta(seconds=n_sec)
    chunk = config.PSD_FULLRES_HDF5_CHUNK_WINDOWS
    for run in runs:
        i0 = max(run.first_row, int(np.ceil((m0 - run.start).total_seconds())))
        i1 = min(run.n_windows, int(np.ceil((m1 - run.start).total_seconds())))
        if i0 >= i1:
            continue
        cols = [j for j, ch in enumerate(run.channels) if ch in slot]
        names = [run.channels[j] for j in cols]
        rows = [slot[c] for c in names]
        offset = int((run.start - m0).total_seconds())
        with h5py.File(run.path, 'r') as fh:
            data = fh[DATA_PATH]
            b0 = i0
            while b0 < i1:
                b1 = min((b0 // chunk + 1) * chunk, i1)
                block = finite_or_nan(data[b0:b1][:, cols][:, :, bins]
                                      .astype(config.CACHE_ACCUMULATE_DTYPE))
                excluded = mask_projection.project_pair_mask_to_windows(
                    sess.mask, f'run-{run.run_id}', names, np.arange(b0, b1, dtype=float))
                block[excluded, :] = np.nan
                out[offset + b0:offset + b1, rows] = band_power(block, bin_table, bands)
                b0 = b1
    return out


def epoch_means(values, starts, n_win, max_excluded_frac):
    """(n_epochs, C, n_bands) mean over windows [k, k + n_win) per start k, and the
    (n_epochs, C) NaN fraction. NaN where that fraction exceeds `max_excluded_frac`."""
    stack = np.stack([values[k:k + n_win] for k in starts])
    frac = np.isnan(stack).all(axis=3).mean(axis=1)
    with np.errstate(invalid='ignore'), warnings.catch_warnings():
        warnings.filterwarnings('ignore', message='Mean of empty slice')
        means = np.nanmean(stack, axis=1)
    means[frac > max_excluded_frac] = np.nan
    return means, frac


def shade_epochs(ax, x):
    epoch_h = config.EPOCH_MINUTES_BEFORE / 60.0
    for tx in x:
        ax.axvspan(tx - epoch_h, tx, color=EPOCH_SHADE, lw=0, zorder=0)


def plot_example(o, scores, doses, minute_h, minute_z, epoch_z, channel, roi, bands,
                 out_path):
    n = 2 + len(bands)
    height = 1.7 * n + 0.9
    fig, axs = plt.subplots(n, 1, sharex=True, figsize=(12, height),
                            gridspec_kw=dict(hspace=0.25))
    x = candidates.draw_pain(axs[0], scores, o['start_h'])
    axs[0].set_ylabel('Pain (0-10)', fontsize=style.TICK_SIZE, color=style.TEXT_PRIMARY)
    candidates.draw_mme(axs[1], doses, o['start_h'], o['end_h'])
    axs[1].set_ylabel('Active MME (mg)', fontsize=style.TICK_SIZE,
                      color=style.TEXT_PRIMARY)
    epoch_h = config.EPOCH_MINUTES_BEFORE / 60.0
    for b, (ax, band) in enumerate(zip(axs[2:], bands)):
        color = BAND_COLOR.get(band, style.TEXT_PRIMARY)
        ax.axhline(0, color=style.ZERO_LINE_COLOR, lw=0.8, zorder=1)
        ax.plot(minute_h, minute_z[:, b], color=color, lw=1, alpha=0.8, zorder=2)
        for tx, v in zip(x, epoch_z[:, b]):
            ax.plot([tx - epoch_h, tx], [v, v], color=color, lw=4,
                    solid_capstyle='butt', zorder=3)
        lo, hi = bands[band]
        ax.set_ylabel(f'{BAND_LABEL[band]}\n{lo:g}-{hi:g} Hz (z)',
                      fontsize=style.TICK_SIZE, color=style.TEXT_PRIMARY)
    for ax in axs:
        shade_epochs(ax, x)
        style.style_axes(ax, grid_axis=None)
    axs[-1].set_xlim(0, o['end_h'] - o['start_h'])
    axs[-1].set_xticks(np.arange(0, o['end_h'] - o['start_h'] + 0.1, 1))
    axs[-1].set_xlabel('Hours from window start', fontsize=style.TICK_SIZE,
                       color=style.TEXT_PRIMARY)
    fig.suptitle(f'Option {o["option"]}: sub-{o["subject"]} ses-{o["session"]}, '
                 f'from h {o["start_h"]:.1f}. Channel {channel} ({roi})',
                 fontsize=style.LABEL_SIZE, color=style.TEXT_PRIMARY, x=0.06, y=1 - 0.15 / height,
                 ha='left')
    handles = candidates.legend_handles() + [
        plt.Rectangle((0, 0), 1, 1, color=EPOCH_SHADE, label='5-min pre-score epoch'),
        plt.Line2D([], [], color=style.TEXT_MUTED, lw=1, label='1-min mean z'),
        plt.Line2D([], [], color=style.TEXT_MUTED, lw=4, label='Epoch mean z')]
    fig.legend(handles=handles, loc='lower center', ncol=7, frameon=False,
               fontsize=style.LEGEND_SIZE, bbox_to_anchor=(0.5, 1 - 1.05 / height))
    fig.subplots_adjust(top=1 - 1.1 / height)
    return style.save(fig, out_path, footnote=(
        'z: log10 band power per 2 s window (linear mean over the band\'s bins, line '
        'noise notched) against every pain-epoch window of the session, per channel. '
        'Masked windows out.'))


def plot_channels(o, scores, minute_h, minute_z, channels, region_of, chosen, bands,
                  out_path):
    order = sorted(range(len(channels)), key=lambda j: (region_of[channels[j]],
                                                        channels[j]))
    names = [channels[j] for j in order]
    rois = [region_of[c] for c in names]
    z = minute_z[:, order]
    x = scores['t_h'] - o['start_h']
    epoch_h = config.EPOCH_MINUTES_BEFORE / 60.0
    lim = float(np.nanpercentile(np.abs(z), 98))
    height = 0.09 * len(names) + 1.8
    fig, axs = plt.subplots(1, len(bands), sharey=True, squeeze=False,
                            figsize=(4.6 * len(bands) + 3.0, height))
    axs = axs[0]
    for b, (ax, band) in enumerate(zip(axs, bands)):
        im = ax.imshow(z[:, :, b].T, aspect='auto', cmap=HEATMAP_CMAP, vmin=-lim,
                       vmax=lim, interpolation='nearest',
                       extent=(minute_h[0], minute_h[-1] + 1 / 60, len(names) - 0.5,
                               -0.5))
        for tx in x:
            ax.axvline(tx - epoch_h, color=style.TEXT_PRIMARY, lw=0.5)
            ax.axvline(tx, color=style.TEXT_PRIMARY, lw=0.5)
        for k in np.flatnonzero(np.r_[False, np.array(rois[1:]) != np.array(rois[:-1])]):
            ax.axhline(k - 0.5, color=style.TEXT_PRIMARY, lw=0.8)
        lo, hi = bands[band]
        ax.set_title(f'{BAND_LABEL[band]} {lo:g}-{hi:g} Hz', fontsize=style.LABEL_SIZE,
                     color=style.TEXT_PRIMARY, loc='left')
        ax.set_xlabel('Hours from window start', fontsize=style.TICK_SIZE,
                      color=style.TEXT_PRIMARY)
        ax.tick_params(labelsize=style.TICK_SIZE, colors=style.TEXT_MUTED)
    axs[0].set_yticks(range(len(names)))
    axs[0].set_yticklabels([f'{r}  {c}' for r, c in zip(rois, names)], fontsize=5)
    axs[0].get_yticklabels()[names.index(chosen)].set_fontweight('bold')
    fig.colorbar(im, ax=axs, shrink=min(1.0, 6 / height), label='z (1-min mean)')
    fig.suptitle(f'Option {o["option"]}: sub-{o["subject"]} ses-{o["session"]}, '
                 f'{len(names)} clean ROI channels. Lines bound each pre-score epoch',
                 fontsize=style.TITLE_SIZE, color=style.TEXT_PRIMARY, x=0.01, ha='left')
    return style.save(fig, out_path)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--options-run', required=True)
    parser.add_argument('--option', type=int, required=True)
    parser.add_argument('--reference-run', required=True)
    parser.add_argument('--bands', nargs='+', default=['high_gamma'])
    parser.add_argument('--channel', default=None)
    parser.add_argument('--run-name', default=None)
    args = parser.parse_args()

    io.warn_if_dirty()

    options_run = Path(args.options_run)
    options = io.read_table(options_run / 'options.csv', on_stale='warn')
    o = options.set_index('option').loc[args.option].to_dict()
    # The CSV round-trip reads the ids as integers.
    o.update(option=args.option, subject=f'{int(o["subject"]):03d}',
             session=f'{int(o["session"]):02d}')
    subject, session, sid = o['subject'], o['session'], f'sub-{o["subject"]}'
    reference = traj.load_reference(args.reference_run)
    vc = view_config.ViewConfig(**reference['view_config'])
    options_mask = json.loads((options_run / 'provenance.json').read_text())['params'][
        'mask_label']
    if options_mask != vc.mask_label:
        raise ValueError(f'options chose clean pairs with {options_mask}, the reference '
                         f'run masks with {vc.mask_label}')
    if vc.baseline != 'all_pain_epochs':
        raise ValueError(f'the reference run baselines on {vc.baseline}, not every pain '
                         'epoch')
    bands = {b: BAND_SETS[BAND_SET][b] for b in args.bands}

    starts, all_doses = load_exposure()
    start = starts[(subject, session)]
    scores = pain_link.load_pain_scores()
    scores = scores[(scores['subject'] == subject) & (scores['session'] == session)].copy()
    scores['t_h'] = (scores['score_dt'] - start).dt.total_seconds() / 3600.0
    scores = candidates.window_scores(scores, o['start_h'], o['end_h'])
    doses = all_doses[(all_doses['subject'] == subject) & (all_doses['session'] == session)]

    zpath = zview.zscore_dir(vc) / f'zmean_{sid}_ses-{session}.parquet'
    roi_by_subject, _ = roi_maps([zpath], {sid}, reference['roi_scheme'],
                                 insula_threshold=reference['insula_threshold'])
    region_of = roi_by_subject[sid]
    sess = zview.open_session(subject, session, vc)
    freq_table, notch = fullres_cells.analysis_freq_table()
    bins = freq_table.index.to_numpy()
    bin_table = freq_table.reset_index()
    mean, sd, cache_epochs = cache_band_power(
        sess, bins, bin_table, bands,
        zview.zscore_params(vc, fullres_reader.n_freqs())['baseline_min_windows'])
    channels = [c for c in sess.session_channels if c in region_of]
    srows = [sess.session_channels.index(c) for c in channels]

    m0 = (start + pd.Timedelta(hours=o['start_h'])).round('s')
    n_sec = int(round((o['end_h'] - o['start_h']) * 3600))
    bp = nwb_band_power(traj.session_runs(subject, session), sess, channels, bins,
                        bin_table, bands, m0, n_sec)
    z = axes.normalize(bp, mean[srows], sd[srows], 'zscore_vs_baseline')
    with np.errstate(invalid='ignore'), warnings.catch_warnings():
        warnings.filterwarnings('ignore', message='Mean of empty slice')
        minute_z = np.nanmean(z.reshape(-1, 60, len(channels), len(bands)), axis=1)
    minute_h = np.arange(minute_z.shape[0]) / 60.0

    epoch = pd.Timedelta(minutes=config.EPOCH_MINUTES_BEFORE)
    n_win = int(epoch.total_seconds())
    max_excl = vc.resolved().max_excluded_frac
    starts_s = [int((t - epoch - m0).total_seconds()) for t in scores['score_dt']]
    epoch_bp, frac = epoch_means(bp, starts_s, n_win, max_excl)
    epoch_z = axes.normalize(epoch_bp, mean[srows], sd[srows], 'zscore_vs_baseline')

    cached = np.full_like(epoch_bp, np.nan)
    slot = {r: j for j, r in enumerate(srows)}
    for i, t in enumerate(scores['score_dt']):
        rows, values = cache_epochs[t]
        keep = [k for k, r in enumerate(rows) if r in slot]
        cached[i, [slot[rows[k]] for k in keep]] = values[keep]
    nan_mismatch = int((np.isnan(cached) != np.isnan(epoch_bp)).sum())
    max_diff = float(np.nanmax(np.abs(cached - epoch_bp)))
    logger.info('epoch band power, NWB vs cache: max |diff| %.3g, %d NaN-pattern '
                'mismatches', max_diff, nan_mismatch)
    if nan_mismatch or max_diff > traj.VALIDATION_TOL:
        raise AssertionError(f'NWB epoch band power does not reproduce the cache: max '
                             f'|diff| {max_diff}, {nan_mismatch} NaN mismatches')

    clean = [c for j, c in enumerate(channels) if (frac[:, j] == 0).all()]
    ci = [channels.index(c) for c in clean]
    spread = np.nanstd(epoch_z[:, ci], axis=0).mean(axis=1)
    channel = args.channel or clean[int(np.argmax(spread))]
    if channel not in clean:
        raise ValueError(f'{channel} is not a clean ROI channel of this window: {clean}')
    j = channels.index(channel)
    logger.info('%d ROI channels, %d clean; plotting %s (%s)', len(channels), len(clean),
                channel, region_of[channel])

    run_name = args.run_name or f'option{args.option}_{sid}'
    run_dir = config.analysis_run_dir(candidates.QUESTION, OUTPUT_TYPE, run_name,
                                      event=candidates.EVENT)
    run_dir.mkdir(parents=True, exist_ok=True)
    plot_example(o, scores, doses, minute_h, minute_z[:, j], epoch_z[:, j], channel,
                 region_of[channel], bands, run_dir / 'example.png')
    plot_channels(o, scores, minute_h, minute_z[:, ci], clean, region_of, channel, bands,
                  run_dir / 'channels.png')

    n_min = minute_z.shape[0]
    names = list(bands)
    epoch_rows = pd.DataFrame([
        dict(score_dt=t, t_h=th, pain_score=p, channel=c, roi=region_of[c],
             clean=c in clean, band=b, z=epoch_z[i, jj, bi], excluded_frac=frac[i, jj])
        for i, (t, th, p) in enumerate(zip(scores['score_dt'], scores['t_h'],
                                           scores['pain_score']))
        for jj, c in enumerate(channels) for bi, b in enumerate(names)])
    mz = minute_z[:, ci]
    minute_rows = pd.DataFrame({
        'minute_h': np.repeat(minute_h, len(clean) * len(names)),
        'channel': np.tile(np.repeat(clean, len(names)), n_min),
        'band': np.tile(names, n_min * len(clean)),
        'z': mz.reshape(-1)})
    params = {**vars(args), 'view_config': reference['view_config'],
              'roi_scheme': reference['roi_scheme'], 'bands': bands,
              'notch_bins_removed': notch, 'channel': channel}
    parents = [str(options_run), str(reference['run_dir']),
               io.manifest_ref(config.fullres_epoch_unit_dir()),
               io.manifest_ref(config.BIPOLAR_PSD_FULLRES_ROOT)]
    common = dict(kind='table', script=SCRIPT, params=params, parents=parents,
                  subjects=[sid])
    io.write_table(epoch_rows, run_dir / 'epoch_band_z.csv', **common)
    io.write_table(minute_rows, run_dir / 'minute_band_z.csv', **common)
    io.write_run_provenance(run_dir, script=SCRIPT, params=params, parents=parents,
                            subjects=[sid],
                            extra={'n_roi_channels': len(channels),
                                   'n_clean_channels': len(clean),
                                   'zscore': 'band-level, vs every pain-epoch window',
                                   'validation_max_abs_diff': max_diff,
                                   'validation_tol': traj.VALIDATION_TOL,
                                   'status': 'EXPLORATORY methods-figure example'})
    io.log_analysis(f'Methods-figure neural example: option {args.option}, {sid}, '
                    f'{channel}, {" ".join(names)} z', run_dir)
    logger.info('-> %s', run_dir)


if __name__ == '__main__':
    main()
