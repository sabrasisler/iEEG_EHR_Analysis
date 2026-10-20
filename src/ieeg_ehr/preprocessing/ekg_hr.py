"""
Heart rate at every second of a session, from the EKG leads, with NeuroKit2.

Writes two tables per subject-session into preprocessed/ekg_hr/:

    sub-XXX_ses-YY_hr.parquet       one row per recorded second
        run, second, time, hr_bpm, quality, rr_sec
    sub-XXX_ses-YY_rpeaks.parquet   one row per R peak
        run, sample, t_run_sec, time, quality

Per run:

1. Candidate leads from the first two columns of ElectricalSeries_EKG:
   ch0 - ch1, ch0, ch1. The difference is a bipolar lead that cancels
   common-mode noise. It gave the most clean beats in 11 of 12 subjects
   checked on 2026-10-19. The single leads cover a run where one electrode
   came off.
2. Polarity: `nk.ecg_invert` votes on 10 s blocks. On a whole run it builds an
   (n_samples x 2 s) sliding window and needs ~28 GB per 30 min at 1 kHz.
3. `nk.ecg_clean(method='neurokit', powerline=60)`, then
   `nk.ecg_peaks(correct_artifacts=True)` (Lipponen & Tarvainen 2019).
4. Beat quality: `nk.ecg_quality(method='templatematch')`, the correlation of
   each beat with the average beat, one template per 5-min block. One template
   per run would build a per-sample pandas frame of the whole run.
5. Keep the lead with the most beats at quality >= LEAD_MIN_QUALITY.
6. `nk.signal_rate` gives 60 / RR at each beat. `nk.signal_interpolate`
   (monotone cubic) evaluates it at the center of each second.

QUALITY IS A METRIC, NOT A MASK. Nothing here thresholds hr_bpm (the
metric/threshold split, as in qc/). A second's `quality` is the lower of the
two beats that bracket it, and `rr_sec` is the interval between them, so a
stretch with no detected beats shows up as a long `rr_sec`. Seconds before
the second beat or after the last one are NaN. On 2026-10-19 a lead with no
ECG in it scored a median of 0.19, clean subjects 0.92-0.99.

`time` is the wall clock at the START of the second: session_start_time +
starting_time + second, UTC offset stripped. Correct on V1 and V2
(docs/dataset_v2.md §3). hr_bpm is evaluated at the second's center.

A V1 run whose EKG fails to decode (corrupt chunk, truncated file; 257 of
7,781 runs on 2026-10-19, docs/dataset_v2.md §2.2) is read from its V2 copy,
which holds the same samples. The sidecar's `runs` records the `source` of
each run. A run that neither version can supply, or that has no EKG series,
is logged, listed in `runs` with its error, and contributes no rows.

Usage:
  python -m ieeg_ehr.preprocessing.ekg_hr --subjects 019 085
  python -m ieeg_ehr.preprocessing.ekg_hr --subjects 019 --session 01 --overwrite
"""

import argparse
import logging
import time
import warnings

import neurokit2 as nk
import numpy as np
import pandas as pd
from pynwb import NWBHDF5IO

from ieeg_ehr import config, io
from ieeg_ehr.io import nwb as nwb_io

logger = logging.getLogger(__name__)

EKG_SERIES = 'ElectricalSeries_EKG'
POWERLINE_HZ = 60
INVERT_BLOCK_SEC = 10
INVERT_EVERY_SEC = 300
QUALITY_BLOCK_SEC = 300
#: `nk.signal_cyclesegment` refuses anything shorter.
QUALITY_MIN_BLOCK_SEC = 4
QUALITY_MIN_BLOCK_BEATS = 10
LEAD_MIN_QUALITY = 0.8

PARAMS = {
    'series': EKG_SERIES,
    'leads': 'ch0-ch1 | ch0 | ch1, most beats at quality >= lead_min_quality',
    'lead_min_quality': LEAD_MIN_QUALITY,
    'invert': f'nk.ecg_invert majority over {INVERT_BLOCK_SEC} s blocks '
              f'every {INVERT_EVERY_SEC} s',
    'clean': f"nk.ecg_clean(method='neurokit', powerline={POWERLINE_HZ})",
    'peaks': "nk.ecg_peaks(method='neurokit', correct_artifacts=True)",
    'quality': f"nk.ecg_quality(method='templatematch') per {QUALITY_BLOCK_SEC} s block",
    'rate': "nk.signal_rate, nk.signal_interpolate(method='monotone_cubic') at second centers",
    'neurokit2': nk.__version__,
}


def read_ekg(nwb_path):
    """The first two EKG columns in volts, their labels, the rate, and the
    run's wall-clock start."""
    with NWBHDF5IO(str(nwb_path), 'r') as handle:
        nwb = handle.read()
        series = nwb.acquisition[EKG_SERIES]
        if series.unit != 'volts':
            raise ValueError(f"unexpected unit {series.unit!r} in {nwb_path}")
        labels = list(nwb.electrodes.to_dataframe()
                      .iloc[series.electrodes.data[:]]['location'])[:2]
        data = series.data[:, :2] * series.conversion
        start = (pd.Timestamp(nwb.session_start_time)
                 + pd.Timedelta(seconds=float(series.starting_time))).tz_localize(None)
    return data, labels, float(series.rate), start


def candidate_leads(data, labels):
    if data.shape[1] == 1:
        return {labels[0]: data[:, 0]}
    return {f'{labels[0]}-{labels[1]}': data[:, 0] - data[:, 1],
            labels[0]: data[:, 0], labels[1]: data[:, 1]}


def fix_polarity(x, fs):
    block = int(INVERT_BLOCK_SEC * fs)
    starts = range(0, max(len(x) - block, 0) + 1, int(INVERT_EVERY_SEC * fs))
    inverted = np.mean([nk.ecg_invert(x[i:i + block], sampling_rate=fs)[1]
                        for i in starts]) > 0.5
    return (-x if inverted else x), bool(inverted)


def quality_blocks(n_samples, fs):
    """[start, stop) blocks of QUALITY_BLOCK_SEC, a short tail merged into the
    block before it."""
    block = int(QUALITY_BLOCK_SEC * fs)
    starts = list(range(0, n_samples, block))
    if len(starts) > 1 and n_samples - starts[-1] < block // 2:
        starts.pop()
    return list(zip(starts, starts[1:] + [n_samples]))


def beat_quality(clean, peaks, fs):
    """templatematch quality at each beat. NaN in a block too short or with too
    few beats to build a template."""
    quality = np.full(len(peaks), np.nan)
    for a, b in quality_blocks(len(clean), fs):
        sel = (peaks >= a) & (peaks < b)
        if b - a < QUALITY_MIN_BLOCK_SEC * fs or sel.sum() < QUALITY_MIN_BLOCK_BEATS:
            continue
        local = peaks[sel] - a
        q = nk.ecg_quality(clean[a:b], rpeaks=local, sampling_rate=fs,
                           method='templatematch')
        quality[sel] = q[local]
    return quality


def detect_beats(x, fs):
    clean = nk.ecg_clean(x, sampling_rate=fs, method='neurokit', powerline=POWERLINE_HZ)
    _, info = nk.ecg_peaks(clean, sampling_rate=fs, method='neurokit',
                           correct_artifacts=True)
    peaks = np.asarray(info['ECG_R_Peaks'], dtype=np.int64)
    return peaks, beat_quality(clean, peaks, fs)


def best_lead(leads, fs):
    best = None
    for name, x in leads.items():
        x, inverted = fix_polarity(x, fs)
        peaks, quality = detect_beats(x, fs)
        n_good = int(np.sum(quality >= LEAD_MIN_QUALITY))
        if best is None or n_good > best['n_good_beats']:
            best = {'lead': name, 'inverted': inverted, 'peaks': peaks,
                    'quality': quality, 'n_good_beats': n_good}
    return best


def hr_per_second(peaks, quality, fs, n_samples):
    """(hr_bpm, quality, rr_sec) at the center of each whole second of the run."""
    centers = np.arange(int(n_samples // fs)) + 0.5
    hr = np.full(len(centers), np.nan)
    q = np.full(len(centers), np.nan)
    rr = np.full(len(centers), np.nan)
    # signal_rate needs more than 3 peaks, and fills the first beat's rate
    # with the mean period, so HR starts at the second beat.
    if len(peaks) > 3:
        t = peaks / fs
        inside = (centers > t[1]) & (centers < t[-1])
        hr[inside] = nk.signal_interpolate(t, nk.signal_rate(peaks, sampling_rate=fs),
                                           x_new=centers[inside], method='monotone_cubic')
        after = np.searchsorted(t, centers[inside])
        q[inside] = np.minimum(quality[after - 1], quality[after])
        rr[inside] = t[after] - t[after - 1]
    return hr, q, rr


def read_ekg_v1_or_v2(subject, session, run, nwb_path):
    """read_ekg on the V1 file, or on the V2 copy when V1 fails to decode."""
    try:
        return read_ekg(nwb_path) + ('V1',)
    except OSError as e:
        v2_path = config.v2_raw_nwb_path(subject, session, run)
        logger.warning('sub-%s ses-%s run-%s: V1 EKG unreadable (%r); reading %s',
                       subject, session, run, e, v2_path)
        return read_ekg(v2_path) + ('V2',)


def process_run(subject, session, run, nwb_path):
    """One run's per-second table, beat table and summary."""
    data, labels, fs, start, source = read_ekg_v1_or_v2(subject, session, run, nwb_path)
    best = best_lead(candidate_leads(data, labels), fs)
    peaks, quality = best['peaks'], best['quality']
    hr, q, rr = hr_per_second(peaks, quality, fs, len(data))

    seconds = np.arange(len(hr))
    hr_df = pd.DataFrame({'run': run, 'second': seconds,
                          'time': start + pd.to_timedelta(seconds, unit='s'),
                          'hr_bpm': hr, 'quality': q, 'rr_sec': rr})
    t_run = peaks / fs
    beats_df = pd.DataFrame({'run': run, 'sample': peaks, 't_run_sec': t_run,
                             'time': start + pd.to_timedelta(t_run, unit='s'),
                             'quality': quality})
    summary = {'run': run, 'source': source, 'sfreq': fs, 'n_samples': len(data), 'labels': labels,
               'lead': best['lead'], 'inverted': best['inverted'],
               'n_beats': len(peaks), 'n_good_beats': best['n_good_beats']}
    return hr_df, beats_df, summary


def build_session(subject, session, runs, overwrite=False):
    hr_path = config.ekg_hr_path(subject, session)
    beats_path = config.ekg_rpeaks_path(subject, session)
    if hr_path.exists() and not overwrite:
        logger.info('sub-%s ses-%s: %s exists, skipping (--overwrite to redo)',
                    subject, session, hr_path.name)
        return

    hr_tables, beat_tables, summaries = [], [], []
    for run, nwb_path in runs:
        t0 = time.time()
        try:
            hr_df, beats_df, summary = process_run(subject, session, run, nwb_path)
        except (OSError, KeyError) as e:
            logger.error('sub-%s ses-%s run-%s: cannot read EKG (%r); no rows',
                         subject, session, run, e)
            summaries.append({'run': run, 'error': repr(e)})
            continue
        hr_tables.append(hr_df)
        beat_tables.append(beats_df)
        summaries.append(summary)
        logger.info('sub-%s ses-%s run-%s: %d Hz, lead %s%s, %d beats (%d good), '
                    'median HR %.1f, %.0f s', subject, session, run, summary['sfreq'],
                    summary['lead'], ' (inverted)' if summary['inverted'] else '',
                    summary['n_beats'], summary['n_good_beats'],
                    np.nanmedian(hr_df['hr_bpm']) if hr_df['hr_bpm'].notna().any() else np.nan,
                    time.time() - t0)

    if not hr_tables:
        raise RuntimeError(f'sub-{subject} ses-{session}: no run had readable EKG')

    parents = [str(p) for _, p in runs] + [
        str(config.v2_raw_nwb_path(subject, session, s['run']))
        for s in summaries if s.get('source') == 'V2']
    sidecar = dict(params=PARAMS, parents=parents,
                   subjects=[f'sub-{subject}'],
                   extra={'runs': summaries,
                          'n_runs_failed': sum('error' in s for s in summaries)})
    io.write_table(pd.concat(hr_tables, ignore_index=True), hr_path, **sidecar)
    io.write_table(pd.concat(beat_tables, ignore_index=True), beats_path, **sidecar)
    logger.info('sub-%s ses-%s: wrote %s and %s', subject, session,
                hr_path.name, beats_path.name)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--subjects', nargs='+', required=True,
                    help='Subject IDs, e.g. 019 085 (or sub-019).')
    ap.add_argument('--session', default=None, help='One session, e.g. 01. Default: all.')
    ap.add_argument('--overwrite', action='store_true')
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    warnings.filterwarnings('ignore', category=UserWarning, module='pynwb')
    io.warn_if_dirty()

    for s in args.subjects:
        subject = s.replace('sub-', '')
        runs = pd.DataFrame(nwb_io.get_session_runs(subject, args.session),
                            columns=['session', 'run', 'path'])
        if runs.empty:
            raise ValueError(f'sub-{subject}: no runs in the registry'
                             + (f' for ses-{args.session}' if args.session else ''))
        for session, grp in runs.groupby('session'):
            build_session(subject, session, list(zip(grp['run'], grp['path'])),
                          overwrite=args.overwrite)

    io.log_analysis('EKG heart rate per second (NeuroKit2)', config.EKG_HR_ROOT)


if __name__ == '__main__':
    main()
