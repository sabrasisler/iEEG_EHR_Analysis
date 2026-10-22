#!/usr/bin/env python3
"""Where each V2 session has iEEG on Sherlock: one row per subject-session.

    <V2_DERIVATIVES_BASE>/recording_coverage/<label>_<timestamp>/
        fig_recording_coverage.png
        nwb_runs.csv       one row per NWB file found, readable or not
        load_errors.csv    the subset with a load problem
        sessions.csv       one row per subject-session

Every number comes from the NWB headers in `iEEG_EHR_V2/raw/`, never from the
status CSVs, which lag the filesystem (docs/dataset_v2.md §1).

TIME. A run starts at `session_start_time + starting_time` and ends
`n_samples / rate` later, taken as the min start and max end over the file's
ElectricalSeries (dataset_v2.md §3). The UTC offset is dropped, which keeps the
wall clock (§3.2). No NWB stores a session end, so the session span runs from
the first run start to the last run end of the readable files.

LOAD PROBLEMS. A file fails when h5py cannot open it or its header lacks a
timing field, and it then has no place on the time axis. A file that opens
can still hold unallocated chunks, which read back as fill value without
raising (§2.2). The chunk count is compared against the shape from metadata
alone; this does not decompress any chunk, so a chunk that is present but
corrupt passes.
"""

import argparse
import logging
import math
import os
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timedelta

import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

from ieeg_ehr import config, io

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

SCRIPT = 'ieeg_ehr/analysis/plot_v2_recording_coverage.py'

RUN_COLOUR = '#2a78d6'
ISSUE_COLOUR = '#d03b3b'
SPAN_COLOUR = '#e4e3df'
TEXT_SECONDARY = '#52514e'

#: Many sessions overlap 5 s at every run boundary (2 h 5 s runs on a 2 h
#: cadence). The plot flags only overlaps well past that, and sessions.csv
#: carries every session's total.
OVERLAP_FLAG_H = 1.0


def _text(v):
    v = v[0] if isinstance(v, np.ndarray) and v.ndim else v
    return v.decode() if isinstance(v, bytes) else str(v)


def read_header(path):
    """Timing and chunk completeness of one NWB, or the error that stopped it."""
    parts = path.split('/')
    row = dict(subject=parts[-4], session=parts[-3],
               run=path.split('run-')[-1][:-len('.nwb')], path=path,
               size_bytes=os.path.getsize(path))
    try:
        with h5py.File(path, 'r') as f:
            t0 = datetime.fromisoformat(_text(f['session_start_time'][()])).replace(tzinfo=None)
            starts, ends, missing = [], [], 0
            for name in sorted(f['acquisition']):
                g = f[f'acquisition/{name}']
                if 'starting_time' not in g:
                    continue
                st = g['starting_time']
                data = g['data']
                start = float(st[()])
                starts.append(start)
                ends.append(start + data.shape[0] / float(st.attrs['rate']))
                if data.chunks is not None:
                    expected = math.prod(math.ceil(n / c) for n, c in zip(data.shape, data.chunks))
                    missing += expected - data.id.get_num_chunks()
            if not starts:
                raise KeyError('no acquisition series carries starting_time')
    except Exception as e:
        return dict(row, error=f'{type(e).__name__}: {e}')
    return dict(row, run_start=t0 + timedelta(seconds=min(starts)),
                run_end=t0 + timedelta(seconds=max(ends)),
                n_series=len(starts), missing_chunks=missing, error=None)


def _union_h(g):
    """Hours covered by at least one run, so overlapping runs count once."""
    total, end = 0.0, None
    for s, e in sorted(zip(g.run_start, g.run_end)):
        if end is None or s > end:
            total += (e - s).total_seconds()
            end = e
        elif e > end:
            total += (e - end).total_seconds()
            end = e
    return total / 3600


def summarize_sessions(runs):
    ok = runs[runs.error.isna()]
    span = ok.groupby(['subject', 'session']).agg(
        session_start=('run_start', 'min'), session_end=('run_end', 'max'),
        summed_h=('duration_h', 'sum'))
    span['recorded_h'] = ok.groupby(['subject', 'session'])[['run_start', 'run_end']].apply(_union_h)
    span['overlap_h'] = span.summed_h - span.recorded_h
    counts = runs.groupby(['subject', 'session']).agg(
        n_files=('run', 'size'),
        n_unreadable=('error', lambda s: int(s.notna().sum())),
        n_missing_chunks=('missing_chunks', lambda s: int((s > 0).sum())))
    s = counts.join(span).reset_index()
    s['span_h'] = (s.session_end - s.session_start).dt.total_seconds() / 3600
    s['coverage'] = s.recorded_h / s.span_h
    return s


def plot(runs, sessions, out_path, dpi):
    n = len(sessions)
    fig, ax = plt.subplots(figsize=(12, 1.6 + 0.2 * n))
    t0 = dict(zip(zip(sessions.subject, sessions.session), sessions.session_start))
    ok = runs[runs.error.isna()]
    max_days = (sessions.span_h.max() / 24) if sessions.span_h.notna().any() else 1
    for i, s in sessions.iterrows():
        if pd.notna(s.span_h):
            ax.barh(i, s.span_h / 24, left=0, height=0.8, color=SPAN_COLOUR, linewidth=0)
        mine = ok[(ok.subject == s.subject) & (ok.session == s.session)]
        start = t0[(s.subject, s.session)]
        for issue, colour in ((False, RUN_COLOUR), (True, ISSUE_COLOUR)):
            sel = mine[(mine.missing_chunks > 0) == issue]
            left = (sel.run_start - start).dt.total_seconds() / 86400
            ax.broken_barh(list(zip(left, sel.duration_h / 24)), (i - 0.4, 0.8),
                           color=colour, linewidth=0, antialiased=False)
        note = f'{s.n_files} files'
        if pd.notna(s.coverage):
            note += f' · {s.coverage:.0%} covered'
        ax.text(1.005, i, note, transform=ax.get_yaxis_transform(),
                va='center', fontsize=7, color=TEXT_SECONDARY)
        problems = []
        if s.n_unreadable:
            problems.append(f'{s.n_unreadable} unreadable')
        if s.n_missing_chunks:
            problems.append(f'{s.n_missing_chunks} missing chunks')
        if s.overlap_h >= OVERLAP_FLAG_H:
            problems.append(f'runs overlap {s.overlap_h:.1f} h')
        if problems:
            ax.text(1.16, i, '✕ ' + ', '.join(problems), transform=ax.get_yaxis_transform(),
                    va='center', fontsize=7, color=ISSUE_COLOUR)

    ax.set_yticks(range(n), [f'{a} {b}' for a, b in zip(sessions.subject, sessions.session)],
                  fontsize=7)
    ax.set_ylim(n - 0.5, -0.5)
    ax.set_xlim(0, max_days * 1.01)
    ax.set_xlabel('Days since session start (first run start)')
    ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(1 if max_days <= 30 else 5))
    ax.tick_params(axis='x', top=True, labeltop=True, labelsize=8)
    ax.grid(axis='x', color='#d8d7d2', linewidth=0.5)
    ax.set_axisbelow(True)
    for side in ('top', 'right', 'left'):
        ax.spines[side].set_visible(False)
    height = fig.get_figheight()
    fig.subplots_adjust(top=1 - 1.1 / height, bottom=0.5 / height)
    fig.suptitle(f'iEEG_EHR_V2 recording coverage from NWB headers: '
                 f'{sessions.subject.nunique()} subjects, {n} sessions, {len(runs)} files',
                 x=0.02, y=1 - 0.15 / height, ha='left', fontsize=10)
    fig.legend(handles=[Patch(color=RUN_COLOUR, label='NWB on Sherlock'),
                        Patch(color=ISSUE_COLOUR, label='NWB with unallocated chunks'),
                        Patch(color=SPAN_COLOUR, label='No NWB data in session span')],
               loc='upper left', bbox_to_anchor=(0.02, 1 - 0.4 / height),
               ncol=3, frameon=False, fontsize=8)
    fig.savefig(out_path, dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('--label', default='v2_nwb_headers')
    p.add_argument('--workers', type=int, default=int(os.environ.get('SLURM_CPUS_PER_TASK', 8)))
    p.add_argument('--dpi', type=int, default=150)
    args = p.parse_args()

    paths = sorted(str(x) for x in config.V2_RAW_DIR.glob('sub-*/ses-*/ieeg/*.nwb'))
    logger.info('%d NWB files under %s', len(paths), config.V2_RAW_DIR)
    with ProcessPoolExecutor(args.workers) as ex:
        runs = pd.DataFrame(list(ex.map(read_header, paths, chunksize=20)))
    runs['duration_h'] = (runs.run_end - runs.run_start).dt.total_seconds() / 3600

    problems = runs[runs.error.notna() | (runs.missing_chunks > 0)]
    for r in problems.itertuples():
        logger.warning('load problem %s %s %s: %s', r.subject, r.session, r.run,
                       r.error if r.error else f'{int(r.missing_chunks)} unallocated chunks')
    logger.info('%d of %d files have a load problem', len(problems), len(runs))

    sessions = summarize_sessions(runs)
    out_dir = config.V2_DERIVATIVES_BASE / 'recording_coverage' / f'{args.label}_{datetime.now():%Y%m%d-%H%M%S}'
    out_dir.mkdir(parents=True)
    plot(runs, sessions, out_dir / 'fig_recording_coverage.png', args.dpi)

    subjects = sorted(runs.subject.unique())
    params = {**vars(args), 'raw_dir': str(config.V2_RAW_DIR)}
    io.write_table(runs, out_dir / 'nwb_runs.csv', script=SCRIPT, params=params,
                   subjects=subjects,
                   extra={'times': 'naive wall clock, UTC offset dropped'})
    io.write_table(problems, out_dir / 'load_errors.csv', script=SCRIPT, params=params,
                   subjects=subjects,
                   extra={'reading': 'error = open or header failure; missing_chunks > 0 = '
                                     'file opens but chunks are unallocated'})
    io.write_table(sessions, out_dir / 'sessions.csv', script=SCRIPT, params=params,
                   subjects=subjects,
                   extra={'span': 'first run start to last run end of readable files'})
    io.write_run_provenance(out_dir, script=SCRIPT, params=params, subjects=subjects,
                            extra={'n_files': len(runs), 'n_load_problems': len(problems)})
    io.log_analysis(f'V2 recording coverage from NWB headers: {len(sessions)} sessions, '
                    f'{len(runs)} files, {len(problems)} with load problems', out_dir)
    logger.info('done -> %s', out_dir)


if __name__ == '__main__':
    main()
