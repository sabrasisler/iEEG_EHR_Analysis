#!/usr/bin/env python3
"""
Orchestrates steps 1 (saturation) -> 2 (flatline) -> 3 (gross artifact) for a
subject subset, writing one per-window CSV per subject per artifact type.

The three steps are fully independent of each other's outputs (no exclusion
masking crosses between them) — this keeps each detector a pure function of
the raw trace + its own config, which matters for analyses (e.g. permutation
testing) that want to isolate/reanalyze one detector without re-running the
others.

Saturation's rail is resolved SESSION-wide (pooled across all runs, with
cross-channel agreement — see detect_saturation.py), but this does NOT mean
every run gets read twice. While a run's trace is loaded (pass 1), we also
classify it against its OWN local peak (classify_saturation_with_rail with
rail=that run's own abs_max) and cache the result — cheap, since the trace is
already in memory. Once the session-wide rail is resolved, most runs' cached
result can be reused directly:
  - if the resolved rail equals this run's own local peak (the common case,
    especially under cross-channel agreement, where the resolved value IS one
    of the channels' own peaks) -> reuse the cached result exactly.
  - if this run's own peak is BELOW the resolved rail -> trivially zero
    saturated windows, no data needed (a peak below the rail can't reach it).
  - only if this run's own peak is ABOVE the resolved rail (rare — this run
    individually saw something more extreme than the session's shared rail)
    does classification actually need the resolved rail instead of the local
    one, which requires a real second read — but only for that specific
    run/channel, not the whole session.

Flatline (no baseline at all) and gross-artifact (session mean/std, computed
unconditionally over every sub-bin) both need exactly one raw-data read per
run, same as before.

Results are streamed to disk per-run (config.append_table) rather than held
in memory for an entire session — some subjects have 100+ runs, and holding
every run's per-channel DataFrame in a Python list until the end scales
memory with run count, which is what caused OOMs on subjects with many runs.
A single bad/corrupted NWB file is logged and skipped rather than aborting
the rest of that subject's runs.

Usage:
  python -m qc_scripts.run_pipeline --n-subjects 3
  python -m qc_scripts.run_pipeline --subjects 217,222
  python -m qc_scripts.run_pipeline --subjects 217,222 --output-dir /path/to/alt/root
"""

import argparse
import warnings

import pandas as pd

# rail_value is legitimately all-NaN for channels with no resolved rail; this
# is expected (see detect_saturation.resolve_session_rails), not a bug.
warnings.filterwarnings(
    'ignore', message='.*DataFrame concatenation with empty or all-NA entries.*',
    category=FutureWarning)

import numpy as np

from qc_scripts import config, io_utils
from qc_scripts.detect_saturation import (
    RTOL, local_extreme_stats, session_rail_for_channel, resolve_session_rails,
    classify_saturation_with_rail, zero_result_like,
)
from qc_scripts.detect_flatline import classify_flatline
from qc_scripts.detect_square_wave import classify_square_wave
from qc_scripts.detect_gross_artifact import (
    new_accumulator, accumulate_and_cache_window_means, finalize_baseline,
)


def _rows_from_result(channel, run, artifact_type, result, array_cols=(), extra=None):
    """
    Build per-window rows storing the CONTINUOUS METRIC ONLY (no `excluded`) —
    thresholding is deferred to build_exclusions.py (metric/threshold split).
    `subject_id`/`session_id` are NOT stored here -- one output file already
    covers exactly one subject/session (see _output_path), so they're
    recoverable from the filename; storing them per-row would just repeat the
    same string across every row on disk for no benefit. `array_cols` names
    extra per-window arrays in `result` to carry (e.g. 'range' for
    square_wave, 'window_max_abs' for saturation); `extra` is a dict of
    scalars broadcast across all rows (e.g. rail_value, session_mean).
    """
    n = len(result['window_start'])
    row = {
        'run_id': [f'run-{run}'] * n,
        'channel': [channel] * n,
        'window_start_time': result['window_start'],
        'window_end_time': result['window_end'],
        'artifact_type': [artifact_type] * n,
        'metric_value': result['metric_value'],
    }
    for col in array_cols:
        row[col] = result[col]
    if extra:
        for k, v in extra.items():
            row[k] = [v] * n
    return pd.DataFrame(row)


def _output_path(subject, session, artifact_type):
    return config.PER_WINDOW_DIR / f'sub-{subject}_ses-{session}_{artifact_type}.csv'


def process_session(subject, session, runs):
    """
    runs: list of (session, run, nwb_path) tuples for this subject-session.
    Streams all three artifact types' rows to disk as they're computed.

    Single raw-data read per run (loaded_runs), in which:
      - flatline is classified and written immediately (fully independent).
      - gross-artifact's session accumulator + cached window means are
        updated (fully independent, unconditional over every sub-bin).
      - saturation is provisionally classified against THIS RUN'S OWN local
        peak (local_extreme_stats), and that result is cached — this is the
        classification that will end up correct for the common case where
        the resolved session rail agrees with a run's own peak.

    After every run has been read once: resolve the session-wide saturation
    rail (cross-channel agreement / per-channel fallback) and the
    gross-artifact baseline, both pure arithmetic on the tiny cached stats.

    Saturation rows are then finalized per (run, channel):
      - resolved rail == this run's own local peak -> reuse the cached result.
      - this run's own local peak < resolved rail -> trivially zero (a peak
        below the rail can't contain any saturated samples), no data needed.
      - this run's own local peak > resolved rail -> the rare disagreement
        case; a genuine second read, but only for this specific run.
    """
    gross_accumulators = {}     # channel -> accumulator dict
    gross_cached_means = {}     # (run, channel) -> cached window-mean dict
    sat_stats_by_channel = {}   # channel -> list of (run, abs_max, count_at_abs_max)
    sat_local_cache = {}         # (run, channel) -> (local_abs_max, classify_saturation_with_rail result)
    run_order = []
    loaded_runs = []             # (session_, run, nwb_path) that actually loaded OK

    # --- Single read per run ---
    for session_, run, nwb_path in runs:
        print(f"  sub-{subject} ses-{session_} run-{run}: loading...", flush=True)
        try:
            data_v, channel_names, sfreq = io_utils.load_all_channels(nwb_path)
        except Exception as e:
            print(f"  WARNING: failed to read {nwb_path} ({e!r}); skipping this run.",
                  flush=True)
            continue

        run_order.append((session_, run))
        loaded_runs.append((session_, run, nwb_path))
        flat_batch = []
        square_batch = []

        for ch_idx, channel in enumerate(channel_names):
            trace_v = data_v[:, ch_idx]

            flat = classify_flatline(trace_v, sfreq)
            flat_batch.append(_rows_from_result(channel, run, 'flatline', flat))

            sq = classify_square_wave(trace_v, sfreq)
            square_batch.append(_rows_from_result(channel, run, 'square_wave',
                                                   sq, array_cols=['range']))

            abs_max, count = local_extreme_stats(trace_v)
            sat_stats_by_channel.setdefault(channel, []).append((run, abs_max, count))
            local_rail = abs_max if abs_max > 0 else None
            sat_local_cache[(run, channel)] = (abs_max, classify_saturation_with_rail(
                trace_v, sfreq, local_rail))

            acc = gross_accumulators.setdefault(channel, new_accumulator())
            acc, cached = accumulate_and_cache_window_means(acc, trace_v, sfreq)
            gross_cached_means[(run, channel)] = cached

        config.append_table(pd.concat(flat_batch, ignore_index=True),
                             _output_path(subject, session_, 'flatline'))
        config.append_table(pd.concat(square_batch, ignore_index=True),
                             _output_path(subject, session_, 'square_wave'))
        del data_v, flat_batch, square_batch

    # --- Resolve baselines/rails: pure arithmetic on cached stats, no data access ---
    gross_baselines = {channel: finalize_baseline(acc) for channel, acc in gross_accumulators.items()}

    per_channel_session_stats = {
        channel: session_rail_for_channel([(a, c) for (_, a, c) in stats])
        for channel, stats in sat_stats_by_channel.items()
    }
    resolved_rails = resolve_session_rails(per_channel_session_stats)

    # --- Saturation: reuse cache where possible, collect the rare disagreements ---
    sat_result_by_run_channel = {}   # (run, channel) -> (result, rail, rail_source)
    reread_needed = {}               # run -> set of channels

    for channel, stats in sat_stats_by_channel.items():
        rail, rail_source = resolved_rails.get(channel, (None, 'none'))
        for run, _, _ in stats:
            local_abs_max, cached_result = sat_local_cache[(run, channel)]

            if rail is None:
                # No usable rail at all for this channel -> nothing can be saturated.
                result = zero_result_like(cached_result)
            elif np.isclose(local_abs_max, rail, rtol=RTOL):
                # Common case: this run's own peak IS the resolved rail.
                result = cached_result
            elif local_abs_max < rail:
                # This run never reached the resolved rail -> trivially zero, no data needed.
                result = zero_result_like(cached_result)
            else:
                # Rare: this run's own peak exceeds the resolved rail. Needs the actual
                # resolved threshold applied to raw samples -> defer to a targeted reread.
                reread_needed.setdefault(run, set()).add(channel)
                continue

            sat_result_by_run_channel[(run, channel)] = (result, rail, rail_source)

    if reread_needed:
        n_reread_channels = sum(len(v) for v in reread_needed.values())
        print(f"  Re-reading {len(reread_needed)} run(s) for {n_reread_channels} disagreeing "
              f"channel/run combo(s) (local peak exceeded the resolved session rail)...", flush=True)

    for session_, run, nwb_path in loaded_runs:
        if run not in reread_needed:
            continue
        data_v, channel_names, sfreq = io_utils.load_all_channels(nwb_path)
        for channel in reread_needed[run]:
            if channel not in channel_names:
                continue
            trace_v = data_v[:, channel_names.index(channel)]
            rail, rail_source = resolved_rails.get(channel, (None, 'none'))
            result = classify_saturation_with_rail(trace_v, sfreq, rail)
            sat_result_by_run_channel[(run, channel)] = (result, rail, rail_source)
        del data_v

    for session_, run in run_order:
        sat_batch = [
            _rows_from_result(channel, run, 'saturation', result,
                               array_cols=['window_max_abs'],
                               extra={'rail_value': rail, 'rail_source': rail_source})
            for (r, channel), (result, rail, rail_source) in sat_result_by_run_channel.items()
            if r == run
        ]
        if sat_batch:
            config.append_table(pd.concat(sat_batch, ignore_index=True),
                                 _output_path(subject, session_, 'saturation'))

    # --- Gross-artifact: store the RAW per-window variance + session baseline (no reload,
    #     no thresholding). build_exclusions.py computes z = (var - mean)/std and applies std_thresh. ---
    for session_, run in run_order:
        gross_batch = []
        for (r, channel), cached in gross_cached_means.items():
            if r != run:
                continue
            session_mean, session_std = gross_baselines[channel]
            gross = {
                'window_start': cached['window_start'],
                'window_end': cached['window_end'],
                'metric_value': cached['window_variance'],
            }
            gross_batch.append(_rows_from_result(
                channel, run, 'gross_artifact', gross,
                extra={'session_mean': session_mean, 'session_std': session_std}))
        if gross_batch:
            config.append_table(pd.concat(gross_batch, ignore_index=True),
                                 _output_path(subject, session_, 'gross_artifact'))


def _write_run_info(subject_sessions):
    """Per-(subject, session) record of how the metrics were produced:
    detection params + git provenance + run_timestamp. Written under
    metrics/run_info/sub-XXX_ses-YY.json (one file per subject/session, so
    parallel Slurm array tasks don't race on a shared file). Thresholds are
    NOT here — those live with build_exclusions. `subject_sessions`: dict
    {subject: [session, ...]}."""
    import json
    prov = config.warn_if_dirty()
    detection_params = {
        'sat_window_sec': config.SAT_WINDOW_SEC,
        'sat_agreement_threshold': config.SAT_AGREEMENT_THRESHOLD,
        'sat_min_repeats': config.SAT_MIN_REPEATS,
        'flatline_window_sec': config.FLATLINE_WINDOW_SEC,
        'square_window_sec': config.SQUARE_WINDOW_SEC,
        'square_eps_frac': config.SQUARE_EPS_FRAC,
        'gross_window_sec': config.GROSS_WINDOW_SEC,
    }
    ts = config.run_timestamp()
    rdir = config.OUTPUT_DIR / 'run_info'
    rdir.mkdir(parents=True, exist_ok=True)
    for subject, sessions in subject_sessions.items():
        for session in sessions:
            info = {
                'subject': f'sub-{subject}',
                'session': f'ses-{session}',
                'artifact_types': config.ARTIFACT_TYPES,
                'detection_params': detection_params,
                'run_timestamp': ts,
                'git': prov,
            }
            path = rdir / f'sub-{subject}_ses-{session}.json'
            with open(path, 'w') as f:
                json.dump(info, f, indent=2, default=str)
            print(f"  Wrote {path}", flush=True)


def run(subjects):
    config.ensure_output_dirs()

    subject_sessions = {}
    for subject in subjects:
        print(f"=== sub-{subject} ===", flush=True)

        session_runs = io_utils.get_session_runs(subject)
        sessions = sorted(set(s for s, _, _ in session_runs))
        subject_sessions[subject] = sessions

        for session in sessions:
            for artifact_type in config.ARTIFACT_TYPES:
                config.reset_table(_output_path(subject, session, artifact_type))

            runs = [(s, r, p) for s, r, p in session_runs if s == session]
            process_session(subject, session, runs)

            for artifact_type in config.ARTIFACT_TYPES:
                out_path = _output_path(subject, session, artifact_type)
                n_rows = sum(1 for _ in open(out_path)) - 1 if out_path.exists() else 0
                print(f"  Wrote {out_path} ({n_rows} rows)", flush=True)

    _write_run_info(subject_sessions)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--n-subjects', type=int, default=None,
                     help=f'Number of subjects to sample (default: {config.N_SUBJECTS})')
    ap.add_argument('--subjects', default=None,
                     help='Comma-separated explicit subject IDs, e.g. 217,222 (overrides --n-subjects)')
    ap.add_argument('--seed', type=int, default=None,
                     help=f'Random seed for subject sampling (default: {config.RANDOM_SEED})')
    ap.add_argument('--level-root', default=None,
                     help=f'QC level root; writes metrics to its metrics/per_window/ '
                          f'(default: {config.DEFAULT_LEVEL_ROOT})')
    ap.add_argument('--output-dir', default=None,
                     help='(alternative to --level-root) write per_window/ directly under this dir')
    args = ap.parse_args()

    if args.output_dir:
        config.set_output_dir(args.output_dir)
    else:
        level_root = args.level_root or config.DEFAULT_LEVEL_ROOT
        config.set_output_dir(config.metrics_root(level_root))

    subject_list = args.subjects.split(',') if args.subjects else None
    subjects = io_utils.sample_subjects(n_subjects=args.n_subjects, subject_list=subject_list,
                                         seed=args.seed)
    print(f"Running on {len(subjects)} subject(s): {subjects}")
    run(subjects)


if __name__ == '__main__':
    main()
