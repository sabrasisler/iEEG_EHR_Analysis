"""Laplacian band-RMS features for one subject-session's pain epochs.

    python -m ieeg_ehr.features.build_pain_epoch_bandpass --subjects 183

Reproduces Prasad et al. 2025's feature pipeline directly off the raw
time-domain signal: notch -> Laplacian re-reference -> six 8th-order zero-phase
Butterworth bandpass filters -> RMS over the 5-min window -> log10. See
`preprocessing/laplacian.py` for the method and the two deliberate departures.

WHY THIS EXISTS. Every other difference between our decoder and theirs is
reproducible from the stored PSD cache -- band edges, the RMS-vs-geometric-mean
averaging, even band-power weighting (DECISIONS 2026-09-09). The RE-REFERENCE is
not, because `run_pipeline_bipolar.py` never persisted the bipolar trace. So this
isolates the one variable the cache cannot answer.

WHY EPOCHS ONLY, not a continuous family. architecture.md's layer model says a
feature family is extracted continuously over the whole run and then sliced. That
would cost ~40x the I/O here for a question that only ever reads 5-min
pre-report windows, and raw NWB is chunked along TIME so a windowed read is
cheap (measured: 2.1 s for the worst subject's 5 min, versus a 1.5 GB whole-run
read). Deliberate departure, recorded in `config.bandpass_unit_dir`'s docstring.

OUTPUT IS DELIBERATELY VIEW-SHAPED. The table carries exactly the columns the
psd view layer emits, so `decoding/features.py` reads it through the same loader
and the decoder runs unchanged. The sidecar records `source` so the two can still
never be confused for one another.

READS V1 (`iEEG_EHR/iEEG_NWB`), DELIBERATELY -- and a V2 port is NOT a path swap.
V2 has 3 subjects, no `ehr/` folder at all (so no pain scores), and every
derivative this depends on -- the epoch definitions, the QC masks -- was built
from V1 (docs/dataset_v2.md §1). Beyond availability, the access pattern would
have to change: V1 chunks `(10000, n_channels)` so a short full-width window is
cheap, which is exactly what this reads; V2 chunks `(120 s, 1 channel)`, which
makes that the SLOW direction. Laplacian re-referencing is cross-channel and
inherently wants time-major reads, so on V2 this would need >=120 s
chunk-aligned blocks rather than a straight port (docs/dataset_v2.md §4,
SCRATCHPAD).
"""

import argparse
import logging
import sys
import time

import numpy as np
import pandas as pd

from ieeg_ehr import config, io
from ieeg_ehr.config import cohorts
from ieeg_ehr.io import nwb as nwb_io
from ieeg_ehr.preprocessing import laplacian
from ieeg_ehr.qc import mask_projection
from ieeg_ehr.views import axes, cache_reader

logger = logging.getLogger(__name__)

SOURCE = 'laplacian_bandpass_rms'


def _run_paths(subject, session):
    """{run_id: raw nwb path} from the file registry."""
    reg = pd.read_csv(config.FILE_REGISTRY_CSV)
    sel = reg[(reg.sub_id == f'sub-{subject}') & (reg.ses_id == f'ses-{session}')]
    return dict(zip(sel.run_id, sel.raw_file_path))


def _sample_mask(mask_df, run_id, contact_names, lap_channels, start_sec,
                 n_samples, sfreq):
    """(n_samples, n_laplacian) True-where-VALID, from the MONOPOLAR mask.

    A Laplacian channel is bad whenever ANY of its three contributing contacts is
    bad -- the same rule the bipolar path applies to anode|cathode, extended to a
    triple. The raw-voltage mask is per-contact and per-60 s bin, so this expands
    those bins to samples.

    Returns all-True when there is no mask, which the caller has to have asked
    for explicitly.
    """
    valid = np.ones((n_samples, len(lap_channels)), dtype=bool)
    if mask_df is None:
        return valid

    seconds = start_sec + np.arange(n_samples) / sfreq
    sub = mask_df[mask_df['run_id'] == run_id]
    if sub.empty:
        return valid
    excl_by_contact = {}
    for contact, grp in sub.groupby('channel'):
        starts = grp['bin_start'].to_numpy(float)
        flags = grp['excluded'].to_numpy(bool)
        order = np.argsort(starts)
        idx = np.searchsorted(starts[order], seconds, side='right') - 1
        idx = np.clip(idx, 0, len(starts) - 1)
        excl_by_contact[contact] = flags[order][idx]

    for j, (name, i_prev, i_ctr, i_next) in enumerate(lap_channels):
        bad = np.zeros(n_samples, dtype=bool)
        for pos in (i_prev, i_ctr, i_next):
            f = excl_by_contact.get(contact_names[pos])
            if f is not None:
                bad |= f
        valid[:, j] = ~bad
    return valid


def build_subject(subject, session='01', epoch_minutes=None, mask_label=None,
                  max_excluded_frac=None, bands=None):
    """Returns (long_table, stats) for one subject-session."""
    t0 = time.time()
    epoch_minutes = epoch_minutes or config.EPOCH_MINUTES_BEFORE
    max_excluded_frac = (config.EPOCH_MAX_EXCLUDED_FRAC if max_excluded_frac is None
                         else max_excluded_frac)
    bands = bands or config.PAPER_BANDS_6_HZ
    dur_sec = epoch_minutes * 60.0

    defs = cache_reader.load_defs(subject, session, epoch_minutes)
    defs = defs.sort_values('epoch_id').reset_index(drop=True)
    defs['pain_bin'] = axes.assign_pain_bins(defs, 'subject_relative')
    runs = _run_paths(subject, session)

    mask_df = None
    if mask_label:
        path = config.mask_csv(subject, session, mask_label)
        if not path.exists():
            raise FileNotFoundError(
                f'no raw-voltage mask at {path}. This pipeline needs the MONOPOLAR '
                'mask (per contact), not the bipolar one -- a Laplacian channel is '
                'built from three contacts and cannot be keyed by a pair name. '
                'Pass --mask-label "" to run deliberately unmasked.')
        mask_df = mask_projection.load_mask(path)

    rows, stats = [], {'n_epochs': 0, 'n_channels': 0, 'n_dropped_coverage': 0,
                       'rows_read': 0, 'missing_runs': set()}

    for _, ep in defs.iterrows():
        run_id = ep['run_id']
        raw_path = runs.get(run_id)
        if raw_path is None:
            stats['missing_runs'].add(run_id)
            continue

        data, contact_names, sfreq, elec_df, _ = nwb_io.load_window_with_electrodes(
            raw_path, float(ep['epoch_start_sec']), dur_sec)
        if data.shape[0] < 0.5 * dur_sec * sfreq:
            logger.warning('epoch %s: only %.0fs of data available, skipping',
                           ep['epoch_id'], data.shape[0] / sfreq)
            continue

        lap_channels = laplacian.build_laplacian_channels(elec_df)
        if not lap_channels:
            raise RuntimeError(f'sub-{subject}: no Laplacian channels could be '
                               'built; check the electrode naming')

        # Notch BEFORE re-referencing, as the paper specifies (note this is the
        # opposite of the usual order, where a common-mode line artefact would
        # largely cancel in the difference).
        notched = laplacian.notch_filter(data.astype(np.float64), sfreq)
        lap, names = laplacian.apply_laplacian(notched, lap_channels)

        valid = _sample_mask(mask_df, run_id, contact_names, lap_channels,
                             float(ep['epoch_start_sec']), lap.shape[0], sfreq)
        band_vals = laplacian.band_rms(lap, sfreq, bands=bands, sample_mask=valid,
                                       min_valid_frac=1.0 - max_excluded_frac)

        for band, vals in band_vals.items():
            finite = np.isfinite(vals)
            stats['n_dropped_coverage'] += int((~finite).sum())
            for ch, v in zip(np.asarray(names)[finite], vals[finite]):
                rows.append((f'sub-{subject}', f'ses-{session}',
                             int(ep['epoch_id']), int(ep['pain_event_id']),
                             float(ep['pain_score']), ep['pain_bin'],
                             ch, band, float(v), 1))
        stats['n_epochs'] += 1
        stats['n_channels'] = len(names)
        stats['rows_read'] += int(data.size)

    table = pd.DataFrame(rows, columns=['subject_id', 'session_id', 'epoch_id',
                                        'pain_event_id', 'pain_score', 'pain_bin',
                                        'region', 'freq_bin_index', 'value',
                                        'n_channels'])
    stats['missing_runs'] = sorted(stats['missing_runs'])
    stats['elapsed_sec'] = round(time.time() - t0, 1)
    stats['n_rows'] = len(table)
    return table, stats


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--subjects', nargs='+', default=None)
    ap.add_argument('--split', default='discovery')
    ap.add_argument('--session', default='01')
    ap.add_argument('--reref', default='laplacian',
                    choices=list(config.BANDPASS_REREF_SCHEMES))
    ap.add_argument('--mask-label', default=config.CANONICAL_MASK_LABEL,
                    help='RAW-VOLTAGE (monopolar) mask label; "" to run unmasked')
    ap.add_argument('--max-excluded-frac', type=float, default=None)
    ap.add_argument('--epoch-minutes', type=float, default=None)
    ap.add_argument('--log-level', default='INFO')
    args = ap.parse_args(argv)

    logging.basicConfig(level=args.log_level,
                        format='%(asctime)s %(levelname)s %(message)s')
    io.warn_if_dirty()

    subjects = args.subjects or cohorts.subjects_for_split(args.split)
    out_dir = config.bandpass_unit_dir(args.reref, args.epoch_minutes)
    out_dir.mkdir(parents=True, exist_ok=True)

    params = {'source': SOURCE, 'reref': args.reref,
              'bands_hz': {k: list(v) for k, v in config.PAPER_BANDS_6_HZ.items()},
              'notch_freqs_hz': list(config.PSD_LINE_NOISE_FREQS_HZ),
              'notch_order': laplacian.NOTCH_ORDER,
              'bandpass_order': laplacian.BANDPASS_ORDER,
              'epoch_minutes': args.epoch_minutes or config.EPOCH_MINUTES_BEFORE,
              'mask_level': 'raw_voltage' if args.mask_label else 'none',
              'mask_label': args.mask_label or None,
              'max_excluded_frac': (config.EPOCH_MAX_EXCLUDED_FRAC
                                    if args.max_excluded_frac is None
                                    else args.max_excluded_frac),
              'downsampled': False, 'statistic': 'log10_rms'}

    failed = []
    for subject in subjects:
        try:
            table, stats = build_subject(
                subject, args.session, args.epoch_minutes,
                args.mask_label or None, args.max_excluded_frac)
        except Exception as exc:                       # noqa: BLE001
            logger.error('sub-%s FAILED: %s: %s', subject, type(exc).__name__, exc)
            failed.append(subject)
            continue
        if table.empty:
            logger.warning('sub-%s produced no rows; skipping write', subject)
            failed.append(subject)
            continue
        path = config.bandpass_table_path(subject, args.session, args.reref,
                                          args.epoch_minutes)
        io.write_table(table, path, params=dict(params, **{'stats': stats}),
                       subjects=[f'sub-{subject}'],
                       script='ieeg_ehr/features/build_pain_epoch_bandpass.py')
        logger.info('sub-%s: %d epochs x %d channels x %d bands = %d rows in %.0fs '
                    '-> %s', subject, stats['n_epochs'], stats['n_channels'],
                    len(config.PAPER_BANDS_6_HZ), stats['n_rows'],
                    stats['elapsed_sec'], path.name)

    if not config.bandpass_manifest_path(args.reref, args.epoch_minutes).exists():
        io.write_manifest(out_dir, params=params,
                          script='ieeg_ehr/features/build_pain_epoch_bandpass.py')
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main())
