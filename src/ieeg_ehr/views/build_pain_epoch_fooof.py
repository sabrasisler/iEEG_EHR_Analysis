"""BG.3: periodic/aperiodic decomposition of each epoch's spectrum, via FOOOF.

    python -m ieeg_ehr.views.build_pain_epoch_fooof --subjects 019 --arm fixed

Separates what `views/aperiodic.py` cannot. That module fits one straight line
through log-power vs log-frequency over the whole stored spectrum, which
`config/psd_params.py` is explicit about: it is a BROADBAND TILT, with the
low-frequency knee, the alpha/beta peaks and any high-frequency amplifier floor
all folded into a single number. FOOOF fits an aperiodic component and the peaks
separately, so the exponent is not contaminated by the oscillations and the
oscillations are measured against their own local background.

WHY THIS ONLY BECAME POSSIBLE NOW. FOOOF refuses a `peak_width_limits` lower
bound below 2x the frequency resolution. On the 50-log-bin axis the effective
resolution is ~1.2 Hz at alpha and ~2.5 Hz at beta, so the floor would be ~5 Hz --
wider than an alpha peak, i.e. the model could not represent the thing it exists
to measure. The native 0.5 Hz grid makes a 1.0 Hz lower bound legal. Storing that
grid is what unblocked this (DECISIONS 2026-09-15).

INPUT IS THE EPOCH-MEAN VIEW, NOT THE PER-WINDOW CACHE, AND NOT RAW.
A single 2 s Hann periodogram is chi-squared with 2 dof, so its standard deviation
EQUALS its mean -- ~100% variance per bin. Fitting peaks on that is fitting noise.
`build_pain_epoch_fullres_mean.py` averages ~300 half-overlapping windows
(~150 independent), cutting that ~12x. And reading the mean view (~1.2 GB) rather
than the per-window cache (~226 GB) is what makes the parameter sweep below
affordable at all -- every arm would otherwise re-read the whole cache.

TWO TRAPS, EACH PINNED BY A TEST.

1. FOOOF WANTS **LINEAR** POWER. `FOOOF.fit(freqs, power_spectrum)` takes power
   and logs it internally. Our cache stores log10 power. Handing it stored values
   fits log-of-log and returns an entirely plausible exponent from nonsense. So
   the mean view is exponentiated in `CACHE_LINEAR_DOMAIN_DTYPE` first.

2. THE FIT RANGE MUST NOT REACH 250 Hz. At 500 Hz sampling, 250 Hz IS Nyquist, so
   the top of that range is anti-alias filter rolloff -- a steep artifactual drop
   that would dominate an aperiodic fit. `SLOPE_FIT_HI_HZ = 250` inherits this
   problem. Hence the two arms below, neither of which goes near it.

THE ARMS (`--arm`), and why there are two rather than one:

  fixed  1-45 Hz   PRIMARY. Below the first 60 Hz harmonic, so NO notch is needed
                   at all. ~1.65 decades, which does not require a knee. The most
                   literature-comparable number and the least artifactual.
  knee   1-150 Hz  The broadband exponent. 150 rather than 250 to clear the
                   500 Hz subjects' rolloff. NOT notched -- FOOOF requires
                   equidistant frequencies and rejects a gap, so the 60/120 Hz
                   harmonics are fit AS peaks and then dropped from the peaks
                   table (see ARMS). The aperiodic fit therefore sees the line
                   noise, which is the main reason `fixed` is primary.

They are not versions of each other, and the DIVERGENCE between them is
informative: it is the knee plus the high-frequency floor that the existing
polyfit slope folds into its single number.

`sfreq` is stored per row on purpose. The cohort mixes 500/1000/2000 Hz sampling,
and an exponent that depends on sampling rate would be a confound, not a finding.
The fit ranges above were chosen to reduce that; verifying it is a separate check
this table's columns make possible.

NORMALIZATION IS REFUSED. Same refusal as `build_pain_epoch_slope.py:19-27`: a
z-scored or baseline-subtracted spectrum's aperiodic component is not the
aperiodic component, and dividing each bin by its own baseline SD rescales the
y-axis per frequency, which is a different curve with a different exponent.
"""

import argparse
import logging
import sys
import time
import warnings

import numpy as np
import pandas as pd

from ieeg_ehr import config, io
from ieeg_ehr.views import build_pain_epoch_fullres_mean as meanview
from ieeg_ehr.views import fullres_reader, view_config

logger = logging.getLogger(__name__)

SCRIPT = 'ieeg_ehr/views/build_pain_epoch_fooof.py'
VIEW_LABEL_PREFIX = 'fooof'

#: The two arms. See the module docstring on why neither range reaches 250 Hz.
#:
#: NEITHER ARM NOTCHES THE INPUT, and that is forced rather than chosen: FOOOF
#: requires EQUIDISTANT frequencies (`fit.py::_prepare_data` raises
#: `DataError: The input frequency values are not evenly spaced`), so removing
#: 60 Hz leaves a gap the model rejects outright. Measured 2026-09-15 by trying it.
#:
#: The three ways out, and why the third wins:
#:   - interpolate across the notch -> fabricates data, and a fabricated value at
#:     60 Hz is a fabricated peak. Refused.
#:   - fit sub-ranges 1-58 and 62-118 separately -> cannot fit ONE knee, which is
#:     the entire point of this arm.
#:   - let FOOOF model the line artifact AS A PEAK, then discard that peak.
#:     A 60 Hz line artifact IS a narrow peak, and separating narrow peaks from
#:     the aperiodic component is precisely what the model is for. The exponent
#:     is then fit on a complete, evenly-spaced spectrum.
#:
#: `line_peak_slots` is the allowance added to `max_n_peaks` so absorbing the
#: harmonics does not crowd out real oscillations; `drop_line_peaks` removes them
#: from the peaks table afterwards, and the count is recorded.
#:
#: THE TRADE, stated plainly: the aperiodic fit SEES the line noise. FOOOF's
#: alternating fit is robust to a couple of narrow peaks, but this is why `fixed`
#: is the PRIMARY arm -- it stops at 45 Hz and never meets a harmonic at all.
ARMS = {
    'fixed': {'aperiodic_mode': 'fixed', 'fit_lo_hz': 1.0, 'fit_hi_hz': 45.0,
              'drop_line_peaks': False, 'line_peak_slots': 0},
    'knee':  {'aperiodic_mode': 'knee',  'fit_lo_hz': 1.0, 'fit_hi_hz': 150.0,
              'drop_line_peaks': True, 'line_peak_slots': 2},
}

#: FOOOF's own parameters. All overridable on the CLI and all hashed into the view
#: directory, which is the entire reason this is a separate job from the
#: extraction: changing `max_n_peaks` must never require re-reading raw NWB.
#:
#: peak_width_limits lower bound = 2 x the 0.5 Hz frequency resolution, which is
#: FOOOF's own floor and is only legal because of the native grid. max_n_peaks=6
#: covers delta/theta/alpha/beta/low-gamma plus one; unbounded invites fitting
#: notch residue as oscillations. min_peak_height is in log10 power units, so 0.10
#: is ~26% above the aperiodic fit.
DEFAULT_PEAK_WIDTH_LIMITS = (1.0, 12.0)
DEFAULT_MAX_N_PEAKS = 6
DEFAULT_MIN_PEAK_HEIGHT = 0.10
DEFAULT_PEAK_THRESHOLD = 2.0


def fooof_params(arm, peak_width_limits, max_n_peaks, min_peak_height,
                 peak_threshold, notch_half_width, mean_view_params):
    spec = dict(ARMS[arm])
    return {
        'metric': 'fooof_decomposition',
        'arm': arm,
        'aperiodic_mode': spec['aperiodic_mode'],
        'fit_lo_hz': spec['fit_lo_hz'],
        'fit_hi_hz': spec['fit_hi_hz'],
        # The input is never notched -- FOOOF requires equidistant frequencies.
        # These describe the POST-FIT peak removal instead.
        'drop_line_peaks': spec['drop_line_peaks'],
        'line_peak_half_width_hz': (notch_half_width if spec['drop_line_peaks']
                                    else None),
        'line_peak_slots': spec['line_peak_slots'],
        'peak_width_limits': list(peak_width_limits),
        'max_n_peaks': max_n_peaks,
        'min_peak_height': min_peak_height,
        'peak_threshold': peak_threshold,
        'normalization': 'none',
        'input_domain': 'linear_from_stored_log',
        'source_view': mean_view_params,
    }


def select_freqs(freqs, arm, notch_half_width=None):
    """(indices, freqs) to fit: a CONTIGUOUS, EVENLY SPACED slice of the arm's range.

    Deliberately does NOT notch. FOOOF requires equidistant frequencies and raises
    `DataError` on a gap, so the line harmonics are handled as peaks after the fit
    (see ARMS). `notch_half_width` is accepted for signature stability and used
    only by `drop_line_peaks`; it does not affect what is fit.

    The even-spacing assertion is the point of this function: a caller that
    reintroduced a gap would otherwise get FOOOF's error three frames deeper, with
    nothing pointing at the cause.
    """
    spec = ARMS[arm]
    keep = (freqs >= spec['fit_lo_hz']) & (freqs <= spec['fit_hi_hz'])
    idx = np.flatnonzero(keep)
    if len(idx) < 20:
        raise ValueError(f'arm {arm!r} leaves only {len(idx)} frequencies to fit')
    sel = freqs[idx]
    steps = np.diff(sel)
    if not np.allclose(steps, steps[0], rtol=0, atol=1e-9):
        raise ValueError(
            f'arm {arm!r} produced a non-uniform frequency grid (steps '
            f'{np.unique(np.round(steps, 6))}). FOOOF requires equidistant '
            'frequencies and raises DataError otherwise -- do not notch the input; '
            'line harmonics are removed from the PEAKS after fitting.')
    return idx, sel


def drop_line_peaks(peaks_df, half_width_hz=None, line_freqs=None):
    """Remove peaks centred on a line-noise harmonic. Returns (kept, n_dropped).

    The counterpart to not notching the input. A peak at 60.0 Hz in an iEEG
    spectrum is mains, not an oscillation, and leaving it in would put a spurious
    'gamma' peak in every channel of every epoch.

    Matches on the peak's CENTRE FREQUENCY rather than on overlap with its
    bandwidth: a genuine broad gamma peak can extend across 60 Hz without being
    line noise, and discarding it would be worse than keeping the artifact.
    """
    half = (config.PSD_NOTCH_HALF_WIDTH_HZ if half_width_hz is None
            else half_width_hz)
    lines = config.PSD_LINE_NOISE_FREQS_HZ if line_freqs is None else line_freqs
    if peaks_df.empty:
        return peaks_df, 0
    bad = np.zeros(len(peaks_df), dtype=bool)
    cf = peaks_df['peak_cf'].to_numpy()
    for lf in lines:
        bad |= np.abs(cf - lf) <= half
    return peaks_df.loc[~bad].reset_index(drop=True), int(bad.sum())


def _fit_group(fit_freqs, linear_spectra, arm, peak_width_limits, max_n_peaks,
               min_peak_height, peak_threshold):
    """Fit many spectra at once. Returns (aperiodic_rows, peak_rows).

    FOOOFGroup rather than a Python loop over FOOOF: same algorithm, and the
    per-model setup cost is paid once instead of ~580,000 times cohort-wide.
    """
    from fooof import FOOOFGroup

    # Extra slots so absorbing the line harmonics (which this arm must, since
    # FOOOF cannot take a notched grid) does not crowd out real oscillations.
    fg = FOOOFGroup(peak_width_limits=tuple(peak_width_limits),
                    max_n_peaks=max_n_peaks + ARMS[arm]['line_peak_slots'],
                    min_peak_height=min_peak_height,
                    peak_threshold=peak_threshold,
                    aperiodic_mode=ARMS[arm]['aperiodic_mode'],
                    verbose=False)
    with warnings.catch_warnings():
        # A flat or all-NaN channel-epoch fails to fit; FOOOF reports NaN params
        # for it, which is the honest answer and is counted by the caller.
        warnings.simplefilter('ignore')
        fg.fit(fit_freqs, linear_spectra)

    ap = fg.get_params('aperiodic_params')          # (n, 2) or (n, 3) with knee
    r2 = fg.get_params('r_squared')
    err = fg.get_params('error')
    peaks = fg.get_params('peak_params')            # (n_peaks_total, 4): CF PW BW idx
    return np.atleast_2d(ap), np.atleast_1d(r2), np.atleast_1d(err), np.atleast_2d(peaks)


def build_subject_session(subject, session, vc, arm, args, epoch_minutes=None,
                          overwrite=False):
    t0 = time.time()
    epoch_minutes = vc.resolved().epoch_minutes if epoch_minutes is None else epoch_minutes

    freqs = fullres_reader.freqs_hz(epoch_minutes)
    mv_params = meanview.mean_params(vc, len(freqs))
    mv_dir = config.fullres_epoch_views_dir(
        f'{meanview.VIEW_LABEL_PREFIX}-{vc.scheme_code}',
        io.config_hash(mv_params), epoch_minutes)
    mv_path = mv_dir / f'mean_sub-{subject}_ses-{session}.parquet'
    if not mv_path.exists():
        logger.warning('sub-%s ses-%s: no epoch-mean view at %s -- run '
                       'views.build_pain_epoch_fullres_mean first. Skipping.',
                       subject, session, mv_path.name)
        return None

    params = fooof_params(arm, args.peak_width_limits, args.max_n_peaks,
                          args.min_peak_height, args.peak_threshold,
                          args.notch_half_width, mv_params)
    out_dir = config.fullres_epoch_views_dir(
        f'{VIEW_LABEL_PREFIX}-{arm}-{vc.scheme_code}', io.config_hash(params),
        epoch_minutes)
    ap_path = out_dir / f'aperiodic_sub-{subject}_ses-{session}.parquet'
    pk_path = out_dir / f'peaks_sub-{subject}_ses-{session}.parquet'
    if ap_path.exists() and not overwrite:
        logger.info('sub-%s ses-%s: exists, skipping', subject, session)
        return None

    mv = io.read_table(mv_path, on_stale='warn')
    freq_cols = fullres_reader.freq_columns(epoch_minutes)
    idx, fit_freqs = select_freqs(freqs, arm, args.notch_half_width)
    fit_cols = [freq_cols[i] for i in idx]

    stored_log = mv[fit_cols].to_numpy()
    # TRAP 1: FOOOF logs the spectrum itself, so it must receive LINEAR power.
    linear = np.power(10.0, stored_log.astype(config.CACHE_LINEAR_DOMAIN_DTYPE))

    usable = np.isfinite(linear).all(axis=1) & (linear > 0).all(axis=1)
    n_unusable = int((~usable).sum())
    if not usable.any():
        logger.warning('sub-%s ses-%s: no usable spectra', subject, session)
        return None

    ap, r2, err, peaks = _fit_group(
        fit_freqs, linear[usable], arm, args.peak_width_limits, args.max_n_peaks,
        args.min_peak_height, args.peak_threshold)

    meta = mv.loc[usable, ['epoch_id', 'channel', 'pain_score', 'pain_bin',
                           'n_windows_used', 'mask_excluded_frac']].reset_index(drop=True)
    out = meta.copy()
    out['subject_id'] = f'sub-{subject}'
    out['session_id'] = f'ses-{session}'
    out['arm'] = arm
    out['aperiodic_offset'] = ap[:, 0]
    if ARMS[arm]['aperiodic_mode'] == 'knee':
        out['aperiodic_knee'] = ap[:, 1]
        out['aperiodic_exponent'] = ap[:, 2]
        # The raw knee is in units of Hz^chi and is not meaningfully averageable.
        # knee**(1/exponent) is in Hz and is.
        with np.errstate(invalid='ignore', divide='ignore'):
            out['knee_frequency_hz'] = np.where(
                (ap[:, 1] > 0) & (ap[:, 2] > 0),
                np.power(np.abs(ap[:, 1]), 1.0 / np.where(ap[:, 2] != 0, ap[:, 2], np.nan)),
                np.nan)
    else:
        out['aperiodic_knee'] = np.nan
        out['aperiodic_exponent'] = ap[:, 1]
        out['knee_frequency_hz'] = np.nan
    # Stored, NEVER thresholded here -- metric/threshold split, same as the slope
    # view's r2. A cutoff belongs downstream where it costs nothing to vary.
    out['r_squared'] = r2
    out['error'] = err
    out['n_freqs_fit'] = len(fit_freqs)

    # Peaks: long, because the count varies per fit. FOOOF's 4th column is the
    # row index into the group, which is how peaks map back to channel-epochs.
    n_line_peaks = 0
    if peaks.size and peaks.shape[1] >= 4:
        pk = pd.DataFrame({'row': peaks[:, 3].astype(int),
                           'peak_cf': peaks[:, 0],
                           'peak_pw': peaks[:, 1],
                           'peak_bw': peaks[:, 2]})
        pk = pk.join(meta[['epoch_id', 'channel']], on='row').drop(columns='row')
        if ARMS[arm]['drop_line_peaks']:
            # This arm's range crosses 60/120 Hz and FOOOF cannot take a notched
            # input, so the harmonics were fit AS peaks. Remove them here or every
            # channel-epoch gets a spurious 'gamma' oscillation.
            pk, n_line_peaks = drop_line_peaks(pk, args.notch_half_width)
        counts = pk.groupby(['epoch_id', 'channel']).size() if len(pk) else {}
        out['n_peaks'] = [int(counts.get((e, c), 0)) if len(pk) else 0
                          for e, c in zip(out['epoch_id'], out['channel'])]
    else:
        pk = pd.DataFrame(columns=['peak_cf', 'peak_pw', 'peak_bw',
                                   'epoch_id', 'channel'])
        out['n_peaks'] = 0

    parents = [str(mv_path),
               io.manifest_ref(config.fullres_epoch_unit_dir(epoch_minutes))]
    extra = {'n_channel_epochs': int(usable.sum()),
             'n_unusable_spectra': n_unusable,
             'n_peaks_total': int(len(pk)),
             'n_line_peaks_dropped': int(n_line_peaks),
             'median_r_squared': float(np.nanmedian(r2)),
             'fit_freq_lo_hz': float(fit_freqs[0]),
             'fit_freq_hi_hz': float(fit_freqs[-1]),
             'n_freqs_fit': int(len(fit_freqs)),
             'elapsed_sec': round(time.time() - t0, 1)}
    io.write_table(out, ap_path, kind='view', script=SCRIPT, params=params,
                   parents=parents, subjects=[f'sub-{subject}'], extra=extra)
    io.write_table(pk, pk_path, kind='view', script=SCRIPT, params=params,
                   parents=parents, subjects=[f'sub-{subject}'], extra=extra)
    io.write_view_sidecar(ap_path, view_config=params,
                          cache_manifest=config.fullres_epoch_unit_dir(epoch_minutes))

    logger.info('sub-%s ses-%s arm=%s: %d channel-epochs, %d peaks, '
                'median r2 %.3f, %d freqs (%.1f-%.1f Hz), %.0fs%s',
                subject, session, arm, int(usable.sum()), len(pk),
                np.nanmedian(r2), len(fit_freqs), fit_freqs[0], fit_freqs[-1],
                time.time() - t0,
                (f'  [{n_unusable} unusable]' if n_unusable else '')
                + (f'  [{n_line_peaks} line peaks dropped]' if n_line_peaks else ''))
    return ap_path, extra


def band_peaks(peaks_df, bands=None):
    """Largest peak per canonical band -> a fixed-width feature vector.

    A recomputed FUNCTION, not a saved artifact: it is a cheap derivation of the
    long peaks table, and which band set to use is a choice the caller makes
    (architecture.md PART 3). Returns one row per (epoch_id, channel) with
    <band>_cf / _pw / _bw columns, NaN where a band has no peak.

    "Largest" is by power (PW), not by proximity to the band centre: a band's
    dominant oscillation is the one carrying the power.
    """
    bands = bands or config.CANONICAL_BANDS_HZ
    if peaks_df.empty:
        return pd.DataFrame()
    out = []
    for (epoch_id, channel), g in peaks_df.groupby(['epoch_id', 'channel']):
        row = {'epoch_id': epoch_id, 'channel': channel}
        for band, (lo, hi) in bands.items():
            sel = g[(g['peak_cf'] >= lo) & (g['peak_cf'] < hi)]
            if len(sel):
                best = sel.loc[sel['peak_pw'].idxmax()]
                row[f'{band}_cf'] = best['peak_cf']
                row[f'{band}_pw'] = best['peak_pw']
                row[f'{band}_bw'] = best['peak_bw']
            else:
                row[f'{band}_cf'] = row[f'{band}_pw'] = row[f'{band}_bw'] = np.nan
        out.append(row)
    return pd.DataFrame(out)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--subjects', nargs='+', required=True)
    ap.add_argument('--session', default='01')
    ap.add_argument('--arm', choices=sorted(ARMS), default='fixed',
                    help="'fixed' 1-45 Hz (primary, no notch needed) or 'knee' "
                         "1-150 Hz notched. NOT 250 Hz: at 500 Hz sampling that "
                         "is Nyquist and the top of the range is filter rolloff.")
    ap.add_argument('--peak-width-limits', nargs=2, type=float,
                    default=list(DEFAULT_PEAK_WIDTH_LIMITS),
                    help='FOOOF refuses a lower bound below 2x the 0.5 Hz '
                         'resolution; 1.0 is that floor.')
    ap.add_argument('--max-n-peaks', type=int, default=DEFAULT_MAX_N_PEAKS)
    ap.add_argument('--min-peak-height', type=float, default=DEFAULT_MIN_PEAK_HEIGHT)
    ap.add_argument('--peak-threshold', type=float, default=DEFAULT_PEAK_THRESHOLD)
    ap.add_argument('--notch-half-width', type=float, default=None,
                    help=f'default: config {config.PSD_NOTCH_HALF_WIDTH_HZ} Hz '
                         '(knee arm only)')
    ap.add_argument('--overwrite', action='store_true')
    view_config.add_view_arguments(ap)
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)s %(message)s')
    io.warn_if_dirty()

    vc = view_config.from_args(args)
    if vc.normalization != 'none':
        ap.error(
            f'--normalization {vc.normalization!r} refused. A z-scored or '
            'baseline-subtracted spectrum\'s aperiodic component is not the '
            'aperiodic component: dividing each bin by its own baseline SD '
            'rescales the y-axis per frequency, giving a different curve with a '
            'different exponent. Same refusal as build_pain_epoch_slope.py.')

    n_ok, out_dir = 0, None
    for s in args.subjects:
        subject = s.replace('sub-', '')
        for session in ([args.session] if args.session != 'all'
                        else meanview._sessions_for(subject)):
            r = build_subject_session(subject, session, vc, args.arm, args,
                                      overwrite=args.overwrite)
            if r is not None:
                n_ok += 1
                out_dir = r[0].parent
    if n_ok and out_dir is not None:
        io.log_analysis(
            f'FOOOF {args.arm} arm, {n_ok} subject-session(s)', out_dir)
    return 0 if n_ok else 1


if __name__ == '__main__':
    sys.exit(main())
