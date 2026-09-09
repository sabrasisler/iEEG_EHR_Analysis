"""Laplacian montage and the target paper's band-RMS extraction.

This is the SECOND re-referencing scheme in the project, and it exists to test
one specific hypothesis: that the gap between our decoding results and Prasad et
al. 2025's comes from the spatial filter rather than from the spectral estimator.
Everything else about the paper's feature pipeline is reproducible from the
stored PSD cache (see DECISIONS 2026-09-09); the re-reference is not, because
the bipolar trace was never persisted.

WHAT THE PAPER DOES, and what this reproduces
---------------------------------------------
  "a fourth order notch filter was first used to attenuate line noise (60, 120,
   and 180 Hz), followed by a laplacian re-referencing scheme to minimize
   far-field volume conduction. Delta (1-4 Hz) ... high-gamma (70-170 Hz)
   signals were obtained by using an 8th order zero-phase IIR Butterworth
   filter."
  "...obtaining the log spectral power (root mean square) of the filtered
   signals for each channel and frequency band 5 min prior to the pain report."

So: notch -> Laplacian -> 6 bandpass filters -> RMS over the 5-min window -> log.

TWO DEPARTURES, both deliberate
-------------------------------
1. NO DOWNSAMPLING TO 100 Hz. The paper downsamples the filtered signals to
   100 Hz, which cannot be right as literally written -- a 70-170 Hz band
   downsampled to 100 Hz (Nyquist 50) aliases completely. The only coherent
   reading is that they downsampled an amplitude ENVELOPE. Since we need only
   the RMS over the whole window, and RMS is invariant to that step, we simply
   skip it and compute at full rate. This removes the ambiguity rather than
   guessing at it.
2. NOTCH BEFORE LAPLACIAN, as written. Note this is the opposite order from what
   most pipelines do (re-reference first, since a common-mode line artifact
   largely cancels in the difference). Following the paper.

THE LAPLACIAN DEFINITION
------------------------
For a contact with immediate neighbours on the SAME shaft:

    lap_i = x_i - (x_{i-1} + x_{i+1}) / 2

Contacts at either end of a shaft have no such pair and are DROPPED, so a shaft
of k contacts yields k-2 channels (bipolar yields k-1). The channel is named for
its CENTRE contact, so a Laplacian channel name is a bare contact name ('LA2')
where a bipolar one is a pair ('LA1-LA2'). They are not interchangeable and must
never be joined on name.
"""

import logging

import numpy as np
from scipy import signal

from ieeg_ehr import config
from ieeg_ehr.preprocessing.bipolar_reref import parse_electrode_shaft

logger = logging.getLogger(__name__)

#: The paper's notch: 4th order, at each line-noise harmonic, +/- this half-width.
NOTCH_ORDER = 4
NOTCH_HALF_WIDTH_HZ = 2.0

#: The paper's bandpass: 8th order zero-phase IIR Butterworth.
BANDPASS_ORDER = 8

#: A band is skipped when its upper edge reaches Nyquist -- the filter would be
#: undefined and scipy would raise. At 500 Hz (Nyquist 250) every paper band is
#: fine; this guards the possibility rather than a known case.
NYQUIST_MARGIN = 0.95


def build_laplacian_channels(elec_df):
    """[(name, i_prev, i_centre, i_next)] over the rows of `elec_df`.

    Indices are POSITIONAL within elec_df (and therefore within the data array's
    channel axis), not electrode-table row labels.
    """
    locs = list(elec_df['location'].values)
    parsed = [parse_electrode_shaft(loc) for loc in locs]

    by_shaft = {}
    for pos, (shaft, num) in enumerate(parsed):
        if shaft is None:
            continue
        by_shaft.setdefault(shaft, []).append((num, pos, locs[pos]))

    out = []
    for shaft, contacts in sorted(by_shaft.items()):
        contacts.sort()
        by_num = {num: (pos, name) for num, pos, name in contacts}
        for num, pos, name in contacts:
            prev_, next_ = by_num.get(num - 1), by_num.get(num + 1)
            if prev_ is None or next_ is None:
                continue           # end contact, or a gap in the numbering
            out.append((name, prev_[0], pos, next_[0]))

    n_unparsed = sum(1 for s, _ in parsed if s is None)
    if n_unparsed:
        logger.warning('%d/%d channel names did not parse as <shaft><number>',
                       n_unparsed, len(locs))
    logger.info('laplacian montage: %d channels from %d contacts across %d shafts',
                len(out), len(locs), len(by_shaft))
    return out


def apply_laplacian(data, channels):
    """(n_samples, n_contacts) -> (n_samples, n_laplacian), plus the names."""
    if not channels:
        return np.empty((data.shape[0], 0), dtype=data.dtype), []
    prev_i = [c[1] for c in channels]
    ctr_i = [c[2] for c in channels]
    next_i = [c[3] for c in channels]
    lap = data[:, ctr_i] - 0.5 * (data[:, prev_i] + data[:, next_i])
    return lap, [c[0] for c in channels]


def notch_filter(data, sfreq, freqs=None, order=NOTCH_ORDER,
                 half_width=NOTCH_HALF_WIDTH_HZ):
    """Zero-phase bandstop at each line-noise harmonic below Nyquist."""
    freqs = config.PSD_LINE_NOISE_FREQS_HZ if freqs is None else freqs
    out = data
    for f0 in freqs:
        if f0 + half_width >= sfreq / 2 * NYQUIST_MARGIN:
            continue
        sos = signal.butter(order, [f0 - half_width, f0 + half_width],
                            btype='bandstop', fs=sfreq, output='sos')
        out = signal.sosfiltfilt(sos, out, axis=0)
    return out


#: A filtered segment shorter than this cannot support an 8th-order zero-phase
#: filter's padding, and its edge transients would dominate it. Mask bins are 60 s
#: so real segments are far longer; this guards the pathological case.
MIN_SEGMENT_SEC = 2.0


def _valid_runs(flags):
    """[(start, stop)] for each contiguous True run in a 1-D bool array."""
    padded = np.concatenate(([False], flags, [False]))
    edges = np.flatnonzero(padded[1:] != padded[:-1])
    return list(zip(edges[0::2], edges[1::2]))


def band_rms(data, sfreq, bands=None, sample_mask=None,
             min_valid_frac=0.5, order=BANDPASS_ORDER,
             min_segment_sec=MIN_SEGMENT_SEC):
    """{band: (n_channels,) log10 RMS} over the window's VALID samples.

    `sample_mask` is (n_samples,) or (n_samples, n_channels) True-where-VALID.

    ARTIFACT SEGMENTS ARE EXCLUDED BEFORE FILTERING, NOT AFTER. An earlier
    version filtered the whole window and masked only the averaging, on the
    reasoning that the filter wanted continuous input. That is wrong, and a test
    caught it: `sosfiltfilt` is ZERO-PHASE, so it runs forward AND backward, and
    a large artifact therefore rings BACKWARD in time into the clean samples that
    precede it. Measured on a step artifact in the second half of a window, the
    supposedly-masked estimate came out 3x too high (RMS 4.23 vs 1.41).

    So instead: channels are grouped by their validity pattern, and each
    contiguous valid RUN is filtered on its own. Grouping keeps this cheap --
    with 60 s mask bins over a 5-min epoch there are at most 5 bins and so a
    handful of distinct patterns, and the common case (nothing masked) collapses
    to exactly one filter call over the whole window, as before.

    Channels with less than `min_valid_frac` of the window surviving get NaN
    rather than an RMS over a handful of samples, matching the cache pipeline's
    `max_excluded_frac` rule.

    Returns log10(RMS). The paper reports "log spectral power (root mean
    square)"; log(RMS) = 0.5*log(mean power), and the factor of 2 is absorbed by
    the decoder's per-feature standardization.
    """
    bands = bands or config.PAPER_BANDS_6_HZ
    n_samples, n_ch = data.shape
    min_len = max(int(min_segment_sec * sfreq), 3 * (2 * order + 1))

    if sample_mask is None:
        valid = np.ones((n_samples, n_ch), dtype=bool)
    elif np.asarray(sample_mask).ndim == 1:
        valid = np.repeat(np.asarray(sample_mask)[:, None], n_ch, axis=1)
    else:
        valid = np.asarray(sample_mask)

    # Channels sharing a validity pattern are filtered together.
    patterns = {}
    for ch in range(n_ch):
        patterns.setdefault(valid[:, ch].tobytes(), []).append(ch)

    out = {}
    for name, (lo, hi) in bands.items():
        if hi >= sfreq / 2 * NYQUIST_MARGIN:
            logger.warning('band %s (%s-%s Hz) reaches Nyquist at sfreq=%s; NaN',
                           name, lo, hi, sfreq)
            out[name] = np.full(n_ch, np.nan)
            continue
        sos = signal.butter(order, [lo, hi], btype='bandpass', fs=sfreq,
                            output='sos')
        sumsq = np.zeros(n_ch, dtype=np.float64)
        count = np.zeros(n_ch, dtype=np.int64)
        for cols in patterns.values():
            flags = valid[:, cols[0]]
            for start, stop in _valid_runs(flags):
                if stop - start < min_len:
                    continue
                seg = signal.sosfiltfilt(sos, data[start:stop, cols], axis=0)
                sumsq[cols] += (seg.astype(np.float64) ** 2).sum(axis=0)
                count[cols] += stop - start
        with np.errstate(divide='ignore', invalid='ignore'):
            rms = np.sqrt(sumsq / np.where(count > 0, count, np.nan))
            value = np.log10(rms)
        value[count < min_valid_frac * n_samples] = np.nan
        out[name] = value
    return out
