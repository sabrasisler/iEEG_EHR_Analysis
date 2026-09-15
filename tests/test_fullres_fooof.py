"""Tests for the epoch-mean full-res view and the FOOOF decomposition.

Same theme as the sibling files: each failure pinned here would be SILENT. A
geometric mean where an arithmetic one belongs, a log-of-log handed to FOOOF, or
a fit range that reaches Nyquist all produce a perfectly plausible exponent from
the wrong quantity.
"""

import numpy as np
import pandas as pd
import pytest

from ieeg_ehr import config
from ieeg_ehr.views import build_pain_epoch_fooof as fooofview
from ieeg_ehr.views import build_pain_epoch_fullres_mean as meanview


# ---------------------------------------------------------------------------
# The epoch mean must be LINEAR-then-log, in float64
# ---------------------------------------------------------------------------

def test_epoch_mean_is_arithmetic_in_linear_power_not_geometric():
    """`log10(mean(10**x))`, not `mean(x)`.

    Averaging stored log values directly is a GEOMETRIC mean in linear terms — a
    different quantity, always smaller (Jensen), and systematically so. A power
    spectrum's mean is the arithmetic one. This is the single most consequential
    line in the mean view.
    """
    # Two windows, one channel, one frequency: linear 1.0 and 100.0.
    block = np.log10(np.array([[[1.0]], [[100.0]]]))
    out = meanview.epoch_mean_log(block)
    assert out.shape == (1, 1)
    assert out[0, 0] == pytest.approx(np.log10(50.5))          # arithmetic
    assert out[0, 0] != pytest.approx(np.log10(10.0))          # geometric would be 10
    # And the geometric mean is strictly smaller, i.e. the bias has a sign.
    assert np.log10(10.0) < out[0, 0]


def test_epoch_mean_skips_masked_windows_without_losing_the_channel_slot():
    """Masking NaNs whole (window, channel) cells; the mean must be nan-aware.

    Every pair keeps its slot on axis 1 so `channels` stays synchronised with the
    array — dropping a channel instead would silently mislabel every one after it.
    """
    block = np.log10(np.array([
        [[1.0], [10.0]],
        [[np.nan], [10.0]],        # channel 0's second window masked
        [[3.0], [10.0]],
    ]))
    out = meanview.epoch_mean_log(block)
    assert out.shape == (2, 1)
    assert out[0, 0] == pytest.approx(np.log10(2.0))   # mean of 1 and 3, not of 3
    assert out[1, 0] == pytest.approx(1.0)


def test_epoch_mean_gives_nan_for_a_fully_masked_channel():
    """All-masked must be NaN, not 0 and not an exception — 0 would read as a real
    (and enormous, in log terms) measurement."""
    block = np.full((4, 2, 3), np.nan)
    block[:, 1, :] = 0.0                               # log10 power 0 => linear 1
    out = meanview.epoch_mean_log(block)
    assert np.isnan(out[0]).all()
    assert np.allclose(out[1], 0.0)


def test_epoch_mean_computes_in_float64():
    """P0.6: a float32 accumulator over ~300 windows holds only ~6 sig figs, and
    10**(-36.8) in float32 is one division from underflowing to exactly zero."""
    block = np.full((300, 1, 1), -36.8, dtype=np.float32)
    out = meanview.epoch_mean_log(block)
    assert out.dtype == np.float64
    # Compare against the FLOAT32 representation of -36.8, not the literal: the
    # stored value is float32, so -36.79999923706055 is the exact right answer and
    # a tighter tolerance would be testing float32's precision, not the mean's.
    assert out[0, 0] == pytest.approx(float(np.float32(-36.8)), abs=1e-12)
    # The point of the float64 path: this must not have underflowed to zero, which
    # is one baseline division away in float32 (P0.6, cache_params.py).
    assert out[0, 0] != 0.0 and np.isfinite(out[0, 0])
    assert np.power(10.0, out[0, 0]) > 0.0


def test_mean_params_pins_linear_then_log_and_refuses_to_record_normalization():
    from ieeg_ehr.views.view_config import ViewConfig
    vc = ViewConfig(normalization='none', mask_level='none')
    p = meanview.mean_params(vc, 499)
    assert p['epoch_agg'] == 'linear_then_log'
    assert p['normalization'] == 'none'
    assert p['n_freqs'] == 499
    assert p['accumulate_dtype'] == str(config.CACHE_ACCUMULATE_DTYPE)


# ---------------------------------------------------------------------------
# The FOOOF fit range: the Nyquist trap
# ---------------------------------------------------------------------------

def test_neither_arm_reaches_250_hz():
    """At 500 Hz sampling, 250 Hz IS Nyquist — the top of that range is anti-alias
    rolloff, a steep artifactual drop that would dominate an aperiodic fit.
    `SLOPE_FIT_HI_HZ = 250` inherits this; these arms must not."""
    for arm, spec in fooofview.ARMS.items():
        assert spec['fit_hi_hz'] <= 150.0, f'arm {arm} reaches {spec["fit_hi_hz"]} Hz'
        # And well clear of 0.4 x Nyquist for the 500 Hz subjects (=100 Hz is the
        # rolloff onset region; 150 is accepted for the broadband arm, 250 is not).
        assert spec['fit_hi_hz'] < 250.0


def test_fixed_arm_needs_no_notch_because_it_stops_below_60_hz():
    """Half the reason the primary arm is 1-45 Hz: no line-noise handling at all,
    so no notch parameter can affect the primary number."""
    assert fooofview.ARMS['fixed']['fit_hi_hz'] < 60.0 - config.PSD_NOTCH_HALF_WIDTH_HZ
    assert fooofview.ARMS['fixed']['notch'] is False
    assert fooofview.ARMS['knee']['notch'] is True


def test_select_freqs_fixed_arm_is_a_contiguous_unnotched_range():
    freqs = np.arange(1.0, 250.5, 0.5)
    idx, ff = fooofview.select_freqs(freqs, 'fixed')
    assert ff[0] == 1.0 and ff[-1] == 45.0
    np.testing.assert_allclose(np.diff(ff), 0.5)          # no gaps
    assert len(ff) == 89


def test_select_freqs_knee_arm_removes_exactly_the_harmonics_in_range():
    freqs = np.arange(1.0, 250.5, 0.5)
    idx, ff = fooofview.select_freqs(freqs, 'knee')
    assert ff[0] == 1.0 and ff[-1] == 150.0
    # 60 and 120 fall inside 1-150; 180 and 240 do not.
    for lf in (60.0, 120.0):
        assert not np.any(np.abs(ff - lf) <= config.PSD_NOTCH_HALF_WIDTH_HZ), \
            f'{lf} Hz survived the notch'
    # The notch is a GAP, not an interpolation: the diff jumps where it bites.
    assert np.any(np.diff(ff) > 0.5)
    # Cost is 4 Hz per harmonic, i.e. 9 bins each, not the log axis's 13.2 Hz.
    full = np.sum((freqs >= 1.0) & (freqs <= 150.0))
    assert full - len(ff) == 18


def test_select_freqs_refuses_a_range_that_leaves_too_little():
    freqs = np.arange(1.0, 250.5, 0.5)
    with pytest.raises(ValueError, match='only'):
        fooofview.select_freqs(np.array([1.0, 1.5, 2.0]), 'fixed')
    # A sane range is fine.
    fooofview.select_freqs(freqs, 'fixed')


def test_notch_half_width_is_overridable_and_widens_the_gap():
    freqs = np.arange(1.0, 250.5, 0.5)
    _, narrow = fooofview.select_freqs(freqs, 'knee', notch_half_width=1.0)
    _, wide = fooofview.select_freqs(freqs, 'knee', notch_half_width=5.0)
    assert len(wide) < len(narrow), 'a wider notch must remove more frequencies'


# ---------------------------------------------------------------------------
# THE trap: FOOOF takes LINEAR power, our cache stores log10
# ---------------------------------------------------------------------------

def test_fooof_recovers_the_exponent_only_from_linear_power():
    """Handing FOOOF our STORED log values fits log-of-log and yields a plausible
    number from nonsense. Pins the float64 exponentiation the view must do."""
    fooof = pytest.importorskip('fooof')
    from fooof import FOOOF

    f = np.arange(1.0, 45.5, 0.5)
    true_exp, true_off = 1.8, -20.0
    linear = 10.0 ** (true_off - true_exp * np.log10(f))
    stored_log = np.log10(linear).astype(config.CACHE_FLOAT_DTYPE)

    kw = dict(peak_width_limits=tuple(fooofview.DEFAULT_PEAK_WIDTH_LIMITS),
              max_n_peaks=fooofview.DEFAULT_MAX_N_PEAKS,
              min_peak_height=fooofview.DEFAULT_MIN_PEAK_HEIGHT,
              peak_threshold=fooofview.DEFAULT_PEAK_THRESHOLD,
              aperiodic_mode='fixed', verbose=False)

    # RIGHT: exponentiate back to linear in float64 first.
    right = np.power(10.0, stored_log.astype(config.CACHE_LINEAR_DOMAIN_DTYPE))
    fm = FOOOF(**kw); fm.fit(f, right)
    assert fm.aperiodic_params_[1] == pytest.approx(true_exp, abs=0.05)

    # WRONG: pass the stored log values as if they were power. FOOOF logs them
    # again, so this fits log-of-log. It must NOT accidentally agree.
    fm2 = FOOOF(**kw)
    with np.errstate(invalid='ignore', divide='ignore'):
        fm2.fit(f, np.abs(stored_log.astype(float)))
    assert abs(fm2.aperiodic_params_[1] - true_exp) > 0.5, (
        'log-of-log should be badly wrong, not subtly wrong — if this ever '
        'passes, the trap has become invisible')


def test_knee_frequency_is_in_hz_and_the_raw_knee_is_not():
    """`knee**(1/exponent)` is in Hz and averages meaningfully; the raw knee is in
    Hz^chi and does not. Stored alongside for exactly that reason."""
    knee, exponent = 80.0, 2.2
    knee_hz = knee ** (1.0 / exponent)
    assert knee_hz == pytest.approx(7.33, abs=0.01)
    # Two subjects with the same knee FREQUENCY but different exponents have very
    # different raw knees, so averaging the raw one mixes incomparable units.
    assert 80.0 ** (1 / 2.2) != pytest.approx(80.0 ** (1 / 1.2), abs=0.5)


# ---------------------------------------------------------------------------
# band_peaks: the fixed-width feature vector
# ---------------------------------------------------------------------------

def test_band_peaks_picks_the_highest_POWER_peak_in_each_band():
    """"Largest" is by PW, not by proximity to the band centre: a band's dominant
    oscillation is the one carrying the power."""
    peaks = pd.DataFrame({
        'epoch_id': [0, 0, 0], 'channel': ['A', 'A', 'A'],
        'peak_cf': [9.0, 11.0, 22.0],
        'peak_pw': [0.2, 0.9, 0.5],       # 11 Hz dominates alpha
        'peak_bw': [1.5, 2.0, 3.0],
    })
    out = fooofview.band_peaks(peaks, bands={'alpha': (8, 12), 'beta': (13, 30)})
    assert len(out) == 1
    r = out.iloc[0]
    assert r['alpha_cf'] == 11.0 and r['alpha_pw'] == 0.9
    assert r['beta_cf'] == 22.0


def test_band_peaks_gives_nan_for_a_band_with_no_peak():
    """NaN, not 0: absence of a detected oscillation is not zero power."""
    peaks = pd.DataFrame({'epoch_id': [0], 'channel': ['A'], 'peak_cf': [10.0],
                          'peak_pw': [0.5], 'peak_bw': [2.0]})
    out = fooofview.band_peaks(peaks, bands={'alpha': (8, 12), 'beta': (13, 30)})
    r = out.iloc[0]
    assert r['alpha_cf'] == 10.0
    assert np.isnan(r['beta_cf']) and np.isnan(r['beta_pw']) and np.isnan(r['beta_bw'])


def test_band_peaks_handles_an_empty_peaks_table():
    assert fooofview.band_peaks(pd.DataFrame()).empty


def test_fooof_params_hash_changes_with_every_swept_parameter():
    """Each is a sweep axis, so each must land in a different view directory —
    otherwise two arms overwrite each other's provenance."""
    from ieeg_ehr import io

    base = dict(arm='fixed', peak_width_limits=(1.0, 12.0), max_n_peaks=6,
                min_peak_height=0.10, peak_threshold=2.0, notch_half_width=2.0,
                mean_view_params={'x': 1})
    h0 = io.config_hash(fooofview.fooof_params(**base))
    for key, val in [('arm', 'knee'), ('max_n_peaks', 8),
                     ('min_peak_height', 0.05), ('peak_threshold', 2.5),
                     ('peak_width_limits', (2.0, 12.0))]:
        h = io.config_hash(fooofview.fooof_params(**{**base, key: val}))
        assert h != h0, f'{key} does not change the config hash'
