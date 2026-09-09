"""Unit tests for the Laplacian band-RMS pipeline.

Same theme as the rest of this repo's tests: every failure guarded here would be
SILENT. A montage that quietly keeps end contacts, a mask that excludes nothing,
or an RMS taken over three surviving samples all produce a perfectly plausible
feature table.
"""

import numpy as np
import pandas as pd
import pytest

from ieeg_ehr import config
from ieeg_ehr.preprocessing import laplacian


def _elec(locations):
    return pd.DataFrame({'location': locations})


# ---------------------------------------------------------------------------
# The montage
# ---------------------------------------------------------------------------

def test_montage_drops_shaft_end_contacts():
    """A Laplacian needs a neighbour on BOTH sides, so a shaft of k contacts
    yields k-2 channels -- one fewer than bipolar's k-1 at each end."""
    ch = laplacian.build_laplacian_channels(_elec(['LA1', 'LA2', 'LA3', 'LA4']))
    assert [c[0] for c in ch] == ['LA2', 'LA3']


def test_montage_is_named_for_the_centre_contact():
    """A Laplacian channel name is a bare contact ('LA2'); a bipolar one is a
    pair ('LA1-LA2'). They must never be joined on name, so this is pinned."""
    ch = laplacian.build_laplacian_channels(_elec(['LA1', 'LA2', 'LA3']))
    assert ch[0][0] == 'LA2'
    assert '-' not in ch[0][0]


def test_montage_respects_gaps_in_contact_numbering():
    """A missing contact is not a neighbour. LA1,LA2,LA4,LA5: only LA4 and LA5
    lack a full pair... LA2 has no LA3, so nothing survives except where both
    neighbours are numerically adjacent."""
    ch = laplacian.build_laplacian_channels(_elec(['LA1', 'LA2', 'LA4', 'LA5']))
    assert [c[0] for c in ch] == []


def test_montage_keeps_shafts_separate():
    """Contacts on different shafts are not neighbours even when adjacent in the
    electrode table -- referencing across shafts would be meaningless."""
    ch = laplacian.build_laplacian_channels(
        _elec(['LA1', 'LA2', 'LA3', 'RB1', 'RB2', 'RB3']))
    assert [c[0] for c in ch] == ['LA2', 'RB2']


def test_apply_laplacian_is_centre_minus_mean_of_neighbours():
    data = np.array([[1.0, 10.0, 3.0]])          # LA1, LA2, LA3
    ch = laplacian.build_laplacian_channels(_elec(['LA1', 'LA2', 'LA3']))
    lap, names = laplacian.apply_laplacian(data, ch)
    assert names == ['LA2']
    assert lap[0, 0] == pytest.approx(10.0 - (1.0 + 3.0) / 2)


def test_unparseable_names_are_skipped_not_crashed():
    ch = laplacian.build_laplacian_channels(_elec(['EKG', 'DC1-x', 'LA1', 'LA2', 'LA3']))
    assert [c[0] for c in ch] == ['LA2']


# ---------------------------------------------------------------------------
# band_rms
# ---------------------------------------------------------------------------

def _sine(freq, amp, sfreq=1000.0, secs=20.0):
    t = np.arange(int(sfreq * secs)) / sfreq
    return (amp * np.sin(2 * np.pi * freq * t))[:, None]


def test_band_rms_recovers_a_known_amplitude_in_the_right_band_only():
    """A 40 Hz sine of amplitude 2 has RMS 2/sqrt(2) = 1.414 in gamma (25-70),
    and essentially nothing in the others. If the band edges or the filter were
    wrong, the energy would show up in the wrong column and still look like data.
    """
    sfreq = 1000.0
    out = laplacian.band_rms(_sine(40.0, 2.0, sfreq), sfreq)
    gamma = 10 ** out['gamma'][0]
    assert gamma == pytest.approx(2 / np.sqrt(2), rel=0.05)
    for other in ('delta', 'theta', 'alpha', 'beta'):
        assert 10 ** out[other][0] < 0.05 * gamma


def test_band_rms_separates_two_simultaneous_rhythms():
    sfreq = 1000.0
    x = _sine(10.0, 1.0, sfreq) + _sine(100.0, 3.0, sfreq)
    out = laplacian.band_rms(x, sfreq)
    assert 10 ** out['alpha'][0] == pytest.approx(1 / np.sqrt(2), rel=0.05)
    assert 10 ** out['high_gamma'][0] == pytest.approx(3 / np.sqrt(2), rel=0.05)


def test_masked_artifact_does_not_ring_backwards_into_clean_samples():
    """THE bug this pipeline is most exposed to.

    `sosfiltfilt` is zero-phase, so it filters forward AND backward. An artifact
    late in the window therefore contaminates the CLEAN samples that precede it,
    and masking only the averaging does not help -- the ringing is already in
    them. The first implementation did exactly that and came out 3x too high.
    Segments must be excluded BEFORE filtering, not after.
    """
    sfreq = 1000.0
    x = _sine(40.0, 2.0, sfreq)
    half = len(x) // 2
    dirty = x.copy()
    dirty[half:] += 500.0                              # an enormous artifact

    valid = np.ones(len(x), dtype=bool)
    valid[half:] = False
    masked = laplacian.band_rms(dirty, sfreq, sample_mask=valid, min_valid_frac=0.4)
    clean = laplacian.band_rms(x[:half], sfreq)
    assert 10 ** masked['gamma'][0] == pytest.approx(10 ** clean['gamma'][0], rel=0.15)


def test_channel_below_min_valid_frac_is_nan_not_a_thin_average():
    """An RMS over a handful of surviving samples is not the 5-minute statistic
    it claims to be, so it must be NaN rather than a number."""
    sfreq = 1000.0
    x = _sine(40.0, 2.0, sfreq)
    valid = np.zeros(len(x), dtype=bool)
    valid[:100] = True                                 # 0.5% of the window
    out = laplacian.band_rms(x, sfreq, sample_mask=valid, min_valid_frac=0.5)
    assert np.isnan(out['gamma'][0])


def test_a_band_reaching_nyquist_is_nan_not_an_exception():
    """At a low sampling rate high_gamma is undefined; scipy would raise, and a
    whole array task must not die because one subject was recorded slower."""
    sfreq = 200.0                                      # Nyquist 100 Hz
    out = laplacian.band_rms(_sine(10.0, 1.0, sfreq, secs=10), sfreq)
    assert np.isnan(out['high_gamma'][0])
    assert np.isfinite(out['alpha'][0])


def test_notch_attenuates_line_noise():
    sfreq = 1000.0
    x = _sine(60.0, 5.0, sfreq) + _sine(40.0, 1.0, sfreq)
    before = laplacian.band_rms(x, sfreq)['gamma'][0]
    after = laplacian.band_rms(laplacian.notch_filter(x, sfreq), sfreq)['gamma'][0]
    assert after < before - 0.3                        # >2x attenuation in log10


def test_bands_are_the_papers_edges():
    """The whole point of this pipeline is matching the paper, so the band set it
    defaults to is pinned here as well as in test_views."""
    assert laplacian.band_rms(_sine(10.0, 1.0), 1000.0).keys() == \
        config.PAPER_BANDS_6_HZ.keys()


if __name__ == '__main__':
    import sys
    sys.exit(pytest.main([__file__, '-v']))
