"""Unit tests for the EKG heart-rate extractor, on NeuroKit2's simulated ECG."""

import neurokit2 as nk
import numpy as np
import pytest

from ieeg_ehr.preprocessing import ekg_hr

FS = 500


def _ecg(heart_rate=70, seconds=120, seed=0):
    return nk.ecg_simulate(duration=seconds, sampling_rate=FS, heart_rate=heart_rate,
                           noise=0.01, random_state=seed) * 1e-4


def test_recovers_simulated_heart_rate():
    x = _ecg(heart_rate=70)
    peaks, quality = ekg_hr.detect_beats(x, FS)
    hr, q, rr = ekg_hr.hr_per_second(peaks, quality, FS, len(x))
    assert len(hr) == 120
    assert np.nanmedian(hr) == pytest.approx(70, abs=2)
    assert np.nanmedian(q) > 0.9
    assert np.nanmedian(rr) == pytest.approx(60 / 70, abs=0.03)


def test_inverted_lead_is_flipped():
    x = _ecg()
    assert not ekg_hr.fix_polarity(x, FS)[1]
    flipped, inverted = ekg_hr.fix_polarity(-x, FS)
    assert inverted
    np.testing.assert_array_equal(flipped, x)


def test_flat_lead_loses_to_ecg_lead():
    x = _ecg()
    common = np.random.default_rng(1).standard_normal(len(x)) * 1e-5
    data = np.column_stack([x + common, common])
    best = ekg_hr.best_lead(ekg_hr.candidate_leads(data, ['EKG1', 'EKG2']), FS)
    assert best['lead'] in ('EKG1-EKG2', 'EKG1')
    assert best['n_good_beats'] > 100


def test_hr_per_second_brackets_beats():
    fs = 100
    peaks = np.arange(0, 10 * fs + 1, fs)              # a beat every 1 s, 0..10 s
    quality = np.linspace(0, 1, len(peaks))
    hr, q, rr = ekg_hr.hr_per_second(peaks, quality, fs, 12 * fs)
    assert len(hr) == 12
    # second 0 sits before the second beat, 10 and 11 after the last
    assert np.isnan(hr[[0, 10, 11]]).all()
    np.testing.assert_allclose(hr[1:10], 60)
    np.testing.assert_allclose(rr[1:10], 1)
    # center 3.5 s lies between the beats at 3 s and 4 s
    assert q[3] == pytest.approx(quality[3])


def test_quality_blocks_merge_short_tail():
    block = ekg_hr.QUALITY_BLOCK_SEC * FS
    assert ekg_hr.quality_blocks(2 * block + 10, FS) == [(0, block), (block, 2 * block + 10)]
    assert ekg_hr.quality_blocks(2 * block + block // 2, FS)[-1] == (2 * block, 2 * block + block // 2)
    assert ekg_hr.quality_blocks(100, FS) == [(0, 100)]
