"""Unit tests for the full-resolution epoch PSD cache and its reader.

Same theme as `test_views.py`: every failure pinned here would be SILENT in
production. A one-window-short read, a transposed reshape, a frequency axis off
by one bin, or a log-of-log handed to specparam all produce a plausible-looking
spectrum with wrong numbers.

Deliberately uses a SMALL, NON-DEFAULT frequency axis (`N_FREQ = 9`, f_max well
below 250 Hz) so that nothing silently depends on 499 the way the 50-bin path
came to depend on `config.PSD_N_LOG_BINS`.
"""

import numpy as np
import pandas as pd
import pytest
from scipy import signal

from ieeg_ehr import config
from ieeg_ehr.features import build_pain_epoch_fullres_psd as fullres
from ieeg_ehr.views import axes
from ieeg_ehr.views.cache_reader import CacheLayoutError

N_FREQ = 9          # 1.0 .. 5.0 Hz at 0.5 Hz -- deliberately not 499
F_MIN, F_MAX = 1.0, 5.0


# ---------------------------------------------------------------------------
# Window geometry: the hop must be integer-consistent
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('sfreq', [500.0, 1000.0, 2000.0])
def test_window_geometry_is_consistent_at_every_sampling_rate(sfreq):
    """nperseg - noverlap must equal the hop the stored PSD `rate` implies.

    The stored NWB rate comes from the float `window_sec * (1 - overlap_frac)`
    while scipy steps the integer `nperseg - noverlap`. They agree for every
    sfreq in this dataset, but a half-sample-per-window drift would misalign
    every epoch in a run and nothing else would report it.
    """
    nperseg, noverlap, hop = fullres.window_geometry(sfreq)
    assert nperseg == int(round(config.PSD_WINDOW_SEC * sfreq))
    assert nperseg - noverlap == hop
    assert hop == int(round(sfreq * config.PSD_WINDOW_SEC
                            * (1.0 - config.PSD_OVERLAP_FRAC)))


def test_window_geometry_refuses_an_inconsistent_hop():
    """An odd nperseg makes the float and integer hops disagree -- must raise."""
    # 2.0 s at 501 Hz -> nperseg 1002, noverlap 501, scipy hop 501; the float
    # expression gives 501 too. Force the mismatch with a 3-way overlap instead.
    with pytest.raises(ValueError, match='hop mismatch'):
        fullres.window_geometry(1000.0, window_sec=2.0, overlap_frac=1.0 / 3.0)


# ---------------------------------------------------------------------------
# THE OFF-BY-ONE. This is the test that matters most.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('sfreq', [500.0, 1000.0, 2000.0])
def test_read_length_must_be_n_windows_plus_one(sfreq):
    """(n_windows + 1) * hop samples yields n_windows windows; n_windows * hop
    yields n_windows - 1.

    PSD row i covers raw samples [i*hop, i*hop + nperseg), so the last window of
    an epoch needs one extra hop's worth of signal beyond the epoch's nominal
    length. `build_pain_epoch_bandpass.py` reads exactly epoch_minutes*60, which
    is correct for its RMS purpose and one window short for this one -- the
    resulting cache would silently hold 299 windows where the 50-bin cache holds
    300, and every cross-cache comparison would be off by a window.
    """
    nperseg, noverlap, hop = fullres.window_geometry(sfreq)
    n_windows = 12
    rng = np.random.default_rng(0)
    x = rng.standard_normal((n_windows + 1) * hop).astype(np.float32)

    _, _, full = signal.spectrogram(x, fs=sfreq, nperseg=nperseg, noverlap=noverlap,
                                    window=config.PSD_WINDOW_FN, detrend='constant',
                                    scaling='density', mode='psd')
    _, _, short = signal.spectrogram(x[:n_windows * hop], fs=sfreq, nperseg=nperseg,
                                     noverlap=noverlap, window=config.PSD_WINDOW_FN,
                                     detrend='constant', scaling='density', mode='psd')
    assert full.shape[1] == n_windows
    assert short.shape[1] == n_windows - 1


# ---------------------------------------------------------------------------
# The frequency axis
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('sfreq', [500.0, 1000.0, 2000.0])
def test_frequency_grid_is_identical_across_sampling_rates(sfreq):
    """0.5 Hz at 500, 1000 AND 2000 Hz -- the reason one fixed axis works cohort-wide.

    df = sfreq/nperseg = 1/PSD_WINDOW_SEC, and nperseg is resolved per-run from
    that run's own sfreq, so the rate cancels. If this ever stopped holding, the
    cohort would need resampling and a fixed frequency axis would be a fiction.
    """
    nperseg, noverlap, hop = fullres.window_geometry(sfreq)
    x = np.zeros(4 * hop, dtype=np.float32)
    freqs, _, _ = signal.spectrogram(x, fs=sfreq, nperseg=nperseg, noverlap=noverlap,
                                     window=config.PSD_WINDOW_FN, detrend='constant',
                                     scaling='density', mode='psd')
    sl, grid = fullres.freq_slice(freqs, F_MIN, F_MAX)
    assert len(grid) == N_FREQ
    np.testing.assert_allclose(grid, np.arange(F_MIN, F_MAX + 0.25, 0.5))
    np.testing.assert_allclose(freqs[sl], grid, atol=1e-9)

    # And the production defaults land on 499 bins, 1.0 .. 250.0 inclusive.
    _, default_grid = fullres.freq_slice(freqs)
    assert len(default_grid) == config.PSD_FULLRES_N_FREQS == 499
    assert default_grid[0] == 1.0 and default_grid[-1] == 250.0


def test_freq_slice_includes_the_top_frequency():
    """f_max is INCLUSIVE, and is selected by index rather than by `freqs <= f_max`.

    rfftfreq values are not exactly representable, so a float comparison can drop
    the top bin depending on the last bit -- which would silently shorten the
    axis by one and shift every downstream column name.
    """
    freqs = np.arange(0, 501) * 0.5
    sl, grid = fullres.freq_slice(freqs, 1.0, 250.0)
    assert grid[-1] == 250.0
    assert freqs[sl][-1] == 250.0
    assert len(grid) == 499


def test_freq_slice_refuses_a_grid_that_is_not_df_spaced():
    with pytest.raises(ValueError, match='not 0.5 Hz-spaced'):
        fullres.freq_slice(np.arange(0, 100) * 0.3, 1.0, 5.0)


def test_freq_slice_refuses_above_nyquist():
    """A 500 Hz-sampled run cannot supply 300 Hz; that must raise, not truncate."""
    freqs = np.arange(0, 501) * 0.5        # Nyquist 250 Hz
    with pytest.raises(ValueError, match='Nyquist'):
        fullres.freq_slice(freqs, 1.0, 300.0)


def test_freq_column_names_are_positional_and_sort_correctly():
    """Zero-padded so lexical order == frequency order; nothing parses the name."""
    names = fullres.freq_column_names(499)
    assert names[:2] == ['f000', 'f001']
    assert names[-1] == 'f498'
    assert names == sorted(names)
    assert len(set(names)) == 499


# ---------------------------------------------------------------------------
# A synthetic sine must land in the right column
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('f_sine', [10.0, 60.0, 137.5])
def test_synthetic_sine_lands_in_its_own_frequency_column(f_sine):
    """The whole point of the cache: column j holds freqs_hz[j].

    An off-by-one in the slice, or a transposed reshape, would put the peak in a
    neighbouring column -- invisible in any band average, fatal for a peak fit.
    """
    sfreq = 1000.0
    nperseg, noverlap, hop = fullres.window_geometry(sfreq)
    n_windows = 8
    t = np.arange((n_windows + 1) * hop) / sfreq
    x = np.sin(2 * np.pi * f_sine * t).astype(np.float32)

    freqs, sxx = fullres.epoch_spectrogram(x[:, None], sfreq, nperseg, noverlap)
    sl, grid = fullres.freq_slice(freqs)
    block = sxx[sl]                                    # (n_freq, n_win, n_pairs)

    assert block.shape == (len(grid), n_windows, 1)
    peak = int(np.argmax(block[:, :, 0].mean(axis=1)))
    assert grid[peak] == pytest.approx(f_sine), (
        f'peak landed at {grid[peak]} Hz, expected {f_sine} Hz')


def test_epoch_spectrogram_matches_the_production_per_channel_call():
    """Bit-exact against `bipolar_reref._welch_one_channel`'s own scipy call.

    This is what makes the audit's check A meaningful: if `epoch_spectrogram`
    batched the FFT over an axis, or upcast to float64, it would be slightly MORE
    accurate and the gate against the existing 50-bin cache would become
    approximate -- unable to distinguish a 1-ulp difference from a one-window
    misalignment.
    """
    sfreq = 1000.0
    nperseg, noverlap, _ = fullres.window_geometry(sfreq)
    rng = np.random.default_rng(7)
    x = rng.standard_normal((5 * nperseg, 3)).astype(np.float32)

    _, mine = fullres.epoch_spectrogram(x, sfreq, nperseg, noverlap)
    for ch in range(x.shape[1]):
        _, _, theirs = signal.spectrogram(
            x[:, ch], fs=sfreq, nperseg=nperseg, noverlap=noverlap,
            window='hann', scaling='density', mode='psd')
        assert np.array_equal(mine[:, :, ch], theirs), f'channel {ch} not bit-exact'


def test_spectrogram_stays_float32_like_production():
    """float32 in -> float32 out. Pinned because the audit's exactness depends on
    it, and because an accidental float64 would double the cache's memory."""
    sfreq = 1000.0
    nperseg, noverlap, _ = fullres.window_geometry(sfreq)
    x = np.zeros((3 * nperseg, 2), dtype=np.float32)
    _, sxx = fullres.epoch_spectrogram(x, sfreq, nperseg, noverlap)
    assert sxx.dtype == np.float32


# ---------------------------------------------------------------------------
# freq_table must feed the EXISTING band-aggregation code unchanged
# ---------------------------------------------------------------------------

def _freq_table(grid, df=0.5):
    """What `fullres_reader.freq_table()` builds, without needing a manifest."""
    return pd.DataFrame({'freq_bin_index': np.arange(len(grid)),
                         'bin_low_hz': grid - df / 2.0,
                         'bin_high_hz': grid + df / 2.0})


def test_freq_table_bins_have_nonzero_width():
    """Zero-width bins would silently zero every weight under band_weighting='width'."""
    tbl = _freq_table(np.arange(1.0, 6.0, 0.5))
    widths = tbl['bin_high_hz'] - tbl['bin_low_hz']
    assert (widths > 0).all()
    np.testing.assert_allclose(widths, 0.5)


def test_aggregate_bands_consumes_the_fullres_table_unchanged():
    """`axes.aggregate_bands` works on the native grid with NO code changes.

    It selects by geometric bin centre from (bin_low_hz, bin_high_hz), which is
    exactly what freq_table supplies. This is the main reuse claim of the whole
    change, so it is pinned rather than asserted in prose.
    """
    grid = np.arange(1.0, 30.5, 0.5)
    tbl = _freq_table(grid)
    # One channel, value == the bin's own frequency, so a band's mean is the mean
    # frequency of its members and can be checked by hand.
    values = grid[None, :].astype(float)
    bands = {'delta': (1, 4), 'theta': (4, 8), 'alpha': (8, 12)}

    out, names = axes.aggregate_bands(values, tbl, bands=bands, is_difference=True,
                                      domain='log', weighting='uniform')
    assert names == ['delta', 'theta', 'alpha']
    assert out.shape == (1, 3)

    centers = np.sqrt(tbl['bin_low_hz'].to_numpy() * tbl['bin_high_hz'].to_numpy())
    for j, (lo, hi) in enumerate(bands.values()):
        members = grid[(centers >= lo) & (centers < hi)]
        assert members.size > 0
        assert out[0, j] == pytest.approx(members.mean())


def test_canonical_and_paper_bands_both_resolve_on_the_fullres_grid():
    """Every band in both published sets must find at least one member bin.

    A band that matched nothing would be dropped with a warning and silently
    absent from the output table.
    """
    grid = np.arange(1.0, 250.5, 0.5)
    tbl = _freq_table(grid)
    values = np.ones((1, len(grid)))
    for bands in (config.CANONICAL_BANDS_HZ, config.PAPER_BANDS_6_HZ):
        out, names = axes.aggregate_bands(values, tbl, bands=bands,
                                          is_difference=True, domain='log')
        assert names == list(bands), f'a band found no bins: {set(bands) - set(names)}'
        assert np.isfinite(out).all()


def test_bands_for_treats_fullres_as_no_aggregation():
    assert axes.bands_for('fullres') is None
    assert axes.bands_for('log_bins_50') is None
    assert axes.bands_for('canonical_bands') is config.CANONICAL_BANDS_HZ
    with pytest.raises(ValueError, match='unknown freq axis'):
        axes.bands_for('native')


# ---------------------------------------------------------------------------
# The notch: narrower than the log-bin axis could ever be
# ---------------------------------------------------------------------------

def _notch(grid, half, lines=config.PSD_LINE_NOISE_FREQS_HZ):
    flagged = np.zeros(len(grid), dtype=bool)
    for lf in lines:
        flagged |= np.abs(grid - lf) <= half
    return np.flatnonzero(flagged)


def test_notch_costs_four_hz_per_harmonic_not_thirteen():
    """The number this whole change exists for.

    On the 50-log-bin axis, excluding 60 Hz means dropping bins 36+37, which span
    53.27-66.44 Hz = 13.2 Hz. On the native grid it means 8 bins spanning 4.0 Hz.
    """
    grid = np.arange(1.0, 250.5, 0.5)
    idx = _notch(grid, config.PSD_NOTCH_HALF_WIDTH_HZ)
    around_60 = grid[idx][(grid[idx] >= 55) & (grid[idx] <= 65)]
    assert around_60.min() == 58.0 and around_60.max() == 62.0
    assert len(around_60) == 9                      # 58.0 .. 62.0 inclusive
    assert around_60.max() - around_60.min() == 4.0

    # And the log-bin cost it replaces, computed from the real edge function.
    from ieeg_ehr.preprocessing import bipolar_reref
    edges = bipolar_reref.log_bin_edges(50, 1.0, 250.0)
    flagged = bipolar_reref.line_noise_mask(edges, (60.0,),
                                            config.PSD_LINE_NOISE_GUARD_HZ)
    spanned = edges[1:][flagged].max() - edges[:-1][flagged].min()
    assert spanned == pytest.approx(13.17, abs=0.05)
    assert spanned > 3 * 4.0, 'the log axis should cost >3x the native notch'


def test_notch_keeps_the_vast_majority_of_the_axis():
    grid = np.arange(1.0, 250.5, 0.5)
    idx = _notch(grid, config.PSD_NOTCH_HALF_WIDTH_HZ)
    assert len(idx) == 4 * 9                         # 4 harmonics x 9 bins
    assert len(grid) - len(idx) == 463               # vs 44 of 50 on the log axis
    assert (len(grid) - len(idx)) / len(grid) > 0.92


# ---------------------------------------------------------------------------
# Masking is frequency-length agnostic
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('n_freq', [5, 50, 499])
def test_apply_mask_broadcasts_over_any_frequency_axis_length(n_freq):
    """`cache_reader.apply_mask` is reused as-is; it must not care about n_freq.

    It NaNs whole (window, channel) cells, which broadcasts over the last axis
    regardless of length -- the reason the entire QC layer transfers to this cache
    without a rewrite.
    """
    from ieeg_ehr.views import cache_reader

    n_win, n_pairs = 10, 4
    block = np.ones((n_win, n_pairs, n_freq))
    excluded = np.zeros((n_win, n_pairs), dtype=bool)
    excluded[0:3, 1] = True          # 30% of channel 1
    excluded[:, 2] = True            # 100% of channel 2

    out, kept, frac = cache_reader.apply_mask(block, excluded, max_excluded_frac=0.5)
    assert out.shape == (n_win, n_pairs, n_freq)
    # channel 1 keeps its slot; only its masked windows are NaN, across ALL freqs
    assert np.isnan(out[0:3, 1, :]).all()
    assert np.isfinite(out[3:, 1, :]).all()
    # channel 2 exceeds the coverage floor and is dropped entirely
    assert not kept[2] and np.isnan(out[:, 2, :]).all()
    assert kept[0] and kept[1] and kept[3]
    np.testing.assert_allclose(frac, [0.0, 0.3, 1.0, 0.0])


# ---------------------------------------------------------------------------
# Cache layout: the wide ravel order must be proven, not assumed
# ---------------------------------------------------------------------------

def _write_fullres_cache(path, epochs, n_freq=N_FREQ, row_group_size=None):
    """A cache with the real schema and the real ravel order.

    Values encode their own coordinates as epoch*1e6 + win*1e3 + pair*10 + freq,
    so a misread is identifiable rather than merely wrong.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    schema = fullres.cache_schema(n_freq)
    names = fullres.freq_column_names(n_freq)
    kwargs = {'row_group_size': row_group_size} if row_group_size else {}
    writer = pq.ParquetWriter(path, schema)
    try:
        for epoch_id, (n_win, n_pairs) in enumerate(epochs):
            w = np.repeat(np.arange(n_win), n_pairs)
            p = np.tile(np.arange(n_pairs), n_win)
            arrays = [
                pa.array(np.full(n_win * n_pairs, epoch_id, dtype=np.int32)),
                pa.array(w.astype(np.int16)),
                pa.array(np.array([f'C{i}' for i in p])).dictionary_encode(),
            ] + [pa.array((epoch_id * 1e6 + w * 1e3 + p * 10 + j
                           ).astype(np.float32)) for j in range(n_freq)]
            writer.write_table(pa.Table.from_arrays(arrays, schema=schema), **kwargs)
    finally:
        writer.close()
    return pd.DataFrame([{'epoch_id': i, 'n_windows': nw, 'n_channels': npr,
                          'run_id': 'run-A', 'epoch_start_sec': 0.0, 'hop_sec': 1.0}
                         for i, (nw, npr) in enumerate(epochs)])


def test_read_epoch_decodes_the_wide_ravel_order(tmp_path):
    """Row r is (win = r // n_pairs, pair = r % n_pairs).

    Getting this backwards transposes windows and channels into each other's
    slots and still averages to something plausible -- the same class of silent
    failure `cache_reader.verify_layout` exists to catch for the long layout.
    """
    import pyarrow.parquet as pq
    from ieeg_ehr.views import fullres_reader

    path = tmp_path / 'c.parquet'
    defs = _write_fullres_cache(path, [(4, 3), (6, 3)])
    pf = pq.ParquetFile(path)

    mapping = fullres_reader.verify_layout(pf, defs, N_FREQ)
    assert set(mapping) == {0, 1}

    names = fullres.freq_column_names(N_FREQ)
    block = fullres_reader.read_epoch(pf, defs.iloc[1], mapping[1], columns=names)
    assert block.shape == (6, 3, N_FREQ)
    assert block.dtype == config.CACHE_ACCUMULATE_DTYPE
    for w in range(6):
        for p in range(3):
            for j in range(N_FREQ):
                assert block[w, p, j] == pytest.approx(1e6 + w * 1e3 + p * 10 + j)


def test_read_epoch_can_slice_columns_off_disk(tmp_path):
    """The payoff of the wide layout: read 2 frequencies instead of 499."""
    import pyarrow.parquet as pq
    from ieeg_ehr.views import fullres_reader

    path = tmp_path / 'c.parquet'
    defs = _write_fullres_cache(path, [(4, 2)])
    pf = pq.ParquetFile(path)
    mapping = fullres_reader.verify_layout(pf, defs, N_FREQ)

    want = ['f003', 'f007']
    block = fullres_reader.read_epoch(pf, defs.iloc[0], mapping[0], columns=want)
    assert block.shape == (4, 2, 2)
    assert block[2, 1, 0] == pytest.approx(2 * 1e3 + 1 * 10 + 3)
    assert block[2, 1, 1] == pytest.approx(2 * 1e3 + 1 * 10 + 7)


def test_verify_layout_handles_an_epoch_split_across_row_groups(tmp_path):
    """epoch:row-group is not guaranteed 1:1, so the walk must not assume it."""
    import pyarrow.parquet as pq
    from ieeg_ehr.views import fullres_reader

    path = tmp_path / 'c.parquet'
    defs = _write_fullres_cache(path, [(8, 4), (8, 4)], row_group_size=7)
    pf = pq.ParquetFile(path)
    mapping = fullres_reader.verify_layout(pf, defs, N_FREQ)
    assert pf.num_row_groups > 2, 'row groups did not actually split'
    assert sum(len(v) for v in mapping.values()) == pf.num_row_groups

    names = fullres.freq_column_names(N_FREQ)
    block = fullres_reader.read_epoch(pf, defs.iloc[0], mapping[0], columns=names)
    assert block[7, 3, 0] == pytest.approx(7 * 1e3 + 3 * 10)


def test_verify_layout_refuses_a_row_count_mismatch(tmp_path):
    """defs and cache out of sync must raise, not silently read the wrong slice."""
    import pyarrow.parquet as pq
    from ieeg_ehr.views import fullres_reader

    path = tmp_path / 'c.parquet'
    defs = _write_fullres_cache(path, [(4, 3)])
    defs.loc[0, 'n_windows'] = 5                       # a lie
    pf = pq.ParquetFile(path)
    with pytest.raises(CacheLayoutError, match='defs implies'):
        fullres_reader.verify_layout(pf, defs, N_FREQ)


def test_verify_layout_refuses_a_frequency_count_mismatch(tmp_path):
    """If the manifest says 499 and the cache has 9 columns, that must raise."""
    import pyarrow.parquet as pq
    from ieeg_ehr.views import fullres_reader

    path = tmp_path / 'c.parquet'
    defs = _write_fullres_cache(path, [(4, 3)])
    pf = pq.ParquetFile(path)
    with pytest.raises(CacheLayoutError, match='columns, expected'):
        fullres_reader.verify_layout(pf, defs, N_FREQ + 1)


def test_chosen_parquet_encoding_is_bitwise_lossless(tmp_path):
    """BYTE_STREAM_SPLIT + zstd must round-trip float32 EXACTLY, NaN and -inf included.

    P0.6 validated float32 round-trip through DEFAULT Parquet; the cache now uses
    a different encoding, so it gets the same check rather than the assumption.
    BYTE_STREAM_SPLIT only transposes the bytes of each float, so this should be
    exact by construction -- which is precisely why a silent failure here would be
    a library bug nothing else would catch.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    rng = np.random.default_rng(3)
    n_freq, n_rows = 12, 500
    names = fullres.freq_column_names(n_freq)
    # Include the pathological values the real cache contains: -inf from an
    # exactly-zero bin, and the ~-36.8 extreme cache_params.py cites.
    vals = (rng.standard_normal((n_freq, n_rows)) * 10 - 20).astype(np.float32)
    vals[0, 0] = -np.inf
    vals[1, 0] = np.float32(-36.8)
    vals[2, 0] = np.nan

    schema = pa.schema([(n, pa.float32()) for n in names])
    path = tmp_path / 'enc.parquet'
    w = pq.ParquetWriter(path, schema, **fullres.parquet_options(names))
    w.write_table(pa.Table.from_arrays([pa.array(vals[j]) for j in range(n_freq)],
                                       schema=schema))
    w.close()

    back = pq.ParquetFile(path).read()
    got = np.stack([back.column(n).to_numpy() for n in names])
    assert got.dtype == np.float32
    # Bitwise, so NaN compares equal to NaN and -0.0 does not equal 0.0.
    assert np.array_equal(got.view(np.uint32), vals.view(np.uint32))


def test_analytic_freqs_matches_the_real_spectrogram_grid():
    """The slice is sized from `rfftfreq` before any spectrogram runs, so that
    only the kept frequencies are allocated. If the two ever disagreed, the cache
    would silently store the wrong frequencies."""
    for sfreq in (500.0, 1000.0, 2000.0):
        nperseg, noverlap, hop = fullres.window_geometry(sfreq)
        x = np.zeros(4 * hop, dtype=np.float32)
        real, _, _ = signal.spectrogram(
            x, fs=sfreq, nperseg=nperseg, noverlap=noverlap,
            window=config.PSD_WINDOW_FN, detrend='constant',
            scaling='density', mode='psd')
        np.testing.assert_allclose(fullres.analytic_freqs(sfreq, nperseg), real,
                                   rtol=0, atol=1e-9)


def test_epoch_spectrogram_slice_matches_the_full_grid():
    """Slicing per channel (to save memory) must give bit-identical values to
    slicing the full grid afterwards."""
    sfreq = 1000.0
    nperseg, noverlap, _ = fullres.window_geometry(sfreq)
    rng = np.random.default_rng(11)
    x = rng.standard_normal((5 * nperseg, 3)).astype(np.float32)

    freqs, full = fullres.epoch_spectrogram(x, sfreq, nperseg, noverlap)
    sl, _ = fullres.freq_slice(freqs)
    _, sliced = fullres.epoch_spectrogram(x, sfreq, nperseg, noverlap, freq_sl=sl)
    assert sliced.shape[0] == 499
    assert np.array_equal(sliced, full[sl])


def test_cache_schema_has_no_int8_bin_ceiling():
    """psd_epochs' `bin` is int8 -- a hard 127-bin ceiling. The wide layout has no
    such column at all, which is half the reason it is wide."""
    import pyarrow as pa

    schema = fullres.cache_schema(499)
    assert 'bin' not in schema.names
    assert schema.names[:3] == fullres.INDEX_COLUMNS
    assert len(schema.names) == 499 + 3
    assert all(schema.field(n).type == pa.float32()
               for n in fullres.freq_column_names(499))


# ---------------------------------------------------------------------------
# specparam's input convention -- pinned before the FOOOF view is written
# ---------------------------------------------------------------------------

def test_specparam_needs_linear_power_not_our_stored_log():
    """The cache stores log10 power; specparam logs the spectrum ITSELF.

    Handing it stored values fits log-of-log and yields a plausible exponent from
    nonsense. This pins the round trip the FOOOF view must do -- exponentiate in
    CACHE_LINEAR_DOMAIN_DTYPE first -- without needing specparam installed.
    """
    grid = np.arange(1.0, 100.5, 0.5)
    true_exponent = 2.0
    linear = grid ** (-true_exponent)
    stored_log = np.log10(linear).astype(config.CACHE_FLOAT_DTYPE)

    # The right thing: back to linear in float64, then fit log10 vs log10.
    recovered = np.power(10.0, stored_log.astype(config.CACHE_LINEAR_DOMAIN_DTYPE))
    slope_ok = np.polyfit(np.log10(grid), np.log10(recovered), 1)[0]
    assert slope_ok == pytest.approx(-true_exponent, abs=1e-5)

    # The wrong thing: passing stored log values as if they were linear power.
    with np.errstate(invalid='ignore'):
        slope_bad = np.polyfit(np.log10(grid),
                               np.log10(np.abs(stored_log.astype(float)) + 1e-12), 1)[0]
    assert abs(slope_bad - (-true_exponent)) > 1.0, (
        'log-of-log should be badly wrong, not subtly wrong')
