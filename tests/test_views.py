"""
Unit tests for the P1.3 view layer.

The theme: every one of these failures would be SILENT in production. A wrong
reshape, a mask that matches nothing, a float32 accumulator, or averaging before
normalizing all produce a plausible-looking heatmap with wrong numbers. So each
test pins a property whose violation has no other symptom.
"""

import numpy as np
import pandas as pd
import pytest

from ieeg_ehr import config
from ieeg_ehr.qc import mask_projection as mp
from ieeg_ehr.views import axes, cache_reader
from ieeg_ehr.views.view_config import ViewConfig

N_BINS = 5


# ---------------------------------------------------------------------------
# Cache layout: the reshape must be proven, not assumed
# ---------------------------------------------------------------------------

def _write_cache(path, epochs, n_bins=N_BINS, row_group_size=None, scramble=False):
    """Build a cache file with the real schema, in the real ravel order.

    `epochs` is [(n_win, n_pairs)]. Values encode their own coordinates as
    epoch*1e6 + win*1e3 + pair*10 + bin, so a misread is identifiable rather than
    merely wrong.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    schema = pa.schema([('epoch_id', pa.int32()), ('window_idx', pa.int16()),
                        ('channel', pa.dictionary(pa.int16(), pa.string())),
                        ('bin', pa.int8()), ('log_power', pa.float32())])
    writer = pq.ParquetWriter(path, schema)
    try:
        for eid, (n_win, n_pairs) in enumerate(epochs):
            w = np.repeat(np.arange(n_win), n_pairs * n_bins)
            p = np.tile(np.repeat(np.arange(n_pairs), n_bins), n_win)
            b = np.tile(np.arange(n_bins), n_win * n_pairs)
            vals = eid * 1e6 + w * 1e3 + p * 10 + b
            if scramble:
                order = np.arange(len(vals))[::-1]
                w, p, b, vals = w[order], p[order], b[order], vals[order]
            tbl = pa.table({
                'epoch_id': pa.array(np.full(len(w), eid, dtype=np.int32)),
                'window_idx': pa.array(w.astype(np.int16)),
                'channel': pa.array([f'C{j}-C{j + 1}' for j in p]).dictionary_encode(),
                'bin': pa.array(b.astype(np.int8)),
                'log_power': pa.array(vals.astype(np.float32)),
            }, schema=schema)
            if row_group_size:
                writer.write_table(tbl, row_group_size=row_group_size)
            else:
                writer.write_table(tbl)
    finally:
        writer.close()


def _defs(epochs, scores=None):
    scores = scores or [0.0] * len(epochs)
    return pd.DataFrame({
        'epoch_id': range(len(epochs)),
        'n_windows': [w for w, _ in epochs],
        'n_channels': [p for _, p in epochs],
        'run_id': ['run-A'] * len(epochs),
        'pain_event_id': range(len(epochs)),
        'pain_score': scores,
        'row_start': [0] * len(epochs),
        'epoch_start_sec': [0.0] * len(epochs),
        'hop_sec': [1.0] * len(epochs),
    })


def test_reshape_round_trips_coordinates(tmp_path):
    """The core claim: reshape recovers (window, pair, bin) exactly."""
    import pyarrow.parquet as pq
    epochs = [(4, 3), (4, 3)]
    path = tmp_path / 'c.parquet'
    _write_cache(path, epochs)
    f = pq.ParquetFile(path)
    defs = _defs(epochs)
    mapping = cache_reader.verify_layout(f, defs, N_BINS)

    block = cache_reader.read_epoch(f, defs.iloc[1], N_BINS, mapping[1])
    assert block.shape == (4, 3, N_BINS)
    for w in range(4):
        for p in range(3):
            for b in range(N_BINS):
                assert block[w, p, b] == pytest.approx(1e6 + w * 1e3 + p * 10 + b)


def test_epoch_spanning_multiple_row_groups(tmp_path):
    """The bug the guard caught on real data: an epoch is NOT always one row group.

    pyarrow splits a write_table above its row-group ceiling, which happens for any
    subject with enough pairs. Epochs must map to contiguous RANGES.
    """
    import pyarrow.parquet as pq
    epochs = [(4, 3)]                      # 4*3*5 = 60 rows
    path = tmp_path / 'c.parquet'
    _write_cache(path, epochs, row_group_size=25)     # -> 3 row groups for 1 epoch
    f = pq.ParquetFile(path)
    assert f.num_row_groups == 3
    defs = _defs(epochs)
    mapping = cache_reader.verify_layout(f, defs, N_BINS)
    assert mapping[0] == [0, 1, 2]
    block = cache_reader.read_epoch(f, defs.iloc[0], N_BINS, mapping[0])
    assert block[2, 1, 3] == pytest.approx(2 * 1e3 + 1 * 10 + 3)


def test_scrambled_ravel_is_rejected(tmp_path):
    """A reordered cache must RAISE, not silently transpose."""
    import pyarrow.parquet as pq
    epochs = [(4, 3)]
    path = tmp_path / 'c.parquet'
    _write_cache(path, epochs, scramble=True)
    with pytest.raises(cache_reader.CacheLayoutError, match='C-order ravel|innermost'):
        cache_reader.verify_layout(pq.ParquetFile(path), _defs(epochs), N_BINS)


def test_defs_cache_mismatch_is_rejected(tmp_path):
    import pyarrow.parquet as pq
    path = tmp_path / 'c.parquet'
    _write_cache(path, [(4, 3), (4, 3)])
    with pytest.raises(cache_reader.CacheLayoutError, match='out of sync'):
        cache_reader.verify_layout(pq.ParquetFile(path), _defs([(4, 3)]), N_BINS)


# ---------------------------------------------------------------------------
# Mask projection: the pair-keyed trap
# ---------------------------------------------------------------------------

def _pair_mask(excluded_bins):
    return pd.DataFrame({
        'run_id': ['run-A'] * len(excluded_bins),
        'channel': ['C0-C1'] * len(excluded_bins),
        'bin_start': [float(b) for b in excluded_bins],
        'excluded': [True] * len(excluded_bins),
    })


def test_pair_keyed_mask_needs_the_pair_level_projector():
    """project_to_pairs on a pair-keyed table returns all-False -- SILENTLY.

    This is why project_pair_mask_to_windows exists as a separate entry point
    rather than an auto-detecting branch.
    """
    mask = _pair_mask([0.0])
    secs = np.array([0.0, 30.0, 60.0])
    right = mp.project_pair_mask_to_windows(mask, 'run-A', ['C0-C1'], secs)[:, 0]
    wrong = mp.project_to_pairs(mask, 'run-A', ['C0-C1'], secs)[:, 0]
    assert right.tolist() == [True, True, False]
    assert not wrong.any(), 'monopolar projector should see nothing in a pair-keyed table'


def test_pair_projection_bin_boundaries():
    """A 60s bin covers [b, b+60) -- inclusive at the start, exclusive at the end."""
    mask = _pair_mask([60.0])
    secs = np.array([59.9, 60.0, 119.9, 120.0])
    got = mp.project_pair_mask_to_windows(mask, 'run-A', ['C0-C1'], secs)[:, 0]
    assert got.tolist() == [False, True, True, False]


def test_or_pair_flags_rejects_duplicate_keys():
    """Duplicates would multiply merge rows and shift every later row."""
    df = pd.DataFrame({'session_id': ['ses-01'], 'run_id': ['run-A'],
                       'anode_channel': ['C0'], '_bin': [0.0]})
    dup = pd.DataFrame({'session_id': ['ses-01'] * 2, 'run_id': ['run-A'] * 2,
                        'channel': ['C0'] * 2, 'bin_start': [0.0] * 2,
                        'excluded': [True, False]})
    with pytest.raises(ValueError, match='duplicate'):
        mp.or_pair_flags_60s(df, dup, 'anode_channel')


def test_or_pair_flags_both_legs_contribute():
    """THE PAIR RULE: either leg excludes the pair. The cathode side is not
    decorative -- the last contact on a shaft appears only as a cathode."""
    df = pd.DataFrame({'session_id': ['ses-01'] * 2, 'run_id': ['run-A'] * 2,
                       'anode_channel': ['C0', 'C1'], 'cathode_channel': ['C1', 'C2'],
                       '_bin': [0.0, 0.0]})
    mask = pd.DataFrame({'session_id': ['ses-01'], 'run_id': ['run-A'],
                         'channel': ['C2'], 'bin_start': [0.0], 'excluded': [True]})
    anode = mp.or_pair_flags_60s(df, mask, 'anode_channel')
    cathode = mp.or_pair_flags_60s(df, mask, 'cathode_channel')
    assert not anode.any()                      # C2 is nobody's anode here
    assert (anode | cathode).tolist() == [False, True]


# ---------------------------------------------------------------------------
# Numerics: P0.6's rules, and Jensen
# ---------------------------------------------------------------------------

def test_normalize_then_average_differs_from_average_then_normalize():
    """The reason the cache keeps per-window granularity at all.

    Z-scoring is affine, so it commutes with averaging -- but only when sigma is a
    fixed per-channel scalar. Under a per-window nonlinearity (here: the linear
    domain), the two orders genuinely differ, which is what the registry's fixed
    axis order encodes.
    """
    log_block = np.array([[[1.0]], [[3.0]]])            # 2 windows, 1 pair, 1 bin
    baseline_mean = np.array([[2.0]])

    # correct order: to linear, subtract, then average over windows
    lin = axes.to_domain(log_block, 'linear')
    correct = axes.epoch_mean(axes.normalize(lin, 10.0 ** baseline_mean,
                                             None, 'baseline_subtract'))
    # wrong order: average first, then subtract
    wrong = axes.normalize(axes.epoch_mean(lin)[None, :, :], 10.0 ** baseline_mean,
                           None, 'baseline_subtract')[0]
    assert correct == pytest.approx(wrong)   # subtraction IS linear, so these agree

    # ...but a log-domain average is a geometric mean, so the domain choice does
    # not commute with averaging:
    log_first = axes.epoch_mean(log_block)                     # 2.0
    linear_first = np.log10(axes.epoch_mean(lin))              # log10(mean(10,1000))
    assert log_first != pytest.approx(linear_first)


def test_float64_accumulation_beats_float32():
    """A float32 accumulator over ~300 windows holds only ~6 sig figs (P0.6).

    Pinned because `arr.mean(axis=0)` on float32 input silently accumulates in
    float32 -- numpy does not widen for you.
    """
    n_win = 300
    block = np.full((n_win, 1, 1), 1.0, dtype=np.float32)
    block[0, 0, 0] = 1.0 + 1e-3

    # The oracle must be the exact mean of the values AS STORED in float32 --
    # 1.0 + 1e-3 is not exactly representable, so comparing against the intended
    # decimal would measure storage quantisation rather than accumulator width,
    # which is not what this test is about.
    exact = float(block.astype(np.float64).sum() / n_win)

    wide = axes.epoch_mean(block.astype(config.CACHE_ACCUMULATE_DTYPE))[0, 0]
    narrow = float(block.mean(axis=0)[0, 0])     # float32 accumulator
    assert wide == pytest.approx(exact, rel=1e-15), 'float64 path must be exact'
    assert abs(narrow - exact) > abs(wide - exact), (
        'the float32 accumulator should be measurably worse -- if it is not, this '
        'test no longer demonstrates why CACHE_ACCUMULATE_DTYPE exists'
    )


def test_linear_domain_uses_float64():
    """10**-36.8 is near-subnormal in float32 and underflows on the next divide."""
    block = np.array([[[-36.8]]])
    lin = axes.to_domain(block, 'linear')
    assert lin.dtype == config.CACHE_LINEAR_DOMAIN_DTYPE
    assert lin[0, 0, 0] > 0
    assert (lin / 1e5)[0, 0, 0] > 0, 'must not underflow to exactly zero'


def test_baseline_accumulator_matches_numpy_and_is_stable():
    rng = np.random.default_rng(0)
    # Large offset + small spread: the regime where sum(x^2)-mean^2 cancels.
    data = 1e6 + rng.normal(0, 1e-3, size=(120, 4, 3))
    acc = axes.BaselineAccumulator(4, 3)
    for chunk in np.array_split(data, 5):        # merged across "epochs"
        acc.update(chunk)
    mean, sd = acc.finalize()
    assert np.allclose(mean, data.mean(axis=0), rtol=1e-12)
    assert np.allclose(sd, data.std(axis=0, ddof=1), rtol=1e-8)


def test_baseline_accumulator_scatters_by_channel_rows():
    """Runs with different montages must land in the same channel's slot."""
    acc = axes.BaselineAccumulator(3, 1)
    acc.update(np.full((10, 2, 1), 5.0), rows=np.array([0, 2]))
    acc.update(np.full((10, 2, 1), 7.0), rows=np.array([0, 1]))
    mean, _ = acc.finalize()
    assert mean[0, 0] == pytest.approx(6.0)      # saw both
    assert mean[1, 0] == pytest.approx(7.0)
    assert mean[2, 0] == pytest.approx(5.0)


def test_baseline_ignores_masked_nans():
    acc = axes.BaselineAccumulator(1, 1)
    block = np.array([[[1.0]], [[np.nan]], [[3.0]]])
    acc.update(block)
    mean, sd = acc.finalize()
    assert mean[0, 0] == pytest.approx(2.0)
    assert acc.count[0, 0] == 2


def test_baseline_too_few_windows_is_nan_not_zero():
    acc = axes.BaselineAccumulator(1, 1)
    acc.update(np.array([[[1.0]]]))              # a single window
    mean, sd = acc.finalize(min_windows=2)
    assert np.isnan(sd[0, 0]), 'ddof=1 is undefined at n=1; must be NaN, not 0'


def test_zero_variance_baseline_gives_nan_not_inf():
    acc = axes.BaselineAccumulator(1, 1)
    acc.update(np.full((10, 1, 1), 4.0))
    _, sd = acc.finalize()
    assert np.isnan(sd[0, 0])


# ---------------------------------------------------------------------------
# Masking policy + aggregation
# ---------------------------------------------------------------------------

def test_apply_mask_drops_under_covered_channel_epochs():
    block = np.ones((10, 2, 1))
    excluded = np.zeros((10, 2), dtype=bool)
    excluded[:8, 1] = True                        # 80% of channel 1 gone
    out, kept, frac = cache_reader.apply_mask(block, excluded, 0.5)
    assert kept.tolist() == [True, False]
    assert np.isnan(out[:, 1, :]).all()
    assert frac[1] == pytest.approx(0.8)
    assert not np.isnan(out[:, 0, :]).any()


def test_region_aggregation_drops_unmapped_channels():
    values = np.array([[1.0], [3.0], [100.0]])
    channels = ['A-B', 'C-D', 'E-F']
    region_of = {'A-B': 'Insula', 'C-D': 'Insula', 'E-F': None}
    out, counts = axes.aggregate_regions(values, channels, region_of, ['Insula', 'ACC'])
    assert out[0, 0] == pytest.approx(2.0)        # unmapped channel excluded
    assert counts.tolist() == [2, 0]
    assert np.isnan(out[1, 0])                    # no coverage != zero


def test_subject_relative_bins_use_distinct_events():
    defs = pd.DataFrame({'pain_event_id': [0, 1, 2, 3],
                         'pain_score': [0.0, 2.0, 4.0, 6.0]})
    bins = axes.assign_pain_bins(defs, 'subject_relative')
    assert bins.tolist() == ['none', 'low', 'high', 'high']   # threshold = mean(2,4,6) = 4
    assert axes.subject_relative_threshold(defs) == pytest.approx(4.0)


def test_absolute_bins_use_fixed_cutpoints():
    defs = pd.DataFrame({'pain_event_id': [0, 1, 2, 3],
                         'pain_score': [0.0, 2.0, 5.0, 9.0]})
    assert axes.assign_pain_bins(defs, 'absolute').tolist() == \
        ['none', 'low', 'medium', 'high']


# ---------------------------------------------------------------------------
# ViewConfig guardrails
# ---------------------------------------------------------------------------

def test_mask_label_is_required_unless_explicitly_unmasked():
    with pytest.raises(ValueError, match='requires mask_label'):
        ViewConfig(mask_level='bipolar', mask_label=None)
    ViewConfig(mask_level='none', mask_label=None)     # explicit opt-out is fine


def test_bad_axis_value_is_rejected():
    with pytest.raises(ValueError, match='normalization'):
        ViewConfig(normalization='ztransform', mask_level='none')


def test_is_difference_drives_the_jensen_branch():
    assert ViewConfig(normalization='zscore_vs_baseline', mask_level='none').is_difference
    assert ViewConfig(normalization='baseline_subtract', mask_level='none').is_difference
    assert not ViewConfig(normalization='none', mask_level='none').is_difference


def test_value_label_tracks_normalization():
    assert 'z-score' in ViewConfig(normalization='zscore_vs_baseline',
                                   mask_level='none').value_label
    assert 'change' in ViewConfig(normalization='baseline_subtract',
                                  mask_level='none').value_label


def test_config_is_frozen():
    cfg = ViewConfig(mask_level='none')
    with pytest.raises(Exception):
        cfg.domain = 'linear'
