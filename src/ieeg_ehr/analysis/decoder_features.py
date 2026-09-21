"""Which features the per-subject decoders actually used, and in which regions.

    python -m ieeg_ehr.analysis.decoder_features --run-timestamp <ts> --arm regression

The analysis half of the Figure-2 panels; plotting lives in
`plot_decoder_features.py` so the numbers can be inspected without rendering.

WHAT "SIGNIFICANT FEATURE" MEANS HERE
-------------------------------------
NOT "nonzero coefficient". Under collinearity an elastic net picks one of a
correlated group somewhat arbitrarily, so a single large coefficient from a
single fit is close to meaningless -- which is exactly why the target paper
defines significance through STABILITY across 100 bootstrap runs and a
comparison against PERMUTED models, not through magnitude.

The paper's recipe is: mean coefficient over 100 runs -> changepoint on the
cumulative-sum curve -> discard features whose coefficient distribution does not
differ from the permuted models'. We reproduce the second and third steps
directly and replace the changepoint with an explicit selection-frequency
threshold, because:

  * selection frequency answers the same question more legibly ("chosen in 87 of
    100 runs" vs "left of an inflection in a cumulative curve"), and
  * the changepoint is a ranking heuristic with no null attached, while the
    permuted comparison is an actual test. Keeping both would double-count.

A feature is SIGNIFICANT when its real selection frequency exceeds the upper
tail of its own shuffled selection frequency by `MIN_FREQ_MARGIN`, AND it clears
`MIN_SELECTION_FREQ` in absolute terms. Both conditions are per FEATURE per
SUBJECT -- nothing is pooled before the test.

THE DENOMINATOR QUESTION (panels F and G)
-----------------------------------------
The published y-axis runs 0-4, which cannot be a proportion of channels. With 6
bands per channel, significant FEATURES / available CHANNELS can reach 6, so
that is the reading. We report BOTH and label them, because the two answer
different questions: feature-rate says "how much of this region's spectrum was
recruited", channel-rate says "what fraction of its contacts contributed
anything". `RECRUITMENT_DENOMINATORS` names them so a figure cannot be ambiguous
about which it plotted.
"""

import argparse
import logging
import sys

import numpy as np
import pandas as pd

from ieeg_ehr import config, io

logger = logging.getLogger(__name__)

#: A feature must be selected at least this often in the REAL runs before it can
#: be significant, however quiet its null. Guards the case where a feature is
#: chosen 4% of the time really and 1% under shuffling -- a ratio, but not a
#: contribution.
MIN_SELECTION_FREQ = 0.50

#: ...and must beat its own shuffled frequency by at least this margin. The null
#: is per feature, so a feature in a subject whose whole model is unstable is
#: held to that subject's own noise level rather than a global one.
MIN_FREQ_MARGIN = 0.20

RECRUITMENT_DENOMINATORS = ('features', 'channels')

#: Bands low -> high. Fixed so H and I stay aligned when assembled.
BAND_ORDER = list(config.PAPER_BANDS_6_HZ)


def load_stability(run_dir, arm, cv_scheme='random'):
    """Per-unit coefficient stability tables, real and shuffled, as one frame."""
    paths = sorted((run_dir / 'units').glob('*_coef_stability.csv'))
    if not paths:
        raise FileNotFoundError(
            f'no *_coef_stability.csv under {run_dir}/units. That artifact was '
            'added on 2026-09-12; runs made before it stored only the index '
            "model's coefficients, from which stability cannot be recovered. "
            'Re-run the decoder.')
    df = pd.concat([io.read_table(p, on_stale='ignore') for p in paths],
                   ignore_index=True)
    return df[(df.arm == arm) & (df.cv_scheme == cv_scheme)]


def mark_significant(stability, min_freq=MIN_SELECTION_FREQ,
                     min_margin=MIN_FREQ_MARGIN):
    """One row per (unit, feature) with `significant` and the two frequencies."""
    real = stability[~stability.shuffled]
    null = stability[stability.shuffled]
    keys = ['unit', 'feature', 'channel', 'band']
    m = real.merge(null[keys + ['selection_frequency', 'coef_mean', 'coef_sd']],
                   on=keys, suffixes=('', '_null'))
    m['freq_margin'] = m.selection_frequency - m.selection_frequency_null
    m['significant'] = ((m.selection_frequency >= min_freq)
                        & (m.freq_margin >= min_margin))
    logger.info('%d/%d features significant across %d units (%.1f%%)',
                int(m.significant.sum()), len(m), m.unit.nunique(),
                100 * m.significant.mean())
    return m


def attach_regions(features, region_of):
    """Join a DK region onto each feature via its CHANNEL.

    For LAPLACIAN features this is exact: the channel IS a contact, so its region
    is that contact's own Desikan-Killiany label -- no virtual-electrode midpoint
    and no anode-vs-cathode choice, which is the approximation
    docs/view_registry.md flags for bipolar pairs. For bipolar features the
    caller must pass an anode-based map and accept that caveat.
    """
    out = features.copy()
    out['region'] = out.channel.map(region_of)
    missing = out.region.isna().sum()
    if missing:
        logger.warning('%d/%d features have no region label; dropped',
                       missing, len(out))
    return out.dropna(subset=['region'])


def region_recruitment(features, denominator='features'):
    """Pooled recruitment per region: significant features over what was available.

    `denominator='features'` divides by available FEATURES (channels x bands) and
    is a true proportion in [0, 1]. `'channels'` divides by available CHANNELS and
    can exceed 1, which is what makes the published axis run past 1.
    """
    if denominator not in RECRUITMENT_DENOMINATORS:
        raise ValueError(f'denominator must be one of {RECRUITMENT_DENOMINATORS}')
    g = features.groupby('region')
    n_sig = g.significant.sum()
    n_feat = g.size()
    n_chan = g.channel.nunique()
    denom = n_feat if denominator == 'features' else n_chan
    out = pd.DataFrame({'n_significant': n_sig, 'n_features': n_feat,
                        'n_channels': n_chan, 'denominator': denom})
    out['proportion'] = out.n_significant / out.denominator
    return out.sort_values('proportion', ascending=False)


def region_recruitment_by_subject(features, denominator='features'):
    """Same quantity computed WITHIN each subject.

    A subject with no coverage in a region is ABSENT, not zero -- imputing zero
    would say "this region was available and contributed nothing", which is a
    different and false claim.
    """
    g = features.groupby(['unit', 'region'])
    n_sig = g.significant.sum()
    denom = g.size() if denominator == 'features' else g.channel.nunique()
    out = (n_sig / denom).rename('proportion').reset_index()
    return out[denom.reset_index(drop=True).to_numpy() > 0]


def band_power_by_state(feature_table, significant, pain_bins=('low', 'high')):
    """Median z-scored power per significant feature, split by pain state.

    Normalization is WITHIN SUBJECT, before pooling -- otherwise a subject with
    systematically larger amplitudes dominates every band. Each returned row is
    one feature's median in one state, which is the unit the paper's per-band n
    counts (and why those n are large and unequal).
    """
    sig = significant[significant.significant][['unit', 'channel', 'band']]
    df = feature_table.merge(
        sig, left_on=['unit', 'region', 'freq_bin_index'],
        right_on=['unit', 'channel', 'band'], how='inner')

    # z within subject x feature, across that subject's epochs.
    grp = df.groupby(['unit', 'channel', 'band'])['value']
    df['z'] = (df.value - grp.transform('mean')) / grp.transform('std').replace(0, np.nan)
    df = df[df.pain_bin.isin(pain_bins)]
    med = (df.groupby(['unit', 'channel', 'band', 'pain_bin'])['z']
             .median().rename('median_z').reset_index())
    return med


def effect_size_band_region(median_z, region_of, min_subjects=3):
    """Band x region effect size (high - low), with a one-sample t-test per cell.

    The test is across SUBJECTS, on each subject's mean feature-level difference
    -- not across features, which would treat 40 contacts in one subject as 40
    independent observations. Cells measured in fewer than `min_subjects`
    subjects return NaN rather than a t-statistic from 2 points.
    """
    from scipy import stats
    wide = median_z.pivot_table(index=['unit', 'channel', 'band'],
                                columns='pain_bin', values='median_z')
    if not {'low', 'high'} <= set(wide.columns):
        raise ValueError('need both low and high pain bins to form a difference')
    wide['diff'] = wide['high'] - wide['low']
    wide = wide.reset_index()
    wide['region'] = wide.channel.map(region_of)
    wide = wide.dropna(subset=['region', 'diff'])

    # One value per subject per cell FIRST, then test across subjects.
    per_subject = (wide.groupby(['band', 'region', 'unit'])['diff']
                       .mean().rename('subject_diff').reset_index())
    rows = []
    for (band, region), g in per_subject.groupby(['band', 'region']):
        vals = g.subject_diff.to_numpy()
        if len(vals) < min_subjects:
            rows.append({'band': band, 'region': region, 'n_subjects': len(vals),
                         'effect_size': np.nan, 'p': np.nan})
            continue
        t, p = stats.ttest_1samp(vals, 0.0)
        rows.append({'band': band, 'region': region, 'n_subjects': len(vals),
                     'effect_size': float(vals.mean() / (vals.std(ddof=1) or np.nan)),
                     'mean_diff': float(vals.mean()), 't': float(t), 'p': float(p)})
    out = pd.DataFrame(rows)
    from ieeg_ehr.decoding.aggregate import benjamini_hochberg
    out['q_fdr'] = benjamini_hochberg(out.p.to_numpy())
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--run-timestamp', required=True)
    ap.add_argument('--arm', default='regression')
    ap.add_argument('--view-scheme', default='perchannel-laplacian-bandrms')
    ap.add_argument('--cv-scheme', default='random')
    ap.add_argument('--run-name', default='per_subject')
    ap.add_argument('--log-level', default='INFO')
    args = ap.parse_args(argv)
    logging.basicConfig(level=args.log_level,
                        format='%(asctime)s %(levelname)s %(message)s')

    run_dir = config.decoding_run_dir(
        output_type=args.arm, run_name=args.run_name,
        view_scheme=args.view_scheme, timestamp=args.run_timestamp)
    stab = load_stability(run_dir, args.arm, args.cv_scheme)
    sig = mark_significant(stab)
    io.write_table(sig, run_dir / 'feature_significance.csv',
                   params={'arm': args.arm, 'cv_scheme': args.cv_scheme,
                           'min_selection_freq': MIN_SELECTION_FREQ,
                           'min_freq_margin': MIN_FREQ_MARGIN})
    print(sig.groupby('band').significant.agg(['sum', 'size', 'mean'])
             .to_string(float_format=lambda v: f'{v:.3f}'))
    return 0


if __name__ == '__main__':
    sys.exit(main())
