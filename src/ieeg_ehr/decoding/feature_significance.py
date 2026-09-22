"""Which spectro-spatial features actually carry a subject's pain signal.

    from ieeg_ehr.decoding import feature_significance as fs
    sig = fs.significant_features(run_dir, cv_scheme='random')

THE DEFINITION, AND WHY IT IS NOT "NONZERO COEFFICIENT"
-------------------------------------------------------
Elastic net zeroes most features by construction, so "was it selected" in a
single fit is almost content-free -- and under collinearity it is worse than
that: among a correlated group the net picks one somewhat arbitrarily, so which
channel appears is partly a coin flip. A single index-model fit therefore cannot
distinguish a feature that matters from a feature that won one toss.

What separates them is CONCENTRATION ACROSS RESAMPLES. Measured on the
2026-09-12 run, per subject:

    real runs      few features ever selected, but the best reaches ~0.33
    shuffled runs  EVERY feature selected at least once, none above ~0.12

The shuffled runs have a HIGHER MEAN selection frequency than the real ones. That
looks backwards until you see why: with no signal the inner CV's score is flat
along the penalty path, so the chosen alpha is arbitrary and sometimes lands
weak, admitting many features at random. Null selections are spread thin over
everything; real selections pile up on a few. A naive "selection_frequency > t"
rule would therefore be actively misled -- it would rank the null ABOVE the
signal on the mean.

So significance here is: does THIS feature's real selection frequency exceed what
the permuted models achieve? The null is the subject's own shuffled-run
distribution of selection frequencies, which is what the target paper means by
"compared the distribution of feature coefficients from the trained models with
those from permuted models".

WHAT THIS IS NOT. It is not a calibrated per-feature p-value -- the null is
empirical, taken over features within a subject, and features are correlated so
the effective number of independent tests is unknown. It ranks and thresholds;
it does not license a claim that feature X is significant at p = 0.05.
"""

import logging

import numpy as np
import pandas as pd

from ieeg_ehr import io

logger = logging.getLogger(__name__)

#: A feature must beat this quantile of the subject's own shuffled selection
#: frequencies. 0.95 is a convention, not a calibrated alpha -- see the module
#: docstring on why this is a ranking rule rather than a test.
NULL_QUANTILE = 0.95

#: ...and must clear this absolute floor. Without it, a subject whose null
#: happens to be degenerate (every shuffled frequency 0) would have every
#: once-selected feature called significant.
MIN_SELECTION_FREQUENCY = 0.10


def load_stability(run_dir):
    """Every unit's coef_stability table in one frame."""
    from pathlib import Path
    paths = sorted(Path(run_dir).glob('units/*_coef_stability.csv'))
    if not paths:
        raise FileNotFoundError(f'no coef_stability.csv under {run_dir}/units/')
    frames = [io.read_table(p, on_stale='ignore') for p in paths]
    logger.info('loaded stability for %d units from %s', len(paths), run_dir)
    return pd.concat(frames, ignore_index=True)


def significant_features(stability, cv_scheme='random',
                         null_quantile=NULL_QUANTILE,
                         min_frequency=MIN_SELECTION_FREQUENCY):
    """Per (unit, feature): is it selected more often than the permuted models?

    Returns the REAL rows for `cv_scheme` with three columns added:
      `null_threshold`  the subject's shuffled quantile it had to beat
      `significant`     bool
      `excess`          selection_frequency - null_threshold (for ranking)

    The threshold is computed WITHIN SUBJECT, because the null's location depends
    on that subject's n, feature count and penalty path -- one cohort-wide cutoff
    would be stricter for some subjects than others for reasons unrelated to the
    brain.
    """
    d = stability[stability['cv_scheme'] == cv_scheme]
    real = d[~d['shuffled']].copy()
    null = d[d['shuffled']]
    if real.empty or null.empty:
        raise ValueError(f'cv_scheme={cv_scheme!r} lacks real or shuffled rows')

    thresh = (null.groupby('unit')['selection_frequency']
                  .quantile(null_quantile).rename('null_threshold'))
    real = real.merge(thresh, on='unit', how='left')
    real['null_threshold'] = real['null_threshold'].fillna(0.0)
    real['significant'] = ((real['selection_frequency'] > real['null_threshold'])
                           & (real['selection_frequency'] >= min_frequency))
    real['excess'] = real['selection_frequency'] - real['null_threshold']
    return real


def attach_regions(features, region_by_channel):
    """Add a `region` column from a {channel -> DK label} map.

    EXACT FOR LAPLACIAN CHANNELS, which is why this analysis uses them. A
    Laplacian channel is named for its CENTRE contact, and the raw electrodes
    table carries a per-contact `Desikan_Killiany` label, so the region is a
    direct lookup -- no virtual-electrode midpoint, no anode-vs-cathode choice.
    Bipolar pairs have no such luxury: docs/view_registry.md flags the
    anode-based assignment as a "deliberate TEMPORARY stand-in" that is wrong
    whenever a pair straddles a parcel boundary.
    """
    out = features.copy()
    out['region'] = out['channel'].map(region_by_channel)
    missing = out['region'].isna().sum()
    if missing:
        logger.warning('%d/%d features have no region label', missing, len(out))
    return out


def recruitment_by_region(features, denominator='channels'):
    """Pooled recruitment per region, and the fraction printed above each bar.

    `denominator` is EXPLICIT because the source figure's is ambiguous: its
    y-axis runs 0-4, which cannot be a proportion of channels in [0,1]. The
    likely reading is significant FEATURES over available CHANNELS, which can
    exceed 1 since a channel yields up to 6 band features. Both are offered, and
    whichever is chosen must be used for the per-subject panel too or the two
    are not comparable.
    """
    d = features.dropna(subset=['region'])
    grp = d.groupby('region')
    n_sig = grp['significant'].sum()
    if denominator == 'channels':
        total = grp['channel'].nunique()
        label = 'Significant features / available channels'
    elif denominator == 'features':
        total = grp.size()
        label = 'Fraction of features significant'
    else:
        raise ValueError("denominator must be 'channels' or 'features'")
    out = pd.DataFrame({'n_significant': n_sig, 'n_total': total})
    out['proportion'] = out['n_significant'] / out['n_total'].replace(0, np.nan)
    out.attrs['ylabel'] = label
    return out.sort_values('proportion', ascending=False)


def recruitment_by_subject_region(features, denominator='channels'):
    """The same quantity computed WITHIN each subject.

    A subject with no coverage in a region is ABSENT, not zero. Imputing zero
    would claim they had electrodes there and nothing was selected, which is a
    different statement from not having looked.
    """
    d = features.dropna(subset=['region'])
    grp = d.groupby(['unit', 'region'])
    n_sig = grp['significant'].sum()
    total = grp['channel'].nunique() if denominator == 'channels' else grp.size()
    out = pd.DataFrame({'n_significant': n_sig, 'n_total': total}).reset_index()
    out['proportion'] = out['n_significant'] / out['n_total'].replace(0, np.nan)
    return out
