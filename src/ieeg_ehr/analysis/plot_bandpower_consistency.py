"""Is the band-power map consistent ACROSS SUBJECTS, or carried by a few?

    python -m ieeg_ehr.analysis.plot_bandpower_consistency --run-dir <run>

`fig_band_map` and `fig_band_forest` describe the GROUP: one beta per
(region, band) with a Wald interval. `fig_check_caterpillar` opens one cell and
shows the 40-odd subjects inside it. Neither answers the question a reader asks
of the map as a whole -- is the delta-down / beta-up pattern something most
patients show, or a group average over a heterogeneous crowd? These five figures
answer it, at five different resolutions:

    signmap     The companion to fig_band_map, same grid: how many subjects
                share the group's sign in each cell, against a SIGN-FLIP NULL.
    percent     The same grid coloured by the RAW percentage instead of the
                excess over that null, with the scale floored at the null so a
                cell at the bottom is a cell at chance. The one to look at;
                signmap is the one to quote.
    matrix      Every subject x region slope, one panel per band. No averaging
                at all -- the honest version of "is it everyone or a few".
    profile     The spectral tilt, then ONE statistic against ONE null: each
                subject's correlation with a LEAVE-ONE-OUT group map, ranked,
                against a WITHIN-BAND permutation bound computed per subject.
                Holding the tilt fixed makes this a claim about anatomy.
    profile_v2  The SUPERSEDED version of the same question, kept so the two can
                be compared: the tilt test and the topography test as the two
                axes of a scatter. It shows the dissociation as a position
                rather than a number, at the cost of giving equal billing to a
                null that is nearly free to pass. Rendered with the CORRECTED
                statistics, so the only difference from `profile` is
                presentation -- see `fig_profile_v2`.
    network     One point per subject: limbic delta against sensorimotor beta.
                Is this one syndrome in most patients, or two disjoint subsets?
    influence   Leave-one-subject-out jackknife: does any significant cell rest
                on one patient?

All of them, plus the two tables behind them, are written into a `consistency/`
subfolder of the run directory -- one self-contained set, kept together because
it describes the SUBJECTS rather than the model that the run's own figures
describe. `--out-subdir ''` puts them beside `band_cells.parquet` instead.

Everything is computed from `subject_slopes.parquet` and `band_cells.parquet`,
which the fit stage already wrote. Nothing is refitted, so this runs in minutes
and can run while a permutation array is still going.

THREE STATISTICAL POINTS, because each one is a trap that a naive version of
these figures falls into:

1. SIGN CONSISTENCY CANNOT BE TESTED AGAINST 0.5. The group sign is estimated
   from the same subjects being counted, so it is dragged toward whichever sign
   the majority already has. Under a true null the expected agreement is
   therefore ABOVE one half -- around 0.56 at n=44 -- and a binomial test
   against 0.5 would call noise significant. `signflip_null` builds the right
   reference: flip each subject's slope sign at random, recompute the group sign
   the same way, and count agreement. The map is coloured by the EXCESS over
   that null, not by the raw fraction.

2. A SUBJECT'S SIMILARITY TO THE GROUP MAP IS CIRCULAR if the group map includes
   that subject. `loo_group_map` rebuilds the reference from the OTHER subjects
   for each subject in turn. The uncorrected version is computed too, and the
   gap between them is reported -- at ~44 subjects per cell it is small, but it
   is not zero and it is always positive.

3. A PATTERN CORRELATION OVER ~100 CELLS IS MOSTLY BAND STRUCTURE unless the
   null is blocked. Nearly every patient has some delta-down/beta-up tilt, and
   a correlation over that many cells responds to COHERENCE rather than to
   magnitude -- so a small shared shape pointing the same way in every region
   carries the correlation even when no patient reproduces the regional
   topography. The profile figure therefore uses ONE statistic (the plain
   subject-to-group correlation) against ONE null that shuffles the group's
   values WITHIN BAND, holding the tilt fixed across every permutation so that
   only anatomy can beat it. Measured here the leak is large: a free cross-band
   shuffle has a median bound of 0.24 against 0.41 blocked, and lets 10 of 51
   extra subjects through. The free version and an equivalent band-mean-removal
   version are both kept in `consistency_subjects.csv` for audit; the blocked
   and band-mean-removed routes agree exactly (28 of 51).

EXPLORATORY. Discovery cohort only. Nominations, not findings.
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from ieeg_ehr import io
from ieeg_ehr.analysis import view_tables

logger = logging.getLogger(__name__)

SCRIPT = 'ieeg_ehr/analysis/plot_bandpower_consistency.py'

#: `profile_v2` writes `fig_consistency_profile_v2.png` -- the superseded
#: two-statistic scatter, kept so it can be held up against the current
#: `profile`. It is in the default set on purpose: a comparison figure nobody
#: regenerates goes stale against the run it claims to describe.
FIGURES = ('signmap', 'percent', 'matrix', 'profile', 'profile_v2', 'network',
           'influence')

#: Everything lands in this subfolder of the run, not beside `band_cells.parquet`.
#: These five figures and their two tables are one self-contained set describing
#: the SUBJECTS rather than the model, and the run directory already holds twenty
#: files. `--out-subdir ''` restores the flat layout.
OUT_SUBDIR = 'consistency'

DISCLAIMER = ('EXPLORATORY -- discovery cohort, NOMINATIONS NOT FINDINGS. '
              'Not confirmed out of sample.')

#: Same two colours the pilot spaghetti uses, so SIGN reads identically across
#: every figure in this run: blue for power falling with pain, red for rising.
NEG_COLOR = '#2c6fad'
POS_COLOR = '#c1442f'

#: Okabe-Ito blue and orange, for encodings that are NOT sign. Used in the
#: profile figure, which previously paired red against green -- the one
#: colourblind-unsafe combination available, and it made red do three jobs at
#: once (positive, one of two series, and a median line). Blue/orange is safe
#: under every common form of colour vision deficiency and keeps red free to
#: mean "power rose with pain" and nothing else. The diverging MAPS are left
#: alone: RdBu_r and PRGn are both already colourblind-safe by construction.
ACCENT_A = '#0072B2'
ACCENT_B = '#D55E00'

#: Draws from the sign-flip and label-permutation nulls. 10k puts the Monte
#: Carlo error on a 95th percentile at well under a percentage point, and the
#: whole thing is vectorised, so there is no reason to be stingy.
N_PERM = 10000

#: A subject needs a few cells before a correlation against the group map means
#: anything. Below this they keep their row in the matrix figure and lose their
#: point in the profile figure.
MIN_CELLS_FOR_R = 12

#: The two halves of the pattern in `fig_band_map`, fixed ANATOMICALLY and not
#: by which cells came out significant. That distinction is the whole point of
#: the network figure: picking the cells by their p-value and then asking how
#: many subjects support them is circular, whereas these two sets are a
#: statement about pain anatomy that could have been written before the fit.
LIMBIC_DEFAULT = ('Hippocampus', 'Amygdala', 'Thalamus', 'Insula', 'rACC', 'PCC',
                  'OFC')
SENSORIMOTOR = ('M1', 'S1', 'S2/PO', 'dlPFC', 'dmPFC/SMA', 'IFG/vlPFC')


# ============================================================================
# SHARED
# ============================================================================

def load_run(run_dir):
    """The two tables every figure here needs, plus the run's own parameters."""
    run_dir = Path(run_dir)
    cells = io.read_table(run_dir / 'band_cells.parquet', on_stale='warn')
    slopes_path = run_dir / 'subject_slopes.parquet'
    if not slopes_path.exists():
        raise SystemExit(
            f'{slopes_path} is missing -- this run has no per-subject slopes, so '
            'there is nothing to check consistency of. Re-run the fit stage.')
    slopes = io.read_table(slopes_path, on_stale='warn')
    try:
        params = json.loads((run_dir / 'provenance.json').read_text())['params']
    except (OSError, ValueError, KeyError):
        params = {}
    # The eligibility chain, so the figures can PRINT their attrition instead of
    # leaving a viewer to wonder what "51 subjects" is 51 out of. Optional: a run
    # without the inventory still draws, it just says "unknown" in the chain.
    inv_path = run_dir / 'inventory_subjects.parquet'
    inventory = (io.read_table(inv_path, on_stale='ignore')
                 if inv_path.exists() else None)
    return run_dir, cells, slopes, params, inventory


def axis_order(cells, params):
    """(regions, bands, band_edges) in the run's own display order.

    Taken from the run's parameters rather than from a constant here, so these
    figures cannot silently disagree with `fig_band_map` about which row is
    which after a `--roi-scheme` or `--band-set` change.
    """
    bands = list(params.get('bands', {})) or sorted(cells['band'].unique())
    edges = params.get('bands', {})
    scheme = params.get('roi_scheme', 'roi_v2_ofc')
    present = set(cells['region'])
    regions = [r for r in view_tables.roi_regions_for({'roi_scheme': scheme})
               if r in present]
    # Anything the scheme does not know about still has to appear -- a silently
    # dropped region is exactly the kind of shrinking denominator CLAUDE.md says
    # must stay visible.
    extra = sorted(present - set(regions))
    if extra:
        logger.warning('regions outside the %s display order, appended: %s',
                       scheme, ', '.join(extra))
    return regions + extra, bands, edges


def band_label(band, edges):
    lo_hi = edges.get(band)
    return f'{band}\n{lo_hi[0]}-{lo_hi[1]} Hz' if lo_hi else band


def slope_matrix(slopes, regions, bands, value='slope'):
    """(subjects, cell_keys, array[n_subj, n_cells]) with NaN where unobserved.

    Cells are ordered region-major so that reshaping to (n_regions, n_bands)
    later is a plain `.reshape`, and the band-mean removal in `region_pattern_r`
    can operate on a clean 2-D view.
    """
    keys = [(r, b) for r in regions for b in bands]
    subjects = sorted(slopes['subject'].unique())
    wide = (slopes.pivot_table(index='subject', columns=['region', 'band'],
                               values=value, aggfunc='first')
            .reindex(index=subjects))
    # Reindex explicitly: a (region, band) with no subject at all is absent from
    # the pivot entirely and must become a NaN column, not a missing one.
    wide = wide.reindex(columns=pd.MultiIndex.from_tuples(keys))
    return subjects, keys, wide.to_numpy(dtype='float64')


def ivw_mean(values, weights):
    """Inverse-variance weighted mean along the last axis, NaN-safe.

    The stand-in for the mixed model's fixed effect everywhere in this module
    that needs to recompute a group estimate thousands of times. It is NOT the
    same estimator -- it has no channel random intercept and no partial pooling
    -- which is why every figure built on it says so on its face.
    """
    ok = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    w = np.where(ok, weights, 0.0)
    v = np.where(ok, values, 0.0)
    denom = w.sum(axis=-1)
    with np.errstate(invalid='ignore', divide='ignore'):
        return np.where(denom > 0, (w * v).sum(axis=-1) / denom, np.nan)


def cell_weights(se, slope):
    """1/se^2, with a floor so one implausibly precise subject cannot dominate.

    A subject whose epochs happen to line up can return an SE two orders of
    magnitude below the rest, and an unfloored inverse-variance weight would
    hand them the entire cell -- which would make the influence figure report
    its own weighting scheme rather than anything about the data. The floor is
    the 5th percentile of the cell's own SEs.
    """
    se = np.asarray(se, dtype='float64')
    ok = np.isfinite(se) & (se > 0) & np.isfinite(slope)
    if not ok.any():
        return np.zeros_like(se)
    floor = float(np.percentile(se[ok], 5))
    safe = np.where(ok, np.maximum(se, floor), np.nan)
    with np.errstate(invalid='ignore', divide='ignore'):
        return np.where(ok, 1.0 / safe ** 2, 0.0)


# ============================================================================
# THE THREE COMPUTATIONS THE FIGURES SHARE
# ============================================================================

def signflip_null(slope, weight, n_perm=N_PERM, rng=None):
    """(observed-style null mean, 95th pct, p) for sign agreement in one cell.

    THE REFERENCE FOR SIGN CONSISTENCY. Under the null that a subject's slope is
    equally likely to be positive or negative, flipping signs at random and
    re-deriving the group sign the same way the real analysis does reproduces
    the upward bias that makes 0.5 the wrong comparison. Returns the null's mean
    agreement and its 95th percentile; the caller supplies the observed value.

    The group sign in each draw is the sign of the inverse-variance weighted
    mean of the FLIPPED slopes -- the cheap analogue of refitting the mixed
    model on them, which is what an exact version would do.
    """
    rng = rng or np.random.default_rng(0)
    ok = np.isfinite(slope) & (weight > 0)
    s, w = slope[ok], weight[ok]
    n = len(s)
    if n < 3:
        return np.nan, np.nan, n
    eps = rng.choice([-1.0, 1.0], size=(n_perm, n))
    flipped = eps * s
    grp = np.sign((flipped * w).sum(axis=1) / w.sum())
    frac = (np.sign(flipped) == grp[:, None]).sum(axis=1) / n
    return float(frac.mean()), float(np.percentile(frac, 95)), n


def cell_consistency(cells, slopes, regions, bands, n_perm=N_PERM, seed=0):
    """One row per cell: sign agreement against its null, and LOSO influence.

    The two questions a reader has about a coloured square, answered with the
    same per-subject slopes the caterpillar figure draws. `frac_sign` recomputes
    what `band_cells.frac_sign_consistent` already holds -- deliberately, so a
    mismatch surfaces rather than being inherited.
    """
    rng = np.random.default_rng(seed)
    beta = cells.set_index(['region', 'band'])['beta_nrs_within']
    rej = (cells.assign(r=cells['p_bh_reject'].fillna(False).astype(bool))
           .set_index(['region', 'band'])['r'])
    stored = (cells.set_index(['region', 'band'])['frac_sign_consistent']
              if 'frac_sign_consistent' in cells.columns else None)
    by_cell = {k: g for k, g in slopes.groupby(['region', 'band'], sort=False)}

    rows = []
    for region in regions:
        for band in bands:
            g = by_cell.get((region, band))
            b = float(beta.get((region, band), np.nan))
            rec = {'region': region, 'band': band, 'beta': b,
                   'p_bh_reject': bool(rej.get((region, band), False)),
                   'frac_sign': np.nan, 'n_with_slope': 0, 'n_sign_match': 0,
                   'null_frac_mean': np.nan, 'null_frac_p95': np.nan,
                   'excess_sign': np.nan, 'p_signflip': np.nan,
                   'ivw_mean': np.nan, 'ivw_sign_agrees_with_model': True,
                   'loso_max_abs_delta': np.nan, 'loso_max_rel_delta': np.nan,
                   'loso_worst_subject': '', 'loso_flips_sign': False,
                   'frac_sign_stored': (float(stored.get((region, band), np.nan))
                                        if stored is not None else np.nan)}
            if g is None or not np.isfinite(b):
                rows.append(rec)
                continue

            s = g['slope'].to_numpy(dtype='float64')
            w = cell_weights(g['se'].to_numpy(), s)
            subj = g['subject'].to_numpy()
            ok = np.isfinite(s)
            n = int(ok.sum())
            rec['n_with_slope'] = n
            if n < 3:
                rows.append(rec)
                continue

            rec['n_sign_match'] = int((np.sign(s[ok]) == np.sign(b)).sum())
            rec['frac_sign'] = rec['n_sign_match'] / n

            m_full = float(ivw_mean(s, w))
            rec['ivw_mean'] = m_full
            rec['ivw_sign_agrees_with_model'] = bool(np.sign(m_full) == np.sign(b))

            null_mean, null_p95, _ = signflip_null(s, w, n_perm=n_perm, rng=rng)
            rec['null_frac_mean'] = null_mean
            rec['null_frac_p95'] = null_p95
            rec['excess_sign'] = rec['frac_sign'] - null_mean
            # p from the same draws: how often does the null reach the observed
            # agreement? Recomputed rather than read off the percentile so the
            # p-value and the band on the figure cannot drift apart.
            rec['p_signflip'] = _signflip_p(s, w, rec['frac_sign'], n_perm, rng)

            # Leave one subject out of the weighted mean. Not a refit -- see the
            # figure's own footnote.
            deltas, names = [], []
            idx = np.where(ok)[0]
            for i in idx:
                keep = np.ones(len(s), dtype=bool)
                keep[i] = False
                m_i = float(ivw_mean(s[keep], w[keep]))
                deltas.append(m_i - m_full)
                names.append(subj[i])
            deltas = np.asarray(deltas)
            j = int(np.nanargmax(np.abs(deltas)))
            rec['loso_max_abs_delta'] = float(abs(deltas[j]))
            rec['loso_max_rel_delta'] = (float(abs(deltas[j]) / abs(m_full))
                                         if m_full else np.nan)
            rec['loso_worst_subject'] = str(names[j])
            rec['loso_flips_sign'] = bool(
                np.any(np.sign(m_full + deltas) != np.sign(m_full)))
            rows.append(rec)
    return pd.DataFrame(rows)


def _signflip_p(slope, weight, observed, n_perm, rng):
    ok = np.isfinite(slope) & (weight > 0)
    s, w = slope[ok], weight[ok]
    n = len(s)
    if n < 3 or not np.isfinite(observed):
        return np.nan
    eps = rng.choice([-1.0, 1.0], size=(n_perm, n))
    flipped = eps * s
    grp = np.sign((flipped * w).sum(axis=1) / w.sum())
    frac = (np.sign(flipped) == grp[:, None]).sum(axis=1) / n
    # (hits + 1) / (draws + 1): a permutation p is never allowed to be zero.
    return float((np.sum(frac >= observed) + 1) / (n_perm + 1))


def loo_group_map(mat, weights):
    """(n_subj, n_cells) leave-one-out group map: cell means WITHOUT that subject.

    Removes the circularity in "how similar is this subject to the group" --
    with the subject in the reference, a patient with many epochs is partly
    being correlated with themselves. Computed by subtracting each subject's own
    contribution from the pooled weighted sums, which is exact and avoids an
    n_subj x n_cells loop.
    """
    ok = np.isfinite(mat) & np.isfinite(weights) & (weights > 0)
    w = np.where(ok, weights, 0.0)
    v = np.where(ok, mat, 0.0)
    tot_w = w.sum(axis=0)
    tot_wv = (w * v).sum(axis=0)
    den = tot_w[None, :] - w
    num = tot_wv[None, :] - w * v
    with np.errstate(invalid='ignore', divide='ignore'):
        return np.where(den > 0, num / den, np.nan)


def _corr(a, b, min_n):
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < min_n:
        return np.nan, int(ok.sum())
    x, y = a[ok], b[ok]
    if np.std(x) == 0 or np.std(y) == 0:
        return np.nan, int(ok.sum())
    return float(np.corrcoef(x, y)[0, 1]), int(ok.sum())


def band_blocked_vectors(s_cube, g_cube, center=False):
    """(subject vector, group vector, band block sizes), grouped band by band.

    ONE code path for the statistic AND its null. They were two before, and they
    disagreed: the observed statistic removed band means and the null then
    shuffled the flattened result ACROSS bands, while the docstring claimed a
    within-band shuffle. Returning the vectors band-blocked makes the correct
    shuffle the only convenient one.

    `center=True` additionally removes each band's mean, over the regions THIS
    subject has, so a subject with three regions is centred on their own three
    and not on the group's. That is the OLD route to a topography-only test and
    is kept for the record; the figure now uses `center=False` and lets the
    within-band shuffle hold the spectral tilt fixed instead (see
    `_within_block_null`), which gives a numerically identical verdict without
    modifying the data.

    A band where the subject has FEWER THAN TWO regions is dropped either way:
    with one region there is nothing for a within-band shuffle to permute, so
    the cell can never contribute to the null and must not contribute to the
    statistic. (Under `center=True` it is worse than useless -- centring a lone
    value on itself makes it exactly zero in both vectors, padding n while
    adding no covariance and dragging |r| toward zero.)
    """
    xs, gs, sizes = [], [], []
    for j in range(s_cube.shape[1]):
        ok = np.isfinite(s_cube[:, j]) & np.isfinite(g_cube[:, j])
        if ok.sum() < 2:
            continue
        sv, gv = s_cube[ok, j], g_cube[ok, j]
        if center:
            sv, gv = sv - sv.mean(), gv - gv.mean()
        xs.append(sv)
        gs.append(gv)
        sizes.append(int(ok.sum()))
    if not xs:
        return np.array([]), np.array([]), []
    return np.concatenate(xs), np.concatenate(gs), sizes


def _within_block_null(x, g, sizes, n_perm, rng):
    """97.5th pct of r under shuffling `g` WITHIN each band block.

    THIS IS THE FIGURE'S ONLY NULL, and the block constraint is what makes it
    the right one. Whatever structure survives every permutation cannot help the
    observed statistic beat the null. A within-band shuffle leaves each band's
    values in their own band, so the SHARED SPECTRAL TILT is present in every
    draw and cancels out of the comparison; the only thing randomised is which
    REGION carries which value. The null therefore means "correct spectrum,
    random anatomy", and beating it is a claim about topography alone.

    A free shuffle across bands would destroy the tilt in the permuted vector,
    making the null mean "no spectrum AND no anatomy" -- which the observed value
    beats if EITHER is present. Measured on this run that leak is large: the free
    bound is 0.235 against 0.405 for the within-band bound, 72% lower, and 10 of
    51 subjects clear the free bound while failing the correct one. That is so
    even though the tilt is modest in absolute terms (a 0.019 log10 swing from
    delta to beta, about 1.08x the between-subject SD within a band) -- a
    correlation over ~100 cells responds to COHERENCE, not magnitude, and a small
    shared shape pointing the same way in every region is a great deal of
    consistent covariance.

    The blocks also differ in SPREAD (delta slopes dwarf high-gamma ones), so
    shuffling across them would decouple the subject's high-variance bands from
    the group's and narrow the null on that count as well.
    """
    if len(x) < MIN_CELLS_FOR_R or not sizes:
        return np.nan
    blocks, off = [], 0
    for n in sizes:
        blk = np.tile(g[off:off + n], (n_perm, 1))
        rng.permuted(blk, axis=1, out=blk)
        blocks.append(blk)
        off += n
    return float(np.percentile(_rowwise_corr(x, np.concatenate(blocks, axis=1)),
                               97.5))


def subject_similarity(mat, weights, regions, bands, n_perm=N_PERM, seed=1):
    """Per-subject correlation with the leave-one-out group map. ONE null.

    THE PRIMARY PAIR IS `r_map` AND `r_map_null_p975`, and it is all the figure
    draws. `r_map` is the plain correlation between this subject's slopes and the
    group's, over every cell in a band where the subject has at least two
    regions -- nothing subtracted, nothing reweighted. Its null shuffles the
    group's values WITHIN BAND, which holds the shared spectral tilt fixed and
    randomises only which region carries which value, so beating it is a claim
    about ANATOMY rather than about spectrum. See `_within_block_null`.

    Two secondary pairs are computed and written to the table, not plotted,
    because the choice between them is a judgement a reader may want to audit:

      r_full / r_full_null_p975      raw statistic, FREE shuffle. The tilt is
                                     left in the test, and it leaks badly -- 10
                                     of 51 subjects pass here and fail `r_map`.
                                     Kept only to document the size of the leak.
      r_region / r_region_null_p975  band means removed, within-band shuffle.
                                     The other route to a topography-only test;
                                     it agrees with `r_map` exactly (28 of 51 on
                                     this run), which is why the simpler `r_map`
                                     is the one shown.

    Every bound is PER SUBJECT, never pooled: `n_cells` runs 18 to 102 here and
    the sampling distribution of r is far wider at 18 than at 102.
    """
    rng = np.random.default_rng(seed)
    n_subj = mat.shape[0]
    n_reg, n_band = len(regions), len(bands)
    loo = loo_group_map(mat, weights)

    rows = []
    for i in range(n_subj):
        s, g = mat[i], loo[i]
        S, G = s.reshape(n_reg, n_band), g.reshape(n_reg, n_band)

        # --- primary: raw statistic, within-band null
        xm, gm, sizes = band_blocked_vectors(S, G, center=False)
        r_map, n_map = _corr(xm, gm, MIN_CELLS_FOR_R)
        null_map = (_within_block_null(xm, gm, sizes, n_perm, rng)
                    if np.isfinite(r_map) else np.nan)

        # --- secondary: the tilt-inclusive version, to document the leak
        r_full, n_full = _corr(s, g, MIN_CELLS_FOR_R)
        null_full = np.nan
        if np.isfinite(r_full):
            ok = np.isfinite(s) & np.isfinite(g)
            gp = np.tile(g[ok], (n_perm, 1))
            rng.permuted(gp, axis=1, out=gp)
            null_full = float(np.percentile(_rowwise_corr(s[ok], gp), 97.5))

        # --- secondary: the band-mean-removal route to the same verdict
        xr, gr, sizes_r = band_blocked_vectors(S, G, center=True)
        r_reg, n_reg_cells = _corr(xr, gr, MIN_CELLS_FOR_R)
        null_reg = (_within_block_null(xr, gr, sizes_r, n_perm, rng)
                    if np.isfinite(r_reg) else np.nan)

        rows.append({'r_map': r_map, 'n_cells': n_map,
                     'r_map_null_p975': null_map, 'n_bands': len(sizes),
                     'r_full': r_full, 'n_cells_full': n_full,
                     'r_full_null_p975': null_full,
                     'r_region': r_reg, 'n_cells_region': n_reg_cells,
                     'r_region_null_p975': null_reg,
                     'n_regions': int(np.isfinite(S).any(axis=1).sum())})
    return pd.DataFrame(rows)


def _rowwise_corr(x, Y):
    """Pearson r between one vector and every row of Y. Used for the null only."""
    xc = x - x.mean()
    Yc = Y - Y.mean(axis=1, keepdims=True)
    denom = np.sqrt((xc ** 2).sum() * (Yc ** 2).sum(axis=1))
    with np.errstate(invalid='ignore', divide='ignore'):
        return np.where(denom > 0, (Yc @ xc) / denom, np.nan)


def network_scores(slopes, subjects):
    """Per-subject mean slope in the two anatomical halves of the pattern.

    Means, not weighted means: the question is "does this patient show the
    pattern", and an inverse-variance weight would let one well-sampled region
    speak for the whole network.
    """
    lim = slopes[(slopes['band'] == 'delta')
                 & slopes['region'].isin(LIMBIC_DEFAULT)]
    sm = slopes[(slopes['band'] == 'beta') & slopes['region'].isin(SENSORIMOTOR)]
    out = pd.DataFrame({'subject': subjects}).set_index('subject')
    for name, d in (('limbic_delta', lim), ('sensorimotor_beta', sm)):
        g = d.dropna(subset=['slope']).groupby('subject')['slope']
        out[name] = g.mean()
        out[f'n_{name}'] = g.size()
    return out.reset_index().fillna({'n_limbic_delta': 0, 'n_sensorimotor_beta': 0})


# ============================================================================
# FIGURES
# ============================================================================

def _footnote(fig, text):
    fig.text(0.01, 0.005, text + '\n' + DISCLAIMER, fontsize=6.5, va='bottom',
             ha='left', color='0.35', wrap=True)


def fig_signmap(ctx, out):
    import matplotlib.pyplot as plt
    from ieeg_ehr.features import common

    regions, bands, edges = ctx['regions'], ctx['bands'], ctx['edges']
    cc = ctx['cell_consistency']
    grid = cc.pivot(index='region', columns='band', values='excess_sign',
                          ).reindex(index=regions, columns=bands)
    frac = cc.pivot(index='region', columns='band', values='frac_sign',
                          ).reindex(index=regions, columns=bands)
    nn = cc.pivot(index='region', columns='band', values='n_with_slope',
                        ).reindex(index=regions, columns=bands)
    match = cc.pivot(index='region', columns='band', values='n_sign_match',
                           ).reindex(index=regions, columns=bands)
    sig = (cc.pivot(index='region', columns='band', values='p_bh_reject',
                          ).reindex(index=regions, columns=bands)
           .fillna(0).astype(bool))

    arr = grid.to_numpy(dtype=float)
    cap = float(np.nanmax(np.abs(arr))) or 0.1
    # Purple/green, NOT the RdBu_r of fig_band_map. The two maps share a grid and
    # a reader will flick between them; giving agreement its own palette makes it
    # impossible to mistake "most subjects agree" for "power went up".
    cm = plt.get_cmap('PRGn').copy()
    cm.set_bad('0.85')

    fig, ax = plt.subplots(figsize=(8.2, 0.42 * len(regions) + 3.2))
    im = ax.imshow(arr, aspect='auto', cmap=cm, vmin=-cap, vmax=cap,
                   interpolation='nearest')
    common.draw_mask_outline(ax, sig.to_numpy())

    for i in range(len(regions)):
        for j in range(len(bands)):
            k, n = match.iat[i, j], nn.iat[i, j]
            if not np.isfinite(k) or not n:
                continue
            # Flip to white once the cell is dark enough to swallow dark text.
            # The strongest cells are exactly the ones a reader wants the count
            # for, so leaving them unreadable defeats the annotation.
            dark = abs(float(arr[i, j])) > 0.62 * cap
            ax.text(j, i, f'{int(k)}/{int(n)}\n{frac.iat[i, j]:.0%}',
                    ha='center', va='center', fontsize=5.6,
                    color='white' if dark else '0.15')

    ax.set_xticks(range(len(bands)))
    ax.set_xticklabels([band_label(b, edges) for b in bands], fontsize=8)
    ax.set_yticks(range(len(regions)))
    ax.set_yticklabels(regions, fontsize=8)
    n_beat = int((cc['p_signflip'] < 0.05).sum())
    ax.set_title('Do individual subjects share the group\'s sign?\n'
                 f'colour = agreement ABOVE the sign-flip null; {n_beat} of '
                 f'{len(cc)} cells beat that null at p<0.05 (uncorrected)',
                 fontsize=11)
    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.03,
                 label='observed - null fraction of subjects sharing group sign')
    fig.tight_layout(rect=(0, 0.13, 1, 1))
    _footnote(fig,
              'Black outlines are the SAME BH-significant cells as fig_band_map, so the two '
              'grids overlay. Text in each cell is the raw count: subjects whose own unpooled '
              'OLS slope has the sign of the group fixed effect, out of subjects with a '
              'fittable slope there. COLOUR IS NOT THAT FRACTION -- it is the fraction minus '
              'the cell\'s own sign-flip null. That subtraction is the point of the figure: '
              'the group sign is estimated from the very subjects being counted, so it is '
              'pulled toward whichever sign the majority already has, and under a true null '
              f'the expected agreement is ABOVE one half (here it averages '
              f'{cc["null_frac_mean"].mean():.2f}). A binomial test against 0.5 would '
              'therefore call noise significant. The null flips each subject\'s slope sign at '
              f'random {N_PERM:,} times and re-derives the group sign as the sign of the '
              'inverse-variance weighted mean -- a cheap stand-in for refitting the mixed '
              'model on the flipped data. A subject casts a full vote no matter how noisy '
              'their slope is, which is what makes the caterpillar figure the necessary '
              'companion: sign counting discards exactly the precision that the forest plot '
              'shows.')
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)


def fig_percent(ctx, out):
    """The RAW percentage of subjects sharing the group's sign, per region x band.

    THE LITERAL QUESTION, drawn literally: what fraction of the patients with a
    fittable slope in this cell slope the way the group does. `fig_signmap`
    answers the DEFENSIBLE version of the same question and is the one to quote
    -- it subtracts each cell's own sign-flip null, because the group sign is
    estimated from the subjects being counted and so is dragged toward whichever
    sign the majority already has, which puts the null expectation well above
    one half. But the raw percentage is what a reader wants to SEE, and reading
    it off the annotations of a map coloured by something else is a figure
    fighting its own caption.

    So both exist, and this one is built to make its own limitation visible
    rather than to hide it:

      - THE COLOUR SCALE STARTS AT THE NULL, not at 0 and not at 0.5. The floor
        is the mean sign-flip null across cells, so a cell rendered at the
        bottom of the scale is a cell at chance -- not a cell at zero, which no
        cell can be. Anything below the null is clipped to the floor and marked,
        because "worse than chance agreement" is noise, not a finding.
      - Each cell's OWN null is annotated under the percentage, since it varies
        with n.
      - The BH-significant outlines are the same ones as every other map in the
        run, so the three overlay.
    """
    import matplotlib.pyplot as plt
    from ieeg_ehr.features import common

    regions, bands, edges = ctx['regions'], ctx['bands'], ctx['edges']
    cc = ctx['cell_consistency']

    def grid_of(col):
        return (cc.pivot(index='region', columns='band', values=col)
                .reindex(index=regions, columns=bands))

    frac = grid_of('frac_sign')
    nn = grid_of('n_with_slope')
    match = grid_of('n_sign_match')
    null = grid_of('null_frac_mean')
    sig = grid_of('p_bh_reject').fillna(0).astype(bool)

    arr = frac.to_numpy(dtype=float)
    floor = float(np.nanmean(null.to_numpy(dtype=float)))
    if not np.isfinite(floor):
        floor = 0.5
    # Sequential, NOT diverging: this quantity has a floor and a ceiling and no
    # meaningful midpoint, so a diverging map would invent one. Different hue
    # family again from both fig_band_map (RdBu_r) and fig_signmap (PRGn), so
    # three maps on the same grid can never be confused for each other.
    cm = plt.get_cmap('cividis').copy()
    cm.set_bad('0.85')

    fig, ax = plt.subplots(figsize=(8.6, 0.42 * len(regions) + 3.4))
    im = ax.imshow(np.clip(arr, floor, 1.0), aspect='auto', cmap=cm,
                   vmin=floor, vmax=1.0, interpolation='nearest')
    common.draw_mask_outline(ax, sig.to_numpy())

    n_below = 0
    for i in range(len(regions)):
        for j in range(len(bands)):
            k, n = match.iat[i, j], nn.iat[i, j]
            if not np.isfinite(k) or not n:
                continue
            f = float(arr[i, j])
            below = f < floor
            n_below += int(below)
            shade = (f - floor) / max(1.0 - floor, 1e-9)
            ax.text(j, i,
                    f'{f:.0%}{"*" if below else ""}\n{int(k)}/{int(n)}\n'
                    f'null {null.iat[i, j]:.0%}',
                    ha='center', va='center', fontsize=5.2,
                    color='white' if shade < 0.55 else '0.12')

    ax.set_xticks(range(len(bands)))
    ax.set_xticklabels([band_label(b, edges) for b in bands], fontsize=8)
    ax.set_yticks(range(len(regions)))
    ax.set_yticklabels(regions, fontsize=8)
    ax.set_title('What PERCENTAGE of subjects slope the way the group does?\n'
                 f'colour = raw fraction, scale floored at the mean sign-flip '
                 f'null ({floor:.0%}), not at 0 or 50%',
                 fontsize=11)
    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.03,
                 label='fraction of subjects sharing the group sign')
    fig.tight_layout(rect=(0, 0.14, 1, 1))
    _footnote(fig,
              'Each cell: the raw percentage, then the count (subjects whose own unpooled OLS '
              'slope has the sign of the group fixed effect, out of subjects with a fittable '
              'slope there), then that cell\'s own sign-flip null. READ THE PERCENTAGE '
              'AGAINST THE NULL, NEVER AGAINST 50%. The group sign is estimated from the very '
              'subjects being counted, so it is pulled toward whichever sign the majority '
              'already has and chance agreement sits well above one half -- here it averages '
              f'{floor:.0%}. That is why the colour scale is floored there rather than at 0.5: '
              'a cell at the bottom of this scale is a cell AT CHANCE. '
              + (f'{n_below} cell(s) fall BELOW their null and are clipped to the floor and '
                 'marked *; below-chance agreement is noise, not a reversed effect. '
                 if n_below else '')
              + 'fig_consistency_signmap is the same data with each cell\'s own null '
              'SUBTRACTED, and it is the version to quote for a claim -- this one is the '
              'version to look at. Black outlines are the same BH-significant cells as '
              'fig_band_map, so all three grids overlay. A subject casts a full vote however '
              'noisy their slope is, which is what makes the forest and caterpillar figures '
              'the necessary companions: sign counting discards precision by construction.')
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)


def fig_matrix(ctx, out):
    import matplotlib.pyplot as plt

    regions, bands, edges = ctx['regions'], ctx['bands'], ctx['edges']
    mat, subjects = ctx['mat'], ctx['subjects']
    order = ctx['subject_order']
    cells = ctx['cells']
    n_reg, n_band = len(regions), len(bands)
    cube = mat.reshape(len(subjects), n_reg, n_band)[order]
    labels = [subjects[i] for i in order]

    cap = float(np.nanpercentile(np.abs(cube), 98)) or 0.05
    cm = plt.get_cmap('RdBu_r').copy()
    cm.set_bad('0.88')

    ncol = 3
    nrow = int(np.ceil(n_band / ncol))
    fig, axes = plt.subplots(nrow, ncol, squeeze=False,
                             figsize=(4.1 * ncol, 0.155 * len(labels) * nrow + 2.6))
    beta = cells.set_index(['region', 'band'])['beta_nrs_within']
    rej = (cells.assign(r=cells['p_bh_reject'].fillna(False).astype(bool))
           .set_index(['region', 'band'])['r'])

    for k, band in enumerate(bands):
        ax = axes[k // ncol][k % ncol]
        im = ax.imshow(cube[:, :, k], aspect='auto', cmap=cm, vmin=-cap, vmax=cap,
                       interpolation='nearest')
        # The group result as marker glyphs along the top, NOT as a second colour
        # scale: subject slopes run an order of magnitude larger than the fixed
        # effect, so sharing a colourbar would flatten the panel to white.
        for x, region in enumerate(regions):
            b = float(beta.get((region, band), np.nan))
            if not np.isfinite(b) or b == 0:
                continue
            is_sig = bool(rej.get((region, band), False))
            ax.plot(x, -0.9, marker='^' if b > 0 else 'v',
                    color=POS_COLOR if b > 0 else NEG_COLOR,
                    ms=5 if is_sig else 3.2,
                    mfc=(POS_COLOR if b > 0 else NEG_COLOR) if is_sig else 'white',
                    mew=0.8, clip_on=False)
        ax.set_ylim(len(labels) - 0.5, -1.6)
        ax.set_xticks(range(n_reg))
        ax.set_xticklabels(regions, fontsize=5.6, rotation=90)
        if k % ncol == 0:
            ax.set_yticks(range(len(labels)))
            ax.set_yticklabels(labels, fontsize=4.2)
        else:
            ax.set_yticks([])
        ax.set_title(band_label(band, edges).replace('\n', '  '), fontsize=9)
    for j in range(n_band, nrow * ncol):
        axes[j // ncol][j % ncol].set_visible(False)

    fig.colorbar(im, ax=axes, fraction=0.016, pad=0.02,
                 label='subject\'s own OLS slope: d log10 band power per pain point')
    fig.suptitle('Every subject, every region, every band -- no averaging\n'
                 'rows sorted by similarity to the leave-one-out group map '
                 '(most typical patient at the top)', fontsize=12)
    _footnote(fig,
              'One row per subject, one column per region, one panel per band; grey is a '
              'region that subject has no electrodes in, or has too few distinct pain scores '
              'to fit a slope in. Colour is the subject\'s OWN unpooled OLS slope, on a scale '
              f'clipped at the 98th percentile of |slope| ({cap:.3f}) so a handful of extreme '
              'cells cannot flatten the rest. Triangles above each column are the GROUP fixed '
              'effect for that cell -- up/red for a rise with pain, down/blue for a fall, '
              'filled when BH-significant -- drawn as glyphs rather than a second colour row '
              'because subject slopes run roughly an order of magnitude larger than the fixed '
              'effect and a shared scale would wash the panel out. The figure is deliberately '
              'unsummarised: a column that looks mostly blue under a blue triangle is a '
              'consistent effect, a column that is half and half under a strongly coloured '
              'triangle is a group average over disagreeing patients, and only an '
              'unaggregated view can tell those apart. Row order is shared across all six '
              'panels, so a patient who is typical in delta can be checked in beta by eye.')
    fig.savefig(out, dpi=170, bbox_inches='tight')
    plt.close(fig)


def _draw_tilt_panel(ax, ctx, title):
    """Panel (a) of either profile version: the per-subject spectral tilt.

    Shared rather than copied, because the two versions of the profile figure
    exist side by side for comparison and a divergence between their (a) panels
    would be a difference the reader would wrongly attribute to the version.
    Returns the count of lines that leave the clipped view, for the caller's
    title.
    """
    regions, bands, edges = ctx['regions'], ctx['bands'], ctx['edges']
    mat, subjects, cells = ctx['mat'], ctx['subjects'], ctx['cells']
    n_reg, n_band = len(regions), len(bands)
    cube = mat.reshape(len(subjects), n_reg, n_band)

    with np.errstate(invalid='ignore'):
        prof = np.nanmean(cube, axis=1)                      # (subj, band)
    x = np.arange(n_band)

    # The y-range is set by the BULK, not by the extremes. One subject at -0.078
    # was taking ~40% of the height and compressing every other line into a
    # band, which is the opposite of what a spaghetti plot is for. Lines still
    # draw and clip at the frame; the count that leaves the view is stated, so
    # nothing is hidden silently.
    finite = prof[np.isfinite(prof)]
    lo, hi = np.percentile(finite, [2.5, 97.5])
    pad = 0.18 * (hi - lo)
    ylim = (lo - pad, hi + pad)
    n_clipped = int(np.sum(np.any((prof < ylim[0]) | (prof > ylim[1]), axis=1)))

    for i in range(len(subjects)):
        if np.isfinite(prof[i]).sum() < 2:
            continue
        ax.plot(x, prof[i], lw=0.7, alpha=0.32, color='0.55', zorder=2)

    # IQR ribbon: the population summary the spaghetti cannot give on its own.
    with np.errstate(invalid='ignore'):
        q25, med, q75 = (np.nanpercentile(prof, 25, axis=0),
                         np.nanmedian(prof, axis=0),
                         np.nanpercentile(prof, 75, axis=0))
    ax.fill_between(x, q25, q75, color=ACCENT_B, alpha=0.22, lw=0, zorder=3,
                    label='subject IQR')
    ax.plot(x, med, lw=2.2, color=ACCENT_B, zorder=5, label='median subject')

    grp = (cells.set_index(['region', 'band'])['beta_nrs_within']
           .unstack().reindex(index=regions, columns=bands).to_numpy(dtype=float))
    with np.errstate(invalid='ignore'):
        grp_prof = np.nanmean(grp, axis=0)
    ax.plot(x, grp_prof, lw=2.6, color='black', zorder=6,
            label='group fixed effect (mean over regions)')

    ax.axhline(0, color='0.75', lw=0.8, zorder=1)
    ax.set_xticks(x)
    ax.set_xticklabels([band_label(b, edges) for b in bands], fontsize=7.5)
    ax.set_ylabel('mean slope across that subject\'s regions\n'
                  '$\\Delta$ log10 band power per NRS point', fontsize=9)
    ax.set_ylim(*ylim)
    ax.set_xlim(-0.25, n_band - 0.75)
    # Say which way is which, on the axis, so nobody has to derive the sign of a
    # log-power slope while standing in front of the figure.
    ax.text(0.015, 0.975, '↑ power INCREASES with pain',
            transform=ax.transAxes, fontsize=7.5, va='top', ha='left',
            color='0.30')
    ax.text(0.015, 0.025, '↓ power DECREASES with pain',
            transform=ax.transAxes, fontsize=7.5, va='bottom', ha='left',
            color='0.30')
    ax.set_title(f'{title}  (y clipped to the middle 95%; {n_clipped} '
                 f'line{"s" if n_clipped != 1 else ""} leave the view)',
                 fontsize=9.5)
    ax.legend(fontsize=7, loc='lower right')
    return n_clipped


def fig_profile(ctx, out):
    """ONE statistic, ONE null: the ranked per-subject topography test.

    THE CURRENT VERSION. `fig_profile_v2` keeps the previous two-statistic
    scatter for side-by-side comparison; see its docstring for why this one
    replaced it.
    """
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    subjects = ctx['subjects']
    sim = ctx['similarity']

    fig, axes = plt.subplots(1, 2, figsize=(13.6, 6.6),
                             gridspec_kw={'width_ratios': [1.0, 1.08]})

    _draw_tilt_panel(axes[0], ctx,
                     '(a) The spectral tilt the null in (b) HOLDS FIXED')

    # --- (b) ONE statistic, ONE null, ranked
    ax = axes[1]
    d = (sim.dropna(subset=['r_map', 'r_map_null_p975'])
         .sort_values('r_map', ascending=False).reset_index(drop=True))
    rv = d['r_map'].to_numpy(dtype=float)
    bv = d['r_map_null_p975'].to_numpy(dtype=float)
    y = np.arange(len(d))
    beats = rv > bv

    def _area(n):
        return np.clip(np.asarray(n, dtype=float) * 1.6, 12, 150)

    # Each subject's OWN null zone, drawn as a bar from 0 to their bound. This
    # is the whole reason the panel is per-subject rather than a single
    # threshold line: the bound depends on how many cells the patient
    # contributes, and it ranges widely here. A dot past the end of its own bar
    # beats its own null -- readable row by row, with no mental arithmetic and no
    # pooled bound standing in for 51 different ones.
    ax.barh(y, bv, height=0.72, color='0.87', edgecolor='0.68', lw=0.5,
            zorder=1)
    ax.scatter(rv[beats], y[beats], s=_area(d['n_cells'])[beats], color=ACCENT_B,
               alpha=0.90, edgecolors='white', lw=0.6, zorder=4)
    ax.scatter(rv[~beats], y[~beats], s=_area(d['n_cells'])[~beats],
               color=ACCENT_A, alpha=0.70, edgecolors='white', lw=0.6, zorder=3)
    ax.axvline(0, color='0.45', lw=0.9, zorder=2)

    med_r = float(np.median(rv))
    ax.axvline(med_r, color='0.30', lw=1.2, ls=':', zorder=2)

    ax.set_ylim(len(d) - 0.5, -0.5)
    xl = (min(0.0, float(rv.min())) - 0.06, max(float(rv.max()), float(bv.max())) + 0.06)
    ax.set_xlim(*xl)
    ax.text(med_r + 0.012 * (xl[1] - xl[0]), len(d) - 1.0,
            f'median r = {med_r:.2f}', fontsize=7.4, color='0.30', va='bottom',
            ha='left', rotation=90)
    # Rank order is the information; sub-131 tells a viewer nothing and 51
    # labels would consume the whole axis.
    ax.set_yticks([])
    ax.set_ylabel(f'{len(d)} subjects, ranked by r', fontsize=9)
    ax.set_xlabel('r between this subject\'s slope map and the leave-one-out '
                  'group map', fontsize=9)
    ax.set_title(f'(b) {int(beats.sum())}/{len(d)} subjects beat their OWN '
                 f'within-band null  (anatomy, not spectrum)', fontsize=9.5)
    ax.legend(handles=[
        Line2D([], [], ls='', marker='o', ms=8, mfc=ACCENT_B, mec='white',
               label='beats own null'),
        Line2D([], [], ls='', marker='o', ms=8, mfc=ACCENT_A, mec='white',
               label='inside own null'),
        Patch(facecolor='0.87', edgecolor='0.68',
              label='that subject\'s own 97.5% null bound'),
        Line2D([], [], ls='', marker='o', ms=np.sqrt(_area(20)),
               mfc='0.55', mec='white', label='20 cells'),
        Line2D([], [], ls='', marker='o', ms=np.sqrt(_area(60)),
               mfc='0.55', mec='white', label='60 cells'),
        Line2D([], [], ls='', marker='o', ms=np.sqrt(_area(100)),
               mfc='0.55', mec='white', label='100 cells')],
        # Kept SHORT and slightly transparent on purpose. The data traces a
        # monotone band from lower-left to upper-right, so the lower-right corner
        # is the only empty one -- but a tall box there reaches up into the
        # mid-rank rows and hides the very bar-ends the panel asks you to compare
        # each dot against.
        fontsize=6.5, loc='lower right', labelspacing=0.5, borderpad=0.55,
        handletextpad=0.5, framealpha=0.92)

    n_free = int((sim['r_full'] > sim['r_full_null_p975']).sum())
    fig.suptitle('Does an individual patient show THE pattern?', fontsize=13)
    fig.text(0.5, 0.925,
             f'{ctx["n_assessed"]} subjects with pain assessments  ->  '
             f'{ctx["n_eligible"]} met the run\'s eligibility filter  ->  '
             f'{len(subjects)} have a fittable slope in at least one cell and '
             f'appear here  ->  {len(d)} enter panel (b)',
             fontsize=8.5, ha='center', va='top', color='0.35')
    fig.tight_layout(rect=(0, 0.20, 1, 0.905))
    _footnote(fig,
              'ONE STATISTIC, ONE NULL. Panel (b) is the plain correlation between a subject\'s '
              'own slopes and the group\'s -- nothing subtracted, nothing reweighted -- against '
              'a single permutation null. The null holds the subject\'s vector FIXED and '
              f'permutes the GROUP vector {N_PERM:,} times WITHIN BAND: a delta value can only '
              'move to another delta cell. That constraint is the entire design. Whatever '
              'survives every permutation cannot help the observed value beat the null, so '
              'leaving each band\'s values inside their own band means the shared spectral tilt '
              'in panel (a) is present in every draw and cancels out; the only thing randomised '
              'is WHICH REGION carries which value. The null therefore reads "correct spectrum, '
              'random anatomy", and beating it is a claim about ANATOMY. '
              'WHY THE TILT IS HELD FIXED RATHER THAN TESTED: it is modest in size -- a 0.019 '
              'log10 swing from delta to beta, about 1.08x the between-subject SD within a band '
              '-- but it is COHERENT, pointing the same way in nearly every region of nearly '
              'every patient, and a correlation over ~100 cells responds to coherence rather '
              'than to magnitude. Left free to be tested it dominates: the same statistic '
              f'against a FREE cross-band shuffle has a median bound of 0.24 against 0.41 here, '
              f'and {n_free} of {len(sim)} subjects clear the free bound while '
              f'{int(beats.sum())} clear the correct one. The free version is kept in '
              'consistency_subjects.csv (r_full) only to document the size of that gap, and an '
              'equivalent route -- subtracting band means and then shuffling within band '
              '(r_region) -- gives an identical verdict, which is why the simpler untouched '
              'statistic is the one plotted. '
              'THE BOUND IS PER SUBJECT. The grey bar in each row is THAT patient\'s own 97.5th '
              f'percentile, not a pooled threshold: it is wider for a patient contributing '
              f'{int(d["n_cells"].min())} cells than for one contributing '
              f'{int(d["n_cells"].max())}, which is why the panel is 51 rows with 51 bars '
              'instead of one vertical line. A dot past the end of its own bar beats its own '
              'null. Subject labels are omitted deliberately -- an anonymised ID tells a viewer '
              'nothing and the rank order is the information. '
              'The reference map is LEAVE-ONE-OUT: each subject is correlated against a group '
              'estimate rebuilt from the OTHER subjects, because with that subject in the '
              'reference they would be partly correlated with themselves and the bias is always '
              'positive. Cells in a band where the subject has fewer than two regions are '
              'dropped, since a within-band shuffle has nothing to permute there. '
              'ATTRITION is printed above the panels; the '
              f'>={MIN_CELLS_FOR_R}-cell floor excludes NOBODY in this run -- the smallest '
              f'contribution is {int(d["n_cells"].min())} cells -- so every subject lost was '
              'lost upstream by the run\'s own eligibility filter (too few pain reports, too '
              'narrow a range, too few non-modal scores), not by this figure. Panel (a) averages '
              'each subject\'s slopes over the regions they happen to have electrodes in, so a '
              'line is a SPECTRAL claim and not a regional one; the black line is the group '
              'fixed effect averaged the same way, and is flatter than the subject lines because '
              'it is a partially pooled estimate rather than a mean of noisy ones. Electrode '
              'coverage is not random across patients, so a low r can mean "this patient is '
              'atypical" or "this patient has three regions and two of them are '
              'quasi-controls" -- which is what the size encoding is for.')
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)


def fig_profile_v2(ctx, out):
    """THE SUPERSEDED TWO-STATISTIC SCATTER, kept for side-by-side comparison.

    Retained deliberately rather than deleted: the move to a single null was a
    judgement call, not a bug fix, and a reader comparing the two should be able
    to see what was given up. What this version shows that `fig_profile` cannot
    is the DISSOCIATION -- a patient can sit at r=0.7 on the spectral tilt and
    near zero on the regional topography, and here that is a point in the lower
    right rather than a number in a footnote.

    Why it was superseded: the x axis is a test whose null is nearly free to
    pass. A free cross-band shuffle lets any patient with some delta-down /
    beta-up shape through, and measured on this run 10 of 51 clear it while
    failing the blocked null that actually tests anatomy. Putting that on an
    equal footing with the y axis invites a reader to average the two, and the
    average is dominated by the weaker claim.

    THE NUMBERS HERE ARE THE CORRECTED ONES, not the ones this panel showed when
    it was first drawn. Two real defects were fixed in between and are not
    reproduced: the y-axis null shuffled ACROSS bands rather than within, which
    made the bound ~11% too narrow, and the colouring compared every subject to
    the MEDIAN bound rather than their own. Reproducing known-wrong numbers for
    the sake of a faithful diff would make the comparison misleading, so the
    only difference between this figure and `fig_profile` is presentation.
    """
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch, Rectangle

    subjects = ctx['subjects']
    sim = ctx['similarity']

    fig, axes = plt.subplots(1, 2, figsize=(13.6, 6.6),
                             gridspec_kw={'width_ratios': [1.0, 1.08]})

    _draw_tilt_panel(axes[0], ctx, '(a) Spectral tilt, one line per subject')

    # --- (b) tilt against topography
    ax = axes[1]
    d = sim.dropna(subset=['r_full', 'r_region'])
    xv = d['r_full'].to_numpy(dtype=float)
    yv = d['r_region'].to_numpy(dtype=float)
    # Each axis carries its own median bound, because the two shuffles differ:
    # free across cells for x, within band for y.
    nx = float(np.nanmedian(d['r_full_null_p975']))
    ny = float(np.nanmedian(d['r_region_null_p975']))

    ax.add_patch(Rectangle((-nx, -ny), 2 * nx, 2 * ny, facecolor='0.87',
                           edgecolor='0.62', lw=0.9, ls='--', zorder=0))
    ax.axhline(0, color='0.45', lw=0.9, zorder=1)
    ax.axvline(0, color='0.45', lw=0.9, zorder=1)

    def _area(n):
        return np.clip(np.asarray(n, dtype=float) * 1.9, 14, 190)

    # Per-subject bounds for the colouring, as in `fig_profile`. The box is only
    # a guide; comparing everyone to the median flatters low-coverage patients
    # and penalises well-covered ones.
    beats_x = xv > d['r_full_null_p975'].to_numpy(dtype=float)
    beats_y = yv > d['r_region_null_p975'].to_numpy(dtype=float)
    beats_both = beats_x & beats_y
    ax.scatter(xv[beats_both], yv[beats_both], s=_area(d['n_cells'])[beats_both],
               color=ACCENT_B, alpha=0.80, edgecolors='white', lw=0.7, zorder=4)
    ax.scatter(xv[~beats_both], yv[~beats_both],
               s=_area(d['n_cells'])[~beats_both], color=ACCENT_A, alpha=0.62,
               edgecolors='white', lw=0.7, zorder=3)

    med_x, med_y = float(np.median(xv)), float(np.median(yv))
    ax.axvline(med_x, color='0.35', lw=1.2, ls=':', zorder=2)
    ax.axhline(med_y, color='0.35', lw=1.2, ls=':', zorder=2)

    xl = (min(xv.min(), -nx) - 0.10, max(xv.max(), nx) + 0.10)
    yl = (min(yv.min(), -ny) - 0.10, max(yv.max(), ny) + 0.16)
    ax.set_xlim(*xl)
    ax.set_ylim(*yl)
    ax.text(med_x + 0.012 * (xl[1] - xl[0]), yl[1] - 0.05 * (yl[1] - yl[0]),
            f'median {med_x:.2f}', fontsize=7.2, color='0.30', va='top',
            ha='left', rotation=90)
    ax.text(xl[0] + 0.015 * (xl[1] - xl[0]), med_y + 0.012 * (yl[1] - yl[0]),
            f'median {med_y:.2f}', fontsize=7.2, color='0.30', va='bottom',
            ha='left')

    q = {'++': int(((xv > 0) & (yv > 0)).sum()),
         '-+': int(((xv < 0) & (yv > 0)).sum()),
         '+-': int(((xv > 0) & (yv < 0)).sum()),
         '--': int(((xv < 0) & (yv < 0)).sum())}
    for key, (px, py, ha, va) in (
            ('++', (xl[1], yl[1], 'right', 'top')),
            ('-+', (xl[0], yl[1], 'left', 'top')),
            ('+-', (xl[1], yl[0], 'right', 'bottom')),
            ('--', (xl[0], yl[0], 'left', 'bottom'))):
        ax.text(px, py, f'{q[key]}', fontsize=17, color='0.62', ha=ha, va=va,
                fontweight='bold')

    ax.set_xlabel('FULL MAP r  (free shuffle -- spectral tilt dominates)',
                  fontsize=9)
    ax.set_ylabel('REGION PATTERN r  (band means removed, within-band shuffle)',
                  fontsize=9)
    n_tilt_only = int((beats_x & ~beats_y).sum())
    ax.set_title(f'(b) {int(beats_both.sum())}/{len(d)} subjects beat their OWN '
                 f'null on both axes; {n_tilt_only} show the tilt WITHOUT the '
                 f'topography', fontsize=9.5)
    ax.legend(handles=[
        Line2D([], [], ls='', marker='o', ms=8, mfc=ACCENT_B, mec='white',
               label='beats own null on both axes'),
        Line2D([], [], ls='', marker='o', ms=8, mfc=ACCENT_A, mec='white',
               label='beats own null on one or neither'),
        Patch(facecolor='0.87', edgecolor='0.62', ls='--',
              label=f'MEDIAN null box ({nx:.2f} x, {ny:.2f} y) -- guide only'),
        Line2D([], [], ls='', marker='o', ms=np.sqrt(_area(20)),
               mfc='0.55', mec='white', label='20 cells'),
        Line2D([], [], ls='', marker='o', ms=np.sqrt(_area(60)),
               mfc='0.55', mec='white', label='60 cells'),
        Line2D([], [], ls='', marker='o', ms=np.sqrt(_area(100)),
               mfc='0.55', mec='white', label='100 cells')],
        # Lifted clear of the bottom-right corner, which belongs to a quadrant
        # count -- flush against the corner the legend hides it.
        fontsize=6.5, loc='lower right', bbox_to_anchor=(1.0, 0.075),
        labelspacing=0.5, borderpad=0.55, handletextpad=0.5, framealpha=0.92)

    n_map = int((sim['r_map'] > sim['r_map_null_p975']).sum())
    fig.suptitle('Does an individual patient show THE pattern?   '
                 '[V2 -- SUPERSEDED, kept for comparison]', fontsize=13)
    fig.text(0.5, 0.925,
             f'{ctx["n_assessed"]} subjects with pain assessments  ->  '
             f'{ctx["n_eligible"]} met the run\'s eligibility filter  ->  '
             f'{len(subjects)} have a fittable slope in at least one cell and '
             f'appear here  ->  {len(d)} enter panel (b)',
             fontsize=8.5, ha='center', va='top', color='0.35')
    fig.tight_layout(rect=(0, 0.20, 1, 0.905))
    _footnote(fig,
              'VERSION 2, SUPERSEDED BY fig_consistency_profile.png -- KEPT SO THE TWO CAN BE '
              'COMPARED. What this version adds is the DISSOCIATION: x and y are two different '
              f'questions, and the {n_tilt_only} points low and to the right are patients who '
              'have the spectral tilt and NOT the regional topography -- visible here as a '
              'position, whereas the current figure can only state it as a number. What it '
              'costs is that the x axis is given equal billing despite being a far weaker test: '
              'its null is a FREE shuffle across all cells, which destroys the shared '
              'delta-down/beta-up shape in every permutation, so any patient carrying some '
              'version of that shape beats it whatever their anatomy does. On this run 10 of 51 '
              'clear the x null and fail the y null. Two axes of unequal strength invite a '
              'reader to average them, and the average is governed by the weaker one, which is '
              f'why the current figure plots the single blocked test ({n_map} of '
              f'{len(sim)} subjects) and reports the rest in consistency_subjects.csv. '
              'THE NUMBERS SHOWN ARE THE CORRECTED ONES, not what this panel displayed when it '
              'was first drawn. Two genuine defects were fixed in between and are deliberately '
              'NOT reproduced here: the y-axis null used to shuffle ACROSS bands rather than '
              'within it, making the bound about 11% too narrow, and the colouring used to '
              'compare every subject to the MEDIAN bound rather than to their own. Both are '
              'fixed in this rendering, so the only difference between this figure and the '
              'current one is how the same corrected statistics are presented. '
              'Everything else matches the current figure: the reference is a LEAVE-ONE-OUT '
              'group map so no subject is correlated against themselves; the box is the median '
              'of the per-axis bounds and is a guide only, while the colouring uses each '
              'subject\'s own bound; marker area is the number of cells the subject '
              'contributes; and the attrition chain is printed above the panels.')
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)


def fig_network(ctx, out):
    import matplotlib.pyplot as plt

    net = ctx['network']
    cells = ctx['cells']
    d = net[(net['n_limbic_delta'] > 0) & (net['n_sensorimotor_beta'] > 0)].copy()

    fig, axes = plt.subplots(1, 2, figsize=(13.0, 6.0),
                             gridspec_kw={'width_ratios': [1.35, 1.0]})
    ax = axes[0]
    x = d['limbic_delta'].to_numpy()
    y = d['sensorimotor_beta'].to_numpy()
    size = np.clip(np.minimum(d['n_limbic_delta'], d['n_sensorimotor_beta']) * 22,
                   18, 170)
    # The predicted quadrant is x<0 (limbic delta falls) and y>0 (sensorimotor
    # beta rises) -- the two halves of fig_band_map read as one syndrome.
    in_quad = (x < 0) & (y > 0)
    ax.scatter(x[in_quad], y[in_quad], s=size[in_quad], color=POS_COLOR, alpha=0.75,
               edgecolors='white', lw=0.6, zorder=3,
               label='both halves of the pattern')
    ax.scatter(x[~in_quad], y[~in_quad], s=size[~in_quad], color='0.55', alpha=0.7,
               edgecolors='white', lw=0.6, zorder=2, label='one half or neither')
    ax.axhline(0, color='0.4', lw=1.0, ls='--', zorder=1)
    ax.axvline(0, color='0.4', lw=1.0, ls='--', zorder=1)

    gb = cells.set_index(['region', 'band'])['beta_nrs_within']
    gx = float(np.nanmean([gb.get((r, 'delta'), np.nan) for r in LIMBIC_DEFAULT]))
    gy = float(np.nanmean([gb.get((r, 'beta'), np.nan) for r in SENSORIMOTOR]))
    ax.plot(gx, gy, marker='*', ms=22, color='black', zorder=5,
            label='group fixed effect')

    xl, yl = ax.get_xlim(), ax.get_ylim()
    counts = {'lower left': int(((x < 0) & (y < 0)).sum()),
              'upper left': int(in_quad.sum()),
              'lower right': int(((x > 0) & (y < 0)).sum()),
              'upper right': int(((x > 0) & (y > 0)).sum())}
    for (name, n), (px, py, ha, va) in zip(
            counts.items(),
            [(xl[0], yl[0], 'left', 'bottom'), (xl[0], yl[1], 'left', 'top'),
             (xl[1], yl[0], 'right', 'bottom'), (xl[1], yl[1], 'right', 'top')]):
        ax.text(px, py, f'{n}', fontsize=17, color='0.6', ha=ha, va=va,
                fontweight='bold')
    ax.set_xlim(xl)
    ax.set_ylim(yl)
    # The region lists live in the footnote, not on the axes: spelled out here
    # the y-label is longer than the axis is tall and overruns the figure.
    ax.set_xlabel('limbic / default DELTA: subject mean slope '
                  f'({len(LIMBIC_DEFAULT)} regions)', fontsize=9)
    ax.set_ylabel('sensorimotor + lateral PFC BETA: subject mean slope '
                  f'({len(SENSORIMOTOR)} regions)', fontsize=9)
    r, n_r = _corr(x, y, 5)
    ax.set_title(f'{int(in_quad.sum())}/{len(d)} subjects show BOTH halves; '
                 f'r = {r:+.2f} across {n_r} subjects', fontsize=10)
    handles, labels = ax.get_legend_handles_labels()

    # --- the two marginals, which say how much of the quadrant count is each axis
    ax = axes[1]
    for k, (col, colour, label) in enumerate((
            ('limbic_delta', NEG_COLOR, 'limbic/default delta'),
            ('sensorimotor_beta', POS_COLOR, 'sensorimotor/lPFC beta'))):
        v = net[col].dropna().to_numpy()
        pos = int((v > 0).sum())
        ax.scatter(v, np.full(len(v), k) + np.random.default_rng(k).normal(0, 0.055, len(v)),
                   s=26, color=colour, alpha=0.6, edgecolors='white', lw=0.5)
        ax.boxplot(v, positions=[k], vert=False, widths=0.34, showfliers=False,
                   medianprops=dict(color='black', lw=2))
        ax.text(0.01, k + 0.29, f'{label}: {len(v) - pos} of {len(v)} negative',
                transform=ax.get_yaxis_transform(), fontsize=8, color='0.3')
    ax.axvline(0, color='0.4', lw=1.0, ls='--')
    ax.set_yticks([0, 1])
    ax.set_yticklabels(['limbic\ndelta', 'sensorimotor\nbeta'], fontsize=8)
    ax.set_xlabel('subject mean slope: d log10 power per pain point')
    ax.set_title('Each half on its own', fontsize=10)

    fig.suptitle('One syndrome, or two subject subsets?\n'
                 'one point per subject: does the SAME patient show both halves '
                 'of the map?', fontsize=12)
    fig.tight_layout(rect=(0, 0.20, 1, 0.92))
    # Below the axes rather than inside them: every corner of the scatter is
    # either occupied by points or by a quadrant count.
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.30, 0.165),
               ncol=3, fontsize=8, frameon=False)
    _footnote(fig,
              f'Limbic/default delta = {", ".join(LIMBIC_DEFAULT)}. Sensorimotor/lateral '
              f'PFC beta = {", ".join(SENSORIMOTOR)}. '
              'THE TWO REGION SETS ARE FIXED ANATOMICALLY, NOT BY WHICH CELLS CAME OUT '
              'SIGNIFICANT. Choosing the cells by their p-value and then asking how many '
              'subjects support them is circular; these two sets are a statement about pain '
              'anatomy that could have been written before the model was fitted, which is '
              'what makes the quadrant count interpretable. A point is one subject\'s plain '
              'mean over the regions they have in each set (unweighted -- the question is '
              'whether the patient shows the pattern, and an inverse-variance weight would '
              'let one well-sampled region speak for the whole network), and marker area is '
              'the smaller of the two region counts. Only subjects with at least one region '
              'in EACH set appear in the scatter; the right panel keeps everyone, which is '
              'why its counts are larger. The upper-left quadrant is the prediction -- delta '
              'down, beta up, in the same patient. A scatter that fills the upper-left is one '
              'syndrome; two separate clouds on the axes would mean the map is an average '
              'over two different kinds of patient, and the group figure could not tell you '
              'which. The correlation is across subjects and is NOT a test of the pattern: '
              'negative r means patients with a deeper limbic delta drop also have a larger '
              'sensorimotor beta rise, which is a stronger claim than the quadrant count and '
              'rests on far fewer effective degrees of freedom.')
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)


def fig_influence(ctx, out):
    import matplotlib.pyplot as plt
    from ieeg_ehr.features import common

    regions, bands, edges = ctx['regions'], ctx['bands'], ctx['edges']
    cc = ctx['cell_consistency']
    grid = (cc.pivot(index='region', columns='band',
                           values='loso_max_rel_delta')
            .reindex(index=regions, columns=bands))
    flips = (cc.pivot(index='region', columns='band',
                            values='loso_flips_sign')
             .reindex(index=regions, columns=bands).fillna(0).astype(bool))
    sig = (cc.pivot(index='region', columns='band', values='p_bh_reject',
                          ).reindex(index=regions, columns=bands)
           .fillna(0).astype(bool))

    fig, axes = plt.subplots(1, 2, figsize=(14.5, 0.40 * len(regions) + 3.4),
                             gridspec_kw={'width_ratios': [1.15, 1.0]})

    ax = axes[0]
    arr = grid.to_numpy(dtype=float) * 100
    # The scale is set by the BH-SIGNIFICANT cells, which are the only ones the
    # figure is really about. Letting the null cells set vmax saturates most of
    # the map to black -- a cell whose estimate is near zero moves by hundreds of
    # percent for arithmetic reasons, and at vmax=100 that noise sets the scale
    # and hides the 20-40% range where the significant cells actually sit.
    sig_vals = arr[sig.to_numpy()]
    sig_vals = sig_vals[np.isfinite(sig_vals)]
    vmax = float(np.ceil(max(40.0, np.percentile(sig_vals, 95)
                             if len(sig_vals) else 40.0) / 10) * 10)
    cm = plt.get_cmap('magma_r').copy()
    cm.set_bad('0.85')
    im = ax.imshow(arr, aspect='auto', cmap=cm, vmin=0, vmax=vmax,
                   interpolation='nearest')
    common.draw_mask_outline(ax, sig.to_numpy())
    for i in range(len(regions)):
        for j in range(len(bands)):
            if flips.iat[i, j]:
                ax.text(j, i, 'x', ha='center', va='center', fontsize=11,
                        color='#39ff14', fontweight='bold')
    ax.set_xticks(range(len(bands)))
    ax.set_xticklabels([band_label(b, edges) for b in bands], fontsize=8)
    ax.set_yticks(range(len(regions)))
    ax.set_yticklabels(regions, fontsize=8)
    n_flip = int(cc['loso_flips_sign'].sum())
    n_flip_sig = int((cc['loso_flips_sign'] & cc['p_bh_reject']).sum())
    ax.set_title('Does one patient carry the cell?\n'
                 f'largest change from dropping a single subject; sign flips in '
                 f'{n_flip} cells ({n_flip_sig} of them BH-significant)',
                 fontsize=10)
    fig.colorbar(im, ax=ax, fraction=0.045, pad=0.03, extend='max',
                 label='max |change| from leaving one subject out (% of estimate)')

    # --- who the influential patients are
    ax = axes[1]
    sub = (cc[cc['p_bh_reject'] & (cc['loso_worst_subject'] != '')]
           .groupby('loso_worst_subject').size().sort_values())
    if len(sub):
        ax.barh(np.arange(len(sub)), sub.to_numpy(), color='0.55')
        ax.set_yticks(np.arange(len(sub)))
        ax.set_yticklabels(sub.index, fontsize=6.5)
        ax.set_xlabel('BH-significant cells where this subject is the most '
                      'influential one', fontsize=8.5)
        expected = int(cc['p_bh_reject'].sum()) / max(len(ctx['subjects']), 1)
        ax.axvline(expected, color=POS_COLOR, lw=1.4, ls='--',
                   label=f'even spread would be {expected:.2f} cells each')
        ax.legend(fontsize=7.5, loc='lower right')
        ax.set_title(f'{len(sub)} distinct subjects are the top influence across '
                     f'the {int(cc["p_bh_reject"].sum())} significant cells',
                     fontsize=10)
    else:
        ax.set_visible(False)

    fig.tight_layout(rect=(0, 0.15, 1, 1))
    _footnote(fig,
              'THIS IS A JACKKNIFE OF AN INVERSE-VARIANCE WEIGHTED MEAN OF THE UNPOOLED '
              'SUBJECT SLOPES, NOT A REFIT OF THE MIXED MODEL. It drops one subject, '
              'recomputes that weighted mean, and reports the largest change over all '
              'subjects as a percentage of the full-sample estimate; a green x marks a cell '
              'where some single subject\'s removal flips the sign of the estimate. The '
              'weighted mean has no channel random intercept and no partial pooling, so it '
              'moves MORE under deletion than the fitted model would -- read this as an upper '
              'bound on fragility, and refit the flagged cells before quoting any of it. '
              'Weights are 1/se^2 with the SE floored at the cell\'s own 5th percentile, '
              'because one subject with an implausibly small SE would otherwise be handed the '
              'entire cell and the figure would be reporting its own weighting scheme. A '
              'large percentage on a cell whose estimate is near zero is arithmetic, not '
              'fragility -- the denominator is small -- so read the outlined '
              'cells first and check them against fig_band_forest. For the same reason the '
              f'colour scale is capped at {vmax:.0f}% (the 95th percentile of the '
              'BH-significant cells; the arrow on the colourbar marks the clip) rather than '
              'at the map maximum: letting the near-zero null cells set the range saturates '
              'almost everything to black and hides the band the significant cells sit in. '
              'The right panel counts, '
              'for each BH-significant cell, which subject moves it most; the question it '
              'answers is whether influence is spread across the cohort or concentrated in a '
              'few patients who happen to have wide electrode coverage. Coverage is a known '
              'confound in this dataset, so concentration there is expected to some degree '
              'and is not by itself evidence of a problem.')
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)


BUILDERS = {'signmap': fig_signmap, 'percent': fig_percent,
            'matrix': fig_matrix, 'profile': fig_profile,
            'profile_v2': fig_profile_v2, 'network': fig_network,
            'influence': fig_influence}


# ============================================================================
# DRIVER
# ============================================================================

def build_context(run_dir, cells, slopes, params, inventory=None, n_perm=N_PERM,
                  seed=0):
    regions, bands, edges = axis_order(cells, params)
    subjects, keys, mat = slope_matrix(slopes, regions, bands)
    _, _, se = slope_matrix(slopes, regions, bands, value='se')

    # Weights for the LOO reference map, cell by cell so the SE floor is local.
    weights = np.zeros_like(se)
    for j in range(se.shape[1]):
        weights[:, j] = cell_weights(se[:, j], mat[:, j])

    cc = cell_consistency(cells, slopes, regions, bands, n_perm=n_perm, seed=seed)
    sim = subject_similarity(mat, weights, regions, bands, n_perm=n_perm,
                             seed=seed + 1)
    sim.insert(0, 'subject', subjects)
    net = network_scores(slopes, subjects)
    sim = sim.merge(net, on='subject', how='left')

    # One row order for every panel of the matrix figure: most typical patient
    # first. NaN r (too few cells) sorts last rather than being dropped -- those
    # subjects still have rows worth looking at.
    key = sim['r_map'].to_numpy(dtype=float)
    order = np.argsort(np.where(np.isfinite(key), -key, np.inf), kind='stable')

    n_assessed = len(inventory) if inventory is not None else len(subjects)
    n_eligible = (int(inventory['included'].sum())
                  if inventory is not None and 'included' in inventory
                  else len(subjects))

    return {'run_dir': run_dir, 'cells': cells, 'slopes': slopes,
            'regions': regions, 'bands': bands, 'edges': edges,
            'subjects': subjects, 'mat': mat, 'se': se, 'weights': weights,
            'cell_consistency': cc, 'similarity': sim, 'network': net,
            'subject_order': order, 'n_assessed': n_assessed,
            'n_eligible': n_eligible}


def report(ctx):
    cc, sim = ctx['cell_consistency'], ctx['similarity']
    sig = cc[cc['p_bh_reject']]
    logger.info('=' * 74)
    logger.info('CONSISTENCY ACROSS SUBJECTS')
    logger.info('  cells: %d fitted, %d BH-significant', len(cc),
                int(cc['p_bh_reject'].sum()))
    logger.info('  sign agreement: median %.2f over all cells, %.2f over the '
                'BH-significant ones (sign-flip null averages %.2f)',
                cc['frac_sign'].median(), sig['frac_sign'].median(),
                cc['null_frac_mean'].mean())
    logger.info('  cells beating their own sign-flip null at p<0.05: %d of %d '
                '(%d of %d BH-significant cells)',
                int((cc['p_signflip'] < 0.05).sum()), len(cc),
                int((sig['p_signflip'] < 0.05).sum()), len(sig))
    if 'frac_sign_stored' in cc.columns and cc['frac_sign_stored'].notna().any():
        gap = (cc['frac_sign'] - cc['frac_sign_stored']).abs().max()
        logger.info('  max |recomputed - stored| frac_sign_consistent: %.4g%s',
                    gap, '  <-- INVESTIGATE' if gap > 1e-6 else '')
    n_disagree = int((~cc['ivw_sign_agrees_with_model']).sum())
    logger.info('  weighted-mean sign disagrees with the model beta in %d cells '
                '(the null uses the weighted-mean sign)', n_disagree)
    # THE HEADLINE: one statistic, one null, each subject against their own bound.
    d = sim.dropna(subset=['r_map', 'r_map_null_p975'])
    beats = d['r_map'] > d['r_map_null_p975']
    logger.info('  subject similarity to the leave-one-out group map: median r '
                '%.2f, %d of %d positive', d['r_map'].median(),
                int((d['r_map'] > 0).sum()), len(d))
    logger.info('  BEATING THEIR OWN WITHIN-BAND NULL: %d of %d subjects '
                '(median bound %.3f) -- this is the anatomy claim',
                int(beats.sum()), len(d),
                float(np.nanmedian(d['r_map_null_p975'])))
    # The two secondary constructions, logged so the choice stays auditable.
    free = sim.dropna(subset=['r_full', 'r_full_null_p975'])
    n_free = int((free['r_full'] > free['r_full_null_p975']).sum())
    reg = sim.dropna(subset=['r_region', 'r_region_null_p975'])
    n_reg = int((reg['r_region'] > reg['r_region_null_p975']).sum())
    # The NET count difference understates the leak. Report the ONE-WAY crossing:
    # subjects the free shuffle passes and the correct blocked null rejects. The
    # traffic goes both ways because a patient with little band structure can
    # have a blocked bound BELOW their free one, so net and one-way differ.
    joint = sim.dropna(subset=['r_full', 'r_full_null_p975', 'r_map',
                               'r_map_null_p975'])
    free_pass = joint['r_full'] > joint['r_full_null_p975']
    wb_pass = joint['r_map'] > joint['r_map_null_p975']
    logger.info('  for reference, the SAME statistic against a FREE cross-band '
                'shuffle: %d of %d pass (median bound %.3f vs %.3f blocked)',
                n_free, len(free),
                float(np.nanmedian(free['r_full_null_p975'])),
                float(np.nanmedian(d['r_map_null_p975'])))
    logger.info('    one-way crossings: %d pass FREE and fail BLOCKED (the tilt '
                'leak), %d the reverse, net %+d',
                int((free_pass & ~wb_pass).sum()),
                int((~free_pass & wb_pass).sum()),
                n_free - int(beats.sum()))
    logger.info('  band-mean-removal route to the same question: %d of %d '
                '(agrees with the plotted statistic: %s)', n_reg, len(reg),
                'YES' if n_reg == int(beats.sum()) else 'NO -- INVESTIGATE')
    logger.info('  attrition: %d assessed -> %d eligible -> %d with a fittable '
                'slope -> %d in the ranked null panel',
                ctx['n_assessed'], ctx['n_eligible'], len(ctx['subjects']),
                len(d))
    net = ctx['network']
    both = net[(net['n_limbic_delta'] > 0) & (net['n_sensorimotor_beta'] > 0)]
    logger.info('  both halves of the pattern in the same subject: %d of %d '
                'subjects with coverage in both sets',
                int(((both['limbic_delta'] < 0)
                     & (both['sensorimotor_beta'] > 0)).sum()), len(both))
    logger.info('  LOSO sign flips: %d cells (%d BH-significant)',
                int(cc['loso_flips_sign'].sum()),
                int((cc['loso_flips_sign'] & cc['p_bh_reject']).sum()))
    logger.info('  MOST FRAGILE BH-significant cells (largest single-subject '
                'influence):')
    for r in (sig.sort_values('loso_max_rel_delta', ascending=False).head(6)
              .itertuples()):
        logger.info('    %-18s %-11s beta %+.5f  sign %d/%d  drop-one moves it '
                    '%.0f%% (%s)', r.region, r.band, r.beta, r.n_sign_match,
                    r.n_with_slope, 100 * r.loso_max_rel_delta,
                    r.loso_worst_subject)
    logger.info('=' * 74)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--run-dir', required=True,
                    help='A band-power mixed-model run written by '
                         'run_bandpower_mixed.py.')
    ap.add_argument('--figure', action='append', choices=FIGURES,
                    help='Draw only these (repeatable). Default: all of them.')
    ap.add_argument('--n-perm', type=int, default=N_PERM,
                    help='Draws for the sign-flip and label-permutation nulls.')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out-subdir', default=OUT_SUBDIR,
                    help="Folder inside the run directory to write into. Pass "
                         "'' to write beside band_cells.parquet as the other "
                         "figures do.")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)s %(message)s')
    io.warn_if_dirty()

    import matplotlib
    matplotlib.use('Agg')

    run_dir, cells, slopes, params, inventory = load_run(args.run_dir)
    logger.info('%d cells, %d subject-slope rows, %d subjects', len(cells),
                len(slopes), slopes['subject'].nunique())
    ctx = build_context(run_dir, cells, slopes, params, inventory=inventory,
                        n_perm=args.n_perm, seed=args.seed)
    report(ctx)

    # Everything this script produces goes in ONE subfolder of the run. These
    # are a self-contained set that answers a different question from the run's
    # own figures -- they describe the SUBJECTS rather than the model -- and the
    # run directory is already twenty files deep, so keeping them together is
    # what makes them findable. `--out-subdir ''` puts them back at the top.
    out_dir = run_dir / args.out_subdir if args.out_subdir else run_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    wanted = args.figure or list(FIGURES)
    written = []
    for name in wanted:
        out = out_dir / f'fig_consistency_{name}.png'
        BUILDERS[name](ctx, out)
        written.append(out.name)
        logger.info('wrote %s', out.relative_to(run_dir))

    # The two tables behind the figures. CSV under analysis/ per the IO contract:
    # small, terminal, read by eye.
    parents = [str(run_dir / 'band_cells.parquet'),
               str(run_dir / 'subject_slopes.parquet')]
    shared = {'n_perm': args.n_perm, 'seed': args.seed,
              'limbic_default': list(LIMBIC_DEFAULT),
              'sensorimotor': list(SENSORIMOTOR),
              'min_cells_for_r': MIN_CELLS_FOR_R}
    io.write_table(ctx['cell_consistency'], out_dir / 'consistency_cells.csv',
                   params={**shared, 'unit': 'region x band cell'},
                   parents=parents, script=SCRIPT)
    io.write_table(ctx['similarity'], out_dir / 'consistency_subjects.csv',
                   params={**shared, 'unit': 'subject'}, parents=parents,
                   subjects=sorted(ctx['subjects']), script=SCRIPT)

    io.log_analysis('band-power mixed model: across-subject consistency of the '
                    'region x band map (sign-flip null, leave-one-out group map, '
                    'LOSO influence) -- EXPLORATORY, nominations not findings',
                    run_dir)
    logger.info('wrote %s', ', '.join(written))


if __name__ == '__main__':
    main()
