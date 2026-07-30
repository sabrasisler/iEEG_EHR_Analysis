"""
Cluster-based permutation testing on a region x frequency map (Maris &
Oostenveld 2007), for "is this different from the 0-pain state?".

Pure statistics: no plotting, no IO, no config imports. That is deliberate --
these functions are the actual deliverable of the analysis, so they are
unit-testable in isolation and the figure script stays a figure script.

THE DESIGN, AND WHY EACH PIECE IS THE WAY IT IS
-----------------------------------------------

**Unit of observation: one value per subject per region per frequency bin.** The
view layer has already averaged channels within an ROI and epochs within a pain
level, so a row of the input matrix is one subject. If channels entered as
independent rows the exchangeability assumption below would be false and the test
would run anticonservative -- more "significant" the more electrodes a subject
happens to have.

**One-sample against zero.** The views are referenced to each subject's own 0-pain
baseline, so testing a level against 0 IS testing it against the 0-pain state. The
high-vs-low contrast is the SAME function applied to the per-subject paired
difference -- not a second code path.

**Sign-flipping, one sign vector per permutation across all regions.** Under the
null, a subject's map is as likely to be negated as not. The vector is shared
across regions on purpose: a subject's regions are correlated, and flipping each
region independently would destroy that correlation and inflate significance.
(Sign-flipping tests the null that the distribution is SYMMETRIC about zero, which
is slightly stronger than "the mean is zero" -- standard for this design, worth
knowing.)

**Adjacency along frequency only, within a region.** Region rows are not
neighbours: the heatmap's row order is a display choice, not an anatomical
adjacency graph. An invalid cell -- an excluded line-noise bin, a cell below the
subject floor, a non-finite t -- TERMINATES a run, so no cluster bridges the 60 Hz
notch.

**Per-cell critical t.** Coverage varies from ~21 subjects (ACC) to ~51
(Temporal), so one shared threshold would be wrong somewhere. `t_crit` depends only
on n, which sign-flipping does not change, so the threshold map is computed once
and reused for every permutation.

**min_extent is enforced inside the permutation loop.** Filtering only the
observed clusters would compare a filtered observation against an unfiltered null
and inflate significance. This is the easiest thing here to get silently wrong.

**Both correction scopes come out of ONE loop**, accumulating per-region and global
max |mass| against the same sign vectors, so the two schemes are directly
comparable rather than two independent randomisations.
"""

import logging

import numpy as np

logger = logging.getLogger(__name__)

# The sentence that must travel with every number this module produces.
BOUNDARY_CAVEAT = (
    "A cluster's p-value applies to the cluster AS A WHOLE, not to its "
    "boundaries. Report 'a significant beta-band effect in S2'; do NOT report "
    "that it spans 15.2-31.7 Hz. The extent is where the statistic happened to "
    "cross an arbitrary threshold in this sample, and it is not itself tested."
)


# ============================================================================
# BIN-LEVEL STATISTICS
# ============================================================================

def onesample_t(x):
    """NaN-aware one-sample t against zero, over axis 0.

    x: (n_subjects, ...) -> (t, n) each with x.shape[1:].

    Cells with fewer than 2 finite observations, or zero variance, get t = NaN
    rather than an inf or a fabricated number: the caller's validity mask is
    where "not enough data" is expressed, and a silent inf would sail through a
    `>` comparison as a cluster.
    """
    x = np.asarray(x, dtype=np.float64)
    finite = np.isfinite(x)
    n = finite.sum(axis=0)

    with np.errstate(invalid='ignore', divide='ignore'):
        # np.nanmean/nanstd warn on all-NaN slices and return NaN; suppressing the
        # warning is safe because n == 0 there and the result is masked below.
        mean = _nanmean_safe(x)
        sd = _nanstd_safe(x)
        t = np.where(n >= 2, mean / (sd / np.sqrt(np.maximum(n, 1))), np.nan)
    t = np.where(np.isfinite(t), t, np.nan)
    return t, n


def _nanmean_safe(x):
    """nanmean that returns NaN for all-NaN slices without warning.

    An explicit guard rather than `warnings.catch_warnings`: subject x region
    combinations with no coverage are EXPECTED here (4 subjects have no DK labels
    at all, and not every subject has an electrode in every ROI), so this is a
    normal path, not an exceptional one.
    """
    finite = np.isfinite(x)
    n = finite.sum(axis=0)
    total = np.where(finite, x, 0.0).sum(axis=0)
    return np.where(n > 0, total / np.maximum(n, 1), np.nan)


def _nanstd_safe(x, ddof=1):
    finite = np.isfinite(x)
    n = finite.sum(axis=0)
    mean = _nanmean_safe(x)
    sq = np.where(finite, (x - mean) ** 2, 0.0).sum(axis=0)
    denom = np.maximum(n - ddof, 1)
    return np.where(n > ddof, np.sqrt(sq / denom), np.nan)


def yuen_onesample_t(x, trim=0.2):
    """One-sample trimmed-mean t analogue (Yuen), over axis 0.

    A SENSITIVITY CHECK, not the primary statistic. Sign-flipping does not by
    itself protect against a single subject driving a cluster, because t is still
    mean/SD and both respond to one extreme value. Substituting a trimmed mean and
    a winsorized SD does, so a cluster that survives both is not an outlier
    artefact.

    Computed per cell on that cell's finite values, so trimming counts adapt to
    the coverage each region actually has.
    """
    x = np.asarray(x, dtype=np.float64)
    flat = x.reshape(x.shape[0], -1)
    t = np.full(flat.shape[1], np.nan)
    n_out = np.zeros(flat.shape[1], dtype=int)

    for j in range(flat.shape[1]):
        col = flat[:, j]
        col = np.sort(col[np.isfinite(col)])
        n = col.size
        n_out[j] = n
        g = int(np.floor(trim * n))
        if n < 4 or n - 2 * g < 2:
            continue
        trimmed = col[g:n - g]
        # Winsorize for the variance: replace the trimmed tails with the nearest
        # retained value rather than dropping them, which is what makes the
        # denominator an estimate of the trimmed mean's own variability.
        wins = np.clip(col, col[g], col[n - 1 - g])
        ssq = ((wins - wins.mean()) ** 2).sum()
        denom = (n - 2 * g) * (n - 2 * g - 1)
        if ssq <= 0 or denom <= 0:
            continue
        t[j] = trimmed.mean() / np.sqrt(ssq / denom)

    return t.reshape(x.shape[1:]), n_out.reshape(x.shape[1:])


def critical_t(n, alpha=0.05):
    """Two-sided critical t PER CELL, from that cell's own df = n - 1.

    NaN where n < 2. Vectorized over the whole map; scipy is imported lazily so
    importing this module stays cheap for callers that only want the clustering.
    """
    from scipy import stats

    n = np.asarray(n)
    df = n - 1
    out = np.full(n.shape, np.nan)
    ok = df >= 1
    out[ok] = stats.t.ppf(1.0 - alpha / 2.0, df[ok])
    return out


# ============================================================================
# CLUSTERING
# ============================================================================

def find_clusters(t, valid, t_crit, min_extent=3):
    """Suprathreshold runs along the FREQUENCY axis, within each region row.

    t, valid, t_crit: (n_region, n_bin). Returns a list of dicts with
    region_idx, bin_lo, bin_hi (inclusive), n_bins, mass, peak_t, sign.

    Positive and negative clusters are formed SEPARATELY -- a run may not change
    sign partway through, or its mass would cancel toward zero and a real
    bidirectional effect would vanish. Invalid cells break runs.
    """
    t = np.asarray(t, dtype=np.float64)
    valid = np.asarray(valid, dtype=bool)
    t_crit = np.asarray(t_crit, dtype=np.float64)

    supra = {
        1: valid & np.isfinite(t) & np.isfinite(t_crit) & (t > t_crit),
        -1: valid & np.isfinite(t) & np.isfinite(t_crit) & (t < -t_crit),
    }

    clusters = []
    for sign, mask in supra.items():
        for r in range(mask.shape[0]):
            for lo, hi in _runs(mask[r]):
                if hi - lo + 1 < min_extent:
                    continue
                seg = t[r, lo:hi + 1]
                clusters.append({
                    'region_idx': r, 'bin_lo': lo, 'bin_hi': hi,
                    'n_bins': hi - lo + 1, 'mass': float(seg.sum()),
                    'peak_t': float(seg[np.argmax(np.abs(seg))]), 'sign': sign,
                })
    return clusters


def _runs(row):
    """[(lo, hi)] inclusive index pairs of each maximal True run in a 1-D mask."""
    idx = np.flatnonzero(row)
    if idx.size == 0:
        return []
    breaks = np.flatnonzero(np.diff(idx) > 1)
    starts = np.r_[0, breaks + 1]
    ends = np.r_[breaks, idx.size - 1]
    return [(int(idx[s]), int(idx[e])) for s, e in zip(starts, ends)]


def max_mass_per_region(t, valid, t_crit, min_extent, n_region):
    """(per_region_max, global_max) of |cluster mass| for ONE map.

    Regions with no surviving cluster contribute 0.0, which is what makes them
    comparable in the null: "no cluster at all" is the weakest possible outcome,
    not a missing value.
    """
    per_region = np.zeros(n_region)
    for c in find_clusters(t, valid, t_crit, min_extent):
        m = abs(c['mass'])
        if m > per_region[c['region_idx']]:
            per_region[c['region_idx']] = m
    return per_region, float(per_region.max()) if per_region.size else 0.0


# ============================================================================
# THE TEST
# ============================================================================

def permutation_p(observed, null):
    """(1 + #{null >= observed}) / (n_perm + 1).

    The +1 counts the observed data as one of its own permutations, so p is never
    exactly 0 -- with 10,000 permutations the floor is 1e-4, and reporting
    "p = 0" from a finite randomisation would be a claim the procedure cannot make.
    """
    null = np.asarray(null, dtype=np.float64)
    return float((1 + np.count_nonzero(null >= observed)) / (null.size + 1))


def bh_fdr(p, q=0.05):
    """Benjamini-Hochberg step-up. Returns (rejected, p_adjusted).

    Applied across REGIONS in the two-stage scheme. Every region is in the family,
    including those with no cluster (p = 1.0): the number of tests is the number of
    regions looked at, not the number that happened to produce something.
    """
    p = np.asarray(p, dtype=np.float64)
    m = p.size
    if m == 0:
        return np.zeros(0, dtype=bool), np.zeros(0)
    order = np.argsort(p)
    ranked = p[order]
    # Monotone (step-up) adjusted p-values, then un-sort.
    adj_sorted = np.minimum.accumulate((ranked * m / np.arange(1, m + 1))[::-1])[::-1]
    adj_sorted = np.clip(adj_sorted, 0.0, 1.0)
    adj = np.empty_like(adj_sorted)
    adj[order] = adj_sorted
    return adj <= q, adj


def cluster_test(x, valid=None, alpha=0.05, min_extent=3, n_perm=10000, q=0.05,
                 seed=0, statistic='t', trim=0.2, chunk=500, min_subjects=1):
    """Full cluster permutation test on a (n_subject, n_region, n_bin) matrix.

    valid: optional (n_region, n_bin) bool -- cells eligible to enter a cluster
        (e.g. False for excluded line-noise bins). Combined with the data-driven
        validity (>= min_subjects finite observations, finite t).

    Returns a dict: clusters (list, each with p_within_region, p_global,
    sig_two_stage, sig_global), region_p, region_p_adj, region_rejected,
    n_map, t_map, t_crit_map, valid_map, n_perm, seed, statistic.
    """
    x = np.asarray(x, dtype=np.float64)
    n_subj, n_region, n_bin = x.shape
    stat_fn = (lambda a: yuen_onesample_t(a, trim=trim)) if statistic == 'yuen' else onesample_t

    t_obs, n_map = stat_fn(x)
    t_crit_map = critical_t(n_map, alpha)

    valid_map = (n_map >= max(min_subjects, 2)) & np.isfinite(t_obs) & np.isfinite(t_crit_map)
    if valid is not None:
        valid_map &= np.asarray(valid, dtype=bool)
    _warn_if_coverage_varies_within_region(n_map, valid_map)

    clusters = find_clusters(t_obs, valid_map, t_crit_map, min_extent)

    # ---------------- the null ----------------
    # ONE sign vector per permutation, applied to every region at once. t_crit is
    # NOT recomputed: sign-flipping cannot change how many finite observations a
    # cell has, so the threshold map is invariant.
    rng = np.random.default_rng(seed)
    null_region = np.empty((n_perm, n_region))
    null_global = np.empty(n_perm)

    for start in range(0, n_perm, chunk):
        stop = min(start + chunk, n_perm)
        signs = rng.choice(np.array([-1.0, 1.0]), size=(stop - start, n_subj))
        for k in range(stop - start):
            t_perm, _ = stat_fn(x * signs[k][:, None, None])
            per_region, g = max_mass_per_region(t_perm, valid_map, t_crit_map,
                                                min_extent, n_region)
            null_region[start + k] = per_region
            null_global[start + k] = g

    # ---------------- p-values ----------------
    for c in clusters:
        m = abs(c['mass'])
        c['p_within_region'] = permutation_p(m, null_region[:, c['region_idx']])
        c['p_global'] = permutation_p(m, null_global)

    # Two-stage: family-wise within region across frequency, then BH across the
    # regions. A region with no cluster still counts as a test.
    region_p = np.ones(n_region)
    for c in clusters:
        region_p[c['region_idx']] = min(region_p[c['region_idx']], c['p_within_region'])
    region_rejected, region_p_adj = bh_fdr(region_p, q)

    for c in clusters:
        r = c['region_idx']
        c['region_p_bh'] = float(region_p_adj[r])
        c['sig_two_stage'] = bool(region_rejected[r] and c['p_within_region'] < alpha)
        c['sig_global'] = bool(c['p_global'] < alpha)

    return {
        'clusters': clusters, 'region_p': region_p, 'region_p_adj': region_p_adj,
        'region_rejected': region_rejected, 't_map': t_obs, 'n_map': n_map,
        # The group MEAN map, not just t. A permutation test answers "is this
        # bigger than chance", which is not the same question as "is this big" --
        # a tiny mean with a tinier SE is highly significant and scientifically
        # empty. Every cluster must be reportable with an effect size beside its p.
        'mean_map': np.where(valid_map, _nanmean_safe(x), np.nan),
        't_crit_map': t_crit_map, 'valid_map': valid_map,
        'null_global': null_global, 'n_perm': n_perm, 'seed': seed,
        'statistic': statistic, 'alpha': alpha, 'min_extent': min_extent, 'q': q,
    }


def _warn_if_coverage_varies_within_region(n_map, valid_map):
    """Coverage is EXPECTED to be constant across bins within a region.

    If it is not, the per-cell df silently differs along a row, which is handled
    correctly here but is a coverage fact worth discovering rather than averaging
    over -- so it is logged rather than absorbed.
    """
    for r in range(n_map.shape[0]):
        ns = np.unique(n_map[r][valid_map[r]])
        if ns.size > 1:
            logger.warning('region index %d: subject count varies across frequency '
                           'bins (%s) -- per-cell df used, but check coverage',
                           r, ns.tolist())


def significant_mask(result, n_region, n_bin, correction='two_stage'):
    """(n_region, n_bin) bool of cells inside a significant cluster.

    This is what gets outlined on the heatmap. `correction` selects which of the
    two scopes decides; both are always available on the clusters themselves.
    """
    key = 'sig_two_stage' if correction == 'two_stage' else 'sig_global'
    mask = np.zeros((n_region, n_bin), dtype=bool)
    for c in result['clusters']:
        if c[key]:
            mask[c['region_idx'], c['bin_lo']:c['bin_hi'] + 1] = True
    return mask


def detrend_over_frequency(x, valid=None):
    """Subtract each subject x region map's mean over its valid bins.

    OPT-IN, because it CHANGES THE HYPOTHESIS: from "differs from the 0-pain
    state" to "the spectral shape is not flat". Its purpose is that the broadband
    low-frequency offset otherwise absorbs into one giant ~1-40 Hz cluster in most
    regions -- significant, and useless, because it spans delta through beta and
    cannot be attributed to a band. The high-minus-low contrast cancels the same
    component naturally, which is a reason to prefer reading that.
    """
    x = np.asarray(x, dtype=np.float64).copy()
    if valid is not None:
        # Invalid cells must not contribute to the mean that gets subtracted --
        # otherwise an excluded line-noise bin would shift the whole row.
        x[:, ~np.asarray(valid, dtype=bool)] = np.nan

    # Mean over the BIN axis (2), per (subject, region). _nanmean_safe reduces over
    # axis 0, so move the bin axis to the front and move the result back.
    mean = _nanmean_safe(np.moveaxis(x, 2, 0))           # -> (n_subject, n_region)
    return x - mean[:, :, None]
