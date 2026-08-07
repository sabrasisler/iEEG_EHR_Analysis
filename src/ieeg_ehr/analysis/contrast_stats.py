"""
Paired pain contrasts and their NOISE FLOOR.

A contrast (`high - none` per subject and region) is easy to compute and easy to
over-read. This module exists so that the number never travels without the scale
it should be judged against, which is the convention `docs/cluster_permutation.md`
section 6 already sets for the cluster test: measure a floor, report a
`floor_ratio` beside the effect, and apply NO hard gate -- any multiplier would be
arbitrary, so the number is reported and the reader judges.

WHY A PERMUTATION AND NOT A t-TEST
----------------------------------
The floor being asked for is "how big a contrast does this pipeline produce from
nothing", and that depends on how many epochs each cell has, how variable that
subject's slopes are, and how unevenly the pain levels are populated -- none of
which a parametric standard error captures well at these sample sizes (a cell can
have 10 epochs in one bin and 40 in another).

The shuffle is WITHIN each (subject, region): the `pain_bin` labels are permuted
among that cell's own epochs. That preserves the cell's epoch count, its
per-bin group sizes and its entire slope distribution, and destroys ONLY the
association with pain. A shuffle across subjects would instead destroy the
between-subject offset too, and would produce a null far wider than the real one
-- it would flatter the result rather than test it.

TWO FLOORS, AND THEY ANSWER DIFFERENT QUESTIONS
-----------------------------------------------
Measured 2026-08-07 on the discovery slope tables, and the gap is the whole point:

- `floor_group` -- the 95th percentile of |group mean| under the null, 0.0064.
  The observed group mean is +0.0515, i.e. 8.1x. THE GROUP EFFECT IS REAL.
- `floor_cell` -- the median |single cell| under the null, 0.0427, against 0.0641
  observed. Only ~1.5x. A SINGLE SUBJECT'S DOT IS BARELY ABOVE NOISE.

Both are reported because a figure that shows one dot per subject invites reading
an individual dot, and that reading is not supported even when the group mean is.
"""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

DEFAULT_N_PERM = 200


def paired_contrast(epoch_tables, a, b, value_col='slope', min_a=10, min_b=0,
                    keys=('subject_id', 'region'), log=True):
    """Per (subject, region): mean(`a`) - mean(`b`), with the per-bin floors applied.

    The floors live HERE rather than in the plot script so a figure cannot forget
    them. `min_b` defaults to 0 because `b` is normally the 0-pain reference, whose
    adequacy is enforced once at the SUBJECT level by
    `view_tables.exclude_thin_baseline_subjects` -- a per-cell floor on the
    reference as well would cut subjects twice for the same reason.

    A cell missing either bin entirely is dropped, not treated as zero.
    """
    keys = list(keys)
    rows = epoch_tables[epoch_tables['pain_bin'].isin([a, b])]
    grouped = (rows.groupby(keys + ['pain_bin'], dropna=False)
               .agg(value=(value_col, 'mean'), n=('epoch_id', 'nunique'))
               .reset_index())

    wide_v = grouped.pivot_table(index=keys, columns='pain_bin', values='value')
    wide_n = grouped.pivot_table(index=keys, columns='pain_bin', values='n')
    for frame in (wide_v, wide_n):
        for col in (a, b):
            if col not in frame.columns:
                frame[col] = np.nan

    out = pd.DataFrame({
        'value': wide_v[a] - wide_v[b],
        f'n_{a}': wide_n[a].fillna(0).astype(int),
        f'n_{b}': wide_n[b].fillna(0).astype(int),
    }).reset_index().dropna(subset=['value'])

    before = len(out)
    out = out[(out[f'n_{a}'] >= min_a) & (out[f'n_{b}'] >= min_b)]
    # `log=False` from inside the permutation loop: the message is identical on
    # every one of the ~200 shuffles (the floors depend on epoch counts, which the
    # shuffle preserves by design), and repeating it buries the actual results.
    if log and before - len(out):
        logger.info('%s-%s: %d/%d cells dropped by the per-bin floors '
                    '(>=%d %s, >=%d %s)', a, b, before - len(out), before,
                    min_a, a, min_b, b)
    out['subject'] = out['subject_id']          # the violin helper's column name
    out['contrast'] = f'{a}-{b}'
    return out


def permutation_null(epoch_tables, a, b, value_col='slope', min_a=10, min_b=0,
                     n_perm=DEFAULT_N_PERM, seed=0, by_region=True):
    """The floor: what this contrast looks like when pain carries no information.

    Shuffles `pain_bin` WITHIN each (subject, region) -- see the module docstring
    for why that is the right exchangeability, and why shuffling across subjects
    would flatter the result.

    Returns a frame with one row per region (or one row overall when
    `by_region=False`), carrying `floor_group` (95th pct of |group mean| under the
    null), `floor_cell` (median |cell| under the null), and the two-sided
    permutation p for the observed group mean.
    """
    rng = np.random.default_rng(seed)
    rows = epoch_tables[epoch_tables['pain_bin'].isin([a, b])].copy()
    cell = rows['subject_id'].astype(str) + '|' + rows['region'].astype(str)

    observed = paired_contrast(rows, a, b, value_col, min_a=min_a, min_b=min_b)
    group_cols = ['region'] if by_region else []

    null_means = {}      # region -> list of null group means
    null_cells = {}      # region -> list of median |cell|
    for _ in range(n_perm):
        shuffled = rows.copy()
        shuffled['pain_bin'] = (shuffled.groupby(cell, sort=False)['pain_bin']
                                .transform(lambda s: rng.permutation(s.to_numpy())))
        perm = paired_contrast(shuffled, a, b, value_col, min_a=min_a, min_b=min_b,
                               log=False)
        if perm.empty:
            continue
        if by_region:
            for region, sub in perm.groupby('region'):
                null_means.setdefault(region, []).append(float(sub['value'].mean()))
                null_cells.setdefault(region, []).append(float(sub['value'].abs().median()))
        else:
            null_means.setdefault('__all__', []).append(float(perm['value'].mean()))
            null_cells.setdefault('__all__', []).append(float(perm['value'].abs().median()))

    out = []
    groups = (observed.groupby('region') if by_region
              else [('__all__', observed)])
    for key, sub in groups:
        means = np.asarray(null_means.get(key, []), dtype=float)
        cells = np.asarray(null_cells.get(key, []), dtype=float)
        obs_mean = float(sub['value'].mean())
        if means.size == 0:
            floor_g, p = np.nan, np.nan
        else:
            floor_g = float(np.percentile(np.abs(means), 95))
            # +1 in numerator and denominator: with n_perm shuffles the smallest
            # honest p is 1/(n_perm+1), and reporting an exact 0 would claim more
            # resolution than the test has.
            p = float((np.sum(np.abs(means) >= abs(obs_mean)) + 1) / (means.size + 1))
        row = {
            'observed_mean': obs_mean,
            'observed_median': float(sub['value'].median()),
            'observed_median_abs_cell': float(sub['value'].abs().median()),
            'frac_positive': float((sub['value'] > 0).mean()),
            'n_subjects': int(sub['subject_id'].nunique()),
            'floor_group': floor_g,
            'floor_cell': float(np.median(cells)) if cells.size else np.nan,
            'perm_p': p,
            'n_perm': int(means.size),
        }
        if by_region:
            row['region'] = key
        out.append(row)

    summary = pd.DataFrame(out)
    summary['floor_ratio'] = floor_ratio(summary['observed_mean'],
                                         summary['floor_group'])
    summary['cell_floor_ratio'] = floor_ratio(summary['observed_median_abs_cell'],
                                              summary['floor_cell'])
    return summary


def floor_ratio(observed, floor):
    """|observed| / floor, with a zero floor giving NaN rather than infinity.

    One definition, matching the vocabulary `cluster_permutation.py` and
    `docs/cluster_permutation.md` already use, so "3x the floor" means the same
    thing in every figure this project produces.
    """
    observed = np.abs(np.asarray(observed, dtype=float))
    floor = np.asarray(floor, dtype=float)
    with np.errstate(invalid='ignore', divide='ignore'):
        return np.where(floor > 0, observed / np.where(floor > 0, floor, np.nan),
                        np.nan)
