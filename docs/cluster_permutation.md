# cluster_permutation.md — the cluster test on the region × frequency map

**Read this before interpreting an outline on a heatmap, or before changing any
parameter of the test.** Implementation: `src/ieeg_ehr/analysis/cluster_permutation.py`
(pure statistics, unit-tested in `tests/test_cluster_permutation.py`), driven by
`--cluster-test` in `analysis/plot_pain_view_heatmaps.py`. Status: v1, built
2026-07-29.

Everything this produces is an **EXPLORATORY NOMINATION, not a finding**
(`CLAUDE.md`). It runs on the discovery cohort only, and nothing from it earns a
`DECISIONS.md` entry before P2.6 FREEZE.

---

## 1. What is being tested

The views are already referenced to each subject's own 0-pain baseline, so
"different from the 0-pain state" is a **one-sample test against zero**. Three
contrasts, all through the same function:

| Contrast | Input |
|---|---|
| `low` | that level's per-subject values |
| `high` | that level's per-subject values |
| `high_minus_low` | the per-subject PAIRED difference |
| `none_control` | the 0-pain bin itself — see §6 |

`high_minus_low` is not a second code path. It is the same one-sample test on the
difference, which is what makes it a paired comparison and removes the large
between-subject offsets that a two-group comparison would leave in.

## 2. Unit of observation

**One value per subject per region per frequency bin.** The view layer has already
averaged channels within an ROI and epochs within a pain level, so a row of the
input matrix is one subject. This is not cosmetic: if channels entered as
independent rows the exchangeability assumption in §4 would be false and the test
would run **anticonservative** — more "significant" the more electrodes a subject
happens to have.

## 3. Bin-level statistic

Student's one-sample t, **NaN-aware**, with **df taken per cell**. Coverage varies
from ~21 subjects (ACC) to ~51 (Temporal), so one shared threshold would be wrong
somewhere. `t_crit` depends only on n, which sign-flipping cannot change, so the
threshold map is computed once and reused across permutations.

Cells with n < 2, zero variance, or fewer than `--min-subjects` (default 8)
contributing subjects are invalid and cannot enter a cluster. `--min-subjects`
affects the **test only**; the heatmap still displays those cells.

## 4. Permutation scheme

**Sign-flipping**: multiply each subject's whole map by ±1. **One sign vector per
permutation, applied across all regions at once** — a subject's regions are
correlated, and flipping each region independently would destroy that correlation
and inflate significance.

Sign-flipping tests the null that the distribution is **symmetric about zero**,
which is slightly stronger than "the mean is zero". Standard for this design; worth
stating in a methods section.

`p = (1 + #{null ≥ observed}) / (n_perm + 1)`, `n_perm = 10,000`. The `+1` counts
the observed data as one of its own permutations, so p is never exactly 0 — a
finite randomisation cannot support that claim.

## 5. Adjacency, clustering, and the two correction scopes

**Adjacency is along frequency only, within a region.** Region rows are not
neighbours: the heatmap's row order is a display choice, not an anatomical
adjacency graph, and defending an arbitrary one is unnecessary. An invalid cell —
an excluded line-noise bin, a low-coverage cell, a non-finite t — **terminates** a
run, so nothing bridges the 60 Hz notch or merges a beta cluster with a gamma one.

Cluster statistic is **mass** (sum of t over the run). Positive and negative
clusters form separately, or a sign change mid-run would cancel toward zero. The
null is built on **max |mass|**.

`--min-cluster-bins` (default 3) is enforced **inside the permutation loop as well
as on the observed map**. Filtering only the observation would compare a filtered
statistic against an unfiltered null and inflate significance. This is the easiest
thing here to get silently wrong, and it has its own test.

Both correction scopes come out of **one** permutation loop against the same sign
vectors, so they are directly comparable:

- **`two_stage`** (default, drives the outlines) — each cluster gets a p from its
  own region's max-|mass| null, i.e. family-wise across frequency within that
  region; then each region contributes its minimum cluster p (or 1.0 if it has
  none) and BH-FDR runs across all regions at q = 0.05. A cluster is significant if
  its region survives BH **and** its own within-region p < α. **Regions with no
  cluster stay in the family** — the number of tests is the number of regions
  looked at, not the number that happened to fire.
- **`global`** — one max-stat null over the whole map. Stricter, and stricter
  unevenly: with n from 21 to 51 the t magnitudes are not comparable across
  regions, so well-covered regions (Temporal) swallow the power from sparse ones
  (ACC, S1).

Both land in `clusters.parquet` for every cluster regardless of which one is
plotted. `p_global ≥ p_within_region` always — a cheap invariant worth checking.

**No correction across the three contrasts.** Each is its own family. Stated rather
than silently assumed.

## 6. The `none` control is a FLOOR, not a pass/fail

The 0-pain bin is tested against zero as a control. It **is expected to produce
significant clusters**, and that is not a failure:

- It is **circular**. Those are the very windows that define the baseline, so it is
  data tested against a statistic computed from itself.
- The baseline pools **windows**; a reported value averages **epochs**. QC masking
  makes epochs retain unequal numbers of windows, so the two differ and the bin
  cannot be exactly zero. Measured 2026-07-29: max |group mean| **0.0201 z**,
  median 0.0037, correlating with masking at r = +0.65
  (`n_channel_epochs_dropped_coverage`).

So the control's job is **quantitative**: it measures how small an effect this
pipeline will happily call significant. At n=56 the `none` bin cleared
significance at a mean of ~0.004 z, because a tiny mean over a tinier standard
error has a large t. For comparison, `high` has a median cell of 0.067 and peaks at
0.66 — roughly 18× and 33× the floor.

**Therefore effect size travels with every p-value.** `clusters.parquet` carries
`mean_signed_z`, `mean_abs_z`, `peak_abs_z` and `floor_ratio` (= `mean_abs_z` /
floor). A cluster that is significant but sits near the floor is bookkeeping. There
is deliberately **no hard gate** on `floor_ratio`, because any multiplier would be
arbitrary; the number is reported and the reader judges. See `SCRATCHPAD.md` for
whether a gate should eventually be adopted, and `TASKS.md` for the split-half
baseline that would make this a real (non-circular) control.

## 7. Line-noise bins are excluded from the test by default

Independently of `--exclude-line-noise-bins`, which is about display. **Not**
because they would be non-significant — the opposite. Line noise is highly
consistent across subjects, so its across-subject variance is small and its t can
be large; those bins already hold the most extreme cells in the group map. Three
consequences, of which the second decides it:

1. A significant 59 Hz cluster is an artifact statement, not physiology.
2. **The null is a max statistic.** Large artifact masses inflate it, which makes
   every real cluster harder to detect. Including them costs power for the science.
3. Left valid, a run can bridge *through* the notch and merge two unrelated
   clusters into one uninterpretable blob.

`--test-includes-line-noise` opts back in, for checking what they do.

## 8. Robustness

`--robust` re-runs with a 20% trimmed-mean (Yuen) statistic at a lower `n_perm`.
Sign-flipping does **not** by itself protect against one subject driving a cluster,
because t is still mean/SD and both respond to a single extreme value. Clusters
surviving both statistics are not outlier artifacts. Reported in the table, never
plotted.

## 9. `--detrend-freq` changes the hypothesis

Off by default. It subtracts each subject × region map's mean over its valid bins,
which turns the question from "differs from the 0-pain state" into "the spectral
shape is not flat". Its purpose: the broadband low-frequency offset otherwise
absorbs into one giant ~1–40 Hz cluster in most regions — significant, and useless,
because it spans delta through beta and cannot be attributed to a band. The
`high_minus_low` contrast cancels the same component naturally, which is a reason
to prefer reading it.

## 10. The reporting constraint

> **A cluster's p-value applies to the cluster AS A WHOLE, not to its boundaries.**
> Report "a significant beta-band effect in S2". Do NOT report that it spans
> 15.2–31.7 Hz.

The extent is where the statistic happened to cross an arbitrary threshold in this
sample; it is not itself tested. This sentence lives in
`cluster_permutation.BOUNDARY_CAVEAT`, is written into the table's sidecar and the
run's `provenance.json`, and is printed as a footnote on every figure — so it is
attached to the numbers rather than remembered.

## 11. Reproducibility

`--seed` (default 0) is recorded in `provenance.json`. Same seed, same data, same
result; the tests assert both halves of that.

## 12. Outlines

Drawn as **explicit cell-edge segments** (`common.draw_mask_outline`), never with
`contour()` — contour interpolates between cell centres and puts the boundary half
a bin inside the cluster, which on a log-spaced frequency axis is a visibly wrong
claim about which frequencies were tested.
