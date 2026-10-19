# Run tokens: how a `pain_change` path names what produced it

A path names a run for a human. The complete record is the run's
`provenance.json`, which holds every parameter below whether or not it is a
token. Read provenance for the full answer, never the folder name.

```
analysis/pain/pain_change/bandpower/<data token>/<run token>_<timestamp>/figures/<figure token>.png
```

The code is the authority. Each token's codes live in a dict beside the code it
describes (the "Defined in" column). The path builders (`pain_change.data_token`,
`pain_change.run_token`) assemble names from those dicts, so nobody types a name
by hand. This file decodes them. On any conflict, the code wins.

## The rule for a new token

Provenance records every parameter of every run. A parameter becomes a token in
a path only when you vary it and want both versions side by side. A token
promoted later appears in a name only when it departs from its default, so
existing paths stay valid (the convention `ViewConfig.scheme_code` already uses
for the ROI scheme). To promote one: add its code to the dict in code, add a row
here, and move it out of "Not tokens yet".

Give a change a NEW NAME when you want to keep both versions and compare them.
Give it a NEW TIMESTAMP when the new run replaces the old one.

## Data tokens: the level-4 folder

Which rows exist and what their d_z values are. Every model you compare shares
them. Joined with `-`, in this order.

| Token | Parameter | Values | Defined in |
|---|---|---|---|
| normalization + baseline | AXIS 3 + AXIS 2 | `zscoreallep` = z-score against all of the session's pain-epoch windows | `views/view_config.py` `NORMALIZATION_CODES`, `BASELINE_CODES` |
| band set | AXIS 5 | `paperbands6hg200`, `fullres` (the `--band-set` name without underscores) | `run_bandpower_mixed.BAND_SETS` |
| ROI scheme | AXIS 6 | `roiv2ofcins`, `roiv3ins`, ... (DK schemes). `hcpexv5` = paper ROIs from an `hcpex_paper` coverage run, version of its mapping file | `views/view_config.py` `ROI_SCHEME_CODES` and `roi_code`, decoded in `docs/view_registry.md` |
| first-hours exclusion | pair filter | `nofirst24h` = pairs whose first epoch starts within 24 h of `session_start` dropped. Absent = none dropped | `pain_change.data_token`, `--exclude-first-h` |
| second-epoch window | pairing | `e2pre10to5` = each pair's later epoch is [t2 - 10, t2 - 5) min (shifted tables, `features/build_pain_change_shifted_z.py`), the earlier stays [t1 - 5, t1). Absent = [t2 - 5, t2) | `pain_change.data_token`, `--second-epoch-shift-min` |

`scheme_code` also carries the pain-bin code (`relpain`). `pain_change` never
bins pain, so its data token leaves that code out.

## Run tokens: the run folder

What you fit to those rows and how. Joined with `_`, in this order, then the
timestamp.

| Token | Parameter | Values | Defined in |
|---|---|---|---|
| model | fixed and random effects | `opioid`, `opioid-pairint`, `opioid-dc-pairint`, `painchange`, `painchange-pairint` | `pain_change.MODELS` |
| exposure version | the recipe for `dC` and `Cbar`. Present only when the formula uses them | `mme1` | `config/opioid_mme.py` `EXPOSURE_VERSION` |
| engine | fitting software | `sm` = statsmodels `MixedLM`, `lme4` = `lme4::lmer` (`pain_change_lmer.R`, same frames and formula) | `pain_change.ENGINES` |
| fit type | full fit or leave-one-subject-out | absent = full fit, `loo` = LOO refit, written beside its parent | `pain_change_loo.py` |

A model ID is a base name plus modifiers in a fixed order, `-<term>` for an
added term and `-no<term>` for a dropped one (`opioid-apap`, `opioid-nogap`).
Each ID is written out in full in `MODELS`, and nothing parses the name.

### Models

| ID | Formula | Since |
|---|---|---|
| `painchange` | `d_z ~ d_pain + pain_1_within + gap_h` | 2026-10-06 |
| `painchange-pairint` | `painchange` + `(1 \| pair_id)`. lme4 only | 2026-10-15 |
| `opioid` | `d_z ~ d_pain + dC + Cbar + d_pain:Cbar + pain_1_within + gap_h` | 2026-10-15 |
| `opioid-pairint` | `opioid` + `(1 \| pair_id)`, one shift shared by a pair's channels. lme4 only | 2026-10-15 |
| `opioid-dc-pairint` | `d_z ~ d_pain + dC + d_pain:dC + pain_1_within + gap_h` + `(1 \| pair_id)`. dC in place of Cbar, not centred. lme4 only | 2026-10-15 |
| `opioid-dc-pairint-nosubjint` | `opioid-dc-pairint` without `(1 \| subject)`: random effects `(0 + d_pain \| subject) + (1 \| pair_id)`. The intercept fitted to zero in all 132 cells of the 2026-10-15 run (a subject offset cancels in d_z). lme4 only | 2026-10-19 |

Every model except a `-nosubjint` one takes the random effects
`(1 \| subject) + (0 + d_pain \| subject)` (`pain_change.RE_SUBJECT`). A model ID
names the fixed AND the random effects.

### Exposure versions

ANY change to the recipe takes a new version, a fix included, so a version
always names one set of curves.

| Version | Recipe | Since |
|---|---|---|
| `mme1` | MAR doses only (doses before `session_start` missing, so early curves are lower bounds). Opioids only, buprenorphine excluded. CDC 2022 oral and Nielsen 2014 IV MME factors. One-compartment PO and IV bolus, three-compartment IV fentanyl. C at an assessment = mean curve over the 5-min epoch before it. `Cbar` = mean curve over [t1, t2), centred on the cohort mean over pairs. Units: typical doses of 10.0 MME (median discovery opioid dose). | 2026-10-15 |
| `mme2` | `mme1`, but C at an assessment = mean curve over the epoch its d_z uses, so a shifted second epoch (`e2pre10to5`) moves C_2 to [t2 - 10, t2 - 5). Used only by shifted runs: equal to `mme1` without a shift (`opioid_mme.exposure_version`). | 2026-10-19 |

## Figure tokens: file names in `figures/`

`<kind>_<term>[_<variant>].png`. PNG only.

| File | Shows | Made by |
|---|---|---|
| `heatmap_dpain.png` | `d_pain` fixed effect, region x band | `plot_pain_change_heatmap --term dpain` |
| `heatmap_dpain-x-cbar.png` | `d_pain:Cbar` fixed effect, region x band | `plot_pain_change_heatmap --term dpain_x_cbar` |
| `heatmap_dpain-x-dc.png` | `d_pain:dC` fixed effect, region x band | `plot_pain_change_heatmap --term dpain_x_dc` |
| `heatmap_dc.png` | `dC` fixed effect (change in exposure), region x band | `plot_pain_change_heatmap --term dc` |
| `heatmap_cbar.png` | `Cbar` fixed effect (mean exposure), region x band | `plot_pain_change_heatmap --term cbar` |

Overwrite a figure when only its look changes (layout, colour, fonts, labels).
Give it a new name when what it shows changes (term, statistic, correction,
subset). Start a new run when anything upstream of `cells.csv` changes.

## Not tokens yet

Recorded in provenance, constant across runs so far.

| Group | Parameter | Current value |
|---|---|---|
| neural | cache | native 0.5 Hz FFT, 5-min pre-score epoch (the second epoch can move, see `e2pre10to5`), 2 s windows with 1 s hop |
| neural | QC mask | `std10_rv-gross-std3_satmargin15_sw_logz4`, bipolar |
| neural | domain, epoch average | log, mean |
| neural | line-noise notch | +-2.0 Hz (`PSD_NOTCH_HALF_WIDTH_HZ`), removed before the band mean |
| neural | band weighting | plain mean over native bins |
| neural | insula cut | pinned with `--insula-from-run` (2026-10-06 run: y > -1.550). DK schemes only: an `hcpex` scheme defines aIns and pIns by parcel |
| rows | cohort | `config.discovery_subjects()`, minus sessions with no MAR export |
| rows | pairs | consecutive assessments, 30-240 min apart |
| rows | coverage floors | 8 subjects, 20 rows per cell |
| model | estimation | REML. `sm`: default optimizer, maxiter 200. `lme4`: bobyqa, maxfun 2e5 |
| analysis | FDR | BH per term (`d_pain`, `d_pain:Cbar`, `d_pain:dC`, `dC`, `Cbar`), across all fitted cells and within region, failed fits excluded |
