# Handoff — full-resolution PSD → band power → processing domains

**Sessions of 2026-09-16 → 2026-09-18.** Written as a continuity document, not an
authority: every number here is reproduced in a run's own `METHODS.md` and
`provenance.json`, and those win on any conflict. Day-by-day narrative is in
`docs/labnotebook/2026-09-16.md`, `-17.md` and `-18.md`; settled calls belong in
`DECISIONS.md` and none of this has earned an entry there yet.

**Everything below is EXPLORATORY, discovery cohort (n=51), nominations not
findings.**

---

## 1. The arc

Four analyses, each answering something the previous one could not.

| # | Analysis | Unit | Question |
|---|---|---|---|
| 1 | Native-resolution grid | region × 0.5 Hz bin | does the old 44-log-bin map survive a 10× finer axis? |
| 2 | Cluster permutation | region × frequency clusters | is any of it distinguishable from chance? |
| 3 | Band-power models | region × band | which BAND, with a standard error per estimate |
| 4 | Domain models | processing domain × band | do pain-matrix domains differ, and does medication explain them? |

---

## 2. Runs on disk

All under `$DERIV = /oak/.../derivatives/sisler/analysis/pain/`.

### Native-resolution grid
`psd_physiology/univariate_analysis/cont_pain_fullres/fullres_grid_mixedlm_20260916-112734/`
21 regions × 463 bins = 9,723 cells, 9,332 converged. No correction, no
significance by design. `fig_fullres_map.png`, `fig_fullres_spectra.png`.

### Cluster permutation
`psd_physiology/univariate_analysis/cont_pain_fullres/fullres_cluster_20260916-135008/`
Two arms, two nulls. Raw 27 clusters / 0 significant; detrended 52 / 17
(sign-flip) / 25 (shuffle).

### Band power, region level
- `bandpower/mixed_effects/paperbands6-roiv2ofc/paperbands6_mixedlm_20260917-112604/`
  — first run, 120 cells, 31 significant
- `bandpower/mixed_effects/paperbands6hg200-roiv2ofc/paperbands6_mixedlm_20260917-115938/`
  — high_gamma extended to 200 Hz, Lateral Temporal excluded, 114 cells, 30 significant
- `bandpower/mixed_effects/paperbands6hg200-roiv2ofc-opioids/paperbands6_mixedlm_20260917-130515/`
  — + opioid terms; `fig_med_effects.png`, `fig_med_attenuation.png`

### Domain models
- `bandpower/domain_model/paperbands6hg200-paindomains/domain_mixedlm_20260917-173208/`
  — v1 domains (superseded)
- `bandpower/domain_model/paperbands6hg200-paindomainsv2/domain_mixedlm_20260918-100249/`
  — **v2 domains, pain only**
- `bandpower/domain_model/paperbands6hg200-paindomainsv2-opioids/domain_mixedlm_20260918-132124/`
- `bandpower/domain_model/paperbands6hg200-paindomainsv2-analgesics/domain_mixedlm_20260918-132041/`
- `bandpower/domain_model/paperbands6hg200-paindomainsv2-non_opioid_analgesics/domain_mixedlm_20260918-135720/`
- `bandpower/domain_model/paperbands6hg200-paindomainsv2-opioids-exclnon_opioid_analgesics/domain_mixedlm_20260918-145947/`
  — **the clean opioid contrast**: opioid-dosed vs ANALGESIC-FREE. 356 of 2406
  epochs dosed with a non-opioid analgesic but not an opioid were dropped, so
  836 medicated vs 1214 unmedicated (was 836 vs 1570). 87,789 rows vs 104,844.
  `bands/*_residuals.parquet` here are INVALID — see `bands/RESIDUALS_INVALID.md`
  and `TASKS.md`; nothing else in the run is affected.

Disposable smoke runs under `cont_pain_fullres/` (`smoke_*`) can be deleted; see
`TASKS.md`.

---

## 3. What the data says

### Pain
The same picture survives every change of axis, region set and model:
**delta down, beta/gamma up**, with mid-frequency effects concentrated in
sensorimotor and lateral/medial frontal cortex. Old 44-bin vs native-resolution
betas correlate **r = 0.978** (OLS slope 1.04, 92.6% sign agreement).

At domain level (v2, pain only), 10 of 30 cells significant:

| domain | delta | beta | gamma |
|---|---|---|---|
| Sensory | −0.0117 | +0.0100 | +0.0065 |
| Affective | −0.0159 | — | — |
| Cognitive | −0.0118 | +0.0114 | +0.0042 |
| Modulatory (M1) | — | **+0.0199** | **+0.0115** |
| Control | −0.0099 | — | — |

Moving M1 out of Sensory into its own Modulatory domain sharpened this: M1 has
the largest mid-frequency effects and no delta effect at all.

### The delta caution
**The quasi-control domain (Occipital + Auditory) shows a delta effect of −0.0099
(p=0.009), the same magnitude as the "real" domains**, and the delta omnibus is
p=0.18 — no evidence domains differ there. The low-frequency effect is not
domain-specific and may be global or artifactual. The mid-frequency effects do
not have this problem (control beta +0.0049, ns).

### Medication
Region level, opioids: **within-patient medication effect significant in 45/114
cells, larger than the pain effect it was meant to be confounding** (|medw|
median 0.0199 vs pain 0.01–0.02), positive and low-frequency-dominant.

But it does **not** explain the pain effect away: adjusting shifts the pain slope
by a **median 2.1%**, max 16.2%, 0 of 30 significant cells above 50%, 0 sign
flips, r = 0.996 between adjusted and unadjusted.

**The non-opioid arm is the surprise.** Non-opioid analgesics are given far less
selectively (NRS gap 0.54 vs opioids' 2.45), so that arm has the least
confounding by indication — and it shows the *strongest* domain-differentiated
medication effects (alpha omnibus 1.7e-14 vs opioids' 0.012). Two readings, not
separable from these fits: something non-pharmacological about being dosed, or
the opioid term being absorbed by the pain terms it is collinear with. Either
way it argues against reading the opioid medication effect as a drug effect.

### Cleaning the comparison stratum (opioid vs analgesic-free)
The original opioid arm compared opioid-dosed epochs against "nothing OR an
acetaminophen/NSAID", which is not a drug-free baseline. Excluding the 356
non-opioid-only epochs costs 16% of the rows and does three separable things:

- **The pain effect does not move.** r = 0.987 across the 30 domain × band
  cells, **100% sign agreement**, betas near-identical (Sensory beta +0.0092 vs
  +0.0099). Significant cells fall 12/30 → 3/30 purely on the lost power — the
  SEs grow, the estimates do not shift. This is the third independent way the
  pain effect has survived a medication control.
- **Part of the high-gamma "opioid" effect was the contaminated baseline.** The
  high_gamma medication betas collapse toward zero (Sensory −0.0014 vs −0.0105,
  Cognitive −0.0036 vs −0.0076) and lose significance. Overall medication cells
  18/30 → 15/30, sign agreement only 87% — the term is the least stable of the
  three.
- **The pain × opioid interaction SHARPENS, on less data.** Omnibus: theta
  0.0028 → 5.5e-05, gamma 0.447 → **0.0097**, high_gamma 0.729 → **0.042**.
  Significant cells 6/30 → 9/30. Expected direction — the old "unmedicated"
  stratum was 23% analgesic-dosed, which blurred the very contrast the
  interaction measures.

Shape of the interaction: **positive at gamma/high-gamma** (Sensory +0.0064,
Cognitive +0.0037, Control +0.0054 — pain's positive high-frequency slope gets
*steeper* when dosed) and **negative at delta/theta, concentrated in Control**
(−0.0157, −0.0130). The delta/theta omnibus is therefore significant largely
because Control is the outlier, not because the pain-matrix domains are large —
the same reading trap as the delta main effect in §3. Treat the low-frequency
interaction as a Control-domain anomaly worth explaining, not as a pain finding.

---

## 4. Code written

| file | what |
|---|---|
| `analysis/fullres_cells.py` | wide-layout loader: view resolution, region matrices, parcel matrices, per-subject slope maps |
| `analysis/run_fullres_grid.py` | native-resolution grid, 3 stages, split gate + cohort policy |
| `analysis/plot_fullres_grid.py` | log-Hz pcolormesh map + per-region spectra |
| `analysis/run_fullres_cluster.py` | cluster permutation, 2 arms × 2 nulls, notch-adjacency choice |
| `analysis/run_bandpower_mixed.py` | band models, medication terms, per-family BH |
| `analysis/plot_bandpower_checks.py` | 7 check/diagnostic figures incl. attenuation |
| `analysis/run_domain_model.py` | domain × band mixed models, 3-way medication design |
| `analysis/plot_domain_model.py` | the 3-figure scheme (A result, B heterogeneity, C todo) |
| `config/roi_schemes.py` | `roi_v2_ofc`, `pain_domains`, `pain_domains_v2`, `_merge_categories` |
| `config/psd_params.py` | `PAPER_BANDS_6_HG200_HZ` |

sbatch: `fullres_grid_{array,collect}`, `fullres_cluster`, `bandpower_{mixed,perm_array,checks}`,
`domain_model`.

20 commits, `7c012eb` → `b760533`, all pushed to `med-analysis`.

---

## 5. Bugs found and fixed

1. **Notch off-by-one** — `±2 Hz` takes 9 bins per harmonic, not 8 (inclusive at
   both ends); 36 of 499, not 8. The repo's own comments in `psd_params` and
   `fullres_reader` were wrong and are corrected.
2. **Conditional residuals double-counted** — `MixedLMResults.fittedvalues`
   ALREADY includes the random effects (verified: differs from `exog @ fe_params`
   by up to 2.6 log units, `resid` SD 0.2140 vs √scale 0.2165). Subtracting Zu
   again produced a spurious residuals-vs-fitted slope of −0.44 that read exactly
   like a failed transform. Now asserted, not assumed.
3. **Multi-session subject counted twice** — sub-209 has two sessions; iterating
   paths entered it as two subjects in the across-subject t (n=52 vs 51). Now
   pooled with within-session centring.
4. **Wald constraint shape** — `res.wald_test` for MixedLM builds against
   `res.params` (fixed effects *then* variance components), so an R matrix sized
   to the fixed effects alone is rejected.
5. **sbatch quoting** — `${EXTRA}` expands unquoted, so `"Lateral Temporal"` was
   split into two arguments and killed a run. Exclusions now travel in their own
   variables.
6. **Level-4 folder name lied** — kept saying `roiv2ofc` after `--roi-scheme` was
   added. Now built from band set + region set + drug set in one place.
7. **Conditional residuals, again, the other way** — `conditional_parts` matched
   hard-coded variance-component NAMES, so it silently omitted the domain model's
   fourth component (`subj_parcel_slope`); the mismatch guard fired and its
   fallback returned the INCOMPLETE vector as the residual. Same class of error
   as #2, opposite sign. Now enumerates `res.k_vc` and uses the model's own
   `exog_vc.mats`, so no `vc_formula` can outrun it, and a mismatch is
   adjudicated by residual SD against `sqrt(scale)` with both numbers logged
   (`32df4d5`).

---

## 6. Live constraints and open questions

**Blocking anything that needs a permutation p:**
The residual ACF is **0.42 at lag 1** (median 2.0 h between assessments),
decaying to 0.17 by lag 6. `mixed_model.permutation_null` shuffles epoch labels
freely within subject, which the data denies — it would give a null too narrow
and p-values too small. Needs a cyclic shift or block permutation first.
(`TASKS.md`)

**Known-bad, excluded:** Lateral Temporal — SEs 6–10× its peers, two
non-convergent cells. Data validation pending.

**Heavy left tail in the residuals** — skew −1.81, excess kurtosis +24.6, lower
tail to −15 SD, no matching upper tail. Looks like dropout or partially-masked
epochs surviving QC; also means the Wald normality assumption is a long way off.
(`SCRATCHPAD.md`)

**Heterogeneity everywhere** — the LRT fires in 118/120 band cells. Every
reported mean is a population average over patients who disagree substantially.

**statsmodels cannot fit crossed random effects** — verified: it silently NESTS
them, with no error. The domain model's parcel term is therefore
`(NRS_within || subject:parcel)`, not crossed with subject. The exact-crossing
check (per-(subject,parcel) OLS + two-way clustered SEs) is in `TASKS.md`.

**Insula and Thalamus** — insula needs the Destrieux/a2009s anterior-posterior
split (labels exist in the collaborator electrodes table but not in
`channel_meta`; a rebuild, not new data). Thalamus is in Sensory by assignment in
v2, carrying the caveat that DK gives one parcel where the ascending pathway is
VPL/VPM.

**Brainstem has zero contacts** — 0 of 7,068 across 51 subjects. "Pain
modulatory" is M1 alone (82 contacts, 20 subjects), the only domain whose
internal consistency cannot be checked.

---

## 7. The plotting scheme (agreed 2026-09-18)

Three figures, one per question, replacing the heatmap-first approach:

- **A — result.** Dot-and-interval matrix, rows band × domain, one panel per term
  (pain | med | pain×med), separate x-scales, omnibus in the footnote. Estimate
  and CI are the primary encoding; colour only groups. *Built.*
- **B — heterogeneity.** Per-subject slope distributions beside the variance
  decomposition: does the average describe anyone, and do the modelling choices
  matter. *Built, not yet run.*
- **C — diagnostics.** Conditional residuals vs fitted, QQ, ACF over real time.
  *Not built for the domain model*; the band-power version exists. Per-band
  residuals are now saved so it needs no refit.

Reading rule that keeps getting lost: both predictors are subject-mean-centred,
so each coefficient is read at the OTHER's patient-specific mean. The pain slope
is at that patient's own average medication level — **not** unmedicated. The
medication effect is at that patient's own average pain — **not** zero pain.

---

## 8. Next

1. Figure B for the five domain runs (~10 min each).
2. Figure C for the domain model — needs a refit at `32df4d5`, since the only
   saved residuals are the invalid ones (`TASKS.md`).
3. Beta band is unidentified in the domain fits (`var_subj_slope` 0.0227 vs
   ~0.0003 elsewhere, Hessian not positive definite). Quote its omnibus, never a
   per-domain beta pain slope, until that is pinned down (`TASKS.md`).
4. Blocked permutation before any permutation p is quoted.
5. Decide whether the >100 Hz nominations are physiology or EMG (`TASKS.md`).
6. Explain the Control domain's low-frequency behaviour — it carries both the
   delta main effect and the largest delta/theta pain × opioid interaction. Until
   it is understood, every low-frequency claim in this project is soft.
