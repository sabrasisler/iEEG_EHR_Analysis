# Region Levels — Channel, Anatomical Parcel, ROI, Domain

The four levels at which a contact can be named, what lives at each, and which
one a given model is fitted at.

> **Authority.** `src/ieeg_ehr/config/roi_schemes.py` is the source of truth for
> every mapping below; `src/ieeg_ehr/analysis/insula_ap.py` owns the one
> coordinate-derived parcel. This doc is derivative — if it disagrees with those
> modules, they win and this file is stale. Read `docs/view_registry.md` for how
> the region axis sits among the other six view axes.

> **Counts** are the 51-subject discovery cohort, from `roi_dk_composition.csv`
> as written by the `pain_domains_v3` run of 2026-09-21, unless a row says
> otherwise. That is the modelling denominator. The full `channel_meta` set
> across all 81 subjects holds 11,267 contacts and 82 distinct label strings;
> those numbers are larger and are NOT what any model saw.

---

## The four levels

| # | Level | Unit | N | Ever a fixed effect? |
|---|---|---|---|---|
| 1 | **Channel** | one bipolar pair | ~2,500 in-model | No — always `(1 \| subject:channel)` |
| 2 | **Anatomical Parcel** | atlas label, hemispheres collapsed, **plus the coordinate-derived insula parcels** | 41 present / 35 usable | Only as a random slope (`--unit parcel`) |
| 3 | **ROI** | a named group of parcels | 21 (`roi_v2`) / 22 (`roi_v2_ins`) | Yes, in the per-cell grids |
| 4 | **Domain** | a processing domain of the pain matrix | 4 fitted | Yes, in the domain model |

Level 2→3 is a **substring match on the atlas label**, and insertion order is
precedence. Level 3→4 is a **membership dict**. The insula is the sole exception
to both: it is split by MNI coordinate, not by label.

The distinction between levels 2 and 3 is load-bearing and is why figures say
"4 ROIs" rather than "4 parcels" — see `run_domain_model.unit_word`.

---

## Level 2 — Anatomical Parcels

The parcellation is **Desikan-Killiany** for cortex (FreeSurfer `aparc`) and the
FreeSurfer **`aseg`** subcortical segmentation, both read from the bipolar pair's
**anode** (`dk_anode` in `channel_meta`).

All **34 DK cortical parcels have at least one contact** in this cohort. One of
them, `frontalpole`, is below the usable floor.

### The coordinate-derived parcels: `aIns` / `pIns`

DK has exactly one `insula` parcel, so an anterior/posterior distinction is not
derivable from the label. `analysis/insula_ap.py` supplies it from the contact's
**MNI y coordinate**: pool every insular bipolar pair across subjects and
hemispheres, take the median, call `y > threshold` anterior.

- Threshold as run: **y > −2.246 → `aIns`**, else `pIns` (ties go posterior).
- Result: **253 aIns / 251 pIns**, 42 / 37 subjects, 0 dropped for missing MNI.
- The threshold is **pinned in provenance as a number**, because it is this
  cohort's median and not an anatomical landmark. A different cohort puts the
  line elsewhere.

**Read this split as coarse.** It is not the Destrieux/a2009s assignment
(`G_insular_short` vs `S_circular_insula_ant/sup/inf`), which remains the correct
eventual fix. The insula is folded, so a plane normal to y is not its
anterior–posterior axis, and contacts near the line are near-arbitrary. It
supports *"on average more anterior"*, never *"this contact is in aIns"*.

A scheme without the coordinate step **drops insula entirely** rather than
assigning it wholesale to one side. The failure mode is a missing region, not a
wrong one — that is deliberate (`roi_schemes._with_coordinate_regions`).

---

## Levels 2 → 3 → 4: the complete map

### Sensory — 887 contacts

| Anatomical Parcel | ROI | n | subj |
|---|---|---|---|
| Thalamus-Proper | Thalamus | 335 | 32 |
| insula *(posterior, by coordinate)* | **pIns** | 251 | 37 |
| supramarginal | S2/PO | 154 | 31 |
| postcentral | S1 | 126 | 20 |
| paracentral | S1 | 21 | 10 |

### Affective — 462 contacts

| Anatomical Parcel | ROI | n | subj |
|---|---|---|---|
| insula *(anterior, by coordinate)* | **aIns** | 253 | 42 |
| Amygdala | Amygdala | 137 | 36 |
| rostralanteriorcingulate | rACC | 40 | 14 |
| caudalanteriorcingulate | dACC | 32 | 13 |

### Cognitive — 1,011 contacts

| Anatomical Parcel | ROI | n | subj |
|---|---|---|---|
| superiorfrontal | dmPFC † | 244 | 34 |
| lateralorbitofrontal | lOFC | 191 | 33 |
| rostralmiddlefrontal | dlPFC | 164 | 25 |
| caudalmiddlefrontal | dlPFC | 46 | 14 |
| parstriangularis | IFG/vlPFC | 120 | 35 |
| parsopercularis | IFG/vlPFC | 116 | 27 |
| parsorbitalis | IFG/vlPFC | 36 | 16 |
| medialorbitofrontal | mOFC | 94 | 30 |

† Currently named `dmPFC/SMA` in the code — see *Pending changes*.

### Modulatory — 82 contacts

| Anatomical Parcel | ROI | n | subj |
|---|---|---|---|
| precentral | M1 | 82 | 20 |

Brainstem patterns (`brain-stem`, `brainstem`) are registered so a future contact
lands correctly, but **zero brainstem contacts exist** in this cohort — measured
across all 7,068 contacts. The descending modulatory arm is therefore **M1
alone**: one parcel, the weakest domain in every figure, and the only one whose
internal consistency cannot be checked, because a single-parcel domain has no
parcel-to-parcel agreement to inspect.

### ROIs in no domain — 1,837 contacts

These carry a full ROI label and **appear in every ROI-level heatmap and
consistency map**. They are simply absent from the domain model.

| ROI | Anatomical Parcels | n |
|---|---|---|
| Lateral Temporal | middletemporal, superiortemporal, inferiortemporal, bankssts | 821 |
| Parietal (other) | precuneus, superiorparietal, inferiorparietal | 334 |
| Hippocampus | Hippocampus | 297 |
| Basal Ganglia | Putamen, Pallidum, Caudate, Accumbens-area | 138 |
| PCC | isthmuscingulate, posteriorcingulate | 129 |
| MTL (other) | fusiform, parahippocampal, temporalpole, entorhinal | 118 |

**PCC is the costly omission** — it carries one of the larger low-frequency pain
effects in this cohort. Its absence from the domain model is a real and
deliberate loss, not an oversight.

`Basal Ganglia` is held out because the framework's affective node is *ventral*
striatum and DK lumps dorsal striatum in with it.

### Dropped before any ROI — 2,531 contacts

Non-neural tissue, unlabeled contacts, and parcels below the 8-subject floor.
These are **counted and logged**, never silently excluded — coverage is a
confound here, so a shrinking denominator has to stay visible.

| Label | n | subj |
|---|---|---|
| Cerebral-White-Matter | 1,631 | 51 |
| Unknown | 680 | 50 |
| undefined | 128 | 39 |
| Inf-Lat-Vent | 25 | 17 |
| *(blank)* | 22 | 1 |
| WM-hypointensities | 15 | 6 |
| Lateral-Ventricle | 12 | 9 |
| choroid-plexus | 8 | 8 |
| VentralDC | 4 | 2 |
| 3rd-Ventricle | 2 | 2 |
| **frontalpole** | **2** | **2** |
| CC_Posterior | 1 | 1 |
| CSF | 1 | 1 |

`frontalpole` is the only DK **cortical** parcel with no ROI — 2 contacts in 2
subjects, far below the 8-subject floor, so it could only ever have been a blank
heatmap row.

---

## Ordering constraints (silent if broken)

Insertion order in the pattern dict **is** precedence, substring match,
case-insensitive. Four constraints are real and pinned by
`tests/test_roi_schemes.py`:

1. Tissue and exclusion categories come **first**, so a malformed or non-neural
   label never reaches anatomy.
2. `Parietal (other)` must precede `Occipital` — **`precuneus` contains
   `cuneus`**.
3. `hippocampus` does **not** match `parahippocampal` (different suffix), which
   is how `MTL (other)` claims the latter.
4. `temporalpole` does **not** match any Lateral Temporal pattern, same reason.

In the older `default` scheme a fifth applies: `vmPFC` must precede
`Frontal (other)`, because both list `frontalpole`.

---

## Scheme registry

### ROI schemes (`ROI_SCHEMES`)

| Name | ROIs | What it is |
|---|---|---|
| `default` | 15 | The original set. `ACC` and `OFC` each one row; `Frontal (other)` and `Temporal` are catch-alls; Occipital and Cerebellum are non-ROI |
| `roi_v2` | 21 | Splits ACC→rACC/dACC and OFC→mOFC/lOFC; breaks the catch-alls into M1, dmPFC/SMA, IFG/vlPFC, MTL (other), Lateral Temporal, Auditory, Parietal (other). No catch-alls remain |
| `roi_v2_ofc` | 20 | `roi_v2` with mOFC and lOFC fused back into one `OFC` |
| `roi_v2_ins` | 22 | `roi_v2` with `Insula` replaced by `aIns` + `pIns` |
| `roi_v2_ofc_ins` | 21 | Both of the above |
| a `.json` path | — | A scheme file on Oak, so a region set changes without a commit |

### Domain schemes (`DOMAIN_SCHEMES`)

| Name | Base ROI scheme | Domains |
|---|---|---|
| `pain_domains` | `roi_v2_ofc` | Sensory, Affective, Cognitive, Memory, Control |
| `pain_domains_v2` | `roi_v2` | Sensory, Affective, Cognitive, Modulatory, Control |
| `pain_domains_v3` | `roi_v2_ins` | Sensory, Affective, Cognitive, Modulatory, Control |

**A scheme is never edited in place once a run cites it.** A logged run records
the scheme by name, and rewriting that name's meaning would make the run's record
misleading. New assignment → new version. (Provenance also stores the scheme's
full *contents*, so an old run stays reconstructable either way.)

`ROI_SCHEMES` holds a **fused** label→domain map for each domain scheme, for
callers that want one-step lookup. `DOMAIN_SCHEMES` holds the same information
with the **ROI layer left in**, which is what `--unit roi` needs: the ROI is the
unit that maps to a domain, and the region-level consistency map has to show ROIs
that no domain contains.

---

## Which level each model is fitted at

| Script | One row is | Fixed effect | Region in the model? |
|---|---|---|---|
| `run_bandpower_mixed.py` | channel × epoch, within one **ROI × band** cell | `NRS_within` | No — the ROI *is* the cell |
| `run_fullres_grid.py` | channel × epoch, within one **ROI × freq-bin** cell | `NRS_within` | No |
| `run_domain_model.py --unit parcel` | channel × epoch, whole cohort | `NRS_within * C(domain)` | Yes — **parcel** as a random slope, nested in subject |
| `run_domain_model.py --unit roi` | channel × epoch, whole cohort | `NRS_within * C(domain)` | No — the ROI only looks up the domain |

Nothing is averaged at any level: one row is always one channel × one epoch. The
level changes what a row is *labelled*, never what it *is*.

### Why `subject:parcel` and not `domain:region`

A parcel term shared across subjects is a **crossed** random effect — parcels
appear in many patients, patients contribute many parcels. `statsmodels.MixedLM`
takes one grouping variable and evaluates every `vc_formula` term *within* it;
verified on synthetic data, the crossed form comes back silently **nested**, with
one variance and a per-subject design matrix, and no warning. So parcel is nested
in subject, which is what the library can honestly fit. The claim is weaker than
it looks: the domain effect is not driven by any one subject-parcel, but the
domain standard error does **not** account for a parcel deviating consistently
*across* patients. The exact-crossing counterpart — per-(subject, parcel) OLS
slopes with subject- and parcel-clustered standard errors — is the robustness
check this model wants and is tracked in `TASKS.md`.

### What `--unit roi` gives up

With no region term, nothing in the fit guards against one well-sampled ROI
carrying its domain. That guard does not disappear, it **moves**: the region-level
consistency map (`run_bandpower_mixed` + `plot_bandpower_consistency`) fits every
ROI separately and reports how many subjects share each one's sign, including the
ROIs no domain contains. **Read the two together; neither alone is the answer.**

---

## Standing caveats

- **Anode-based assignment is a temporary stand-in.** A bipolar pair takes the
  parcel of its *anode*, which is wrong whenever the two contacts straddle a
  boundary. The intended replacement is a lookup on the pair's midpoint
  coordinate. Treat assignment near parcel boundaries as approximate. Note the
  asymmetry: the insula split uses the **midpoint** coordinate while the label
  comes from the **anode**.
- **Thalamus is one parcel.** The ascending sensory pathway is VPL/VPM
  specifically, while medial and dorsal nuclei sit in the affective pathway, so a
  Sensory effect partly reflects the latter. Destrieux will not fix this — it is a
  surface parcellation. This needs a nucleus-level atlas. The caveat is
  **accepted deliberately**, not avoided.
- **`S2/PO` is a proxy.** True S2 sits in the parietal operculum, which DK does
  not isolate; `supramarginal` is the stand-in.
- **Coverage is a confound.** Which subjects contribute to a region is not random.
  Always report contributing channel and subject n, and read
  `provenance.json` `subjects[]` for a run's membership — **never the folder
  name**.
- **Reading the region list off `config.ROI_REGIONS` is a bug.** That constant is
  the `default` scheme's 15 regions, resolved at import. Filtering a `roi_v2`
  view against it silently keeps 8 of 21 with no error. Use
  `analysis/view_tables.roi_regions_for(view_params)`.

---

## Pending changes — decided, NOT yet in the code

As of 2026-09-22 these are agreed but unimplemented. The code still ships
`pain_domains_v3` with a `Control` domain and an ROI named `dmPFC/SMA`.

**Update 2026-09-23:** the schemes are now registered in
`config/roi_schemes.py` — `roi_v3` / `roi_v3_ins` (the rename), `ROI_RENAMES`
(old → new label, for displaying older runs), and `pain_domains_v4` (in
`DOMAIN_SCHEMES` only, `Other` not a domain). Still unimplemented: the domain
model's contrast coding without a `Control` reference, and any refit under v4.
`plot_band_map_domains` already displays with them.

### 1. `Control` is removed; everything unassigned becomes `Other`

Occipital cortex is not a defensible negative control for pain. The `Control`
domain goes away. Every ROI outside a real domain — including Auditory and
Occipital — is labelled **`Other`**, which is **not fitted and not included in any
domain analysis by default**, but **is** shown in ROI-level heatmaps.

Proposed `pain_domains_v4`, 4 fitted domains over 14 ROIs:

| Domain | ROIs | contacts |
|---|---|---|
| Sensory | S1, S2/PO, Thalamus, pIns | 887 |
| Affective | aIns, Amygdala, rACC, dACC | 462 |
| Cognitive | dmPFC, lOFC, dlPFC, IFG/vlPFC, mOFC | 1,011 |
| Modulatory | M1 | 82 |
| *`Other` — not fitted* | Auditory, Occipital, PCC, Parietal (other), Hippocampus, MTL (other), Basal Ganglia, Lateral Temporal | ~1,923 |

Two implementation notes:

- **`FALLBACK = 'Other'` already exists** in `roi_schemes.py` as the no-match
  category. Naming a domain `Other` risks silently collecting unmatched labels
  into it via `_merge_categories`. Register `pain_domains_v4` in
  `DOMAIN_SCHEMES` only, and keep `Other` out of `display` — the existing
  "absent from display is dropped" mechanism then does the right thing without a
  fused scheme being built.
- **Removing `Control` breaks the formula literally.**
  `C(domain, Treatment('Control'))` raises once `Control` is not a level, and a
  new contrast coding has to be chosen. Contrast coding is to become a CLI flag
  recorded in provenance, defaulting to **sum coding** (`NRS_within` = the mean
  slope across domains, each interaction a deviation from it), so no domain is
  privileged — which is what dropping the reference implies.

**Cost, stated plainly:** the model loses its negative control. The control
function survives at ROI level, where Occipital and Auditory keep their heatmap
rows.

### 2. `dmPFC/SMA` → `dmPFC`

The parcel does **not** straddle SMA and dmPFC in this cohort. Measured over all
81 subjects' `channel_meta` (378 `superiorfrontal` contacts, 49 subjects):

| | contacts | subjects |
|---|---|---|
| Anterior to the VAC line (MNI y > 0) | **339** | **49** |
| Posterior (y ≤ 0) — the SMA side | 39 | 9 |
| Medial *and* posterior (\|x\| ≤ 15, y < 0) — SMA proper | **26** | **6** |

Median MNI y is **+34**; the 10th percentile is **−0**. A clean anatomical
landmark exists (y = 0, the VAC line — genuinely stronger than the insula's
data-dependent median), but **there is nothing on the posterior side to fit**: 6
subjects, below the 8-subject floor.

So the ROI is renamed **`dmPFC`**, all 378 contacts, staying in Cognitive. This
**retires the standing caveat** that its placement in Cognitive was a judgment
call because `superiorfrontal` straddles SMA — the data says it does not.

Because a run cites its scheme by name, this needs a new base scheme (`roi_v3`,
`roi_v3_ins`) rather than an edit to `roi_v2`.

### 3. Modulatory cannot support a between-subject contrast

M1 alone, 20 subjects. Split into MDD arms it gives 7/13, below the 8-subject
floor, and the dx fit is refused in all 6 bands. With `Control` gone this is the
one remaining domain that cannot answer a between-subject question. The MDD
contrast is answerable only in **Sensory (43), Affective (47), Cognitive (46)**.

---

## See also

- `docs/view_registry.md` — the seven view axes; region aggregation is AXIS 6 and
  the ROI scheme is a separate choice within it
- `docs/cluster_permutation.md` — how an outline on a region × frequency heatmap
  is computed and what it does and does not license
- `docs/architecture.md` — the layer model the regions sit inside
- `DECISIONS.md` — the settled calls and their reasons, append-only
