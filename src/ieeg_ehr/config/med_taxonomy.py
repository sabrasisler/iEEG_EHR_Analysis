"""
Medication name -> (Level 1 class, Level 2 subclass), held as data in config.

Same shape as `roi_schemes.py`: a hand-curated mapping that is *configuration*,
not data, so it lives in the repo where it is version-controlled, diffable, and
reviewable. Drug names are not PHI. The code/data boundary in CLAUDE.md is about
derivatives and outputs, not about a lookup table that decides what a script
computes.

PROVENANCE
----------
Seeded from a colleague's hand-maintained table,
`/home/groups/ckeller1/sandon/med_admin/medications_classified.csv` (170 rows,
columns `Medication, Acts on CNS, Level 1 Class, Level 2 Class, Primary
Receptor, Acts on GABA-A, Acts on D2`). That file is NOT read at runtime — it is
another user's home directory and could change under us. It is recorded as a
provenance parent by `med_analysis` runs via `TAXONOMY_SOURCE_CSV` so the
lineage stays traceable, and copied to Oak once as a frozen snapshot.

WHAT CHANGED FROM THE SOURCE TABLE
----------------------------------
The source table populates class fields only where `Acts on CNS == Yes`, and it
was built for a benzodiazepine analysis. That leaves the non-opioid analgesics
completely unclassified — ACETAMINOPHEN, IBUPROFEN, KETOROLAC, the LIDOCAINEs
and ROCURONIUM all sit at `('', '', 'No')`, so every loader in that repo filters
them straight out. Opioids were already correct (`Analgesics / Opioids`).

Additions made here, and nothing else:

- `Analgesics / Acetaminophen` — ACETAMINOPHEN
- `Analgesics / NSAIDs`        — IBUPROFEN, KETOROLAC
- `Anesthetics / Local anesthetics`     — the four LIDOCAINE entries
- `Anesthetics / Neuromuscular blockers` — ROCURONIUM, SUGAMMADEX (reversal)

`Acts on CNS` is deliberately NOT carried over. It was the source table's
filtering mechanism, and it is exactly what made acetaminophen and the NSAIDs
invisible. Membership here is decided by class, which is the question actually
being asked.

BUTALBITAL-ACETAMINOPHEN-CAFF is left where the source put it, under
`Anticonvulsants / Barbiturates-GABAergic`, rather than being pulled into the
analgesic set. It is a genuine borderline call: the product is a tension-headache
analgesic containing acetaminophen, but its active CNS component is a
barbiturate. Keeping the source's call means our opioid/acetaminophen counts stay
comparable to the benzodiazepine figures. Twenty administrations turn on it. Flip
`BUTALBITAL_IS_ANALGESIC` to move it and re-run; nothing else needs to change.
"""

from pathlib import Path

# Recorded as a provenance parent; never read at runtime. See module docstring.
TAXONOMY_SOURCE_CSV = Path(
    '/home/groups/ckeller1/sandon/med_admin/medications_classified.csv')

BUTALBITAL_IS_ANALGESIC = False


# ============================================================================
# THE MAPPING
# ============================================================================
# Grouped by (Level 1, Level 2) rather than one row per drug: the grouping IS
# the information, and a flat 170-line dict hides which classes are thin. The
# flat drug -> class dict is built once at import, below.

_CLASS_MEMBERS = {
    ('Analgesics', 'Opioids'): (
        'ACETAMINOPHEN-CODEINE', 'BUPRENORPHINE-NALOXONE', 'FENTANYL',
        'FENTANYL (PF)', 'FENTANYL CITRATE (PF)', 'HYDROCODONE-ACETAMINOPHEN',
        'HYDROMORPHONE', 'MORPHINE', 'MORPHINE INJECTABLE SYRINGE',
        'OXYCODONE', 'OXYCODONE-ACETAMINOPHEN', 'TRAMADOL',
    ),
    # ADDED HERE — absent from the source table (see module docstring).
    ('Analgesics', 'Acetaminophen'): (
        'ACETAMINOPHEN',
    ),
    ('Analgesics', 'NSAIDs'): (
        'IBUPROFEN', 'KETOROLAC',
    ),
    ('Anesthetics', 'General anesthetics'): (
        'PROPOFOL',
    ),
    ('Anesthetics', 'Local anesthetics'): (
        'LIDOCAINE', 'LIDOCAINE (PF)', 'LIDOCAINE HCL', 'LIDOCAINE-EPINEPHRINE',
    ),
    ('Anesthetics', 'Neuromuscular blockers'): (
        'ROCURONIUM', 'SUGAMMADEX',
    ),

    # --- unchanged from the source table ---------------------------------
    ('Anxiolytics', 'Benzodiazepines'): (
        'ALPRAZOLAM', 'CLOBAZAM', 'CLONAZEPAM', 'CLORAZEPATE DIPOTASSIUM',
        'DIAZEPAM', 'LORAZEPAM', 'MIDAZOLAM', 'MIDAZOLAM (PF)',
    ),
    ('Anxiolytics', 'Gabapentinoids (Calcium channel alpha2delta ligands)'): (
        'GABAPENTIN', 'PREGABALIN',
    ),
    ('Anticonvulsants', 'Na channel blockers'): (
        'CARBAMAZEPINE', 'ESLICARBAZEPINE', 'LACOSAMIDE', 'LACOSAMIDE IVPB',
        'LAMOTRIGINE', 'OXCARBAZEPINE', 'PHENYTOIN', 'RUFINAMIDE',
        'TOPIRAMATE', 'ZONISAMIDE',
    ),
    ('Anticonvulsants', 'Broad/multiple mechanism'): (
        'CANNABIDIOL', 'CENOBAMATE', 'DIVALPROEX', 'FELBAMATE', 'VALPROIC ACID',
    ),
    ('Anticonvulsants', 'SV2A ligands'): (
        'BRIVARACETAM', 'LEVETIRACETAM',
    ),
    ('Anticonvulsants', 'Barbiturates/GABAergic'): (
        'BUTALBITAL-ACETAMINOPHEN-CAFF', 'PHENOBARBITAL',
    ),
    ('Anticonvulsants', 'AMPA antagonist'): (
        'PERAMPANEL',
    ),
    ('Antiemetics', 'Antiemetics'): (
        'APREPITANT', 'GRANISETRON', 'METOCLOPRAMIDE HCL', 'ONDANSETRON',
        'ONDANSETRON HCL (PF)', 'PROCHLORPERAZINE',
    ),
    ('Antidepressants', 'SSRIs'): (
        'CITALOPRAM', 'ESCITALOPRAM', 'FLUOXETINE', 'SERTRALINE',
    ),
    ('Antidepressants', 'SNRIs'): (
        'DULOXETINE', 'VENLAFAXINE',
    ),
    ('Antidepressants', 'Other'): (
        'BUPROPION HCL', 'MIRTAZAPINE', 'VORTIOXETINE',
    ),
    ('Antipsychotics', 'Second-generation antipsychotics'): (
        'ASENAPINE MALEATE', 'QUETIAPINE', 'RISPERIDONE',
    ),
    ('Antipsychotics', 'First-generation antipsychotics'): (
        'CHLORPROMAZINE', 'HALOPERIDOL LACTATE',
    ),
    ('Muscle relaxants', 'Muscle relaxants'): (
        'CARISOPRODOL', 'CYCLOBENZAPRINE',
    ),
    ('First generation antihistamines', 'First generation antihistamines'): (
        'DIPHENHYDRAMINE HCL', 'HYDROXYZINE HCL',
    ),
    ('Hypnotics', 'Melatonergic agonists'): (
        'MELATONIN', 'RAMELTEON',
    ),
    ('Hypnotics', 'Trazodone'): ('TRAZODONE',),
    ('Hypnotics', 'Daridorexant'): ('DARIDOREXANT',),
    ('Anticholinergics', 'Anticholinergics'): ('BENZTROPINE',),
    ('Alcohols', 'Alcohols'): ('BEER', 'VODKA'),
}

if BUTALBITAL_IS_ANALGESIC:                                  # pragma: no cover
    _CLASS_MEMBERS[('Anticonvulsants', 'Barbiturates/GABAergic')] = (
        'PHENOBARBITAL',)
    _CLASS_MEMBERS[('Analgesics', 'Acetaminophen')] += (
        'BUTALBITAL-ACETAMINOPHEN-CAFF',)

#: canonical drug name -> (level1, level2). Built from _CLASS_MEMBERS.
DRUG_CLASS = {
    drug: classes
    for classes, drugs in _CLASS_MEMBERS.items()
    for drug in drugs
}

# Drugs present in the MAR export that are deliberately unclassified: supportive
# care, antibiotics, cardiac, vitamins, IV fluids. Listed rather than left to
# fall through, so `classify()` can distinguish "known, no class" from "we have
# never seen this drug" — the second is a data change that should be loud.
UNCLASSIFIED_DRUGS = frozenset({
    'ACIDOPHILUS-L.B-BB-S.THERMOPHL', 'ALUM-MAG HYDROXIDE-SIMETH', 'AMLODIPINE',
    'ARTIFICIAL TEARS(HYPROMELLOSE)', 'ATENOLOL', 'ATORVASTATIN', 'AZELASTINE',
    'BACITRACIN-POLYMYXIN B', 'BENZOCAINE', 'BENZOCAINE-MENTHOL', 'BISACODYL',
    'BRIMONIDINE', 'CALAMINE-ZINC OXIDE', 'CALCIUM CARBONATE',
    'CAMPHOR-MENTHOL', 'CARBOXYMETHYLCELLULOSE SODIUM', 'CARVEDILOL',
    'CEFAZOLIN', 'CEFTRIAXONE', 'CEPHALEXIN', 'CETIRIZINE',
    'CHOLECALCIFEROL (VITAMIN D3)', 'CLINDAMYCIN HCL',
    'CYANOCOBALAMIN (VITAMIN B-12)', 'DEXAMETHASONE', 'DOCUSATE SODIUM',
    'DOXAZOSIN', 'ENALAPRIL MALEATE', 'ENOXAPARIN', 'EPHEDRINE SULFATE',
    'ERGOCALCIFEROL (VITAMIN D2)', 'FAMOTIDINE', 'FERROUS SULFATE',
    'FINASTERIDE', 'FLU VACC QS2017-18', 'FLUTICASONE',
    'FLUTICASONE PROPIONATE', 'FOLIC ACID', 'FUROSEMIDE', 'GLIMEPIRIDE',
    'GLYCOPYRROLATE', 'HEPARIN, PORCINE (PF)', 'HYDRALAZINE', 'INSULIN ASPART',
    'INSULIN LISPRO', 'INSULIN REGULAR HUMAN', 'LACTULOSE', 'LEVOTHYROXINE',
    'LISINOPRIL', 'LOSARTAN', 'MAGNESIUM CITRATE PO SOLN',
    'MAGNESIUM HYDROXIDE', 'MAGNESIUM SULFATE IV SCALE', 'METFORMIN',
    'METHOTREXATE SODIUM (PF)', 'METOPROLOL SUCCINATE', 'METOPROLOL TARTRATE',
    'MONTELUKAST', 'MULTIVITAMIN (GENERIC) PO TABS',
    'NEOMYCIN-BACITRACNZN-POLYMYXNB', 'NS IV BOLUS', 'NS IV BOLUS -',
    'OTHER DRUG', 'OXYMETAZOLINE', 'PANTOPRAZOLE', 'PHENYLEPHRINE HCL',
    'POLYETHYLENE GLYCOL', 'POTASSIUM CHLORIDE', 'POTASSIUM CHLORIDE IV SCALE',
    'POTASSIUM CHLORIDE ORAL SCALE', 'POTASSIUM, SODIUM PHOSPHATES',
    'PRO-STAT SUGAR FREE TF MODULAR', 'PROGESTERONE MICRONIZED', 'PROPRANOLOL',
    'PSEUDOEPHEDRINE HCL', 'PYRIDOXINE (VITAMIN B6)', 'ROSUVASTATIN',
    'SENNOSIDES', 'SIMETHICONE', 'SULFASALAZINE', 'TAMSULOSIN',
    'THERAPEUTIC MULTIVITAMIN (GENERIC) PO TABS',
    'THERAPEUTIC MULTIVITAMIN PO TABS', 'THYROID (PORK)', 'TIMOLOL MALEATE',
    'TRIMETHOPRIM-POLYMYXIN B', 'VALSARTAN',
})


# ============================================================================
# NAME CLEANING
# ============================================================================
# Product strings carry strength and formulation ("FENTANYL CITRATE (PF) 50
# MCG/ML INJ SOLN"). `med_analysis.load.extract_medication_name` cuts at the
# first numeric token, then consolidates on the leading word using this map, so
# formulation variants of one drug become one drug.

#: leading word -> canonical name. Extends the colleague's map with the opioid
#: variants his benzodiazepine analysis never had to resolve. Without MORPHINE
#: here, "MORPHINE 2 MG/ML INJ SYRG" and "MORPHINE INJECTABLE SYRINGE" (no
#: numeric token to cut at) stay two different drugs; without FENTANYL, the
#: three fentanyl products stay three.
CONSOLIDATE_FIRST_WORD = {
    'CEFAZOLIN': 'CEFAZOLIN',
    'DEXAMETHASONE': 'DEXAMETHASONE',
    'ESCITALOPRAM': 'ESCITALOPRAM',
    'FAMOTIDINE': 'FAMOTIDINE',
    'FENTANYL': 'FENTANYL',
    'GRANISETRON': 'GRANISETRON',
    'LACOSAMIDE': 'LACOSAMIDE',
    'LEVETIRACETAM': 'LEVETIRACETAM',
    'LIDOCAINE': 'LIDOCAINE',
    'MIDAZOLAM': 'MIDAZOLAM',
    'MORPHINE': 'MORPHINE',
    'ONDANSETRON': 'ONDANSETRON',
    'PHENYTOIN': 'PHENYTOIN',
    'PROCHLORPERAZINE': 'PROCHLORPERAZINE',
}

#: Products with more than one active ingredient. Flagged on every row as
#: `is_combination` and, critically, kept OUT of the acetaminophen PETH column:
#: hydrocodone-acetaminophen co-occurs with acetaminophen in the zero bin 100%
#: of the time by definition, which is an artifact of the product, not a
#: prescribing pattern. The class mapping already does this for us (these are
#: classified as Opioids, not Acetaminophen); the flag makes it assertable.
COMBINATION_DRUGS = frozenset({
    'ACETAMINOPHEN-CODEINE',
    'BUPRENORPHINE-NALOXONE',
    'BUTALBITAL-ACETAMINOPHEN-CAFF',
    'HYDROCODONE-ACETAMINOPHEN',
    'LIDOCAINE-EPINEPHRINE',
    'OXYCODONE-ACETAMINOPHEN',
})


# ============================================================================
# DRUG SETS
# ============================================================================

#: The Level 2 subclasses that make up the analgesic drug set.
ANALGESIC_SUBCLASSES = ('Opioids', 'Acetaminophen', 'NSAIDs')

#: Excluded from the analgesic set. Kept in the taxonomy rather than deleted so
#: the exclusion is a visible predicate instead of a missing row. The MAR export
#: does not capture OR/procedural medication: across all 98 sessions there is 1
#: propofol administration, 3 rocuronium, 21 lidocaine (mostly topical), no
#: ketamine / dexmedetomidine / remifentanil, and not one row with an
#: `infusion_rate`. There is no anesthetic exposure to analyze here.
ANESTHETIC_SUBCLASSES = ('General anesthetics', 'Local anesthetics',
                         'Neuromuscular blockers')

#: Plot/column order for the analgesic subclasses, commonest first.
ANALGESIC_SUBCLASS_ORDER = ('Opioids', 'Acetaminophen', 'NSAIDs')


def drugs_in_subclasses(subclasses):
    """Canonical drug names whose Level 2 class is in `subclasses`."""
    wanted = set(subclasses)
    return frozenset(d for d, (_l1, l2) in DRUG_CLASS.items() if l2 in wanted)


ANALGESIC_DRUGS = drugs_in_subclasses(ANALGESIC_SUBCLASSES)
ANESTHETIC_DRUGS = drugs_in_subclasses(ANESTHETIC_SUBCLASSES)


# ============================================================================
# CO-ADMINISTRATION CLASSES  (the PETH columns)
# ============================================================================
# Mostly Level 1, except the two places where Level 1 pools drugs that behave
# differently: Anxiolytics splits into Benzodiazepines vs Gabapentinoids, and
# Analgesics splits into its three subclasses. Both splits exist because the
# question is about co-prescribing, and a gabapentinoid handed out with an
# opioid means something different from a benzodiazepine handed out with one.

_LEVEL1_PASSTHROUGH = frozenset({
    'Anticonvulsants', 'Antiemetics', 'Antidepressants', 'Muscle relaxants',
    'First generation antihistamines', 'Hypnotics', 'Antipsychotics',
})

_LEVEL2_PASSTHROUGH = frozenset(ANALGESIC_SUBCLASSES)

#: PETH column order. Opioids first (breakthrough dosing on top of a scheduled
#: opioid is the point of including it), then the other analgesics, then the
#: classes an analgesic is plausibly co-prescribed with.
COADMIN_CLASS_ORDER = (
    'Opioids', 'Acetaminophen', 'NSAIDs', 'Anticonvulsants',
    'Benzodiazepines', 'Gabapentinoids', 'Antiemetics',
)

#: Human-readable column headers. Acetaminophen is labelled explicitly because
#: combination products are excluded from it by construction (see
#: COMBINATION_DRUGS) and the reader has no way to know that from "Acetaminophen".
COADMIN_CLASS_LABELS = {
    'Acetaminophen': 'Acetaminophen\n(single-ingredient)',
}


def coadmin_class(level1, level2):
    """Map a (level1, level2) pair onto a PETH column, or None if untracked."""
    if level2 in _LEVEL2_PASSTHROUGH:
        return level2
    if level1 == 'Anxiolytics':
        l2 = (level2 or '').lower()
        if 'benzodiazepine' in l2:
            return 'Benzodiazepines'
        if 'gabapentinoid' in l2:
            return 'Gabapentinoids'
        return None
    if level1 in _LEVEL1_PASSTHROUGH:
        return level1
    return None


def classify(drug):
    """(level1, level2) for a canonical drug name.

    Returns `('', '')` for a drug that is known-but-unclassified (supportive
    care, antibiotics, ...). Raises for a drug that is not in the taxonomy at
    all — that means the MAR export gained a product we have never seen, which
    is a data change worth stopping for rather than a row to silently drop.
    """
    if drug in DRUG_CLASS:
        return DRUG_CLASS[drug]
    if drug in UNCLASSIFIED_DRUGS:
        return ('', '')
    raise KeyError(
        f'{drug!r} is not in the medication taxonomy. It is a new product in the '
        f'MAR export. Add it to _CLASS_MEMBERS (if it has a class) or to '
        f'UNCLASSIFIED_DRUGS (if it does not) in config/med_taxonomy.py — do not '
        f'drop it silently.')


def known_drugs():
    """Every drug name the taxonomy accounts for, classified or not."""
    return frozenset(DRUG_CLASS) | UNCLASSIFIED_DRUGS
