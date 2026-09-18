"""
Desikan-Killiany parcel -> ROI category mapping, as DATA rather than control flow.

WHY THIS IS A DICT AND NOT AN if/elif CHAIN
-------------------------------------------
This replaces a ~90-line `categorize_desikan_killiany()` if/elif ladder in
pain_params.py whose *branch order* was load-bearing (specific parcels had to be
tested before general catch-alls) and whose display list lived separately, so
changing one ROI meant editing two places and reasoning about control flow.

Here, insertion order IS the precedence order. That is the whole trick: the
ordering constraints the old comments described are now expressible as "where the
key sits". Three of them are real and easy to break, so they are called out at
the point they matter below:

  - `Parietal` must precede `Occipital`, because 'precuneus' contains 'cuneus'.
  - `vmPFC` must precede `Frontal (other)`, because both list 'frontalpole'.
  - the tissue/exclusion categories must precede everything anatomical.

BEHAVIOUR IS FROZEN, NOT REINTERPRETED. tests/test_roi_schemes.py carries a
verbatim copy of the original if-chain and asserts this module returns identical
categories for every DK label present in the cohort, plus the edge cases above.
If you change a pattern, that test is what tells you what you changed.

SWAPPING SCHEMES
----------------
`resolve_roi_scheme()` takes a built-in name OR a path to a JSON file on Oak, so
an ROI set can be added or edited without a commit -- which is what "easy to
change" has to mean when the alternative is a code change plus a review. A run's
provenance records the scheme's full CONTENTS, not just its name, because a named
scheme read from a file is not reconstructable later from the name alone.

WHAT IS DELIBERATELY NOT SOLVED HERE
------------------------------------
`S2 (supramarginal)` is a proxy: true S2 sits in the parietal operculum, which DK
does not isolate. And the sensory/affective theory sets in docs/view_registry.md
need anterior vs posterior insula, which DK has as ONE parcel. Both are
definitional questions, not configuration ones. This module makes them cheap to
revisit; it does not answer them.
"""

import json
from pathlib import Path

import pandas as pd

# Categories that exist so that a channel can be positively identified as
# something we are choosing NOT to analyse, rather than silently vanishing. A
# label mapping here is dropped from region-level output, and callers MUST log how
# many they dropped (see features/common.add_region).
NON_ROI_CATEGORIES = (
    'Exclude', 'White Matter', 'CSF/Ventricles', 'Occipital', 'Cerebellum',
    'Other', 'Unlabeled',
)

# Returned when the DK label is null/absent. Not a pattern match -- there is
# nothing to match against -- so it is handled before pattern testing.
UNLABELED = 'Unlabeled'

# The fallthrough when nothing matches. Named rather than inlined so a scheme
# read from JSON gets the same behaviour without having to encode it.
FALLBACK = 'Other'


# ---------------------------------------------------------------------------
# The default scheme
# ---------------------------------------------------------------------------
# ORDER IS PRECEDENCE. Substring match, case-insensitive, on the DK label.

_DEFAULT_PATTERNS = {
    # -- Exclude / artifacts. First, so a malformed label never reaches anatomy.
    'Exclude': ['empty', 'unknown', 'undefined'],

    # -- Non-neural tissue. Before anatomy for the same reason.
    'White Matter': ['white-matter', 'ventraldc', 'cc_', 'wm-'],
    'CSF/Ventricles': ['ventricle', 'csf', 'choroid-plexus', 'hypointensities'],

    # -- Subcortical. NOTE 'hippocampus' does NOT catch 'parahippocampal'
    # (different suffix), which is why Temporal can claim the latter below.
    'Hippocampus': ['hippocampus'],
    'Amygdala': ['amygdala'],
    'Thalamus': ['thalamus'],
    'Basal Ganglia': ['caudate', 'putamen', 'pallidum', 'accumbens'],

    # -- Insula. ONE parcel in DK; anterior/posterior is not derivable from the
    # label, which is what blocks the theory sets (see module docstring).
    'Insula': ['insula'],

    # -- Cingulate
    'ACC': ['caudalanteriorcingulate', 'rostralanteriorcingulate'],
    'PCC': ['posteriorcingulate', 'isthmuscingulate'],

    # -- Prefrontal: specific parcels BEFORE the general frontal catch-all.
    # ORDERING CONSTRAINT: 'frontalpole' appears here AND in Frontal (other);
    # vmPFC must win, so it must stay above it.
    'vmPFC': ['medialorbitofrontal', 'frontalpole'],
    'OFC': ['lateralorbitofrontal'],
    'dlPFC': ['rostralmiddlefrontal', 'caudalmiddlefrontal'],

    # -- Somatosensory
    'S1': ['postcentral'],
    'S2 (supramarginal)': ['supramarginal'],   # proxy, not true S2 -- see docstring

    # -- Remaining frontal (SFG etc.). Catch-all: must stay below vmPFC/OFC/dlPFC.
    'Frontal (other)': ['frontal', 'frontalpole', 'precentral', 'paracentral',
                        'parsopercularis', 'parsorbitalis', 'parstriangularis'],

    # -- Temporal
    'Temporal': ['temporal', 'fusiform', 'entorhinal', 'parahippocampal',
                 'bankssts', 'transversetemporal', 'temporalpole'],

    # -- Parietal. ORDERING CONSTRAINT: must precede Occipital, because
    # 'precuneus' contains the substring 'cuneus'.
    'Parietal': ['parietal', 'precuneus'],

    # -- Occipital
    'Occipital': ['occipital', 'cuneus', 'pericalcarine', 'lingual'],

    'Cerebellum': ['cerebellum'],
}

# Rows of the region x freq-bin figures, in plot order. Everything else
# _DEFAULT_PATTERNS can return is a NON_ROI_CATEGORY and is dropped.
_DEFAULT_DISPLAY = [
    'Hippocampus', 'Amygdala', 'Thalamus', 'Basal Ganglia', 'Insula', 'ACC', 'PCC',
    'vmPFC', 'OFC', 'dlPFC', 'S1', 'S2 (supramarginal)', 'Frontal (other)',
    'Temporal', 'Parietal',
]

# ---------------------------------------------------------------------------
# roi_v2 -- the finer scheme (2026-07-29)
# ---------------------------------------------------------------------------
# 21 regions. Splits three things `default` merged, because they are functionally
# distinct in the pain literature and DK can separate them:
#   ACC  -> rACC / dACC          OFC -> mOFC / lOFC
#   the `Frontal (other)` and `Temporal` CATCH-ALLS -> M1, dmPFC/SMA, IFG/vlPFC,
#   MTL (other), Lateral Temporal, Auditory, Parietal (other)
#
# Two departures from `default` worth knowing:
#   - Occipital is a REAL ROI here, not a NON_ROI category. It is a useful
#     quasi-control: an occipital pain effect argues for a global or artifactual
#     driver rather than nociception.
#   - There are NO catch-alls. Anything unmatched lands in FALLBACK ('Other') and
#     is dropped WITH A LOGGED COUNT, which is how `frontalpole` (2 subjects) and
#     `cerebellum` (0) leave: both were measured below the 8-subject floor on
#     2026-07-29 and would only ever have been a blank or untestable row.
#
# ORDER IS PRECEDENCE (substring, case-insensitive). Three constraints hold here;
# all three are pinned by tests because breaking one is silent:
#   - the tissue/exclusion categories come FIRST, so a malformed or non-neural
#     label never reaches anatomy;
#   - `Parietal (other)` MUST precede `Occipital`, because 'precuneus' contains
#     'cuneus';
#   - 'hippocampus' does not match 'parahippocampal', and 'temporalpole' does not
#     match any Lateral Temporal pattern, so MTL (other) can claim both.
_ROI_V2_PATTERNS = {
    'Exclude': ['empty', 'unknown', 'undefined'],
    'White Matter': ['white-matter', 'ventraldc', 'cc_', 'wm-'],
    'CSF/Ventricles': ['ventricle', 'csf', 'choroid-plexus', 'hypointensities'],

    'Hippocampus': ['hippocampus'],
    'Amygdala': ['amygdala'],
    'Thalamus': ['thalamus'],
    'Basal Ganglia': ['caudate', 'putamen', 'pallidum', 'accumbens'],
    'Insula': ['insula'],
    'rACC': ['rostralanteriorcingulate'],
    'dACC': ['caudalanteriorcingulate'],
    'PCC': ['posteriorcingulate', 'isthmuscingulate'],
    'mOFC': ['medialorbitofrontal'],
    'lOFC': ['lateralorbitofrontal'],
    'dlPFC': ['rostralmiddlefrontal', 'caudalmiddlefrontal'],
    'dmPFC/SMA': ['superiorfrontal'],
    'IFG/vlPFC': ['parsopercularis', 'parstriangularis', 'parsorbitalis'],
    'M1': ['precentral'],
    'S1': ['postcentral', 'paracentral'],
    'S2/PO': ['supramarginal'],
    # Before Occipital: 'precuneus' contains 'cuneus'.
    'Parietal (other)': ['superiorparietal', 'inferiorparietal', 'precuneus'],
    'MTL (other)': ['entorhinal', 'parahippocampal', 'temporalpole', 'fusiform'],
    'Lateral Temporal': ['superiortemporal', 'middletemporal', 'inferiortemporal',
                         'bankssts'],
    'Auditory': ['transversetemporal'],
    'Occipital': ['lateraloccipital', 'cuneus', 'pericalcarine', 'lingual'],
}

# Display order is anatomical/functional, not the pattern order: subcortical,
# limbic/cingulate, orbitofrontal, lateral+medial prefrontal, sensorimotor,
# parietal, temporal, occipital. Fixed, so every figure puts a region in the same
# row and two runs can be compared side by side.
_ROI_V2_DISPLAY = [
    'Hippocampus', 'Amygdala', 'Thalamus', 'Basal Ganglia', 'Insula',
    'rACC', 'dACC', 'PCC',
    'mOFC', 'lOFC', 'dlPFC', 'dmPFC/SMA', 'IFG/vlPFC',
    'M1', 'S1', 'S2/PO',
    'Parietal (other)', 'MTL (other)', 'Lateral Temporal', 'Auditory', 'Occipital',
]

# ---------------------------------------------------------------------------
# roi_v2_ofc -- roi_v2 with the two orbitofrontal parcels fused (2026-09-16)
# ---------------------------------------------------------------------------
# 20 regions. `mOFC` and `lOFC` become one `OFC`, at Sabra's request for the
# band-power analysis. The two were split in roi_v2 because DK can separate them,
# but medial and lateral OFC are frequently reported together in the pain
# literature and splitting them halves the contacts behind each estimate -- roi_v2
# measured 94 in mOFC and 191 in lOFC against 504 in Insula.
#
# DERIVED from _ROI_V2_PATTERNS rather than copy-pasted, so an edit to roi_v2
# cannot silently leave this variant behind. The fused category takes the POSITION
# of its first member in both dicts, which preserves precedence (pattern order)
# and figure row order (display order).

def _merge_categories(patterns, display, merges):
    """(patterns, display) with each named group of categories fused into one.

    `merges` is {new name: (member, member, ...)}. A member that is absent is an
    ERROR rather than a no-op: silently producing the un-merged scheme would give
    a run whose provenance claims a region set it does not have.
    """
    members = {m: new for new, group in merges.items() for m in group}
    for m in members:
        if m not in patterns:
            raise ValueError(f'cannot merge {m!r}: not a category of the parent scheme')

    out_patterns, seen = {}, set()
    for name, pats in patterns.items():
        new = members.get(name)
        if new is None:
            out_patterns[name] = list(pats)
        elif new not in seen:
            seen.add(new)
            out_patterns[new] = [p for m in merges[new] for p in patterns[m]]

    out_display, seen_d = [], set()
    for name in display:
        new = members.get(name, name)
        if new not in seen_d:
            seen_d.add(new)
            out_display.append(new)
    return out_patterns, out_display


_ROI_V2_OFC_PATTERNS, _ROI_V2_OFC_DISPLAY = _merge_categories(
    _ROI_V2_PATTERNS, _ROI_V2_DISPLAY, {'OFC': ('mOFC', 'lOFC')})

# ---------------------------------------------------------------------------
# pain_domains -- the pain matrix grouped by PROCESSING DOMAIN (2026-09-17)
# ---------------------------------------------------------------------------
# Regions collapsed into the domains of the pain matrix, following Shirvalkar et
# al.'s framework (their Figure: regions implicated in different domains of pain
# perception, superimposed on the ascending transmission and descending
# modulation pathways). "Domain" is used there, and here, for a collection of
# related psychological processes -- somatosensation, emotion, cognition, memory
# -- not for an anatomical grouping.
#
# THE POINT OF THIS SCHEME is that it makes the domain the UNIT OF THE MODEL. Two
# contacts in S1 and S2 of the same patient now pool into one `Sensory` estimate
# with one standard error, instead of being two region rows a reader has to
# average by eye. Nothing else about the pipeline changes -- it is a region set,
# and region sets are data here.
#
# WHAT IS DELIBERATELY NOT ASSIGNED, and why each one is held back:
#
#   Insula    -- the framework splits ANTERIOR insula (affective) from POSTERIOR
#                insula (sensory), and DK has exactly one `insula` parcel. Putting
#                it in either domain would fabricate the distinction the domains
#                rest on. Needs the Destrieux/a2009s assignment
#                (G_insular_short, S_circular_insula_ant/sup/inf), which our
#                channel_meta does not yet carry.
#   Thalamus  -- same problem one level deeper: the ascending sensory pathway is
#                VPL/VPM specifically, while medial/dorsal nuclei sit in the
#                affective pathway. DK gives one `thalamus`, and Destrieux will
#                not fix this either -- it is a surface parcellation. Needs a
#                nucleus-level atlas.
#   PCC, Parietal (other), Lateral Temporal
#             -- not in the framework's region set. Dropping them is honest;
#                assigning them to a domain to avoid dropping them is not. PCC in
#                particular carries one of the larger low-frequency pain effects
#                in this cohort, so its absence here is a real loss and a
#                deliberate one.
#
# These stay as CATEGORIES (so a contact is still labelled) but are absent from
# `display`, which is what drops them from region-level output -- the same
# mechanism that drops White Matter.
#
# ONE JUDGEMENT CALL WORTH OVERRULING IF YOU DISAGREE: `dmPFC/SMA` is DK's
# `superiorfrontal`, which straddles SMA (sensorimotor) and dorsomedial
# prefrontal (cognitive/affective). It is placed in Cognitive because medial PFC
# is the dominant reading of that parcel, but a defensible alternative is to hold
# it out with the insula. `Basal Ganglia` is similarly impure: the framework's
# affective node is ventral striatum, and DK's caudate/putamen/pallidum/accumbens
# lumps dorsal striatum in with it.
_PAIN_DOMAINS = {
    'Sensory': ('M1', 'S1', 'S2/PO'),
    'Affective': ('rACC', 'dACC', 'Amygdala', 'Basal Ganglia'),
    'Cognitive': ('dlPFC', 'dmPFC/SMA', 'IFG/vlPFC', 'OFC'),
    'Memory': ('Hippocampus', 'MTL (other)'),
    'Control': ('Occipital', 'Auditory'),
}

_PAIN_DOMAIN_PATTERNS, _ = _merge_categories(
    _ROI_V2_OFC_PATTERNS, _ROI_V2_OFC_DISPLAY, _PAIN_DOMAINS)

#: Display order follows the pathway: ascending sensory, then the affective and
#: cognitive domains it projects to, then memory, then the quasi-controls last so
#: a reader meets them after the regions the hypothesis is about.
_PAIN_DOMAIN_DISPLAY = ['Sensory', 'Affective', 'Cognitive', 'Memory', 'Control']

# ---------------------------------------------------------------------------
# pain_domains_v2 -- Sabra's domain assignment (2026-09-17)
# ---------------------------------------------------------------------------
# Supersedes `pain_domains` for new work. The v1 scheme is KEPT rather than
# edited, because a logged run cites it by name and rewriting the name's meaning
# would make that run's record misleading (its provenance carries the contents,
# so v1 stays reconstructable either way).
#
# FIVE CHANGES FROM v1, all Sabra's assignment:
#   THALAMUS joins Sensory. v1 held it out because DK gives one thalamus parcel
#     while the ascending pathway is VPL/VPM specifically. The caveat does not go
#     away -- a Sensory effect now partly reflects medial/dorsal nuclei that sit
#     in the affective pathway -- it is accepted deliberately, and Thalamus is one
#     of the stronger cells in the region-level runs, so it is carrying weight.
#   M1 leaves Sensory for its own PAIN MODULATORY domain, with Brainstem.
#   BASAL GANGLIA becomes Unassigned, which resolves v1's impurity: the
#     framework's affective node is ventral striatum and DK lumps dorsal striatum
#     with it.
#   HIPPOCAMPUS and MTL (other) become Unassigned; v1's `Memory` domain is gone.
#   INSULA remains absent, pending the Destrieux/a2009s anterior-posterior split.
#
# BRAINSTEM HAS ZERO CONTACTS IN THIS COHORT -- measured 2026-09-17 over 7,068
# contacts in 51 subjects: no brain-stem, pons, medulla, midbrain or
# periaqueductal label at all. So `Pain modulatory` is M1 ALONE today: 82
# contacts in 20 subjects, one parcel. It is the weakest domain in the scheme and
# the only one whose internal consistency cannot be checked, because a
# single-parcel domain has no parcel-to-parcel agreement to inspect. The
# brainstem pattern is registered anyway so a future subject with such a contact
# lands in the right place rather than in FALLBACK.
#
# `paracentral` stays with Sensory because roi_v2's `S1` is postcentral PLUS
# paracentral, and the assignment above names S1. It is medial sensorimotor and
# arguably straddles this scheme's Sensory/Modulatory boundary; 30 contacts in 10
# subjects.
_PAIN_DOMAINS_V2 = {
    'Sensory': ('S1', 'S2/PO', 'Thalamus'),
    'Affective': ('rACC', 'dACC', 'Amygdala'),
    'Cognitive': ('mOFC', 'lOFC', 'dlPFC', 'IFG/vlPFC', 'dmPFC/SMA'),
    'Modulatory': ('M1',),
    'Control': ('Auditory', 'Occipital'),
}

_PAIN_DOMAIN_V2_PATTERNS, _ = _merge_categories(
    _ROI_V2_PATTERNS, _ROI_V2_DISPLAY, _PAIN_DOMAINS_V2)
# Registered for the future, not for today -- see the note above.
_PAIN_DOMAIN_V2_PATTERNS['Modulatory'] = (
    list(_PAIN_DOMAIN_V2_PATTERNS['Modulatory']) + ['brain-stem', 'brainstem'])

#: Display order follows the pathway: ascending sensory, the affective and
#: cognitive domains it projects to, the descending modulatory arm, then the
#: quasi-controls last so a reader meets them after the hypothesis.
_PAIN_DOMAIN_V2_DISPLAY = ['Sensory', 'Affective', 'Cognitive', 'Modulatory',
                           'Control']

ROI_SCHEMES = {
    'default': {'patterns': _DEFAULT_PATTERNS, 'display': _DEFAULT_DISPLAY},
    'roi_v2': {'patterns': _ROI_V2_PATTERNS, 'display': _ROI_V2_DISPLAY},
    'roi_v2_ofc': {'patterns': _ROI_V2_OFC_PATTERNS,
                   'display': _ROI_V2_OFC_DISPLAY},
    'pain_domains': {'patterns': _PAIN_DOMAIN_PATTERNS,
                     'display': _PAIN_DOMAIN_DISPLAY},
    'pain_domains_v2': {'patterns': _PAIN_DOMAIN_V2_PATTERNS,
                        'display': _PAIN_DOMAIN_V2_DISPLAY},
}

DEFAULT_ROI_SCHEME = 'default'


# ---------------------------------------------------------------------------
# Resolution + validation
# ---------------------------------------------------------------------------

def resolve_roi_scheme(scheme=None):
    """A scheme dict from: None (default), a built-in name, a JSON path, or a dict.

    JSON shape mirrors the built-ins::

        {"display": ["Insula", "ACC"],
         "patterns": {"Insula": ["insula"], "ACC": ["caudalanteriorcingulate"]}}

    JSON objects preserve insertion order through `json.load`, so a file's key
    order is its precedence order -- same contract as the built-ins.
    """
    if scheme is None:
        scheme = DEFAULT_ROI_SCHEME
    if isinstance(scheme, dict):
        return _validated(scheme, 'inline dict')
    if scheme in ROI_SCHEMES:
        return _validated(ROI_SCHEMES[scheme], f'built-in {scheme!r}')

    path = Path(scheme)
    if path.suffix == '.json' or path.exists():
        if not path.exists():
            raise FileNotFoundError(f'ROI scheme file not found: {path}')
        with open(path) as fh:
            return _validated(json.load(fh), f'file {path}')

    raise ValueError(
        f'Unknown ROI scheme {scheme!r}. Built-ins: {sorted(ROI_SCHEMES)}; '
        'otherwise pass a path to a .json scheme file.'
    )


def _validated(scheme, origin):
    for key in ('patterns', 'display'):
        if key not in scheme:
            raise ValueError(f'ROI scheme from {origin} is missing {key!r}')
    patterns, display = scheme['patterns'], scheme['display']
    # A display row with no patterns would be a permanently empty heatmap row --
    # far better to fail at config time than to ship a figure with a blank band.
    unknown = [d for d in display if d not in patterns]
    if unknown:
        raise ValueError(
            f'ROI scheme from {origin}: display names {unknown} have no patterns, so '
            f'they could never match a channel. Known: {sorted(patterns)}'
        )
    empty = [k for k, v in patterns.items() if not v]
    if empty:
        raise ValueError(f'ROI scheme from {origin}: categories {empty} have no patterns')
    return {'patterns': patterns, 'display': list(display), 'origin': origin}


def roi_regions(scheme=None):
    """Display-order ROI rows for a scheme."""
    return list(resolve_roi_scheme(scheme)['display'])


def scheme_provenance(scheme=None):
    """Full scheme contents, for a run's provenance.json.

    The contents, not just the name: a scheme loaded from a JSON file on Oak can
    be edited afterwards, so a recorded name would not reconstruct what actually
    ran.
    """
    resolved = resolve_roi_scheme(scheme)
    return {'origin': resolved['origin'], 'display': resolved['display'],
            'patterns': {k: list(v) for k, v in resolved['patterns'].items()}}


# ---------------------------------------------------------------------------
# The mapping itself
# ---------------------------------------------------------------------------

def categorize_desikan_killiany(dk_label, scheme=None):
    """One DK label -> its category (including NON_ROI_CATEGORIES).

    First matching category wins, so precedence is the scheme's key order.
    """
    if dk_label is None or (not isinstance(dk_label, str) and pd.isna(dk_label)):
        return UNLABELED
    label = str(dk_label).lower().strip("'\"")
    for category, patterns in resolve_roi_scheme(scheme)['patterns'].items():
        if any(p in label for p in patterns):
            return category
    return FALLBACK


def region_for_dk_label(dk_label, scheme=None):
    """One DK label -> a DISPLAYED ROI, or None if it falls outside the ROI set.

    None means "deliberately not analysed" (white matter, occipital, unlabeled,
    ...). Callers must drop those AND log how many, never silently exclude them --
    coverage is a confound in this dataset, so a shrinking denominator has to be
    visible.
    """
    resolved = resolve_roi_scheme(scheme)
    category = categorize_desikan_killiany(dk_label, resolved)
    return category if category in resolved['display'] else None
