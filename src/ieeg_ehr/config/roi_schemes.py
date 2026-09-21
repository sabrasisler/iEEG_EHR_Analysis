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

# ---------------------------------------------------------------------------
# COORDINATE-DERIVED REGIONS: the insula split (2026-09-21)
# ---------------------------------------------------------------------------
# `aIns` and `pIns` are the first regions in this module that CANNOT be derived
# from an atlas label. DK has one `insula` parcel; the anterior/posterior split
# comes from the contact's MNI y coordinate (`analysis.insula_ap`), which is a
# different kind of fact from a substring match and is therefore computed
# elsewhere.
#
# They are declared here as `coordinate_regions`, a map from the LABEL-derived
# region they replace to the parts it splits into. Three things follow, and the
# third is the one that makes this safe:
#
#   - `_validated` accepts them as display names even though they have no
#     patterns, which the un-extended validator would reject (correctly -- a
#     display row with no way to be filled is normally a bug).
#   - `region_for_dk_label` still returns None for an insular label, because
#     `Insula` is not in `display`. The label layer therefore cannot assign an
#     insular contact to anything.
#   - So a caller that FORGETS the coordinate step drops insula entirely, which
#     is the honest status quo these schemes already had, rather than silently
#     assigning every insular contact to one part. The failure mode is a missing
#     region, not a wrong one.
#
# `analysis.insula_ap.apply_split` is the one function that fills them in, and it
# consumes `coordinate_regions()` so the two cannot name different parts.

#: The label-derived region that the coordinate step replaces -> its parts.
_INSULA_SPLIT = {'Insula': ('aIns', 'pIns')}


def _with_coordinate_regions(patterns, display, coordinate_regions):
    """(patterns, display) with each split region's parts in its display slot.

    The patterns dict is UNCHANGED: the parent region keeps its patterns and
    keeps matching labels, it simply stops being displayed. That is what makes
    the un-split fallback drop insula rather than mislabel it.
    """
    out_display = []
    for name in display:
        out_display.extend(coordinate_regions.get(name, (name,)))
    return dict(patterns), out_display


_ROI_V2_INS_PATTERNS, _ROI_V2_INS_DISPLAY = _with_coordinate_regions(
    _ROI_V2_PATTERNS, _ROI_V2_DISPLAY, _INSULA_SPLIT)
_ROI_V2_OFC_INS_PATTERNS, _ROI_V2_OFC_INS_DISPLAY = _with_coordinate_regions(
    _ROI_V2_OFC_PATTERNS, _ROI_V2_OFC_DISPLAY, _INSULA_SPLIT)

ROI_SCHEMES = {
    'default': {'patterns': _DEFAULT_PATTERNS, 'display': _DEFAULT_DISPLAY},
    'roi_v2': {'patterns': _ROI_V2_PATTERNS, 'display': _ROI_V2_DISPLAY},
    'roi_v2_ofc': {'patterns': _ROI_V2_OFC_PATTERNS,
                   'display': _ROI_V2_OFC_DISPLAY},
    'roi_v2_ins': {'patterns': _ROI_V2_INS_PATTERNS,
                   'display': _ROI_V2_INS_DISPLAY,
                   'coordinate_regions': _INSULA_SPLIT},
    'roi_v2_ofc_ins': {'patterns': _ROI_V2_OFC_INS_PATTERNS,
                       'display': _ROI_V2_OFC_INS_DISPLAY,
                       'coordinate_regions': _INSULA_SPLIT},
    'pain_domains': {'patterns': _PAIN_DOMAIN_PATTERNS,
                     'display': _PAIN_DOMAIN_DISPLAY},
    'pain_domains_v2': {'patterns': _PAIN_DOMAIN_V2_PATTERNS,
                        'display': _PAIN_DOMAIN_V2_DISPLAY},
}

DEFAULT_ROI_SCHEME = 'default'

# ---------------------------------------------------------------------------
# DOMAIN SCHEMES -- the ROI layer kept, not fused away (2026-09-21)
# ---------------------------------------------------------------------------
# `_merge_categories` builds a domain scheme by FUSING ROI categories into one,
# which is exactly right when the domain is the only level you need: the result
# is an ordinary scheme and every label-based caller works unchanged. It has one
# cost, and the cost is now load-bearing: the intermediate ROI names are GONE
# from the fused dict, so a caller holding `pain_domains_v2` cannot ask which ROI
# a contact is in -- only which domain.
#
# The parcel-level domain model did not need to ask, because it went straight
# from the DK label to a parcel and carried the parcel itself as a random slope.
# Modelling at the ROI LEVEL does need to ask: the ROI is the unit that maps to a
# domain, and the same ROI assignment has to be available to the region-level
# consistency map, which shows ROIs the domains do not contain at all.
#
# So a domain scheme is registered here as (base ROI scheme, ROI -> domain
# membership, display order) rather than only as its fused product. The fused
# product is still registered in ROI_SCHEMES for every caller that wants a
# label -> domain map in one step; this is the same information with the middle
# level left in.
#
# CONSISTENCY BETWEEN THE TWO IS NOT ASSUMED: `domain_scheme` rebuilds the fused
# patterns from the members here, so a membership edit that forgot to update the
# fused scheme is a failed lookup rather than two schemes quietly disagreeing.

# -- pain_domains_v3: v2 with the insula split back in (2026-09-21) ----------
# Sabra's assignment. The ONLY change from v2 is that insula returns, split by
# coordinate rather than by label: aIns joins AFFECTIVE and pIns joins SENSORY,
# which is the anterior/posterior dissociation the framework rests on and the
# exact thing DK's single `insula` parcel made impossible to state. See
# `analysis.insula_ap` for what the split actually is and how coarse it is --
# a median cut on MNI y, not the Destrieux assignment.
#
# The caveat that blocked insula in v1 and v2 has NOT been dissolved, it has been
# TRADED: the parcel is no longer being assigned wholesale to one domain it only
# half belongs to, at the price of a boundary that is this cohort's median rather
# than an anatomical landmark. That is the same class of trade v2 already made
# for Thalamus, and it is accepted on the same terms -- deliberately, in writing,
# with the threshold recorded as a number.
_PAIN_DOMAINS_V3 = {
    'Sensory': ('S1', 'S2/PO', 'Thalamus', 'pIns'),
    'Affective': ('rACC', 'dACC', 'Amygdala', 'aIns'),
    'Cognitive': ('mOFC', 'lOFC', 'dlPFC', 'IFG/vlPFC', 'dmPFC/SMA'),
    'Modulatory': ('M1',),
    'Control': ('Auditory', 'Occipital'),
}

#: Same pathway order as v2: ascending sensory, the affective and cognitive
#: domains it projects to, the descending modulatory arm, quasi-controls last.
_PAIN_DOMAIN_V3_DISPLAY = ['Sensory', 'Affective', 'Cognitive', 'Modulatory',
                           'Control']

DOMAIN_SCHEMES = {
    'pain_domains': {'base': 'roi_v2_ofc', 'members': _PAIN_DOMAINS,
                     'display': _PAIN_DOMAIN_DISPLAY},
    'pain_domains_v2': {'base': 'roi_v2', 'members': _PAIN_DOMAINS_V2,
                        'display': _PAIN_DOMAIN_V2_DISPLAY},
    'pain_domains_v3': {'base': 'roi_v2_ins', 'members': _PAIN_DOMAINS_V3,
                        'display': _PAIN_DOMAIN_V3_DISPLAY},
}


def domain_scheme(name):
    """{'base', 'members', 'display', 'roi_to_domain', 'rois'} for a domain scheme.

    `base` is the ROI scheme the members are ROIs OF -- resolve it to get the
    label patterns and any coordinate regions. `roi_to_domain` is the inverted
    membership, which is the direction every caller actually wants.

    VALIDATED AGAINST THE BASE SCHEME, because the failure this prevents is
    silent: a typo in a member name ('S2/P0' for 'S2/PO') would simply never
    match a contact, and the domain would come back one ROI light with no error
    anywhere. An ROI named here that the base scheme does not display is a hard
    error instead.
    """
    if name not in DOMAIN_SCHEMES:
        raise ValueError(f'Unknown domain scheme {name!r}. '
                         f'Known: {sorted(DOMAIN_SCHEMES)}')
    spec = DOMAIN_SCHEMES[name]
    base_regions = set(roi_regions(spec['base']))
    roi_to_domain = {roi: dom for dom, rois in spec['members'].items()
                     for roi in rois}
    unknown = sorted(r for r in roi_to_domain if r not in base_regions)
    if unknown:
        raise ValueError(
            f'domain scheme {name!r}: ROIs {unknown} are not displayed regions of '
            f'its base scheme {spec["base"]!r}. Known: {sorted(base_regions)}')
    missing = [d for d in spec['display'] if d not in spec['members']]
    if missing:
        raise ValueError(f'domain scheme {name!r}: display names {missing} have '
                         'no member ROIs')
    return {'name': name, 'base': spec['base'],
            'members': {k: tuple(v) for k, v in spec['members'].items()},
            'display': list(spec['display']), 'roi_to_domain': roi_to_domain,
            'rois': [r for r in roi_regions(spec['base']) if r in roi_to_domain]}


def domain_scheme_provenance(name):
    """A domain scheme's full contents, for a run's provenance.

    Includes the BASE scheme's contents, because "which ROIs are in Sensory" is
    only half the story -- which atlas labels are in S1 is the other half, and a
    run recording only the domain membership could not be reproduced.
    """
    spec = domain_scheme(name)
    unassigned = [r for r in roi_regions(spec['base'])
                  if r not in spec['roi_to_domain']]
    return {'name': name, 'base_scheme': spec['base'],
            'display': spec['display'],
            'members': {k: list(v) for k, v in spec['members'].items()},
            'unassigned_rois': unassigned,
            'base_scheme_contents': scheme_provenance(spec['base'])}


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
    coord = scheme.get('coordinate_regions') or {}
    # A COORDINATE region is allowed to have no patterns -- that is the whole
    # point of it -- but its PARENT must exist, or the coordinate step has
    # nothing to replace and the region can never be filled either way.
    missing_parent = [p for p in coord if p not in patterns]
    if missing_parent:
        raise ValueError(
            f'ROI scheme from {origin}: coordinate_regions parents {missing_parent} '
            f'are not categories of this scheme. Known: {sorted(patterns)}')
    coord_parts = {part for parts in coord.values() for part in parts}

    # A display row with no patterns would be a permanently empty heatmap row --
    # far better to fail at config time than to ship a figure with a blank band.
    unknown = [d for d in display if d not in patterns and d not in coord_parts]
    if unknown:
        raise ValueError(
            f'ROI scheme from {origin}: display names {unknown} have no patterns, so '
            f'they could never match a channel. Known: {sorted(patterns)}'
        )
    empty = [k for k, v in patterns.items() if not v]
    if empty:
        raise ValueError(f'ROI scheme from {origin}: categories {empty} have no patterns')
    return {'patterns': patterns, 'display': list(display), 'origin': origin,
            'coordinate_regions': {k: tuple(v) for k, v in coord.items()}}


def roi_regions(scheme=None):
    """Display-order ROI rows for a scheme."""
    return list(resolve_roi_scheme(scheme)['display'])


def coordinate_regions(scheme=None):
    """{parent region: (part, ...)} for regions that a COORDINATE step fills in.

    Empty for every label-only scheme. A caller that handles coordinate regions
    reads this rather than hard-coding 'Insula', so adding a second split later
    (a thalamic one, say) does not mean finding every site that knew about the
    first.
    """
    return dict(resolve_roi_scheme(scheme).get('coordinate_regions') or {})


def scheme_provenance(scheme=None):
    """Full scheme contents, for a run's provenance.json.

    The contents, not just the name: a scheme loaded from a JSON file on Oak can
    be edited afterwards, so a recorded name would not reconstruct what actually
    ran.
    """
    resolved = resolve_roi_scheme(scheme)
    out = {'origin': resolved['origin'], 'display': resolved['display'],
           'patterns': {k: list(v) for k, v in resolved['patterns'].items()}}
    if resolved.get('coordinate_regions'):
        out['coordinate_regions'] = {k: list(v) for k, v
                                     in resolved['coordinate_regions'].items()}
    return out


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


def region_for_dk_label(dk_label, scheme=None, include_coordinate_parents=False):
    """One DK label -> a DISPLAYED ROI, or None if it falls outside the ROI set.

    None means "deliberately not analysed" (white matter, occipital, unlabeled,
    ...). Callers must drop those AND log how many, never silently exclude them --
    coverage is a confound in this dataset, so a shrinking denominator has to be
    visible.

    `include_coordinate_parents` also returns the PARENT of a coordinate region
    (`Insula`, which is never itself displayed), so a caller can hand those
    channels to the coordinate step that splits them. It is off by default
    because the returned name is then NOT a displayed region, and anything that
    treats it as one -- a heatmap row, a domain lookup -- would be wrong. Turn it
    on only immediately before calling `insula_ap.apply_split`, which is what
    removes the parent names again.
    """
    resolved = resolve_roi_scheme(scheme)
    category = categorize_desikan_killiany(dk_label, resolved)
    if category in resolved['display']:
        return category
    if include_coordinate_parents and category in (
            resolved.get('coordinate_regions') or {}):
        return category
    return None
