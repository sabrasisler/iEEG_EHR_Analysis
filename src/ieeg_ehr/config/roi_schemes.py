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

ROI_SCHEMES = {
    'default': {'patterns': _DEFAULT_PATTERNS, 'display': _DEFAULT_DISPLAY},
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
