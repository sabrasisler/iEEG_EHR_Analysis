"""Tests for the shared drug selector and the violin eligibility guard.

`select_drugs` exists so Figs 2, 3, 5 and 6 cannot drift onto different drug
sets, so its contract — explicit list wins, outright — is what matters here.
"""

import pandas as pd
import pytest

from ieeg_ehr.med_analysis import load
from ieeg_ehr.med_analysis import plot_pain_score_bars as psb


def admin(counts):
    """A frame with `counts[drug]` administrations of each drug."""
    rows = []
    for drug, n in counts.items():
        rows += [{'drug': drug, 'subject': f'{i:03d}'} for i in range(n)]
    return pd.DataFrame(rows)


def test_falls_back_to_most_administered_above_threshold():
    df = admin({'A': 50, 'B': 30, 'C': 5})
    assert load.select_drugs(df, min_admin=20) == ['A', 'B']


def test_limit_truncates_the_fallback():
    df = admin({'A': 50, 'B': 30, 'C': 25})
    assert load.select_drugs(df, min_admin=20, limit=2) == ['A', 'B']


def test_explicit_list_wins_over_min_admin():
    """Naming a drug set must return that set, small members included."""
    df = admin({'A': 50, 'B': 30, 'C': 5})
    assert load.select_drugs(df, drugs=['A', 'C'], min_admin=20) == ['A', 'C']


def test_explicit_list_wins_over_limit():
    df = admin({'A': 50, 'B': 30, 'C': 25})
    got = load.select_drugs(df, drugs=['A', 'B', 'C'], min_admin=20, limit=1)
    assert got == ['A', 'B', 'C']


def test_explicit_list_preserves_the_given_order():
    """Bar position and panel order must be the caller's, not value_counts'."""
    df = admin({'A': 50, 'B': 30, 'C': 25})
    assert load.select_drugs(df, drugs=['C', 'A', 'B']) == ['C', 'A', 'B']


def test_unknown_drug_raises_rather_than_being_dropped():
    """A name absent from the data is a typo, not a small sample."""
    df = admin({'A': 50})
    with pytest.raises(ValueError, match='not present'):
        load.select_drugs(df, drugs=['A', 'ACETAMINOPHENN'])


def test_violin_eligibility_drops_only_the_too_small():
    summary = pd.DataFrame([
        {'drug': 'A', 'n_linked': 400},
        {'drug': 'B', 'n_linked': 23},
        {'drug': 'C', 'n_linked': 2},
        {'drug': 'D', 'n_linked': 0},
    ])
    assert psb.violin_eligible(summary, min_n=5) == ['A', 'B']


def test_violin_eligibility_threshold_is_inclusive():
    summary = pd.DataFrame([{'drug': 'A', 'n_linked': 5}])
    assert psb.violin_eligible(summary, min_n=5) == ['A']
