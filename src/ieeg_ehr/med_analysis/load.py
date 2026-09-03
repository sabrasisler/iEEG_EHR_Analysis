"""
MAR export -> one tidy table, one row per administration.

The MAR schema, verified against the files on disk:

    sub_id, ses_id, taken_date, session_start, session_end, medication, line,
    mar_action, sig, route, site, infusion_rate, infusion_rate_unit,
    dose_unit, mar_duration, mar_duration_unit

Things about this table that are not obvious and will silently corrupt results:

1. **There is no `dose` column.** The administered amount is `sig`, with its unit
   in `dose_unit`. `medication` carries the PRODUCT strength, which is a
   different number ("HYDROCODONE-ACETAMINOPHEN 10-325 MG PO TABS" given as
   `sig=1` means one tablet, not 10 mg).

2. **One dose can be several rows.** A dose dispensed as two tablet strengths
   appears once per product, each repeating the same total `sig`. Counting raw
   rows overcounts by ~5.7% corpus-wide. `_collapse_multiproduct` handles it, and
   the dedup key MUST include sub_id/ses_id — `taken_date` sits on a shared
   de-identified epoch, so two subjects genuinely collide on the other fields.

3. **`taken_date` is only comparable within a session.** Every subject's
   timeline is shifted onto the same epoch. Anything that computes a time
   difference does it within (subject, session) and nowhere else.

4. **`mar_action` is a forward guard, not a filter.** All 7,340 rows in the
   current export are already `Given`. Keep the filter anyway; the day it stops
   being a no-op is the day it matters.

5. **Doses are not poolable across drugs.** 536 administrations (every
   combination product) are dosed in `tablet` or `Film`; fentanyl is in `mcg`;
   everything else in `mg`. `assert_single_unit` enforces that no pooled dose
   column ever mixes them. There is no MME conversion here and no strength
   parsing out of product names — a dose axis is always per (drug, route).

WHY-DIFFERENT from the source analysis: routes are kept verbatim. The source
collapses everything that is not Oral or Intravenous into "Other", which for
analgesics would bury sublingual buprenorphine, feeding-tube administration and
nerve blocks — 69 administrations, 3.8% of the set, and the most clinically
distinct routes in it.

No PRN / as-needed field exists in this schema, and nothing here infers one.
"""

import logging
import re

import pandas as pd

from ieeg_ehr import config
from ieeg_ehr.config import med_taxonomy

logger = logging.getLogger(__name__)

MAR_COLUMNS = ('sub_id', 'ses_id', 'taken_date', 'session_start', 'session_end',
               'medication', 'line', 'mar_action', 'sig', 'route', 'site',
               'infusion_rate', 'infusion_rate_unit', 'dose_unit',
               'mar_duration', 'mar_duration_unit')

#: A dose token starts with a digit. "VITAMIN B-12" and "TB24" must not match.
_DOSE_TOKEN_RE = re.compile(r'^\d')

#: The one MAR action that means the drug reached the patient.
GIVEN = 'Given'

#: Identifies one physical administration. See note 2 above.
DEDUP_KEY = ('subject', 'session', 'taken_date_raw', 'drug', 'line', 'sig_raw',
             'route')


def extract_medication_name(raw):
    """Canonical drug name from a product string.

    Cuts at the first standalone numeric token, then consolidates formulation
    variants on the leading word (`med_taxonomy.CONSOLIDATE_FIRST_WORD`), so the
    three fentanyl products become one FENTANYL and "MORPHINE 2 MG/ML INJ SYRG"
    joins "MORPHINE INJECTABLE SYRINGE".

    Combination products survive the cut intact:
    "OXYCODONE-ACETAMINOPHEN 5-325 MG PO TABS" cuts at "5-325", not inside the
    hyphenated name, because the cut is on whitespace tokens.
    """
    tokens = str(raw).split()
    cut_idx = None
    for i, tok in enumerate(tokens):
        if _DOSE_TOKEN_RE.match(tok.lstrip('(')):
            cut_idx = i
            break
    name_tokens = tokens[:cut_idx] if cut_idx is not None else tokens
    name = ' '.join(name_tokens).strip().strip('(').strip()
    name = name if name else str(raw).strip()

    words = name.split()
    first_word = words[0].strip('(),').upper() if words else name.upper()
    return med_taxonomy.CONSOLIDATE_FIRST_WORD.get(first_word, name)


def _read_one(path):
    """One MAR file -> DataFrame, or None if it has no usable rows.

    `encoding='utf-8-sig'` is not optional: without it the first column name
    comes back with a BOM attached and every downstream lookup misses.
    """
    df = pd.read_csv(path, encoding='utf-8-sig', dtype=str, keep_default_na=False)
    missing = set(MAR_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f'{path} is missing MAR columns {sorted(missing)} — the '
                         f'export schema changed; med_analysis.load needs updating')
    if df.empty:
        return None
    return df


def _collapse_multiproduct(df):
    """Drop the extra rows a multi-strength dose generates. Returns (df, n_dropped)."""
    before = len(df)
    df = df.drop_duplicates(subset=list(DEDUP_KEY), keep='first')
    return df, before - len(df)


def load_administrations(paths=None, drugs=None, subclasses=None,
                         require_class=True):
    """The tidy table: one row per administration.

    Args:
        paths: MAR files to read. Default `config.med_admin_files()` — the whole
            cohort, 98 files across 96 subjects.
        drugs: restrict to these canonical drug names. Applied after
            classification, so an unknown drug still raises rather than being
            filtered away unseen.
        subclasses: restrict to these Level 2 classes (e.g.
            `med_taxonomy.ANALGESIC_SUBCLASSES`). Combines with `drugs`.
        require_class: raise on a product the taxonomy has never seen. Leave it
            True — a new drug in the export is a data change, not a row to drop.

    Returns a DataFrame with columns:
        subject, session, drug, level1, level2, is_combination, route,
        dose, dose_unit, taken_dt, session_start, session_end,
        hospital_day, hour_of_day
    """
    paths = list(paths) if paths is not None else config.med_admin_files()
    if not paths:
        raise FileNotFoundError(
            f'no *_med-admin.csv under {config.RAW_DIR} — check the Oak mount')

    frames = [f for f in (_read_one(p) for p in paths) if f is not None]
    raw = pd.concat(frames, ignore_index=True)
    n_rows = len(raw)

    given = raw[raw['mar_action'].str.strip() == GIVEN].copy()
    n_not_given = n_rows - len(given)

    given['subject'] = given['sub_id'].str.replace('sub-', '', regex=False)
    given['session'] = given['ses_id'].str.replace('ses-', '', regex=False)
    given['drug'] = given['medication'].map(extract_medication_name)
    given['taken_date_raw'] = given['taken_date']
    given['sig_raw'] = given['sig']
    given['line'] = given['line'].fillna('')

    if require_class:
        unknown = sorted(set(given['drug']) - med_taxonomy.known_drugs())
        if unknown:
            raise KeyError(
                f'{len(unknown)} drug name(s) in the MAR export are not in the '
                f'taxonomy: {unknown}. Add them to config/med_taxonomy.py — see '
                f'med_taxonomy.classify() for where. Never drop them silently.')

    classes = given['drug'].map(
        lambda d: med_taxonomy.classify(d) if require_class
        else med_taxonomy.DRUG_CLASS.get(d, ('', '')))
    given['level1'] = [c[0] for c in classes]
    given['level2'] = [c[1] for c in classes]
    given['is_combination'] = given['drug'].isin(med_taxonomy.COMBINATION_DRUGS)

    given, n_collapsed = _collapse_multiproduct(given)

    # Timestamps. errors='coerce' then an explicit count, so an unparseable
    # timestamp is reported rather than silently becoming NaT and vanishing.
    given['taken_dt'] = pd.to_datetime(given['taken_date'], errors='coerce')
    given['session_start'] = pd.to_datetime(given['session_start'], errors='coerce')
    given['session_end'] = pd.to_datetime(given['session_end'], errors='coerce')
    n_bad_ts = int(given['taken_dt'].isna().sum())
    if n_bad_ts:
        logger.warning('%d administration(s) have an unparseable taken_date and '
                       'are dropped', n_bad_ts)
        given = given[given['taken_dt'].notna()]

    given['dose'] = pd.to_numeric(given['sig'], errors='coerce')
    given['dose_unit'] = given['dose_unit'].str.strip().replace('', 'unspecified')
    given['route'] = given['route'].str.strip().replace('', 'Unspecified')

    given['hospital_day'] = hospital_day(given['taken_dt'], given['session_start'])
    given['hour_of_day'] = (given['taken_dt'].dt.hour
                            + given['taken_dt'].dt.minute / 60.0
                            + given['taken_dt'].dt.second / 3600.0)

    logger.info('MAR: %d files, %d rows, %d given, %d multi-product rows '
                'collapsed, %d administrations', len(paths), n_rows,
                n_rows - n_not_given, n_collapsed, len(given))

    if subclasses is not None:
        given = given[given['level2'].isin(set(subclasses))]
    if drugs is not None:
        given = given[given['drug'].isin(set(drugs))]

    cols = ['subject', 'session', 'drug', 'level1', 'level2', 'is_combination',
            'route', 'dose', 'dose_unit', 'taken_dt', 'session_start',
            'session_end', 'hospital_day', 'hour_of_day']
    out = given[cols].sort_values(['subject', 'session', 'taken_dt'])
    return out.reset_index(drop=True)


def hospital_day(when, session_start):
    """Days since midnight of the session's own start date.

    WHY-DIFFERENT: the source analysis hardcodes `EPOCH_DATE = 2000-01-01` for
    every subject, on the grounds that de-identification shifts every admission
    onto that date. That holds for 95 of 98 sessions — two start 2000-01-05 and
    one starts 1999-12-31, and those three get shifted or negative day indices
    under a global constant. Anchoring per-session is identical everywhere the
    assumption is true and correct where it is not.

    Day 0 is therefore the calendar day the iEEG session began. Say so in the
    caption: it is not the same claim as "day 0 is admission".
    """
    return (when.dt.normalize() - session_start.dt.normalize()).dt.days


def formulation_label(drug, route):
    """'IV Hydromorphone' — the row label used by Figs 2 and 4."""
    short = {'Intravenous': 'IV', 'Oral': 'PO', 'Sublingual': 'SL',
             'Feeding Tube': 'FT', 'Topical': 'TOP', 'Nerve Block': 'NB',
             'Swish & Spit': 'S&S', 'Injection': 'INJ'}.get(route, route)
    return f'{short} {drug.title()}'


def assert_single_unit(df, context):
    """Refuse to pool doses that are not in the same unit.

    The one guard between "median dose" and a number that averages milligrams
    with tablets. Call it before every dose summary and every dose axis.
    """
    units = sorted(set(df['dose_unit'].dropna()))
    if len(units) > 1:
        raise ValueError(
            f'{context}: dose column mixes units {units}. Doses in this dataset '
            f'are not convertible (tablets and films have no mg equivalent '
            f'without parsing product strength). Group by (drug, route) before '
            f'summarizing dose.')
    return units[0] if units else 'unspecified'


def drug_route_counts(df):
    """(drug, route) -> administrations and distinct subjects, commonest first."""
    grouped = df.groupby(['drug', 'route'], as_index=False).agg(
        n_admin=('subject', 'size'),
        n_subjects=('subject', 'nunique'),
    )
    return grouped.sort_values('n_admin', ascending=False).reset_index(drop=True)


def top_formulations(df, n=5, min_admin=20):
    """The top n (drug, route) pairs with at least `min_admin` administrations.

    Used to pick the rows of Figs 2 and 4. Returns a list of
    (drug, route, label) tuples.
    """
    counts = drug_route_counts(df)
    counts = counts[counts['n_admin'] >= min_admin].head(n)
    return [(r.drug, r.route, formulation_label(r.drug, r.route))
            for r in counts.itertuples()]
