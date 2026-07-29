"""
One choice per view axis, validated -- the `view:` block of docs/view_registry.md
made executable.

A ViewConfig is the complete description of how the per-window cache becomes an
analysis-ready table. It is hashed into every materialized view's sidecar and
recorded in every run's provenance, so "which view produced this figure" has a
single answer.

THE MASK IS PART OF THE CONFIG, not a side channel. The cache stores raw unmasked
slices (P1.1, 2026-07-27), so the mask is chosen at view time -- which means a
figure is uninterpretable without knowing which mask produced it, and it therefore
belongs in the thing that gets hashed and recorded.

AXIS ORDER IS FIXED BY THE REGISTRY and enforced by the code that consumes this,
not by the config: baseline -> normalize (per window) -> epoch-average ->
frequency-aggregate -> region-aggregate -> binarize. Normalizing after averaging
is not the same as before (Jensen), which is the entire reason the cache keeps
per-window granularity.
"""

from dataclasses import asdict, dataclass, field

# Allowed values per axis (docs/view_registry.md AXIS 1-7). Kept here rather than
# as bare strings at call sites so a typo fails at construction, not silently in
# an `if` that never matches.
DOMAINS = ('log', 'linear')
BASELINES = ('zero_pain_epochs', 'whole_session')
NORMALIZATIONS = ('none', 'baseline_subtract', 'zscore_vs_baseline')
EPOCH_AGGS = ('mean',)
FREQ_AGGS = ('log_bins_50', 'canonical_bands')
REGION_AGGS = ('none', 'individual_dk')
PAIN_BINS = ('absolute', 'subject_relative')
MASK_LEVELS = ('bipolar', 'raw_voltage', 'none')

# Short codes for the two axes that go into a DIRECTORY NAME (see `scheme_code`).
# Only these two: everything else about a view lives in the hashed sidecar, and a
# folder name that tried to spell out all seven axes would be unreadable and would
# still not be a complete description.
#
# WORDS, NOT INITIALS. These were 'blsub'/'rel'/'abs' for one day (2026-07-29) and
# were unreadable to the person who has to browse the tree -- which defeats the
# whole point of naming the view in the path. 'delta' because baseline_subtract in
# the log domain IS delta log power, which is already what the sbatch prose calls
# it; 'relpain'/'abspain' because 'rel' alone does not say relative to WHAT.
NORMALIZATION_CODES = {'zscore_vs_baseline': 'zscore',
                       'baseline_subtract': 'delta',
                       'none': 'raw'}
PAIN_BIN_CODES = {'subject_relative': 'relpain', 'absolute': 'abspain'}


@dataclass(frozen=True)
class ViewConfig:
    """Frozen so a config cannot be mutated after it has been hashed into a
    sidecar -- the hash and the object must not be able to disagree."""

    domain: str = 'log'
    baseline: str = 'zero_pain_epochs'
    normalization: str = 'zscore_vs_baseline'
    epoch_agg: str = 'mean'
    freq: str = 'log_bins_50'
    region: str = 'individual_dk'
    pain_bins: str = 'subject_relative'

    # AXIS 0, in effect: which QC mask is applied at load time.
    mask_level: str = 'bipolar'
    mask_label: str = None

    # ROI scheme name or path to a JSON scheme (config/roi_schemes.py). Only
    # meaningful when region != 'none'.
    roi_scheme: str = 'default'

    # Drop a channel-epoch when more than this fraction of its windows are
    # mask-excluded. Part of the config because it changes the output rows.
    max_excluded_frac: float = None

    # Drop frequency bins flagged contains_line_noise in the unit manifest.
    # Default False: the cache deliberately keeps them (P1.1), and whether to drop
    # them is a view choice, not a baked-in one.
    drop_line_noise_bins: bool = False

    epoch_minutes: float = None

    def __post_init__(self):
        for value, allowed, name in (
            (self.domain, DOMAINS, 'domain'),
            (self.baseline, BASELINES, 'baseline'),
            (self.normalization, NORMALIZATIONS, 'normalization'),
            (self.epoch_agg, EPOCH_AGGS, 'epoch_agg'),
            (self.freq, FREQ_AGGS, 'freq'),
            (self.region, REGION_AGGS, 'region'),
            (self.pain_bins, PAIN_BINS, 'pain_bins'),
            (self.mask_level, MASK_LEVELS, 'mask_level'),
        ):
            if value not in allowed:
                raise ValueError(f'{name}={value!r} not one of {allowed}')

        if self.normalization != 'none' and self.baseline is None:
            raise ValueError(f'normalization={self.normalization!r} needs a baseline')
        if self.mask_level != 'none' and not self.mask_label:
            raise ValueError(
                f'mask_level={self.mask_level!r} requires mask_label. Refusing to '
                'default it: an unmasked view silently keeps artifact windows, and '
                'that is invisible in the output.'
            )
        if self.baseline == 'whole_session':
            raise NotImplementedError(
                "baseline='whole_session' is a registry axis value but is not "
                'implemented yet; only zero_pain_epochs is.'
            )

    # ------------------------------------------------------------------
    @property
    def is_difference(self):
        """True when the value is already a DIFFERENCE of logs.

        Drives region/frequency aggregation: a difference of logs averages
        arithmetically, but raw log-power does not (a mean of logs is a geometric
        mean -- registry AXIS 5/6's linear-then-log rule). Getting this wrong is
        silent, so it is one property rather than a repeated inline test.
        """
        return self.normalization in ('baseline_subtract', 'zscore_vs_baseline')

    @property
    def scheme_code(self):
        """Compact human label for a directory name, e.g. 'blsub-rel'.

        ONE definition, used for BOTH the materialized view's directory
        (`config.pain_epoch_views_dir`) and the analysis tree's level-4
        view_scheme folder. If those were built separately they could drift, and
        then two folder names would claim to describe the same view.

        Normalization and pain binning specifically, out of the seven axes: the
        first is what most changes the numbers (and used to be invisible in the
        path -- the old level 4 was `subject_relative` alone), the second is what
        changes how many lines a figure has. The exact, complete view identity is
        the sidecar's `config_hash`; this is the part a human reads.
        """
        return (f'{NORMALIZATION_CODES[self.normalization]}'
                f'-{PAIN_BIN_CODES[self.pain_bins]}')

    @property
    def value_label(self):
        """Axis/colourbar label, so a figure cannot mislabel its own units."""
        if self.normalization == 'zscore_vs_baseline':
            return 'Mean z-score vs 0-pain baseline'
        if self.normalization == 'baseline_subtract':
            unit = 'log10(V^2/Hz)' if self.domain == 'log' else 'V^2/Hz'
            return f'Mean change vs 0-pain baseline ({unit})'
        return 'Mean log10(V^2/Hz)' if self.domain == 'log' else 'Mean power (V^2/Hz)'

    def to_dict(self):
        return asdict(self)

    def resolved(self):
        """Config with None placeholders filled from the config module.

        Deferred rather than done in defaults so importing this module does not
        require the config package, and so the recorded params show what actually
        applied instead of a null.
        """
        from ieeg_ehr import config
        return ViewConfig(
            **{**asdict(self),
               'max_excluded_frac': (config.EPOCH_MAX_EXCLUDED_FRAC
                                     if self.max_excluded_frac is None
                                     else self.max_excluded_frac),
               'epoch_minutes': (config.EPOCH_MINUTES_BEFORE if self.epoch_minutes is None
                                 else self.epoch_minutes)}
        )

    def provenance(self):
        """Config plus the ROI scheme's full CONTENTS, for a run's provenance.json.

        Contents, not just the name: a JSON scheme on Oak can be edited after the
        run, so the name alone would not reconstruct what was used.
        """
        out = self.resolved().to_dict()
        if self.region != 'none':
            from ieeg_ehr.config import roi_schemes
            out['roi_scheme_contents'] = roi_schemes.scheme_provenance(self.roi_scheme)
        return out


def add_view_arguments(parser):
    """Attach --domain/--normalization/... to an argparse parser.

    Shared so the view builder and the plot script cannot drift into offering
    different axis vocabularies.
    """
    g = parser.add_argument_group('view axes (docs/view_registry.md)')
    g.add_argument('--domain', choices=DOMAINS, default='log')
    g.add_argument('--baseline', choices=['zero_pain_epochs'], default='zero_pain_epochs')
    g.add_argument('--normalization', choices=NORMALIZATIONS, default='zscore_vs_baseline')
    g.add_argument('--epoch-agg', choices=EPOCH_AGGS, default='mean')
    g.add_argument('--freq', choices=FREQ_AGGS, default='log_bins_50')
    g.add_argument('--region', choices=REGION_AGGS, default='individual_dk')
    g.add_argument('--pain-bins', choices=PAIN_BINS, default='subject_relative')
    g.add_argument('--roi-scheme', default='default',
                   help='Built-in ROI scheme name or path to a JSON scheme file')
    g.add_argument('--mask-level', choices=MASK_LEVELS, default='bipolar')
    g.add_argument('--mask-label', default=None,
                   help='QC mask label. Required unless --mask-level none.')
    g.add_argument('--max-excluded-frac', type=float, default=None)
    g.add_argument('--drop-line-noise-bins', action='store_true')
    g.add_argument('--epoch-minutes', type=float, default=None)
    return parser


def from_args(args):
    return ViewConfig(
        domain=args.domain, baseline=args.baseline, normalization=args.normalization,
        epoch_agg=args.epoch_agg, freq=args.freq, region=args.region,
        pain_bins=args.pain_bins, roi_scheme=args.roi_scheme,
        mask_level=args.mask_level, mask_label=args.mask_label,
        max_excluded_frac=args.max_excluded_frac,
        drop_line_noise_bins=args.drop_line_noise_bins,
        epoch_minutes=args.epoch_minutes,
    ).resolved()
