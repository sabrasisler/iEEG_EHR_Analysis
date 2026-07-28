"""
The view layer (P1.3): cheap, deterministic transforms from the per-window cache
to analysis-ready tables.

A view is a FUNCTION, recomputed at load by default and NOT saved
(architecture.md PART 2 / CLAUDE.md). Materialize only when recompute is measured
slow and something depends on it -- or, temporarily, when you want to read the
numbers before trusting a figure.

The seven axes and their fixed order live in docs/view_registry.md; ViewConfig
encodes one choice per axis, plus the QC mask, which is a view-time choice now
that the cache stores raw unmasked slices (P1.1, 2026-07-27).

    from ieeg_ehr.views import ViewConfig, build_subject_view

Submodules: `view_config` (the axes) · `channel_meta` (pair order + DK labels from
NWB metadata) · `cache_reader` (row-group streaming + masking) · `axes` (the seven
transforms) · `build_pain_epoch_view` (orchestration + CLI).
"""

from ieeg_ehr.views.view_config import (   # noqa: F401
    ViewConfig,
    add_view_arguments,
    from_args,
)

__all__ = ['ViewConfig', 'add_view_arguments', 'from_args', 'build_subject_view']


def __getattr__(name):
    """Lazily expose build_subject_view.

    Deferred because build_pain_epoch_view pulls in pyarrow and (via
    channel_meta) pynwb, and a caller that only wants to construct a ViewConfig --
    an argparse setup, a unit test of the axes -- should not pay for those
    imports. Same reasoning as ieeg_ehr.io not re-exporting its nwb submodule.
    """
    if name == 'build_subject_view':
        from ieeg_ehr.views.build_pain_epoch_view import build_subject_view
        return build_subject_view
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
