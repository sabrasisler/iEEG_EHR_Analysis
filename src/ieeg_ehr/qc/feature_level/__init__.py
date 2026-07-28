"""
Feature-level QC (P2.1): data-quality facts computed from the stored bipolar PSD
rather than from the raw voltage trace.

Same three-stage shape as the raw-voltage level, so the two are readable as one
system (see `config/paths.py`, FEATURE-LEVEL QC):

    detect_power_outlier.py   metrics  (expensive; reads NWB; run once)
    build_feature_exclusions  exclusions/<type>/<label>/  (cheap; owns K and B)
    build_feature_mask        masks/<label>/              (cheap; OR across types)

The epoch-level half of the cascade (thresholds X, Y, Z in
`config/feature_qc_params.py`) is NOT here — it is defined over epochs, which
only exist once an epoch definition does, so it lives in the view layer and is
applied to the epoch cache at load time.
"""
