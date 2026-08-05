# Methods summary — slide bullets

Short, presentation-ready bullets on the preprocessing / artifact-rejection and
feature-extraction steps, each with the reason it was chosen. Written for a
slideshow, so every bullet is one line of claim + one clause of rationale.

Descriptive, not normative. Source of truth for any parameter is the config
module (`src/ieeg_ehr/config/`); source of truth for design rationale is
`docs/architecture.md`, `docs/qc_context.md`, and `DECISIONS.md`. If this file
disagrees with those, they win.

Parameters below are the **pinned mask**
(`ieeg_ehr.config.CANONICAL_MASK_LABEL` = `gross-std3_satmargin15_sw_logz4`),
not the raw config defaults in `config/qc_params.py`.

---

## Artifact rejection — raw voltage

Four **independent** detectors (no detector's output feeds another's baseline),
all per-channel on 2 s windows, except gross artifact at 60 s:

- **Amplifier saturation** — flag any window with ≥1 sample at the rail. The
  rail is *inferred from the data* (a repeated identical extreme value), pooled
  over the whole session and confirmed by cross-channel agreement (≥25% of
  channels), so channels that never clip still get a rail.
  *Why: clipped samples are not signal, and rails vary by subject gain
  (~3200 µV, a few subjects ~50 µV) — a fixed assumed cutoff would be wrong.*
- **Flatline** — per-window variance below an absolute floor (5e-13 V²), OR'd
  with a **per-channel relative** cutoff (z on log₁₀ variance, one-sided low).
  *Why: the absolute cutoff alone misclassified genuinely-quiet-but-real
  channels as dead; variance is lognormal and spans many orders of magnitude,
  so the z-score is taken in log space, where the low end is actually
  resolvable.*
- **Square-wave / two-level artifact** — fraction of samples pinned within 5% of
  the window's own min/max > 0.9, with a range floor so flat windows aren't
  re-flagged.
  *Why: this digital/relay artifact is invisible to the other three (variance is
  high, it never reaches the rail, it is mean-neutral). The metric is
  dimensionless, hence amplitude- AND frequency-independent — no per-subject
  tuning.*
- **Gross artifact** — session-relative high variance on 60 s windows
  (z > 3, one-sided high only).
  *Why: catches unplug/replug-style bursts; session-relative because absolute
  amplitude is not comparable across subjects. One-sided because low variance is
  flatline's job.*

Structural choices behind the pipeline:

- **Metric/threshold split** — the expensive raw-NWB pass stores only
  *continuous metrics*, never an `excluded` flag; cheap CSV-only steps own
  thresholding and bucketing.
  *Why: retuning any threshold re-runs minutes of CSV work instead of hours of
  re-reading raw recordings.*
- **Detection at 2 s, exclusion rolled up to 60 s bins** — a 60 s bin is
  excluded if ANY 2 s window inside it is flagged.
  *Why: gives a conservative margin around each artifact without a slow
  per-channel absolute-time padding step.*
- **Thresholds set on structural grounds, before looking at any pain
  relationship**, and validated with threshold sweeps plus raw-trace review.
  *Why: keeps QC choices independent of the effect being measured.*

## Preprocessing

- **Bipolar re-referencing** — adjacent contacts within each shaft
  (anode − cathode).
  *Why: removes the shared reference and volume-conducted / global noise,
  yielding a local, spatially specific signal — standard for sEEG.*
- **The bipolar time series is never persisted**, only recomputed on demand.
  *Why: it is cheap to regenerate from raw; storing it would cost terabytes for
  no benefit.*
- **Bipolar-pair variance** is computed as a *secondary / diagnostic* exclusion
  (with a mask-aware baseline), not a primary filter.
  *Why: as assessed it flags more non-artifactual than artifactual events, so
  the raw-voltage mask remains the operative QC layer.*

## Feature extraction

- **PSD in 50 log-spaced frequency bins, 1–250 Hz.**
  *Why log spacing: neural power is roughly 1/f, so log bins carry approximately
  equal information per bin instead of oversampling high frequencies. Why
  250 Hz: Nyquist-safe for the rare 500 Hz-sampled subjects — the range can be
  restricted downstream, but truncated data cannot be recovered.*
- **2 s hann window, 50% overlap (1 s hop).**
  *Why: 0.5 Hz frequency resolution, and time granularity matched to the 2 s QC
  grid. Deliberate tradeoff — a single-segment periodogram per window is noisier
  than a multi-segment Welch average, accepted in exchange for much finer time
  resolution.*
- **Average linear power within a bin, then take the log.**
  *Why: averaging logs first biases the estimate low (Jensen's inequality).*
- **Line noise is flagged, not removed** — bins overlapping 60/120/180/240 Hz
  ± 2 Hz carry a `contains_line_noise` flag, and canonical band edges are
  constructed to fall strictly between harmonics.
  *Why: flagging is reversible and lets each downstream analysis decide; it is
  also why gamma is split into several narrow bands rather than one wide one.*
- **Canonical bands are not precomputed** — they are aggregated from the stored
  50 bins on demand.
  *Why: retuning band definitions never requires re-reading raw data.*
- **The cache stores per-2 s-window log-power, QC-masked,
  pre-normalization and pre-averaging.**
  *Why: normalization must happen per-window before averaging (Jensen again),
  and keeping the cache un-normalized makes any view or normalization choice a
  cheap recompute rather than an expensive re-extraction.*
- **Stored float32, computed float64.**
  *Why (measured, P0.6 dtype audit): float32 storage costs only ~8 significant
  figures end to end, but a float32 accumulator over a ~300-window epoch drops
  to 6, and exponentiating log-power near −36.8 in float32 can underflow a quiet
  channel to exactly zero.*

## Caveats before any of this is presented as settled

- The gross-artifact and flatline thresholds are **pending formal pinning**
  (P0.1); the current mask label is the working candidate, not a frozen choice.
- `docs/architecture.md` and `config/psd_params.py` **disagree on the canonical
  band edges** (flagged 2026-07-27, unresolved — band choice is a P2.2 sweep
  axis). Avoid putting specific band edges on a slide until that is settled.
- Bipolar-pair variance exclusion is not validated; do not describe it as a
  filter the analyses are gated on.
