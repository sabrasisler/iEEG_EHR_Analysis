"""Per-subject pain state decoding (PLANNING "Pain state decoding").

A replication of Prasad et al. 2025 (doi 10.1038/s41467-025-59756-5) on the
discovery cohort at ~4x their n: one model per subject-session, predicting that
subject's own reported pain score from the 5-min pre-report spectro-spatial
feature set.

PREDICTION, NOT INFERENCE. This asks whether a per-subject decoder works. It is
a different object from Phase 3's confirmation GLMM, and its output is
NOMINATIONS -- nothing here is a finding before P2.6.

THE ONE THING TO KNOW BEFORE READING ANY NUMBER OUT OF THIS
-----------------------------------------------------------
Every quantity estimated from data -- the per-feature mean and SD, the imputation
value, the elastic-net penalty -- is fitted INSIDE THE TRAINING FOLD ONLY, and it
is fitted there structurally rather than by remembering to: everything lives in
an sklearn `Pipeline` wrapped in the CV, so leakage would take deliberate effort.
The one transform deliberately NOT applied is a 0-pain baseline, because that is
fitted using the labels and leaks straight into any cross-validated score
(DECISIONS 2026-09-08).

Module map:
  cascade.py   channel/epoch eligibility -- Y/Z, and coverage vs artifact
  features.py  the (epoch x channel-band) matrix, assembled from a view
  cv.py        nested CV, bootstraps, the shuffled-label null
  arms.py      the three model families
  run_decoder.py   CLI, ONE subject-session per invocation (Slurm array task)
  aggregate.py     per-unit outputs -> the run's tables
"""
