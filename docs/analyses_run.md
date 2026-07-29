# analyses_run.md — the terse index of every run

Machine-appended by `log_analysis()` (`src/ieeg_ehr/io/analysis_log.py`). One
line per analysis run. **Append-only. Do not hand-edit** except to flip a
`[ ]` to `[x]`, which is what `/lognote` does.

No interpretation lives here — this is the bare "what have I run" index. The
narrative goes in `docs/labnotebook/YYYY-MM-DD.md`.

Format:

```
- YYYY-MM-DD HH:MM | <one-sentence description> | <output_path> | <git_hash> | [logged?]
```

`$DERIV/` abbreviates the Oak derivatives base. `+dirty` on a hash means the
working tree had uncommitted changes, so that commit does NOT describe what ran.
`[x]` means a `docs/labnotebook/` entry references this run.

---

- 2026-07-27 15:39 | P0.6 dtype audit: float32 cache round-trip + epoch-average precision, 1 subject(s) | $DERIV/qc/feature_level/validation/dtype_audit/smoke_2026-07-27T153908 | 34da3ee180a1+dirty | [ ]
- 2026-07-27 15:52 | P0.6 dtype audit: float32 cache round-trip + epoch-average precision, 3 subject(s) | $DERIV/qc/feature_level/validation/dtype_audit/p0.6_2026-07-27T155201 | 34da3ee180a1+dirty | [ ]
- 2026-07-27 16:00 | P0.6 dtype audit: float32 cache round-trip + epoch-average precision, 3 subject(s) | $DERIV/qc/feature_level/validation/dtype_audit/p0.6_2026-07-27T160009 | 34da3ee180a1+dirty | [ ]
- 2026-07-28 12:46 | P1.3 view heatmaps (zscore_vs_baseline, mask std10_rv-gross-std3_satmargin15_sw_logz4), n=4 | $DERIV/analysis/scratch/view_heatmap/subject_relative/p13_std10_zscore_vs_baseline_20260728-124613 | 8cd9e87958d7+dirty | [ ]
- 2026-07-28 12:46 | P1.3 view heatmaps (baseline_subtract, mask std10_rv-gross-std3_satmargin15_sw_logz4), n=4 | $DERIV/analysis/scratch/view_heatmap/subject_relative/p13_std10_baseline_subtract_20260728-124617 | 8cd9e87958d7+dirty | [ ]
- 2026-07-28 12:48 | P1.3 view heatmaps (zscore_vs_baseline, mask std10_rv-gross-std3_satmargin15_sw_logz4), n=4 | $DERIV/analysis/scratch/view_heatmap/subject_relative/p13_std10_zscore_nolinenoise_20260728-124839 | 8cd9e87958d7+dirty | [ ]
- 2026-07-28 10:41 | raw-voltage mask coverage audit, feature_level level: 3/3 subject-sessions fully covered, 0 with no mask file, 0 with a mask but missing runs | /home/users/sisler/.claude/jobs/272aee09/tmp/smoke_mask_cov | 3c7af732b7a2+dirty | [ ]
- 2026-07-28 11:42 | raw-voltage mask coverage audit, feature_level level: 89/89 subject-sessions fully covered, 0 with no mask file, 0 with a mask but missing runs | $DERIV/qc/feature_level/validation/mask_coverage/20260728T114248 | 7191f370c9ca+dirty | [ ]
- 2026-07-28 11:42 | raw-voltage mask coverage audit, bipolar level: 53/106 subject-sessions fully covered, 16 with no mask file, 37 with a mask but missing runs | $DERIV/qc/bipolar/validation/mask_coverage/20260728T114254 | 7191f370c9ca+dirty | [ ]
- 2026-07-28 11:51 | raw-voltage mask coverage audit, feature_level level: 89/89 subject-sessions fully covered, 0 with no mask file, 0 with a mask but missing runs | $DERIV/qc/feature_level/validation/mask_coverage/20260728T115123 | c4900a7f7127+dirty | [ ]
- 2026-07-28 11:51 | raw-voltage mask coverage audit, bipolar level: 88/106 subject-sessions fully covered, 18 with no mask file, 0 with a mask but missing runs | $DERIV/qc/bipolar/validation/mask_coverage/20260728T115127 | c4900a7f7127+dirty | [ ]
- 2026-07-28 11:57 | raw-voltage mask coverage audit, bipolar level: 53/106 subject-sessions fully covered, 16 with no mask file, 37 with a mask but missing runs | $DERIV/qc/bipolar/validation/mask_coverage/20260728T115748 | 7b5346ad48e8+dirty | [ ]
- 2026-07-28 12:19 | raw-voltage mask coverage audit, feature_level level: 89/89 subject-sessions fully covered, 0 with no mask file, 0 with a mask but missing runs | $DERIV/qc/feature_level/validation/mask_coverage/20260728T121916 | 7b5346ad48e8+dirty | [ ]
- 2026-07-29 11:39 | P1.3 region spectra (baseline_subtract, mask std10_rv-gross-std3_satmargin15_sw_logz4), 15 regions >= 8 subj, n=55 | $DERIV/analysis/scratch/region_spectrum/blsub-rel/smoke_blsub_20260729-113909 | 2e7589eb8031+dirty | [ ]
- 2026-07-29 11:49 | P1.3 region spectra (baseline_subtract, mask std10_rv-gross-std3_satmargin15_sw_logz4), 15 regions >= 8 subj, n=55 | $DERIV/analysis/scratch/region_spectrum/blsub-rel/smoke_blsub_20260729-114856 | 2e7589eb8031+dirty | [ ]
- 2026-07-29 11:53 | P1.3 region spectra (baseline_subtract, mask std10_rv-gross-std3_satmargin15_sw_logz4), 15 regions >= 8 subj, n=55 | $DERIV/analysis/scratch/region_spectrum/blsub-rel/smoke_blsub_20260729-115353 | 2e7589eb8031+dirty | [ ]
- 2026-07-29 11:56 | P1.3 region spectra (baseline_subtract, mask std10_rv-gross-std3_satmargin15_sw_logz4), 15 regions >= 8 subj, n=55 | $DERIV/analysis/scratch/region_spectrum/blsub-rel/smoke_blsub_20260729-115609 | 2e7589eb8031+dirty | [ ]
- 2026-07-29 11:57 | P1.3 view heatmaps (baseline_subtract, mask std10_rv-gross-std3_satmargin15_sw_logz4), n=55 | $DERIV/analysis/scratch/region_freq_heatmap/blsub-rel/smoke_blsub_20260729-115617 | 2e7589eb8031+dirty | [ ]
- 2026-07-29 15:07 | P1.3 region spectra (baseline_subtract, mask std10_rv-gross-std3_satmargin15_sw_logz4), 15 regions >= 8 subj, n=55 | $DERIV/analysis/scratch/region_spectrum/delta-relpain/recolor_check_20260729-150732 | 705ba7695eab+dirty | [ ]
- 2026-07-29 15:25 | P1.3 view heatmaps (zscore_vs_baseline, mask std10_rv-gross-std3_satmargin15_sw_logz4), n=56 | $DERIV/analysis/pain/psd_physiology/region_freq_heatmap/zscore-relpain/discovery_std10_zscore_vs_baseline_20260729-152430 | 4938c78eeff1+dirty | [ ]
- 2026-07-29 15:25 | P1.3 region spectra (zscore_vs_baseline, mask std10_rv-gross-std3_satmargin15_sw_logz4), 15 regions >= 8 subj, n=56 | $DERIV/analysis/pain/psd_physiology/region_spectrum/zscore-relpain/discovery_std10_zscore_vs_baseline_20260729-152530 | 4938c78eeff1+dirty | [ ]
- 2026-07-29 15:26 | P1.3 view heatmaps (baseline_subtract, mask std10_rv-gross-std3_satmargin15_sw_logz4), n=56 | $DERIV/analysis/pain/psd_physiology/region_freq_heatmap/delta-relpain/discovery_std10_baseline_subtract_20260729-152540 | 4938c78eeff1+dirty | [ ]
- 2026-07-29 15:26 | P1.3 region spectra (baseline_subtract, mask std10_rv-gross-std3_satmargin15_sw_logz4), 15 regions >= 8 subj, n=56 | $DERIV/analysis/pain/psd_physiology/region_spectrum/delta-relpain/discovery_std10_baseline_subtract_20260729-152640 | 4938c78eeff1+dirty | [ ]
