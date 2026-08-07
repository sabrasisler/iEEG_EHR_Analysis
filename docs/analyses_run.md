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
- 2026-07-29 15:51 | P1.3 view heatmaps (zscore_vs_baseline, mask std10_rv-gross-std3_satmargin15_sw_logz4), n=56 | $DERIV/analysis/pain/psd_physiology/region_freq_heatmap/zscore-relpain/discovery_std10_zscore_vs_baseline_line_noise_excl_20260729-155051 | 062548c874ba+dirty | [ ]
- 2026-07-29 18:08 | P1.3 view heatmaps (zscore_vs_baseline, mask std10_rv-gross-std3_satmargin15_sw_logz4), n=56 | $DERIV/analysis/scratch/region_freq_heatmap/zscore-relpain/csmoke_20260729-180749 | 291966948b57+dirty | [ ]
- 2026-07-29 18:36 | P1.3 view heatmaps (zscore_vs_baseline, mask std10_rv-gross-std3_satmargin15_sw_logz4), n=56 | $DERIV/analysis/scratch/region_freq_heatmap/zscore-relpain/csmoke_20260729-183528 | 291966948b57+dirty | [ ]
- 2026-07-29 18:41 | P1.3 view heatmaps (zscore_vs_baseline, mask std10_rv-gross-std3_satmargin15_sw_logz4), n=56 | $DERIV/analysis/scratch/region_freq_heatmap/zscore-relpain/csmoke_20260729-184013 | 291966948b57+dirty | [ ]
- 2026-07-29 18:46 | P1.3 view heatmaps (zscore_vs_baseline, mask std10_rv-gross-std3_satmargin15_sw_logz4), n=56 | $DERIV/analysis/pain/psd_physiology/region_freq_heatmap/zscore-relpain/discovery_std10_zscore_vs_baseline_20260729-184452 | 5e3ce07a0ea1+dirty | [ ]
- 2026-07-29 18:46 | P1.3 region spectra (zscore_vs_baseline, mask std10_rv-gross-std3_satmargin15_sw_logz4), 15 regions >= 8 subj, n=56 | $DERIV/analysis/pain/psd_physiology/region_spectrum/zscore-relpain/discovery_std10_zscore_vs_baseline_20260729-184643 | 5e3ce07a0ea1+dirty | [ ]
- 2026-07-29 18:48 | P1.3 view heatmaps (baseline_subtract, mask std10_rv-gross-std3_satmargin15_sw_logz4), n=56 | $DERIV/analysis/pain/psd_physiology/region_freq_heatmap/delta-relpain/discovery_std10_baseline_subtract_20260729-184659 | 5e3ce07a0ea1+dirty | [ ]
- 2026-07-29 18:48 | P1.3 region spectra (baseline_subtract, mask std10_rv-gross-std3_satmargin15_sw_logz4), 15 regions >= 8 subj, n=56 | $DERIV/analysis/pain/psd_physiology/region_spectrum/delta-relpain/discovery_std10_baseline_subtract_20260729-184845 | 5e3ce07a0ea1+dirty | [ ]
- 2026-08-05 08:54 | P1.3 region spectra (zscore_vs_baseline, mask std10_rv-gross-std3_satmargin15_sw_logz4), 21 regions >= 8 subj, n=56 | $DERIV/analysis/pain/psd_physiology/region_spectrum/zscore-relpain-roiv2/discovery_std10_zscore_vs_baseline_20260805-085413 | 519020a176d4+dirty | [ ]
- 2026-08-05 08:54 | P1.3 region spectra (baseline_subtract, mask std10_rv-gross-std3_satmargin15_sw_logz4), 21 regions >= 8 subj, n=56 | $DERIV/analysis/pain/psd_physiology/region_spectrum/delta-relpain-roiv2/discovery_std10_baseline_subtract_20260805-085421 | 519020a176d4+dirty | [ ]
- 2026-08-05 08:56 | P1.3 region spectra (none, mask std10_rv-gross-std3_satmargin15_sw_logz4), 21 regions >= 8 subj, n=56 | $DERIV/analysis/pain/psd_physiology/region_spectrum/raw-relpain-roiv2/discovery_std10_none_20260805-085612 | 519020a176d4+dirty | [ ]
- 2026-08-05 08:57 | P1.3 view heatmaps (zscore_vs_baseline, mask std10_rv-gross-std3_satmargin15_sw_logz4), n=56 | $DERIV/analysis/pain/psd_physiology/region_freq_heatmap/zscore-relpain-roiv2/discovery_std10_zscore_vs_baseline_20260805-085613 | 519020a176d4+dirty | [ ]
- 2026-08-05 08:57 | P1.3 view heatmaps (baseline_subtract, mask std10_rv-gross-std3_satmargin15_sw_logz4), n=56 | $DERIV/analysis/pain/psd_physiology/region_freq_heatmap/delta-relpain-roiv2/discovery_std10_baseline_subtract_20260805-085614 | 519020a176d4+dirty | [ ]
- 2026-08-05 09:18 | electrode coverage glass brain (roi_v2), 4631 pairs, n=55 | $DERIV/analysis/pain/psd_physiology/electrode_locations/roiv2/discovery_20260805-091838 | 519020a176d4+dirty | [ ]
- 2026-08-05 09:46 | beta-band violins by region, within-subject z, 21 regions, n=55 | $DERIV/analysis/pain/psd_physiology/band_violin/raw-relpain-roiv2-beta/discovery_beta_20260805-094534 | 3bfa9b47caac+dirty | [ ]
- 2026-08-05 09:55 | beta-band violins by region, within-subject z, 21 regions, n=55 | $DERIV/analysis/pain/psd_physiology/band_violin/raw-relpain-roiv2-beta/discovery_beta_20260805-095459 | 7a53bb8ac460+dirty | [ ]
- 2026-08-05 10:48 | theta-band violins by region, within-subject z, 21 regions, n=55 | $DERIV/analysis/pain/psd_physiology/band_violin/raw-relpain-roiv2-theta/discovery_theta_20260805-104805 | 3e147074a157+dirty | [ ]
- 2026-08-05 10:48 | delta-band violins by region, within-subject z, 21 regions, n=55 | $DERIV/analysis/pain/psd_physiology/band_violin/raw-relpain-roiv2-delta/discovery_delta_20260805-104806 | 3e147074a157+dirty | [ ]
- 2026-08-05 10:55 | beta-band violins by region, within-subject z, 21 regions, n=56 | $DERIV/analysis/pain/psd_physiology/band_violin/zscore-relpain-roiv2-beta/discovery_beta_20260805-105451 | 3e147074a157+dirty | [ ]
- 2026-08-05 10:55 | theta-band violins by region, within-subject z, 21 regions, n=56 | $DERIV/analysis/pain/psd_physiology/band_violin/zscore-relpain-roiv2-theta/discovery_theta_20260805-105450 | 3e147074a157+dirty | [ ]
- 2026-08-05 10:55 | delta-band violins by region, within-subject z, 21 regions, n=56 | $DERIV/analysis/pain/psd_physiology/band_violin/zscore-relpain-roiv2-delta/discovery_delta_20260805-105451 | 3e147074a157+dirty | [ ]
- 2026-08-06 16:14 | 1/f slope violins by region (1-250 Hz fit), 21 regions, n=55 | $DERIV/analysis/pain/psd_physiology/slope_violin/raw-relpain-roiv2/discovery_std10_slope_20260806-161319 | 18a9273cad80+dirty | [ ]
- 2026-08-06 16:43 | 1/f slope trajectory: paired + per-subject regression on the continuous score + ribbon (1-250 Hz fit), 21 regions, n=55 | $DERIV/analysis/scratch/slope_trajectory/raw-relpain-roiv2/traj_test_20260806-164343 | 405f31221567+dirty | [ ]
- 2026-08-07 13:32 | 1/f slope trajectory: paired + per-subject regression on the continuous score + ribbon (1-250 Hz fit), 21 regions, n=55 | $DERIV/analysis/pain/psd_physiology/slope_trajectory/raw-relpain-roiv2/discovery_std10_slope_20260807-133149 | 92c77737e8d9+dirty | [ ]
- 2026-08-07 13:32 | 1/f slope trajectory: paired + per-subject regression on the continuous score + ribbon (1-250 Hz fit), 21 regions, n=55 | $DERIV/analysis/pain/psd_physiology/slope_trajectory/raw-relpain-roiv2/discovery_std10_slope_nonzero_20260807-133216 | 92c77737e8d9+dirty | [ ]
- 2026-08-07 14:55 | 1/f slope CONTRAST violins (low-none & high-none) vs a permutation noise floor, 18 regions, n=30 | $DERIV/analysis/pain/psd_physiology/slope_contrast/raw-relpain-roiv2/discovery_std10_slope_20260807-145426 | 3e76acf5f8ed+dirty | [ ]
- 2026-08-07 15:01 | 1/f slope CONTRAST violins (low-none & high-none) vs a permutation noise floor, 18 regions, n=30 | $DERIV/analysis/pain/psd_physiology/slope_contrast/raw-relpain-roiv2/discovery_std10_slope_20260807-150035 | 3e76acf5f8ed+dirty | [ ]
