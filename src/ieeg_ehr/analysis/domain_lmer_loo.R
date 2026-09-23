#!/usr/bin/env Rscript
#
# ONE leave-one-subject-out refit of the lme4 PROCESSING DOMAIN model, for one
# band, returning the model's ROI-level pain slopes.
#
#   log10_power ~ 0 + domain + domain:NRS_within + domain:NRS_submean
#                 + (1 + NRS_within || ROI) + (1 + NRS_within || subject)
#                 + (1 + NRS_within || subj_roi) + (1 | chan_id)
#
# The formula is IDENTICAL to domain_lmer.R's all-subjects model -- the only
# difference is the rows: `--exclude <subject>` drops that subject before the
# fit (`--exclude none` fits everyone, the reference for the LOO maps).
#
# WHAT COMES OUT. The group's slope at every (ROI, band) cell, read off the
# refitted model as
#
#     roi_slope = fixef(domain_d:NRS_within) + ranef(ROI)[r, NRS_within]
#
# i.e. the ROI's domain slope plus that ROI's own deviation from it. This is the
# "group map" the per-subject consistency statistic correlates against, rebuilt
# WITHOUT the subject being scored, so a patient is never partly correlated with
# themselves (plot_domain_lmer_consistency.py).
#
# Called by run_domain_lmer_loo.py, one Slurm array task per (band, excluded
# subject). See domain_lmer.R for why this is a subprocess and why lmerTest is
# not needed here (no df / p-values are computed -- only point slopes).

suppressPackageStartupMessages({
  library(data.table)
  library(lme4)
})

args <- commandArgs(trailingOnly = TRUE)
arg_of <- function(flag, default = NA_character_) {
  i <- match(flag, args)
  if (is.na(i) || i == length(args)) default else args[i + 1L]
}

in_csv    <- arg_of("--in")
out_csv   <- arg_of("--out")
band      <- arg_of("--band")
exclude   <- arg_of("--exclude", "none")
optimizer <- arg_of("--optimizer", "bobyqa")
stopifnot(!is.na(in_csv), !is.na(out_csv), !is.na(band))

d <- fread(in_csv)
if ("parcel" %in% names(d) && !("ROI" %in% names(d))) setnames(d, "parcel", "ROI")

if (exclude != "none") {
  if (!(exclude %in% d$subject)) stop(sprintf("--exclude %s not in frame", exclude))
  d <- d[subject != exclude]
}

d[, subj_roi := paste(subject, ROI, sep = "_")]
d[, chan_id  := channel_uid]
for (v in c("domain", "ROI", "subject", "subj_roi", "chan_id")) set(d, j = v, value = factor(d[[v]]))

cat(sprintf("[%s | -%s] rows=%d subjects=%d ROIs=%d\n", band, exclude,
            nrow(d), nlevels(d$subject), nlevels(d$ROI)))

t0 <- Sys.time()
m <- lmer(log10_power ~ 0 + domain + domain:NRS_within + domain:NRS_submean +
            (1 + NRS_within || ROI) + (1 + NRS_within || subject) +
            (1 + NRS_within || subj_roi) + (1 | chan_id),
          data = d, REML = TRUE,
          control = lmerControl(optimizer = optimizer,
                                optCtrl = list(maxfun = 2e5),
                                calc.derivs = TRUE))
fit_seconds <- as.numeric(difftime(Sys.time(), t0, units = "secs"))

fe <- fixef(m)

# `||` splits the ROI term into separate grouping entries (`ROI`, `ROI.1`, ...)
# in lme4, so the NRS_within column is looked for in EVERY ROI entry rather than
# assumed to sit in `ranef(m)$ROI`. `subj_roi` is excluded by the anchor.
re <- ranef(m)
roi_dev <- NULL
for (nm in grep("^ROI(\\.[0-9]+)?$", names(re), value = TRUE)) {
  if ("NRS_within" %in% names(re[[nm]])) {
    roi_dev <- setNames(re[[nm]][["NRS_within"]], rownames(re[[nm]]))
  }
}
if (is.null(roi_dev)) stop("no ROI random slope on NRS_within found in ranef(m)")

roi_dom <- unique(d[, .(ROI = as.character(ROI), domain = as.character(domain))])
if (anyDuplicated(roi_dom$ROI)) stop("an ROI maps to more than one domain")
roi_dom[, fixed_slope := fe[paste0("domain", domain, ":NRS_within")]]
roi_dom[, roi_dev := roi_dev[ROI]]
roi_dom[, roi_slope := fixed_slope + roi_dev]
if (anyNA(roi_dom$roi_slope)) stop("NA ROI slope -- coefficient name mismatch")

warns <- m@optinfo$conv$lme4$messages
roi_dom[, `:=`(band = band, excluded = exclude,
               n_subjects = nlevels(d$subject), n_rows = nrow(d),
               fit_seconds = fit_seconds,
               singular = isSingular(m, tol = 1e-4),
               n_warnings = length(warns),
               warnings = paste(warns, collapse = " | "))]

dir.create(dirname(out_csv), recursive = TRUE, showWarnings = FALSE)
fwrite(roi_dom, out_csv)
cat(sprintf("[%s | -%s] fitted in %.1f s -> %s\n", band, exclude, fit_seconds, out_csv))
