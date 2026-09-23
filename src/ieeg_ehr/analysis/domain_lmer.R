#!/usr/bin/env Rscript
#
# One lme4 fit per band for the PROCESSING DOMAIN model, with the CROSSED
# region random effects that statsmodels cannot express.
#
#   log10_power ~ 0 + domain + domain:NRS_within + domain:NRS_submean
#                 + (1 + NRS_within || ROI)         # ROI deviates from its domain
#                 + (1 + NRS_within || subject)     # brain-wide pain sensitivity
#                 + (1 + NRS_within || subj_roi)    # subject's slope varies by region
#                 + (1 | chan_id)                   # channel intercept
#
# WHY R AT ALL. `statsmodels.MixedLM` takes ONE grouping variable and evaluates
# every `vc_formula` term WITHIN it, so a crossed ROI term is SILENTLY NESTED --
# verified on synthetic data, no error and no warning (see run_domain_model.py).
# The ROI term above is the guard that keeps one well-sampled ROI from carrying
# its domain, and it is only a guard if it is crossed. lme4 fits it directly.
#
# WHY A SUBPROCESS AND NOT rpy2. rpy2 links against libR at build time, so it
# must be compiled against the exact R module in use and breaks when that module
# moves; an R-level error takes the Python process down with it; and the fit
# cannot be re-run on its own. A frame on disk plus `Rscript` is debuggable by
# hand, re-runnable without re-reading the view, and has no ABI coupling. The
# model specification below is the only thing that matters and it is identical
# either way.
#
# NOTE ON `||`. For a NUMERIC predictor `(1 + x || g)` correctly expands to
# `(1 | g) + (0 + x | g)`, which is what is wanted here: uncorrelated intercept
# and slope. `NRS_within` is numeric. The well-known `||` trap applies to
# FACTOR predictors and does not bite here.
#
# NOTE ON lmerTest. `library(lmerTest)` is loaded AFTER lme4 deliberately: it
# masks `lmer()` with a version returning `lmerModLmerTest`, which is what
# carries the Jacobian that Satterthwaite degrees of freedom need. Fitting with
# bare `lme4::lmer` and then asking emmeans for Satterthwaite silently falls
# back to asymptotic (z) tests.

suppressPackageStartupMessages({
  library(data.table)
  library(lme4)
  library(lmerTest)
  library(emmeans)
})

args <- commandArgs(trailingOnly = TRUE)
arg_of <- function(flag, default = NA_character_) {
  i <- match(flag, args)
  if (is.na(i) || i == length(args)) default else args[i + 1L]
}

in_csv   <- arg_of("--in")
out_dir  <- arg_of("--out")
band     <- arg_of("--band")
df_mode  <- arg_of("--df", "satterthwaite")
drop_dom <- arg_of("--drop-domain", "")
optimizer <- arg_of("--optimizer", "bobyqa")

stopifnot(!is.na(in_csv), !is.na(out_dir), !is.na(band))
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

d <- fread(in_csv)

# The frame arrives with the column names the Python pipeline uses. `parcel`
# holds the ROI when the run was built with `--unit roi` -- renamed here so the
# model formula reads the way the specification is written.
if ("parcel" %in% names(d) && !("ROI" %in% names(d))) setnames(d, "parcel", "ROI")

if (nzchar(drop_dom)) {
  for (dom in strsplit(drop_dom, ",")[[1]]) d <- d[domain != dom]
}

# EXPLICIT nesting IDs built here rather than relying on R's ":" operator, which
# builds an interaction of FACTOR LEVELS and would create empty cells for every
# subject x ROI combination that does not exist.
d[, subj_roi := paste(subject, ROI, sep = "_")]
d[, chan_id  := if ("channel_uid" %in% names(d)) channel_uid
                else paste(subj_roi, channel, sep = "_")]

d[, domain   := factor(domain)]
d[, ROI      := factor(ROI)]
d[, subject  := factor(subject)]
d[, subj_roi := factor(subj_roi)]
d[, chan_id  := factor(chan_id)]

cat(sprintf("[%s] rows=%d subjects=%d ROIs=%d domains=%d channels=%d\n",
            band, nrow(d), nlevels(d$subject), nlevels(d$ROI),
            nlevels(d$domain), nlevels(d$chan_id)))

# THE DIAGNOSIS STRATUM, when one is present.
#
#   0 + domain:dx + domain:dx:NRS_within + domain:NRS_submean
#
# Full cell means over domain x dx: every (domain, stratum) gets its own
# intercept and its own pain slope, so `case - control WITHIN a domain` is a
# contrast of two estimated slopes rather than a coefficient read off a
# three-way interaction.
#
# `dx` GETS NO RANDOM EFFECT. It is constant within a subject, so a by-subject
# dx term is confounded with the subject intercept and is not estimable.
#
# `NRS_submean` is NOT split by dx -- deliberately (analyst's call). It is a
# BETWEEN-subject term and there are 51 subjects; splitting it would spend five
# more degrees of freedom on a nuisance parameter. The consequence to be aware
# of is that a dx difference in BASELINE pain level is absorbed into a shared
# term rather than allowed to differ by stratum.
#
# WHAT THIS CONTRAST COSTS. `dx` is a SUBJECT-level label, so case-vs-control
# is a BETWEEN-subject comparison at n=51 -- unlike the pain slope, which is
# within-subject. Expect wide intervals and small Satterthwaite df here.
has_dx <- "dx_state" %in% names(d)

# THE HANDSHAKE. `--expect-dx` is what the CALLER believes it sent. Selecting
# the model from the column alone is how a dropped column once produced six
# converged fits of the WRONG model with a sidecar that named the right one.
# A disagreement is a bug in the caller, so it stops here rather than being
# resolved by guessing.
expect_dx <- arg_of("--expect-dx", "")
if (nzchar(expect_dx)) {
  want <- expect_dx == "1"
  if (want != has_dx) {
    stop(sprintf(
      paste("--expect-dx=%s but dx_state is %s in the input frame.",
            "Refusing to choose a model by guessing:",
            "fix the caller's column selection."),
      expect_dx, if (has_dx) "PRESENT" else "ABSENT"))
  }
}

# THE MEDICATION MODEL, when a medication state is present.
#
#   0 + domain:med + domain:med:NRS_within + domain:NRS_submean + med_submean
#
# STRUCTURALLY DIFFERENT FROM dx, and the difference is the whole point.
# `med_state` is EPOCH-level: it varies WITHIN a patient as doses come and go,
# so "on-drug minus off-drug pain slope" is a WITHIN-subject contrast. The dx
# contrast was between-subject at n=51 and had almost nothing to work with;
# this one is powered by epochs.
#
# `med_submean` IS NOT OPTIONAL. Without it the `med` contrast absorbs the
# between-patient difference between heavily and lightly medicated patients,
# which is confounded with why they were medicated at all. Holding it lets the
# on/off comparison stay inside a patient.
#
# `med_within` GETS A SUBJECT RANDOM SLOPE, which dx could not have: it varies
# within subject, so patients can genuinely differ in how a dose moves their
# power, and pretending otherwise understates the standard error the same way
# the missing subject x ROI slope did.
has_med <- "med_state" %in% names(d)

expect_med <- arg_of("--expect-med", "")
if (nzchar(expect_med)) {
  want_med <- expect_med == "1"
  if (want_med != has_med) {
    stop(sprintf(
      paste("--expect-med=%s but med_state is %s in the input frame.",
            "Refusing to choose a model by guessing."),
      expect_med, if (has_med) "PRESENT" else "ABSENT"))
  }
}

if (has_dx) {
  d[, dx := factor(ifelse(dx_state > 0.5, "case", "control"))]
  cat(sprintf("[%s] dx strata: %s\n", band,
              paste(names(table(d$dx)), table(d$dx), sep = "=", collapse = " ")))
  fml <- log10_power ~ 0 + domain:dx + domain:dx:NRS_within +
    domain:NRS_submean +
    (1 + NRS_within || ROI) + (1 + NRS_within || subject) +
    (1 + NRS_within || subj_roi) + (1 | chan_id)
} else if (has_med) {
  d[, med := factor(ifelse(med_state > 0.5, "on", "off"), levels = c("off", "on"))]
  cat(sprintf("[%s] med epochs: %s\n", band,
              paste(names(table(d$med)), table(d$med), sep = "=", collapse = " ")))
  fml <- log10_power ~ 0 + domain:med + domain:med:NRS_within +
    domain:NRS_submean + med_submean +
    (1 + NRS_within || ROI) +
    (1 + NRS_within + med_within || subject) +
    (1 + NRS_within || subj_roi) + (1 | chan_id)
} else {
  fml <- log10_power ~ 0 + domain + domain:NRS_within + domain:NRS_submean +
    (1 + NRS_within || ROI) + (1 + NRS_within || subject) +
    (1 + NRS_within || subj_roi) + (1 | chan_id)
}

t0 <- Sys.time()
m <- lmer(
  fml,
  data = d, REML = TRUE,
  control = lmerControl(optimizer = optimizer,
                        optCtrl = list(maxfun = 2e5),
                        calc.derivs = TRUE)
)
fit_seconds <- as.numeric(difftime(Sys.time(), t0, units = "secs"))
cat(sprintf("[%s] fitted in %.1f s\n", band, fit_seconds))

warns <- m@optinfo$conv$lme4$messages
if (is.null(warns)) warns <- character(0)
singular <- isSingular(m, tol = 1e-4)

emm_options(lmer.df = df_mode, lmerTest.limit = 2e6, pbkrtest.limit = 2e6)

# Satterthwaite on a frame this size can exhaust memory computing the gradient
# of the covariance; asymptotic (z) is the honest fallback and is exactly what
# the statsmodels pipeline was already doing. Record WHICH was used -- a p-value
# whose reference distribution is unknown is not reportable.
#: With a dx stratum the marginal grid is `dx | domain`, so `pairs()` below
#: gives case-minus-control WITHIN each domain -- the circuit-level question --
#: rather than pooling strata or contrasting domains across them.
spec <- if (has_dx) ~ dx | domain else if (has_med) ~ med | domain else ~ domain

df_used <- df_mode
tr <- tryCatch(
  emtrends(m, spec, var = "NRS_within"),
  error = function(e) {
    cat(sprintf("[%s] %s df failed (%s); falling back to asymptotic\n",
                band, df_mode, conditionMessage(e)))
    df_used <<- "asymptotic"
    emm_options(lmer.df = "asymptotic")
    emtrends(m, spec, var = "NRS_within")
  }
)

slopes <- as.data.frame(summary(tr, infer = TRUE))
pw     <- as.data.frame(pairs(tr))

# NO OMNIBUS IS COMPUTED, DELIBERATELY -- do not "helpfully" add one back.
# `joint_tests(m)` was removed at the analyst's instruction. It is also the
# wrong test under the `0 + domain` cell-means coding used here: its
# `domain:NRS_within` row has df1 = (number of domains) and asks whether ALL
# the domain slopes are ZERO, not whether the domains DIFFER, so it is not the
# counterpart of the statsmodels run's Wald test on the interaction block.
# `pairs()` above is kept: pairwise domain contrasts are specific comparisons,
# not an omnibus, and they are nearly free once `tr` exists.

vc <- as.data.frame(VarCorr(m))

outs <- list(list(slopes, "slopes"), list(pw, "pairs"), list(vc, "varcorr"))

# THE MEDICATION SHIFT ON POWER ITSELF (panel F1), not on the pain slope.
# `emmeans(~ med | domain)` at NRS_within = 0 -- the patient's OWN mean pain,
# because NRS_within is subject-mean-centred -- then on-minus-off within each
# domain. NRS_submean and med_submean sit at their grid means, and neither
# interacts with `med`, so they cancel out of the contrast. Uses the df mode
# the slopes settled on, so a Satterthwaite fallback applies here too.
if (has_med) {
  em <- emmeans(m, ~ med | domain, at = list(NRS_within = 0))
  medeff <- as.data.frame(summary(pairs(em, reverse = TRUE), infer = TRUE))
  outs[[length(outs) + 1L]] <- list(medeff, "medeff")
}

for (x in outs) {
  tab <- x[[1]]
  tab$band <- band
  fwrite(tab, file.path(out_dir, sprintf("%s_%s.csv", band, x[[2]])))
}

fwrite(data.frame(
  band = band, fit_seconds = fit_seconds, n_rows = nrow(d),
  n_subjects = nlevels(d$subject), n_roi = nlevels(d$ROI),
  n_channels = nlevels(d$chan_id), df_method = df_used,
  optimizer = optimizer, singular = singular,
  n_warnings = length(warns),
  warnings = paste(warns, collapse = " | "),
  logLik = as.numeric(logLik(m)), REML = TRUE,
  r_version = paste(R.version$major, R.version$minor, sep = "."),
  lme4_version = as.character(packageVersion("lme4")),
  emmeans_version = as.character(packageVersion("emmeans"))
), file.path(out_dir, sprintf("%s_fitinfo.csv", band)))

cat(sprintf("[%s] wrote %s\n", band, out_dir))
