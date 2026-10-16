#!/usr/bin/env Rscript
#
# One lme4 fit per pain_change cell: the `lme4` engine of pain_change.py.
#
#   Rscript pain_change_lmer.R --in frames.csv --out <dir> --formula "<formula>" \
#       --cores 4 --optimizer bobyqa
#
# The Python side builds every cell frame (same rows, floors and centring as the
# statsmodels engine) and passes the formula from pain_change.MODELS plus the
# random effects, so this script holds no model of its own. `frames.csv` stacks
# the cells, keyed by `cell_id`.
#
# Writes `lmer_cells.csv` (one row per cell: convergence, isSingular, variance
# components with `var_pair` NA when the formula has no `(1 | pair_id)`, warnings,
# error) and `lmer_terms.csv` (cell_id, term, beta, se).
#
# CONVERGED means the optimizer returned code 0 and lme4 raised no "failed to
# converge" or "unable to evaluate" message. Every warning and message of the
# fit is captured, the singular-fit message included, so the Python side sees
# the whole record.

suppressPackageStartupMessages({
  library(data.table)
  library(lme4)
  library(parallel)
})

args <- commandArgs(trailingOnly = TRUE)
arg_of <- function(flag) {
  i <- match(flag, args)
  if (is.na(i) || i == length(args)) stop("missing argument ", flag)
  args[i + 1L]
}

d <- fread(arg_of("--in"))
out_dir <- arg_of("--out")
fml <- as.formula(arg_of("--formula"))
cores <- as.integer(arg_of("--cores"))
optimizer <- arg_of("--optimizer")
d[, subject := factor(subject)]
d[, pair_id := factor(pair_id)]

# NA for a model without the term, so every model writes the same columns.
var_of <- function(vc, group) {
  v <- vc[grp == group, vcov]
  if (length(v)) v else NA_real_
}

fit_one <- function(dc) {
  id <- dc$cell_id[1]
  notes <- character(0)
  t0 <- Sys.time()
  m <- tryCatch(
    withCallingHandlers(
      lmer(fml, data = dc, REML = TRUE,
           control = lmerControl(optimizer = optimizer, optCtrl = list(maxfun = 2e5))),
      warning = function(w) {
        notes <<- c(notes, conditionMessage(w))
        invokeRestart("muffleWarning")
      },
      message = function(msg) {
        notes <<- c(notes, trimws(conditionMessage(msg)))
        invokeRestart("muffleMessage")
      }),
    error = function(e) e)
  secs <- round(as.numeric(difftime(Sys.time(), t0, units = "secs")), 1)
  if (inherits(m, "error")) {
    return(list(cell = data.table(cell_id = id, converged = FALSE, singular = NA,
                                  var_subj_int = NA_real_, var_subj_dpain = NA_real_,
                                  var_pair = NA_real_, var_resid = NA_real_,
                                  warnings = paste(notes, collapse = " | "),
                                  error = conditionMessage(m), fit_seconds = secs),
                terms = NULL))
  }
  notes <- unique(c(notes, m@optinfo$conv$lme4$messages))
  failed <- grepl("failed to converge|unable to evaluate", notes)
  vc <- as.data.table(VarCorr(m))
  co <- summary(m)$coefficients
  list(
    cell = data.table(
      cell_id = id,
      converged = m@optinfo$conv$opt == 0 && !any(failed),
      singular = isSingular(m, tol = 1e-4),
      var_subj_int = vc[var1 == "(Intercept)" & grepl("^subject", grp), vcov],
      var_subj_dpain = vc[var1 == "d_pain" & grepl("^subject", grp), vcov],
      var_pair = var_of(vc, "pair_id"),
      var_resid = vc[grp == "Residual", vcov],
      warnings = paste(notes, collapse = " | "),
      error = "", fit_seconds = secs),
    terms = data.table(cell_id = id, term = rownames(co),
                       beta = co[, "Estimate"], se = co[, "Std. Error"]))
}

fits <- mclapply(split(d, by = "cell_id"), fit_one, mc.cores = cores)
bad <- vapply(fits, inherits, logical(1), "try-error")
if (any(bad)) stop("worker error: ", as.character(fits[[which(bad)[1]]]))
fwrite(rbindlist(lapply(fits, `[[`, "cell")), file.path(out_dir, "lmer_cells.csv"))
fwrite(rbindlist(lapply(fits, `[[`, "terms")), file.path(out_dir, "lmer_terms.csv"))
cat(sprintf("fitted %d cells, %d converged, %d singular\n", length(fits),
            sum(vapply(fits, function(f) isTRUE(f$cell$converged), logical(1))),
            sum(vapply(fits, function(f) isTRUE(f$cell$singular), logical(1)))))
