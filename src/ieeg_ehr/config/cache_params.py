"""
Cache storage dtype, and the precision rules that come with it (P0.6).

Settled by the P0.6 dtype audit (`ieeg_ehr/features/dtype_audit.py`, run
2026-07-27). The audit is re-runnable; the reasons are in `DECISIONS.md`.

The cache stores float32. What that costs was measured, not assumed: a full
float64 recompute of one run's PSD, compared against the production float32
path's epoch averages, agreed to 8.1 significant figures — a worst-case
fractional error of 2.5e-07 in LINEAR power, or 0.14 float32 half-ulps. float32
halves the cache versus float64 for an error four orders of magnitude below
anything an effect size could notice.

Storage quantisation error AVERAGES DOWN over an epoch (independent per-window
rounding, ~300 windows), which is why the end-to-end number is *better* than
float32's own ~7.2 digits. Accumulator error does the opposite — it GROWS with
the number of terms. Hence the two dtypes below: cheap to store narrow, but
never compute narrow.
"""

import numpy as np

# ============================================================================
# WHAT THE CACHE STORES
# ============================================================================

# The per-window cache's log-power dtype (P1.1 writes this). Round-trips
# bit-exactly through both Parquet and HDF5 — verified, not inferred, since both
# formats carry IEEE-754 binary32 natively.
CACHE_FLOAT_DTYPE = np.dtype('float32')

# ============================================================================
# WHAT CODE THAT READS THE CACHE MUST DO
# ============================================================================
# These are REQUIREMENTS ON VIEWS, not on the cache. The audit found both while
# answering the storage question; neither is an argument for storing float64.

# Upcast to this before any epoch average / sum / reduction over windows.
# MEASURED: a float32 accumulator over a ~5 min epoch (~300 windows) holds only
# 6.0 significant figures — at or just below the 6-sig-fig bar P0.6 set, and the
# single largest precision loss anywhere in the chain. It is also the cheapest
# possible fix: `arr.astype(np.float64).mean(...)`, or `mean(dtype=np.float64)`.
# numpy does NOT do this for you — for float32 input it accumulates in float32.
CACHE_ACCUMULATE_DTYPE = np.dtype('float64')

# Exponentiating log-power back to linear (view axis 1, `domain`) must also
# happen in float64. MEASURED: the worst stored log-power seen across the
# sampled cohort was -36.8 (a near-dead channel), leaving only ~1.1 decades
# above float32's smallest NORMAL value (~1.18e-38). 10**(-36.8) is
# representable in float32 but nearly subnormal, so any further scaling — a
# baseline division, a band average — can underflow to zero and silently turn a
# quiet channel into an exact zero. float64 has ~270 decades of headroom there.
CACHE_LINEAR_DOMAIN_DTYPE = np.dtype('float64')
