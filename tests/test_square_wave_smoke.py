"""Quick synthetic sanity test for the new detectors + threshold split. No NWB."""
import numpy as np
from ieeg_ehr import config
from ieeg_ehr.qc.detect_square_wave import classify_square_wave
from ieeg_ehr.qc import build_exclusions
import pandas as pd

sfreq = 1000.0
n = int(60 * sfreq)  # 60s
t = np.arange(n) / sfreq

def report(name, trace_v, artifact='square_wave'):
    res = classify_square_wave(trace_v, sfreq)
    df = pd.DataFrame({'metric_value': res['metric_value'], 'range': res['range']})
    excl = build_exclusions.compute_excluded('square_wave', df, build_exclusions.default_params('square_wave'))
    print(f"{name}: mean_frac={res['metric_value'].mean():.3f} "
          f"mean_range_uV={res['range'].mean()*1e6:.2f} excluded_windows={int(excl.sum())}/{len(excl)}")

# fast square wave: 2.3s period, 0..50uV
sq_fast = ((np.floor(t / 1.15) % 2) * 50e-6)
report("fast square (0-50uV, ~2.3s period)", sq_fast)

# slow square wave: 8s period, 0..50uV
sq_slow = ((np.floor(t / 4.0) % 2) * 50e-6)
report("slow square (0-50uV, 8s period)", sq_slow)

# clean neural-ish noise, 50uV std
rng = np.random.default_rng(0)
clean = rng.standard_normal(n) * 50e-6
report("clean noise (50uV std)", clean)

# flat (constant) -> guard must prevent square-wave flag
flat = np.full(n, 20e-6)
report("flat constant (20uV)", flat)

print("SQUARE_MIN_RANGE_V =", config.SQUARE_MIN_RANGE_V, "V")
print("ARTIFACT_TYPES =", config.ARTIFACT_TYPES)
print("git_provenance =", config.git_provenance()['commit'], "dirty=", config.git_provenance()['dirty'])
