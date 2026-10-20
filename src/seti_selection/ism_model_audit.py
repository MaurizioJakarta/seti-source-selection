from __future__ import annotations

import numpy as np


def model_ratio_summary(a, b):
    """Summary of b/a for positive finite arrays."""
    aa = np.asarray(a, dtype=float)
    bb = np.asarray(b, dtype=float)
    if aa.shape != bb.shape:
        raise ValueError("a and b must have the same shape")
    m = np.isfinite(aa) & np.isfinite(bb) & (aa != 0)
    if not np.any(m):
        raise ValueError("no finite non-zero reference values")
    r = bb[m] / aa[m]
    return {
        "n": int(r.size),
        "min": float(np.min(r)),
        "median": float(np.median(r)),
        "mean": float(np.mean(r)),
        "max": float(np.max(r)),
    }


def max_abs_difference(a, b):
    aa = np.asarray(a, dtype=float)
    bb = np.asarray(b, dtype=float)
    if aa.shape != bb.shape:
        raise ValueError("a and b must have the same shape")
    m = np.isfinite(aa) & np.isfinite(bb)
    if not np.any(m):
        raise ValueError("no finite paired values")
    return float(np.max(np.abs(bb[m] - aa[m])))


def weak_across_band(transition_frequency_ghz, low_ghz=1.15):
    """True when the model transition frequency is below the whole band."""
    nu_t = np.asarray(transition_frequency_ghz, dtype=float)
    if low_ghz <= 0:
        raise ValueError("low_ghz must be positive")
    return np.isfinite(nu_t) & (nu_t < low_ghz)


def weak_margin_factor(transition_frequency_ghz, low_ghz=1.15):
    """Frequency margin low_band_edge / nu_transition (>1 means weak at edge)."""
    nu_t = np.asarray(transition_frequency_ghz, dtype=float)
    if np.any(nu_t <= 0) or low_ghz <= 0:
        raise ValueError("frequencies must be positive")
    return low_ghz / nu_t
