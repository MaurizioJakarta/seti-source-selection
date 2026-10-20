from __future__ import annotations

import numpy as np

from .inference import eta_upper_exact, n_eff


def persistent_hard_completenesses(
    eirp_w: float,
    source_eirp_thresholds_w,
    *,
    pipeline_efficiency: float = 0.94,
):
    """Source-level completeness for a persistent transmitter under a hard threshold.

    Each source is assigned completeness ``pipeline_efficiency`` when its EIRP is
    at or above that source's beam+distance-corrected threshold, and zero otherwise.
    This intentionally excludes temporal, sub-channel and Doppler-smearing effects;
    it is the validation bridge between a geometric source count and the full
    selection-function calculation.
    """
    thresholds = np.asarray(source_eirp_thresholds_w, dtype=float)
    if eirp_w < 0:
        raise ValueError("eirp_w must be non-negative")
    if np.any(thresholds <= 0):
        raise ValueError("source thresholds must be positive")
    if not (0.0 <= pipeline_efficiency <= 1.0):
        raise ValueError("pipeline_efficiency must lie in [0,1]")
    return pipeline_efficiency * (eirp_w >= thresholds).astype(float)


def persistent_hard_summary(
    eirp_w: float,
    source_eirp_thresholds_w,
    *,
    pipeline_efficiency: float = 0.94,
    alpha: float = 0.05,
):
    """Return hard-threshold persistent-signal survey metrics at one EIRP."""
    c = persistent_hard_completenesses(
        eirp_w,
        source_eirp_thresholds_w,
        pipeline_efficiency=pipeline_efficiency,
    )
    return {
        "eirp_w": float(eirp_w),
        "n_detectable": int(np.count_nonzero(c > 0)),
        "n_eff": n_eff(c),
        "eta95_exact": eta_upper_exact(c, alpha=alpha),
    }
