from __future__ import annotations
import math
import numpy as np


def n_eff(source_completenesses):
    """Effective fully searched source count."""
    return float(np.sum(np.asarray(source_completenesses, dtype=float)))


def zero_detection_probability(eta: float, source_completenesses):
    c = np.asarray(source_completenesses, dtype=float)
    if eta < 0 or eta > 1:
        raise ValueError("eta must be between 0 and 1")
    terms = 1.0 - eta * c
    if np.any(terms < 0):
        return 0.0
    return float(np.prod(terms))


def eta_upper_exact(source_completenesses, alpha: float = 0.05, tol: float = 1e-12):
    c = np.asarray(source_completenesses, dtype=float)
    if np.any((c < 0) | (c > 1)):
        raise ValueError("completeness values must lie in [0,1]")
    if not (0 < alpha < 1):
        raise ValueError("alpha must lie in (0,1)")
    if zero_detection_probability(1.0, c) > alpha:
        return 1.0
    lo, hi = 0.0, 1.0
    for _ in range(200):
        mid = (lo + hi) / 2.0
        p0 = zero_detection_probability(mid, c)
        if p0 > alpha:
            lo = mid
        else:
            hi = mid
        if hi - lo < tol:
            break
    return (lo + hi) / 2.0


def eta_upper_poisson(source_completenesses, alpha: float = 0.05):
    ne = n_eff(source_completenesses)
    if ne <= 0:
        return 1.0
    return min(1.0, -math.log(alpha) / ne)
