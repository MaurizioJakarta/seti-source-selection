from __future__ import annotations

import math
import numpy as np
from scipy.special import erfc

_C_M_S = 299792458.0
_PC_M = 3.085677581491367e16


def weak_modulation_index(sm_kpc_m20_3, distance_kpc, frequency_ghz):
    """Cordes & Lazio (1991) weak-scattering point-source modulation index.

    Eq. 52 approximation for phi_rms < 1:
        m_I ~ 10^3.3 SM^(1/2) D_kpc^(5/12) nu_GHz^(-17/12).
    """
    sm = np.asarray(sm_kpc_m20_3, dtype=float)
    d = np.asarray(distance_kpc, dtype=float)
    nu = np.asarray(frequency_ghz, dtype=float)
    if np.any(sm < 0) or np.any(d <= 0) or np.any(nu <= 0):
        raise ValueError("SM must be non-negative; distance and frequency positive")
    return 10.0**3.3 * np.sqrt(sm) * d**(5.0/12.0) * nu**(-17.0/12.0)


def fresnel_scale_m(distance_pc, frequency_ghz):
    """First Fresnel scale sqrt(lambda D / 2pi) for a point-source LoS."""
    d = np.asarray(distance_pc, dtype=float)
    nu = np.asarray(frequency_ghz, dtype=float)
    if np.any(d <= 0) or np.any(nu <= 0):
        raise ValueError("distance and frequency must be positive")
    lam = _C_M_S / (nu * 1e9)
    return np.sqrt(lam * d * _PC_M / (2.0 * np.pi))


def weak_scintillation_timescale_s(distance_pc, frequency_ghz, transverse_velocity_kms):
    """Weak-scattering Fresnel crossing time l_F / V_perp."""
    v = float(transverse_velocity_kms)
    if v <= 0:
        raise ValueError("transverse_velocity_kms must be positive")
    return fresnel_scale_m(distance_pc, frequency_ghz) / (v * 1e3)


def lognormal_gain_survival(threshold_ratio, modulation_index):
    """P(G >= r) for unit-mean lognormal gain with rms/mean = m."""
    r = np.asarray(threshold_ratio, dtype=float)
    m = np.asarray(modulation_index, dtype=float)
    if np.any(r < 0) or np.any(m < 0):
        raise ValueError("threshold ratio and modulation index must be non-negative")
    out = np.ones(np.broadcast(r, m).shape, dtype=float)
    rr, mm = np.broadcast_arrays(r, m)
    zero = mm == 0
    out[zero] = (rr[zero] <= 1.0).astype(float)
    nz = ~zero
    if np.any(nz):
        s2 = np.log1p(mm[nz]**2)
        sig = np.sqrt(s2)
        mu = -0.5 * s2
        z = (np.log(np.maximum(rr[nz], np.finfo(float).tiny)) - mu) / sig
        out[nz] = 0.5 * erfc(z / np.sqrt(2.0))
        out[nz & (rr == 0)] = 1.0
    return out


def gaussian_gain_survival(threshold_ratio, modulation_index):
    """First-order weak-scintillation Gaussian intensity approximation."""
    r = np.asarray(threshold_ratio, dtype=float)
    m = np.asarray(modulation_index, dtype=float)
    if np.any(r < 0) or np.any(m < 0):
        raise ValueError("threshold ratio and modulation index must be non-negative")
    rr, mm = np.broadcast_arrays(r, m)
    out = np.empty(rr.shape, dtype=float)
    zero = mm == 0
    out[zero] = (rr[zero] <= 1.0).astype(float)
    nz = ~zero
    out[nz] = 0.5 * erfc((rr[nz] - 1.0) / (mm[nz] * np.sqrt(2.0)))
    return out


def required_pair_survival(single_scan_survival, *, gain_relation="correlated"):
    """Survival of the persistence requirement for the two source scans.

    correlated: the same weak-scintillation gain applies to both scans.
    independent: deliberately conservative decorrelated-gain stress limit.
    """
    p = np.asarray(single_scan_survival, dtype=float)
    if np.any((p < 0) | (p > 1)):
        raise ValueError("survival probability must lie in [0,1]")
    if gain_relation == "correlated":
        return p
    if gain_relation == "independent":
        return p**2
    raise ValueError("gain_relation must be 'correlated' or 'independent'")
