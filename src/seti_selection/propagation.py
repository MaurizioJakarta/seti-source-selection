from __future__ import annotations

import math
import numpy as np


def airmass_plane_parallel(elevation_deg: float) -> float:
    """Low-opacity plane-parallel airmass, sec(z)=1/sin(elevation)."""
    e = float(elevation_deg)
    if not 0.0 < e <= 90.0:
        raise ValueError("elevation_deg must lie in (0, 90]")
    return 1.0 / math.sin(math.radians(e))


def neutral_atmosphere_attenuation_db(elevation_deg: float, *, zenith_db: float = 0.06) -> float:
    """Conservative L-band neutral-atmosphere stress model."""
    if zenith_db < 0:
        raise ValueError("zenith_db must be non-negative")
    return float(zenith_db) * airmass_plane_parallel(elevation_deg)


def ionosphere_midlatitude_upper_db(
    elevation_deg: float,
    frequency_ghz: float,
    *,
    ref_db_at_1ghz_30deg: float = 0.01,
) -> float:
    """ITU-P.531-style upper-bound scaling for mid-latitude absorption."""
    f = float(frequency_ghz)
    if f <= 0:
        raise ValueError("frequency_ghz must be positive")
    if ref_db_at_1ghz_30deg < 0:
        raise ValueError("reference attenuation must be non-negative")
    am = airmass_plane_parallel(elevation_deg)
    am_ref = airmass_plane_parallel(30.0)
    return float(ref_db_at_1ghz_30deg) * (am / am_ref) / (f * f)


def db_to_power_transmission(attenuation_db: float) -> float:
    a = float(attenuation_db)
    if a < 0:
        raise ValueError("attenuation_db must be non-negative")
    return 10.0 ** (-a / 10.0)


def threshold_multiplier_from_loss_db(attenuation_db: float) -> float:
    """Multiply an above-atmosphere EIRP threshold by this factor."""
    a = float(attenuation_db)
    if a < 0:
        raise ValueError("attenuation_db must be non-negative")
    return 10.0 ** (a / 10.0)


def saturated_scintillation_detection_probability(threshold_ratio, *, independent_draws: int = 1):
    """Exploratory strong-scintillation stress test."""
    r = np.asarray(threshold_ratio, dtype=float)
    n = int(independent_draws)
    if n < 1 or n != independent_draws:
        raise ValueError("independent_draws must be a positive integer")
    if np.any(r < 0):
        raise ValueError("threshold_ratio must be non-negative")
    return np.exp(-n * r)
