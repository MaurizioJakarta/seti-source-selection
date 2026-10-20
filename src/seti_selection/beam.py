"""Primary-beam response utilities for off-axis SETI sensitivity."""
import numpy as np


def gaussian_beam_gain(offset_arcmin, fwhm_arcmin=8.4):
    off = np.asarray(offset_arcmin, dtype=float)
    if np.any(off < 0):
        raise ValueError("offset_arcmin must be non-negative")
    if fwhm_arcmin <= 0:
        raise ValueError("fwhm_arcmin must be positive")
    return np.exp(-4.0 * np.log(2.0) * (off / float(fwhm_arcmin)) ** 2)


def half_power_radius(fwhm_arcmin=8.4):
    if fwhm_arcmin <= 0:
        raise ValueError("fwhm_arcmin must be positive")
    return 0.5 * float(fwhm_arcmin)


def effective_eirp_threshold(distance_pc, offset_arcmin,
                             on_axis_eirp_at_ref_w=1.35e13,
                             ref_distance_pc=100.0,
                             fwhm_arcmin=8.4):
    d = np.asarray(distance_pc, dtype=float)
    if np.any(d <= 0):
        raise ValueError("distance_pc must be positive")
    if ref_distance_pc <= 0 or on_axis_eirp_at_ref_w <= 0:
        raise ValueError("reference distance and EIRP must be positive")
    g = gaussian_beam_gain(offset_arcmin, fwhm_arcmin=fwhm_arcmin)
    return float(on_axis_eirp_at_ref_w) * (d / float(ref_distance_pc)) ** 2 / g


def eirp_margin(eirp_w, distance_pc, offset_arcmin,
                on_axis_eirp_at_ref_w=1.35e13,
                ref_distance_pc=100.0,
                fwhm_arcmin=8.4):
    if eirp_w <= 0:
        raise ValueError("eirp_w must be positive")
    return float(eirp_w) / effective_eirp_threshold(
        distance_pc, offset_arcmin, on_axis_eirp_at_ref_w,
        ref_distance_pc, fwhm_arcmin
    )


def gbt_fwhm_arcmin(frequency_ghz, reference_fwhm_arcmin=8.4, reference_frequency_ghz=1.42):
    """Diffraction-scaled GBT L-band FWHM approximation."""
    nu = np.asarray(frequency_ghz, dtype=float)
    if np.any(nu <= 0) or reference_frequency_ghz <= 0 or reference_fwhm_arcmin <= 0:
        raise ValueError("frequencies and reference FWHM must be positive")
    return float(reference_fwhm_arcmin) * float(reference_frequency_ghz) / nu


def gaussian_beam_gain_chromatic(offset_arcmin, frequency_ghz,
                                  reference_fwhm_arcmin=8.4,
                                  reference_frequency_ghz=1.42):
    fwhm = gbt_fwhm_arcmin(
        frequency_ghz, reference_fwhm_arcmin=reference_fwhm_arcmin,
        reference_frequency_ghz=reference_frequency_ghz,
    )
    off = np.asarray(offset_arcmin, dtype=float)
    if np.any(off < 0):
        raise ValueError("offset_arcmin must be non-negative")
    return np.exp(-4.0 * np.log(2.0) * (off / fwhm) ** 2)


def effective_eirp_threshold_chromatic(distance_pc, offset_arcmin, frequency_ghz,
                                        on_axis_eirp_at_ref_w=1.35e13,
                                        ref_distance_pc=100.0,
                                        reference_fwhm_arcmin=8.4,
                                        reference_frequency_ghz=1.42,
                                        on_axis_sensitivity_ratio=1.0):
    d = np.asarray(distance_pc, dtype=float)
    if np.any(d <= 0) or ref_distance_pc <= 0 or on_axis_eirp_at_ref_w <= 0:
        raise ValueError("distance and reference threshold must be positive")
    sr = np.asarray(on_axis_sensitivity_ratio, dtype=float)
    if np.any(~np.isfinite(sr)) or np.any(sr <= 0):
        raise ValueError("on_axis_sensitivity_ratio must be finite and positive")
    g = gaussian_beam_gain_chromatic(
        offset_arcmin, frequency_ghz, reference_fwhm_arcmin=reference_fwhm_arcmin,
        reference_frequency_ghz=reference_frequency_ghz,
    )
    return float(on_axis_eirp_at_ref_w) * sr * (d / float(ref_distance_pc)) ** 2 / g
