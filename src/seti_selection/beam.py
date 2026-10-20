"""Primary-beam response utilities for off-axis SETI sensitivity."""
import numpy as np


def gaussian_beam_gain(offset_arcmin, fwhm_arcmin=8.4):
    """Return normalized circular-Gaussian power gain at angular offset.

    ``fwhm_arcmin`` is the full width at half maximum. Therefore the
    half-power radius is FWHM/2 and gain(FWHM/2) = 0.5.
    """
    off = np.asarray(offset_arcmin, dtype=float)
    if np.any(off < 0):
        raise ValueError("offset_arcmin must be non-negative")
    if fwhm_arcmin <= 0:
        raise ValueError("fwhm_arcmin must be positive")
    return np.exp(-4.0 * np.log(2.0) * (off / float(fwhm_arcmin)) ** 2)


def half_power_radius(fwhm_arcmin=8.4):
    """Half-power radius corresponding to a full FWHM beamwidth."""
    if fwhm_arcmin <= 0:
        raise ValueError("fwhm_arcmin must be positive")
    return 0.5 * float(fwhm_arcmin)


def effective_eirp_threshold(distance_pc, offset_arcmin,
                             on_axis_eirp_at_ref_w=1.35e13,
                             ref_distance_pc=100.0,
                             fwhm_arcmin=8.4):
    """EIRP needed at distance/offset for the same received-flux threshold.

    Scales as distance squared and inversely with primary-beam gain.
    """
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
    """Dimensionless EIRP margin above the local off-axis threshold."""
    if eirp_w <= 0:
        raise ValueError("eirp_w must be positive")
    return float(eirp_w) / effective_eirp_threshold(
        distance_pc, offset_arcmin, on_axis_eirp_at_ref_w,
        ref_distance_pc, fwhm_arcmin
    )


def gbt_fwhm_arcmin(frequency_ghz, reference_fwhm_arcmin=8.4, reference_frequency_ghz=1.42):
    """Diffraction-scaled GBT L-band FWHM approximation.

    The 8.4 arcmin value is treated as a reference beamwidth at 1.42 GHz.
    For a fixed illuminated aperture, FWHM scales approximately as lambda,
    hence as 1/frequency.  This isolates beam chromaticity; it is not a
    frequency-resolved calibration of gain, SEFD, or pipeline recovery.
    """
    nu = np.asarray(frequency_ghz, dtype=float)
    if np.any(nu <= 0) or reference_frequency_ghz <= 0 or reference_fwhm_arcmin <= 0:
        raise ValueError("frequencies and reference FWHM must be positive")
    return float(reference_fwhm_arcmin) * float(reference_frequency_ghz) / nu


def gaussian_beam_gain_chromatic(offset_arcmin, frequency_ghz,
                                  reference_fwhm_arcmin=8.4,
                                  reference_frequency_ghz=1.42):
    """Gaussian power gain using the diffraction-scaled frequency-dependent FWHM."""
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
    """Frequency-dependent off-axis EIRP threshold.

    ``on_axis_sensitivity_ratio`` multiplies the reference on-axis threshold
    and can be a scalar or array.  Leaving it at unity deliberately isolates
    beam chromaticity when no frequency-resolved sensitivity calibration is
    available.
    """
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
