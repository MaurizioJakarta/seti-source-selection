from __future__ import annotations

import numpy as np

GBT_SCAN_DURATION_S = 150.0


def _active_integral(z, period_s: float, duty_cycle: float):
    """Integral of a unit-amplitude periodic square-wave beacon from 0 to z.

    The beacon is ON on ``[0, qP)`` modulo ``P``.  The implementation also
    works for negative ``z`` through floor division.
    """
    if period_s <= 0:
        raise ValueError("period_s must be positive")
    if not 0.0 <= duty_cycle <= 1.0:
        raise ValueError("duty_cycle must lie in [0, 1]")

    z = np.asarray(z, dtype=float)
    n = np.floor(z / period_s)
    r = z - n * period_s
    return n * duty_cycle * period_s + np.minimum(
        np.maximum(r, 0.0), duty_cycle * period_s
    )


def scan_active_fraction(
    scan_start_s,
    phase_s,
    period_s: float,
    duty_cycle: float,
    scan_duration_s: float = GBT_SCAN_DURATION_S,
):
    """Fraction of one scan overlapping the periodic beacon ON state."""
    if scan_duration_s <= 0:
        raise ValueError("scan_duration_s must be positive")

    scan_start_s = np.asarray(scan_start_s, dtype=float)
    phase_s = np.asarray(phase_s, dtype=float)

    a = _active_integral(
        scan_start_s + scan_duration_s - phase_s,
        period_s,
        duty_cycle,
    )
    b = _active_integral(
        scan_start_s - phase_s,
        period_s,
        duty_cycle,
    )
    return (a - b) / scan_duration_s


def observation_detectable_phase_mask(
    scan_starts_s,
    period_s: float,
    duty_cycle: float,
    required_active_fraction: float,
    *,
    scan_duration_s: float = GBT_SCAN_DURATION_S,
    phase_samples: int = 40000,
    detection_rule: str = "both",
):
    """Return phases for which a two/multi-scan observation is detectable."""
    if phase_samples < 1:
        raise ValueError("phase_samples must be >= 1")
    if required_active_fraction < 0:
        raise ValueError("required_active_fraction must be non-negative")
    if detection_rule not in {"both", "any"}:
        raise ValueError("detection_rule must be 'both' or 'any'")

    starts = np.asarray(scan_starts_s, dtype=float)
    if starts.size == 0:
        return np.zeros(phase_samples, dtype=bool)
    if required_active_fraction > 1.0:
        return np.zeros(phase_samples, dtype=bool)

    phase = np.arange(phase_samples, dtype=float) / phase_samples * period_s
    if detection_rule == "both":
        hit = np.ones(phase_samples, dtype=bool)
        for t in starts:
            frac = scan_active_fraction(
                t, phase, period_s, duty_cycle, scan_duration_s=scan_duration_s,
            )
            hit &= frac + 1e-12 >= required_active_fraction
    else:
        hit = np.zeros(phase_samples, dtype=bool)
        for t in starts:
            frac = scan_active_fraction(
                t, phase, period_s, duty_cycle, scan_duration_s=scan_duration_s,
            )
            hit |= frac + 1e-12 >= required_active_fraction
    return hit


def cadence_detectable_phase_mask(
    on_scan_starts_s,
    period_s: float,
    duty_cycle: float,
    required_active_fraction: float,
    scan_duration_s: float = 300.0,
    phase_samples: int = 40000,
):
    return observation_detectable_phase_mask(
        on_scan_starts_s,
        period_s,
        duty_cycle,
        required_active_fraction,
        scan_duration_s=scan_duration_s,
        phase_samples=phase_samples,
        detection_rule="both",
    )


def source_temporal_completeness(
    observation_scan_starts_s,
    period_s: float,
    duty_cycle: float,
    required_active_fraction: float,
    scan_duration_s: float = GBT_SCAN_DURATION_S,
    phase_samples: int = 40000,
    detection_rule: str = "both",
):
    """Phase-marginalized completeness over one or more observing epochs."""
    covered = np.zeros(phase_samples, dtype=bool)
    for starts in observation_scan_starts_s:
        covered |= observation_detectable_phase_mask(
            starts,
            period_s,
            duty_cycle,
            required_active_fraction,
            scan_duration_s=scan_duration_s,
            phase_samples=phase_samples,
            detection_rule=detection_rule,
        )
    return float(covered.mean())


def source_temporal_completeness_from_mjd(
    observation_scan_mjd,
    period_s: float,
    duty_cycle: float,
    required_active_fraction: float,
    *,
    scan_duration_s: float = GBT_SCAN_DURATION_S,
    phase_samples: int = 40000,
    detection_rule: str = "both",
):
    """As :func:`source_temporal_completeness`, using exact scan-start MJDs."""
    observations = [np.asarray(x, dtype=float) for x in observation_scan_mjd]
    nonempty = [x for x in observations if x.size]
    if not nonempty:
        return 0.0
    base_mjd = min(float(x.min()) for x in nonempty)
    starts_s = [(x - base_mjd) * 86400.0 for x in observations]
    return source_temporal_completeness(
        starts_s,
        period_s,
        duty_cycle,
        required_active_fraction,
        scan_duration_s=scan_duration_s,
        phase_samples=phase_samples,
        detection_rule=detection_rule,
    )


def required_active_fraction_from_eirp(
    eirp_w: float,
    full_scan_threshold_w: float,
) -> float:
    """First-order scan occupancy needed to cross a full-scan power threshold."""
    if eirp_w <= 0:
        return float("inf")
    if full_scan_threshold_w <= 0:
        raise ValueError("full_scan_threshold_w must be positive")
    return float(full_scan_threshold_w / eirp_w)


def pair_pipeline_efficiency_bounds(eta_scan: float) -> tuple[float, float, float]:
    """Legacy alias for the two-scan marginal-recovery benchmark envelope."""
    from .pipeline_calibration import two_scan_marginal_recovery_benchmark_envelope
    return two_scan_marginal_recovery_benchmark_envelope(eta_scan)


def build_template_cadences_from_epoch_mjd(
    epoch_mjd,
    scan_duration_s: float = 300.0,
    transition_overhead_s: float = 30.0,
    on_scan_indices=(0, 2, 4),
):
    """Legacy template helper retained for non-Margot schedules."""
    epoch_mjd = np.asarray(epoch_mjd, dtype=float)
    base = float(epoch_mjd.min())
    anchors = (epoch_mjd - base) * 86400.0
    step = scan_duration_s + transition_overhead_s
    offsets = np.asarray(on_scan_indices, dtype=float) * step
    return [anchor + offsets for anchor in anchors]


def spectral_window_power(epoch_mjd, periods_s):
    """Normalized spectral-window power of cadence start times."""
    t = (np.asarray(epoch_mjd, dtype=float) - np.min(epoch_mjd)) * 86400.0
    periods_s = np.asarray(periods_s, dtype=float)
    f = 1.0 / periods_s
    z = np.exp(-2j * np.pi * f[:, None] * t[None, :]).mean(axis=1)
    return np.abs(z) ** 2
