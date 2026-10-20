from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Margot2023InjectionCalibration:
    """Publicly reported narrowband injection calibration used as an audit record.

    The 9400/10000 recovery fraction is a *marginal injected-hit recovery* for
    coherent chirps.  It is not a measured joint A--B--A--B event-acceptance
    probability, not an intermittent-waveform calibration, and not a recovery
    surface at the S/N=10 search threshold.  The published injections were
    adjusted to a recovered S/N of approximately 20.
    """

    injected_count: int = 10_000
    recovered_count: int = 9_400
    target_recovered_snr_approx: float = 20.0
    search_snr_threshold: float = 10.0
    calibrated_joint_persistence: bool = False
    calibrated_intermittent_waveforms: bool = False
    calibrated_broadened_waveforms: bool = False

    @property
    def marginal_hit_recovery(self) -> float:
        return self.recovered_count / self.injected_count


def two_scan_marginal_recovery_benchmark_envelope(
    marginal_recovery: float,
) -> tuple[float, float, float]:
    """Conditional two-scan benchmark envelope from one marginal recovery rate.

    Returns ``(frechet_lower, independent, fully_correlated_upper)`` for the
    intersection of two hypothetical scan-recovery events that each have the
    supplied marginal probability.  These are rigorous Fréchet bounds only for
    *that intersection problem*.  They are **not** bounds on the complete survey
    event-acceptance probability unless the published marginal recovery is known
    to apply to the required scan-level events and all other acceptance stages.
    """
    x = float(marginal_recovery)
    if not 0.0 <= x <= 1.0:
        raise ValueError("marginal_recovery must lie in [0, 1]")
    return max(0.0, 2.0 * x - 1.0), x * x, x
