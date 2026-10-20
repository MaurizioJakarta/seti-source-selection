from __future__ import annotations
import numpy as np


def sinc2(x):
    """Normalized sinc² channel response."""
    return np.sinc(x) ** 2


def spectral_efficiency(
    subbin_offset,
    drift_hz_s: float,
    fine_channel_hz: float = 2.79,
    time_resolution_s: float = 18.25,
    integration_samples: int = 801,
):
    """
    Fraction of narrowband signal power recovered in the tracked fine channel
    during one time integration.

    Parameters
    ----------
    subbin_offset:
        Frequency offset from the fine-channel center, in channel widths.
        Typically in [-0.5, 0.5).
    drift_hz_s:
        Absolute/ signed linear Doppler drift.
    fine_channel_hz, time_resolution_s:
        Spectrometer resolution.
    integration_samples:
        Numerical samples used to average sinc² over intraintegration drift.

    Notes
    -----
    For large drift, this approaches approximately:
        fine_channel_hz / (|drift| * time_resolution_s)
    """
    eps = np.asarray(subbin_offset, dtype=float)
    span_channels = abs(drift_hz_s) * time_resolution_s / fine_channel_hz

    if span_channels < 1e-14:
        return sinc2(eps)

    u = np.linspace(
        -span_channels / 2.0,
        +span_channels / 2.0,
        integration_samples,
    )
    return sinc2(eps[..., None] - u).mean(axis=-1)


def median_spectral_efficiency(
    drift_hz_s: float,
    fine_channel_hz: float = 2.79,
    time_resolution_s: float = 18.25,
    subbin_samples: int = 20001,
):
    """Median over an unknown uniform fine-channel sub-bin phase."""
    eps = np.linspace(-0.5, 0.5, subbin_samples, endpoint=False)
    eta = spectral_efficiency(
        eps,
        drift_hz_s,
        fine_channel_hz=fine_channel_hz,
        time_resolution_s=time_resolution_s,
    )
    return float(np.median(eta))


def required_scan_active_fraction(
    eirp_over_nominal: float,
    drift_hz_s: float,
    kappa: float = 3.25,
    fine_channel_hz: float = 2.79,
    time_resolution_s: float = 18.25,
    subbin_samples: int = 20001,
    exactly_zero_drift_special: bool = False,
    zero_drift_extra_loss: float = 2.0,
):
    """
    Benchmark phase-averaged fraction of an ON scan that must contain signal.

    This uses the median fine-channel efficiency. A production injection-
    calibrated pipeline should replace this scalar threshold by C_pipe().
    """
    eta = median_spectral_efficiency(
        drift_hz_s,
        fine_channel_hz=fine_channel_hz,
        time_resolution_s=time_resolution_s,
        subbin_samples=subbin_samples,
    )
    effective_kappa = kappa
    if exactly_zero_drift_special and abs(drift_hz_s) < 1e-15:
        effective_kappa *= zero_drift_extra_loss

    denom = eirp_over_nominal * eta
    if denom <= 0:
        return float("inf")
    return effective_kappa / denom
