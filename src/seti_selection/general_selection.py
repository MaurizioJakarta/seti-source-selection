from __future__ import annotations

"""General hierarchical selection-function utilities.

This module deliberately separates three objects that are often conflated:

1. ``K_j(theta)``: source-level detection probability for a *specific* physical
   transmitter/system state ``theta`` after marginalizing survey nuisance
   variables (unknown phase, beam orientation/scan law, propagation gain,
   instrumental noise, pipeline stochasticity, repeated epochs, ...).
2. ``Cbar_j(Lambda)``: population-averaged completeness for source ``j`` after
   marginalizing the joint population distribution ``p(theta | Lambda)``.
3. The occupancy likelihood: a model for how often physical sources host the
   transmitter population at all (Bernoulli prevalence or Poisson multiplicity).

The main design rule is that physical correlations belong in the *joint*
population measure.  EIRP, beam width, duty cycle, revisit period, and other
engineering quantities must not be factorized unless the population model
actually asserts independence.
"""

from dataclasses import dataclass
import math
from typing import Sequence

import numpy as np


def _as_completeness(values) -> np.ndarray:
    c = np.asarray(values, dtype=float)
    if c.ndim != 1:
        raise ValueError("source completenesses must be a 1-D array")
    if np.any(~np.isfinite(c)) or np.any((c < 0.0) | (c > 1.0)):
        raise ValueError("source completenesses must be finite and lie in [0,1]")
    return c


def normalize_weights(weights) -> np.ndarray:
    w = np.asarray(weights, dtype=float)
    if w.ndim != 1 or np.any(~np.isfinite(w)) or np.any(w < 0.0):
        raise ValueError("weights must be a finite non-negative 1-D array")
    total = float(np.sum(w))
    if total <= 0.0:
        raise ValueError("weights must have positive sum")
    return w / total


def population_average_from_joint_samples(
    source_completeness_samples,
    sample_weights,
) -> np.ndarray:
    """Marginalize an arbitrary *joint* transmitter population.

    Parameters
    ----------
    source_completeness_samples : array, shape (n_sample, n_source)
        ``K_j(theta_m)`` evaluated at joint population samples ``theta_m``.
        A sample may contain correlated EIRP, beam width, period, duty cycle,
        bandwidth, carrier frequency, drift rate, etc.
    sample_weights : array, shape (n_sample,)
        Quadrature/Monte-Carlo/mixture weights proportional to
        ``p(theta_m | Lambda)``.

    Returns
    -------
    ndarray, shape (n_source,)
        ``Cbar_j(Lambda) = int K_j(theta) p(theta|Lambda) dtheta``.
    """
    K = np.asarray(source_completeness_samples, dtype=float)
    if K.ndim != 2:
        raise ValueError("source_completeness_samples must have shape (sample, source)")
    if np.any(~np.isfinite(K)) or np.any((K < 0.0) | (K > 1.0)):
        raise ValueError("kernel values must be finite and lie in [0,1]")
    w = normalize_weights(sample_weights)
    if len(w) != K.shape[0]:
        raise ValueError("sample_weights length must match sample axis")
    return np.sum(K * w[:, None], axis=0)


def mixture_population_average(component_source_completenesses, mixture_weights) -> np.ndarray:
    """Average source completeness over discrete transmitter sub-populations."""
    return population_average_from_joint_samples(
        component_source_completenesses,
        mixture_weights,
    )


def bernoulli_zero_loglikelihood(eta: float, source_completenesses) -> float:
    """Exact zero-detection log likelihood for Bernoulli source occupancy.

    Conditional on occupancy, source ``j`` is detected with probability ``C_j``.
    With source prevalence ``eta`` and independent physical sources,

        log L(0 | eta, Lambda) = sum_j log(1 - eta C_j).

    ``log1p`` is used for numerical stability.
    """
    eta = float(eta)
    if not 0.0 <= eta <= 1.0:
        raise ValueError("eta must lie in [0,1]")
    c = _as_completeness(source_completenesses)
    if eta == 1.0 and np.any(c == 1.0):
        return -math.inf
    return float(np.sum(np.log1p(-eta * c)))


def bernoulli_zero_probability(eta: float, source_completenesses) -> float:
    logp = bernoulli_zero_loglikelihood(eta, source_completenesses)
    if logp == -math.inf:
        return 0.0
    return float(math.exp(logp))


def eta_upper_bernoulli_exact(
    source_completenesses,
    *,
    alpha: float = 0.05,
    tol: float = 1e-13,
) -> float:
    """Exact upper prevalence satisfying ``P(0|eta)=alpha``.

    Returns 1 when even universal occupancy would leave a zero-detection
    probability above ``alpha``.
    """
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie in (0,1)")
    c = _as_completeness(source_completenesses)
    target = math.log(alpha)
    if bernoulli_zero_loglikelihood(1.0, c) > target:
        return 1.0
    lo, hi = 0.0, 1.0
    for _ in range(256):
        mid = 0.5 * (lo + hi)
        if bernoulli_zero_loglikelihood(mid, c) > target:
            lo = mid
        else:
            hi = mid
        if hi - lo <= tol:
            break
    return 0.5 * (lo + hi)


def poisson_multiplicity_zero_loglikelihood(mu: float, source_completenesses) -> float:
    """Zero-detection log likelihood for Poisson transmitter multiplicity.

    If each physical source hosts ``N_j ~ Poisson(mu)`` independent transmitters,
    with iid marks from the population and average single-transmitter detection
    probability ``C_j``, thinning gives

        P(0 | mu, Lambda) = exp[-mu sum_j C_j].

    In this model the familiar ``-ln(alpha)/N_eff`` expression is exact, not an
    approximation.
    """
    mu = float(mu)
    if mu < 0.0 or not math.isfinite(mu):
        raise ValueError("mu must be finite and non-negative")
    c = _as_completeness(source_completenesses)
    return -mu * float(np.sum(c))


def mu_upper_poisson_multiplicity(source_completenesses, *, alpha: float = 0.05) -> float:
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie in (0,1)")
    c = _as_completeness(source_completenesses)
    neff = float(np.sum(c))
    if neff <= 0.0:
        return math.inf
    return -math.log(alpha) / neff


def bernoulli_loglikelihood_series(
    eta: float,
    source_completenesses,
    *,
    order: int = 2,
) -> float:
    """Power-series diagnostic for the Bernoulli zero-detection likelihood.

    log L = -eta sum C_j - eta^2/2 sum C_j^2 - eta^3/3 sum C_j^3 - ...

    This makes explicit why ``N_eff=sum C_j`` alone is only first-order
    sufficient for Bernoulli prevalence inference.
    """
    eta = float(eta)
    if not 0.0 <= eta <= 1.0:
        raise ValueError("eta must lie in [0,1]")
    if int(order) != order or order < 1:
        raise ValueError("order must be a positive integer")
    c = _as_completeness(source_completenesses)
    out = 0.0
    for k in range(1, int(order) + 1):
        out -= (eta**k / k) * float(np.sum(c**k))
    return float(out)


def eta_upper_series(source_completenesses, *, alpha: float = 0.05, order: int = 2) -> float:
    """Upper prevalence using a truncated log-likelihood series (diagnostic)."""
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie in (0,1)")
    c = _as_completeness(source_completenesses)
    target = math.log(alpha)
    if bernoulli_loglikelihood_series(1.0, c, order=order) > target:
        return 1.0
    lo, hi = 0.0, 1.0
    for _ in range(256):
        mid = 0.5 * (lo + hi)
        if bernoulli_loglikelihood_series(mid, c, order=order) > target:
            lo = mid
        else:
            hi = mid
        if hi - lo < 1e-13:
            break
    return 0.5 * (lo + hi)


@dataclass(frozen=True)
class BernoulliSelectionSummary:
    n_sources: int
    n_eff: float
    sum_c2: float
    eta_upper_exact: float
    eta_upper_poisson_first_order: float
    eta_upper_second_order: float
    eta_upper_third_order: float
    alpha: float


def summarize_bernoulli_selection(source_completenesses, *, alpha: float = 0.05) -> BernoulliSelectionSummary:
    c = _as_completeness(source_completenesses)
    neff = float(np.sum(c))
    poisson = 1.0 if neff <= 0.0 else min(1.0, -math.log(alpha) / neff)
    return BernoulliSelectionSummary(
        n_sources=int(c.size),
        n_eff=neff,
        sum_c2=float(np.sum(c*c)),
        eta_upper_exact=eta_upper_bernoulli_exact(c, alpha=alpha),
        eta_upper_poisson_first_order=poisson,
        eta_upper_second_order=eta_upper_series(c, alpha=alpha, order=2),
        eta_upper_third_order=eta_upper_series(c, alpha=alpha, order=3),
        alpha=float(alpha),
    )


def bernoulli_zero_probability_global_mixture(
    eta: float,
    source_completeness_by_global_state,
    global_state_weights,
) -> float:
    """Marginalize a *shared* survey/calibration nuisance at likelihood level.

    ``source_completeness_by_global_state`` has shape (n_state, n_source).
    State ``m`` represents one realization of a nuisance variable shared by all
    sources (for example a common calibration scale or beam model).  The correct
    marginalized zero-detection probability is

        sum_m w_m prod_j [1 - eta C_j(psi_m)],

    not ``prod_j [1 - eta sum_m w_m C_j(psi_m)]``.  The latter would silently
    redraw the shared nuisance independently for every source.
    """
    eta = float(eta)
    if not 0.0 <= eta <= 1.0:
        raise ValueError("eta must lie in [0,1]")
    C = np.asarray(source_completeness_by_global_state, dtype=float)
    if C.ndim != 2 or C.shape[0] == 0 or C.shape[1] == 0:
        raise ValueError("source_completeness_by_global_state must be a non-empty 2-D array")
    if np.any(~np.isfinite(C)) or np.any((C < 0.0) | (C > 1.0)):
        raise ValueError("completeness values must lie in [0,1]")
    w = normalize_weights(global_state_weights)
    if w.size != C.shape[0]:
        raise ValueError("global_state_weights length must match state axis")
    logp = np.sum(np.log1p(-eta * C), axis=1)
    finite = np.isfinite(logp)
    if not np.any(finite):
        return 0.0
    m = float(np.max(logp[finite]))
    return float(np.exp(m) * np.sum(w[finite] * np.exp(logp[finite] - m)))


def eta_upper_bernoulli_global_mixture(
    source_completeness_by_global_state,
    global_state_weights,
    *,
    alpha: float = 0.05,
    tol: float = 1e-13,
) -> float:
    """Exact Bernoulli upper limit after marginalizing a shared global nuisance."""
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie in (0,1)")
    C = np.asarray(source_completeness_by_global_state, dtype=float)
    w = normalize_weights(global_state_weights)
    if C.ndim != 2 or C.shape[0] != w.size:
        raise ValueError("completeness matrix/state weights mismatch")
    if bernoulli_zero_probability_global_mixture(1.0, C, w) > alpha:
        return 1.0
    lo, hi = 0.0, 1.0
    for _ in range(256):
        mid = 0.5 * (lo + hi)
        if bernoulli_zero_probability_global_mixture(mid, C, w) > alpha:
            lo = mid
        else:
            hi = mid
        if hi - lo <= tol:
            break
    return 0.5 * (lo + hi)
