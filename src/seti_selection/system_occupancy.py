from __future__ import annotations

"""Utilities for distinguishing catalogued stellar sources from physical systems.

A catalog source is not always an independent Bernoulli trial: resolved components
of a bound multiple system can appear as separate Gaia entries.  This module keeps
the catalog-source calculation intact (useful for reproducing published star-count
benchmarks) and provides an explicit system-collapsed sensitivity analysis.

For a physical system with member-source completenesses C_s, a single-transmitter
system model additionally needs a conditional probability for which component
hosts the transmitter.  If those component weights are unknown, the attainable
system completeness is bracketed by min(C_s) and max(C_s); equal member weights
provide a transparent benchmark, not an empirical prior.
"""

from collections import defaultdict
from typing import Iterable, Mapping, Sequence

import numpy as np


def _validate_completeness(values) -> np.ndarray:
    c = np.asarray(values, dtype=float)
    if c.ndim != 1 or np.any(~np.isfinite(c)) or np.any((c < 0.0) | (c > 1.0)):
        raise ValueError("completeness values must be a finite 1-D array in [0,1]")
    return c


def build_system_labels(source_ids: Sequence[int], confirmed_bound_pairs: Iterable[tuple[int, int]]) -> dict[int, str]:
    """Return deterministic system labels after unioning confirmed bound pairs."""
    ids = [int(x) for x in source_ids]
    parent = {x: x for x in ids}

    def find(x: int) -> int:
        if x not in parent:
            raise ValueError(f"pair member {x} is absent from source_ids")
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(int(a)), find(int(b))
        if ra != rb:
            # Canonicalize to the smaller source id for reproducible labels.
            lo, hi = min(ra, rb), max(ra, rb)
            parent[hi] = lo

    for a, b in confirmed_bound_pairs:
        union(int(a), int(b))

    groups: dict[int, list[int]] = defaultdict(list)
    for x in ids:
        groups[find(x)].append(x)
    # A second pass handles path compression after unions whose root changed.
    groups = defaultdict(list)
    for x in ids:
        groups[find(x)].append(x)

    labels = {}
    for root, members in groups.items():
        tag = "SYS-" + "-".join(str(x) for x in sorted(members))
        for x in members:
            labels[x] = tag
    return labels


def collapse_source_completeness_to_systems(
    source_ids: Sequence[int],
    source_completenesses,
    system_labels: Mapping[int, str],
    *,
    component_model: str = "equal",
    component_weights: Mapping[int, float] | None = None,
):
    """Collapse catalog-source completenesses to physical-system completenesses.

    Parameters
    ----------
    component_model : {'equal', 'minimum', 'maximum', 'weighted'}
        Conditional model for the component hosting a single system-level
        transmitter. ``minimum`` and ``maximum`` give a prior-free location
        bracket. ``equal`` is a transparent sensitivity benchmark. ``weighted``
        requires ``component_weights`` keyed by source id.

    Returns
    -------
    tuple[list[str], ndarray]
        Sorted physical-system labels and their completenesses.
    """
    ids = [int(x) for x in source_ids]
    c = _validate_completeness(source_completenesses)
    if len(ids) != c.size:
        raise ValueError("source_ids and source_completenesses must have equal length")
    if component_model not in {"equal", "minimum", "maximum", "weighted"}:
        raise ValueError("unknown component_model")

    members: dict[str, list[tuple[int, float]]] = defaultdict(list)
    for sid, ci in zip(ids, c):
        if sid not in system_labels:
            raise ValueError(f"missing system label for source {sid}")
        members[str(system_labels[sid])].append((sid, float(ci)))

    labels = sorted(members)
    out = []
    for label in labels:
        vals = members[label]
        cv = np.asarray([x[1] for x in vals], dtype=float)
        if component_model == "minimum":
            csys = float(np.min(cv))
        elif component_model == "maximum":
            csys = float(np.max(cv))
        elif component_model == "equal":
            csys = float(np.mean(cv))
        else:
            if component_weights is None:
                raise ValueError("component_weights are required for weighted model")
            w = np.asarray([float(component_weights.get(x[0], 0.0)) for x in vals], dtype=float)
            if np.any(~np.isfinite(w)) or np.any(w < 0.0) or float(np.sum(w)) <= 0.0:
                raise ValueError(f"invalid component weights for {label}")
            w /= np.sum(w)
            csys = float(np.sum(w * cv))
        out.append(csys)
    return labels, np.asarray(out, dtype=float)


def angular_separation_arcsec(ra1_deg, dec1_deg, ra2_deg, dec2_deg) -> float:
    """Great-circle angular separation in arcseconds without astropy dependency."""
    ra1, dec1, ra2, dec2 = np.deg2rad([ra1_deg, dec1_deg, ra2_deg, dec2_deg])
    h = np.sin((dec2 - dec1) / 2.0) ** 2 + np.cos(dec1) * np.cos(dec2) * np.sin((ra2 - ra1) / 2.0) ** 2
    return float(2.0 * np.arcsin(np.sqrt(np.clip(h, 0.0, 1.0))) * 206264.80624709636)
