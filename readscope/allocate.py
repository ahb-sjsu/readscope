"""Spend a bit budget against a response spectrum.

Reverse water-filling. Given per-direction source variances ``sigma_i^2`` and
the consumer's sensitivity ``lambda_i`` along the same directions, the
distortion the consumer feels is

    D(b) = sum_i lambda_i sigma_i^2 2^{-2 b_i}

and minimizing it under ``sum_i b_i = B`` gives

    b_i = max(0, 0.5 log2(lambda_i sigma_i^2 / theta))

with ``theta`` set by the budget. Directions whose sensitivity-weighted
variance falls below the water level get no bits at all.

This is the same optimization as power allocation across frequency bins. The
quantity being allocated against is a downstream task's sensitivity rather
than a signal's power, which changes what the numbers mean and not the shape
of the problem.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Allocation:
    """A bit allocation and the distortion it buys."""

    bits: np.ndarray
    water_level: float
    distortion: float
    budget: float
    n_starved: int
    """Directions that received no bits because they sit below the water."""

    def to_dict(self) -> dict:
        return {
            "bits": [float(b) for b in self.bits],
            "water_level": self.water_level,
            "distortion": self.distortion,
            "budget": self.budget,
            "n_starved": self.n_starved,
        }


def _distortion(weights: np.ndarray, bits: np.ndarray) -> float:
    return float(np.sum(weights * np.power(2.0, -2.0 * bits)))


def water_fill(
    sensitivity: np.ndarray,
    variance: np.ndarray | None = None,
    *,
    budget: float,
    max_bits: float | None = None,
    tol: float = 1e-12,
) -> Allocation:
    """Allocate ``budget`` total bits across directions.

    Parameters
    ----------
    sensitivity:
        Eigenvalues of the read operator, one per direction. These are what
        make the allocation consumer-relative rather than signal-relative.
    variance:
        Source variance per direction. Defaults to all ones, which is the
        whitened case.
    budget:
        Total bits to spend, summed over directions.
    max_bits:
        Optional per-direction ceiling, for a fixed-width codec.
    """
    lam = np.asarray(sensitivity, dtype=float).ravel()
    if np.any(lam < 0):
        raise ValueError("sensitivities must be non-negative")
    if variance is None:
        var = np.ones_like(lam)
    else:
        var = np.asarray(variance, dtype=float).ravel()
        if var.shape != lam.shape:
            raise ValueError("variance and sensitivity must align")
        if np.any(var < 0):
            raise ValueError("variances must be non-negative")
    if budget < 0:
        raise ValueError("budget must be non-negative")

    w = lam * var
    live = w > 0
    if not np.any(live) or budget == 0.0:
        bits = np.zeros_like(w)
        return Allocation(
            bits=bits,
            water_level=float(w.max()) if w.size else 0.0,
            distortion=_distortion(w, bits),
            budget=budget,
            n_starved=int(w.size),
        )

    def bits_at(theta: float) -> np.ndarray:
        b = np.zeros_like(w)
        idx = live & (w > theta)
        b[idx] = 0.5 * np.log2(w[idx] / theta)
        if max_bits is not None:
            b = np.minimum(b, max_bits)
        return b

    hi = float(w.max())
    lo = hi
    while bits_at(lo).sum() < budget:
        lo *= 0.5
        if lo < 1e-300:
            break
    for _ in range(400):
        mid = 0.5 * (lo + hi)
        if bits_at(mid).sum() > budget:
            lo = mid
        else:
            hi = mid
        if hi - lo <= tol * max(hi, 1.0):
            break
    theta = 0.5 * (lo + hi)
    bits = bits_at(theta)

    return Allocation(
        bits=bits,
        water_level=theta,
        distortion=_distortion(w, bits),
        budget=budget,
        n_starved=int(np.sum(bits <= 0.0)),
    )


def uniform_allocation(n_dims: int, budget: float) -> np.ndarray:
    """Equal bits everywhere, the baseline water-filling has to beat."""
    if n_dims < 1:
        raise ValueError("n_dims must be positive")
    return np.full(n_dims, budget / n_dims, dtype=float)
