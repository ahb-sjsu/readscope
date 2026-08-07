"""Displacement geometry: what a scale-discarding quotient can still see.

Ported from `turboquant_pro.a2_probe`, which established this and carries the
production version. Ported rather than imported to keep this package
numpy-only, with the debt stated here.

**Why it belongs next to the read operator.** The probe answers "which
directions does the consumer read". This answers a different and
complementary question: "of the variation actually present in the data, how
much survives normalization". A quotient that discards scale is safe exactly
when the consumer's metric is carried by the tangential part of the
displacement. Reading the spectrum without checking that is how a quantizer
scores well on reconstruction and destroys the ranking anyway, which is the
failure the source program was built around.
"""

from __future__ import annotations

import numpy as np


def tangential_fraction(x: np.ndarray, y: np.ndarray) -> float:
    """Share of the displacement ``x - y`` that survives row-normalization.

    ``(|x - y|^2 - (|x| - |y|)^2) / |x - y|^2``, in ``[0, 1]``. Near one, the
    pair differs in direction and an angular quotient can see it. Near zero,
    the pair differs mostly in norm and an angular quotient is blind to it.
    NaN for coincident vectors.
    """
    a = np.asarray(x, dtype=np.float64).ravel()
    b = np.asarray(y, dtype=np.float64).ravel()
    d2 = float(((a - b) ** 2).sum())
    if d2 <= 0.0:
        return float("nan")
    dr = float(np.linalg.norm(a)) - float(np.linalg.norm(b))
    return float(min(max((d2 - dr * dr) / d2, 0.0), 1.0))


def tangential_fractions(
    batch: np.ndarray, n_pairs: int = 2000, seed: int = 0
) -> np.ndarray:
    """Tangential fractions over sampled row pairs of ``batch``."""
    A = np.asarray(batch, dtype=np.float64)
    n = A.shape[0]
    if n < 2:
        return np.empty(0)
    rng = np.random.default_rng(seed)
    i = rng.integers(0, n, size=n_pairs)
    j = rng.integers(0, n, size=n_pairs)
    keep = i != j
    i, j = i[keep], j[keep]
    diff = A[i] - A[j]
    d2 = (diff**2).sum(axis=1)
    dr = np.linalg.norm(A[i], axis=1) - np.linalg.norm(A[j], axis=1)
    ok = d2 > 0
    return np.clip((d2[ok] - dr[ok] ** 2) / d2[ok], 0.0, 1.0)


def displacement_decomposition(
    batch: np.ndarray, n_pairs: int = 2000, seed: int = 0
) -> dict:
    """Summary of the tangential and radial split of pairwise displacement.

    A falling median under drift, meaning norm-dominated variation, is the
    early warning that angular quantization is about to damage ranking.
    """
    frac = tangential_fractions(batch, n_pairs=n_pairs, seed=seed)
    if len(frac) == 0:
        nan = float("nan")
        return {
            "median_tangential_fraction": nan,
            "mean_tangential_fraction": nan,
            "median_radial_fraction": nan,
            "n_pairs": 0,
        }
    med = float(np.median(frac))
    return {
        "median_tangential_fraction": med,
        "mean_tangential_fraction": float(np.mean(frac)),
        "median_radial_fraction": 1.0 - med,
        "n_pairs": int(len(frac)),
    }
