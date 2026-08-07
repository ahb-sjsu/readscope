"""The response spectrum of a recovered read operator.

A spectrum analyzer decomposes a signal into a basis and shows where the
energy sits. This decomposes a consumer into its eigendirections and shows
where the sensitivity sits. The object differs and the shape does not.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Spectrum:
    """Eigenvalues of a read operator, largest first."""

    eigenvalues: np.ndarray
    eigenvectors: np.ndarray

    @property
    def dim(self) -> int:
        return int(self.eigenvalues.size)

    @property
    def normalized(self) -> np.ndarray:
        """Eigenvalues as a fraction of total sensitivity."""
        total = float(self.eigenvalues.sum())
        if total <= 0.0:
            return np.zeros_like(self.eigenvalues)
        return self.eigenvalues / total

    @property
    def effective_rank(self) -> float:
        """Participation ratio of the spectrum.

        ``(sum lam)^2 / sum lam^2``. One when a single direction carries all
        the sensitivity, ``d`` when the consumer reads isotropically. This is
        the honest answer to "how many directions does this consumer actually
        read", and it is continuous, so it does not need a cutoff chosen after
        looking at the data.
        """
        lam = self.eigenvalues
        s1 = float(lam.sum())
        s2 = float((lam**2).sum())
        if s2 <= 0.0:
            return 0.0
        return s1 * s1 / s2

    def energy_rank(self, fraction: float = 0.9) -> int:
        """Smallest ``r`` whose top-``r`` directions carry ``fraction``."""
        if not 0.0 < fraction <= 1.0:
            raise ValueError("fraction must lie in (0, 1]")
        c = np.cumsum(self.normalized)
        idx = int(np.searchsorted(c, fraction) + 1)
        return min(idx, self.dim)

    def to_dict(self) -> dict:
        return {
            "eigenvalues": [float(v) for v in self.eigenvalues],
            "normalized": [float(v) for v in self.normalized],
            "effective_rank": self.effective_rank,
            "energy_rank_90": self.energy_rank(0.9),
            "dim": self.dim,
        }


def spectrum_of(S: np.ndarray) -> Spectrum:
    """Eigendecompose a read operator into its response spectrum."""
    A = np.asarray(S, dtype=float)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("S must be square")
    A = 0.5 * (A + A.T)
    vals, vecs = np.linalg.eigh(A)
    order = np.argsort(vals)[::-1]
    vals = np.clip(vals[order], 0.0, None)
    vecs = np.ascontiguousarray(vecs[:, order])
    return Spectrum(eigenvalues=vals, eigenvectors=vecs)
