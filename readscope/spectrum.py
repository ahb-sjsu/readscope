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


def spectrum_of(S) -> Spectrum:
    """Eigendecompose a read operator into its response spectrum.

    Accepts numpy or CuPy operators; the decomposition runs where the
    data lives and the returned arrays share its namespace.
    """
    from readscope import _xp

    xp = _xp.of(S)
    A = xp.asarray(S, dtype=xp.float64)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("S must be square")
    A = 0.5 * (A + A.T)
    vals, vecs = xp.linalg.eigh(A)
    order = xp.argsort(vals)[::-1]
    vals = xp.clip(vals[order], 0.0, None)
    vecs = xp.ascontiguousarray(vecs[:, order])
    return Spectrum(eigenvalues=vals, eigenvectors=vecs)


@dataclass
class TopSpectrum:
    """The leading ``r`` directions of a read operator, plus the exact
    whole-spectrum aggregates that do not need a decomposition.

    ``effective_rank`` here is exact — ``trace^2 / ||S||_F^2`` needs no
    eigenvalues — while ``energy_rank`` is answerable only as far as the
    computed directions reach, and raises rather than guessing beyond
    them. The honest split: aggregates about the whole spectrum come
    from invariants; statements about individual directions come only
    from directions actually computed.
    """

    eigenvalues: np.ndarray  # (r,), largest first
    eigenvectors: np.ndarray  # (d, r)
    trace: float  # exact sum of ALL eigenvalues
    fro_sq: float  # exact sum of ALL squared eigenvalues

    @property
    def dim(self) -> int:
        return int(self.eigenvectors.shape[0])

    @property
    def r(self) -> int:
        return int(self.eigenvalues.size)

    @property
    def coverage(self) -> float:
        """Fraction of total sensitivity carried by the computed top-r."""
        if self.trace <= 0.0:
            return 0.0
        return float(self.eigenvalues.sum()) / self.trace

    @property
    def effective_rank(self) -> float:
        """Exact participation ratio, from trace and Frobenius norm."""
        if self.fro_sq <= 0.0:
            return 0.0
        return self.trace * self.trace / self.fro_sq

    def energy_rank(self, fraction: float = 0.9) -> int:
        if not 0.0 < fraction <= 1.0:
            raise ValueError("fraction must lie in (0, 1]")
        if fraction > self.coverage:
            raise ValueError(
                f"top-{self.r} directions carry only "
                f"{self.coverage:.3f} of the sensitivity; energy_rank"
                f"({fraction}) needs more directions — recompute with a "
                f"larger r rather than extrapolating"
            )
        lam = self.eigenvalues
        c = (
            np.cumsum(
                np.asarray(lam) if not hasattr(lam, "get") else lam.get()
            )
            / self.trace
        )
        return int(np.searchsorted(c, fraction) + 1)


def top_spectrum(
    S, r: int, *, iters: int = 40, oversample: int = 8, seed: int = 0
) -> TopSpectrum:
    """Leading-``r`` directions by block subspace iteration — no full
    eigendecomposition, no dependency beyond numpy, GPU-generic.

    At large ``d`` (thousands), a full ``eigh`` is the dominant cost of
    reading a spectrum; subspace iteration on a PSD operator converges
    geometrically in the spectral gaps and costs ``iters`` matmuls of
    ``(d, r+oversample)``. **This accelerates the linear algebra only:
    the probe's consumer-call budget law (the cliff at ``k = d``) is a
    property of observation, not of FLOPs, and no solver moves it.**
    """
    from readscope import _xp

    xp = _xp.of(S)
    A = xp.asarray(S, dtype=xp.float64)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("S must be square")
    d = int(A.shape[0])
    if not 1 <= r <= d:
        raise ValueError("need 1 <= r <= dim")
    A = 0.5 * (A + A.T)
    b = min(d, r + int(oversample))
    # draw in numpy for cross-backend reproducibility, then transfer
    Q = _xp.to_xp(xp, np.random.default_rng(seed).standard_normal((d, b)))
    Q = xp.linalg.qr(Q)[0]
    for _ in range(int(iters)):
        Q = xp.linalg.qr(A @ Q)[0]
    T = Q.T @ A @ Q
    vals, vecs = xp.linalg.eigh(0.5 * (T + T.T))
    order = xp.argsort(vals)[::-1][:r]
    vals = xp.clip(vals[order], 0.0, None)
    vecs = xp.ascontiguousarray((Q @ vecs)[:, order])
    return TopSpectrum(
        eigenvalues=vals,
        eigenvectors=vecs,
        trace=float(xp.trace(A)),
        fro_sq=float((A * A).sum()),
    )
