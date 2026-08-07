"""How well a recovered subspace matches a known one, and what chance is.

A reading is meaningless without its noise floor. Every overlap this package
reports is accompanied by the chance overlap for the same shape, because a
0.647 means one thing against a floor of 0.126 and nothing at all against a
floor of 0.6.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class OverlapReading:
    """A subspace overlap together with the floor it must be read against."""

    overlap: float
    chance: float
    rank: int
    dim: int

    @property
    def ratio(self) -> float:
        """Overlap in units of chance. The figure that survives rescaling."""
        return self.overlap / self.chance if self.chance > 0 else float("inf")

    def to_dict(self) -> dict:
        return {
            "overlap": self.overlap,
            "chance": self.chance,
            "ratio_to_chance": self.ratio,
            "rank": self.rank,
            "dim": self.dim,
        }


def _orthonormalize(U: np.ndarray) -> np.ndarray:
    Q, _ = np.linalg.qr(np.asarray(U, dtype=float))
    return Q


def subspace_overlap(U_hat: np.ndarray, U_true: np.ndarray) -> OverlapReading:
    """Mean squared canonical correlation between two subspaces.

    ``overlap = ||U_true^T U_hat||_F^2 / r``. It is one when the subspaces
    coincide and ``r / d`` in expectation for a uniformly random ``r``-frame,
    which is the chance value reported alongside.
    """
    A = _orthonormalize(U_hat)
    B = _orthonormalize(U_true)
    if A.shape[0] != B.shape[0]:
        raise ValueError("subspaces live in different ambient dimensions")
    if A.shape[1] != B.shape[1]:
        raise ValueError("subspaces have different ranks")
    d, r = A.shape
    val = float(np.linalg.norm(B.T @ A, ord="fro") ** 2) / r
    return OverlapReading(overlap=val, chance=r / d, rank=r, dim=d)


def chance_overlap(rank: int, dim: int) -> float:
    """Expected overlap of a uniformly random rank-``r`` frame in ``d`` dims."""
    if rank < 1 or dim < 1 or rank > dim:
        raise ValueError("require 1 <= rank <= dim")
    return rank / dim


def consumer_distortion(P: np.ndarray, Sigma_delta: np.ndarray) -> float:
    """``tr(P Sigma_delta)``, the distortion the consumer actually feels.

    This is the quantity that predicted the worse arm 16/16 on Llama heads
    where reconstruction error, tied to 7.5e-9 by construction, predicted 2/16.
    """
    A = np.asarray(P, dtype=float)
    B = np.asarray(Sigma_delta, dtype=float)
    if A.shape != B.shape:
        raise ValueError("P and Sigma_delta must have the same shape")
    return float(np.trace(A @ B))
