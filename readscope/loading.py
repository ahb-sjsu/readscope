"""Probe loading: the instrument perturbs what it measures.

A scope's input impedance draws current from the node it is attached to, so
what you read is not quite what was there. The effect is not a flaw that
invalidates the instrument. It is characterized, specified on the datasheet,
and corrected for.

This probe has the same error term with a different mechanism. The blind probe
estimates ``E[g g^T]`` over the operating points it is given, and those points
come from a probing distribution that is not the activation distribution the
consumer actually sees in service. The recovered operator is the read operator
averaged over the wrong measure.

The functions here quantify how far the probing distribution sits from the
activation distribution, so that recovery quality can be plotted against it.
That plot is the calibration curve, and it is what turns one measurement into
an accuracy specification.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class LoadingReading:
    """How far a probing distribution sits from an activation distribution."""

    jeffreys: float
    """Symmetrized KL between the fitted Gaussians, in nats."""

    bhattacharyya: float
    """Bhattacharyya distance between the fitted Gaussians."""

    mean_shift: float
    """Mahalanobis distance between the means, in activation metric."""

    spectral_ratio: float
    """Worst-direction variance ratio, ``max(l, 1/l)`` over eigendirections."""

    def to_dict(self) -> dict:
        return {
            "jeffreys": self.jeffreys,
            "bhattacharyya": self.bhattacharyya,
            "mean_shift": self.mean_shift,
            "spectral_ratio": self.spectral_ratio,
        }


def _moments(X: np.ndarray, ridge: float) -> tuple[np.ndarray, np.ndarray]:
    A = np.atleast_2d(np.asarray(X, dtype=float))
    if A.shape[0] < 2:
        raise ValueError("need at least two samples to fit a covariance")
    mu = A.mean(axis=0)
    C = np.cov(A, rowvar=False)
    C = np.atleast_2d(C)
    return mu, C + ridge * np.eye(C.shape[0])


def _quad(v: np.ndarray, M: np.ndarray) -> float:
    """``v^T M v`` as a scalar. A (1, 1) array is not 0-d, so float() on
    the raw product raises under numpy 2."""
    return float((v.T @ M @ v).reshape(()))


def _logdet(C: np.ndarray) -> float:
    sign, val = np.linalg.slogdet(C)
    if sign <= 0:
        raise ValueError("covariance is not positive definite")
    return float(val)


def probe_loading(
    probe_points: np.ndarray,
    activation_points: np.ndarray,
    *,
    ridge: float = 1e-9,
) -> LoadingReading:
    """Measure the divergence between probing and activation distributions.

    Both are summarized by their first two moments, which is the honest level
    of description for a quantity meant to be a single datasheet axis. A
    heavier divergence estimator would be more faithful and much harder to
    read off a curve.
    """
    mu_p, C_p = _moments(probe_points, ridge)
    mu_a, C_a = _moments(activation_points, ridge)
    if mu_p.shape != mu_a.shape:
        raise ValueError("probe and activation dimensions differ")
    d = mu_p.size

    inv_a = np.linalg.inv(C_a)
    inv_p = np.linalg.inv(C_p)
    dmu = (mu_p - mu_a).reshape(-1, 1)

    kl_pa = 0.5 * (
        float(np.trace(inv_a @ C_p))
        + _quad(dmu, inv_a)
        - d
        + _logdet(C_a)
        - _logdet(C_p)
    )
    kl_ap = 0.5 * (
        float(np.trace(inv_p @ C_a))
        + _quad(dmu, inv_p)
        - d
        + _logdet(C_p)
        - _logdet(C_a)
    )

    C_mix = 0.5 * (C_p + C_a)
    bhat = 0.125 * _quad(dmu, np.linalg.inv(C_mix)) + 0.5 * (
        _logdet(C_mix) - 0.5 * (_logdet(C_p) + _logdet(C_a))
    )

    mean_shift = float(np.sqrt(max(_quad(dmu, inv_a), 0.0)))

    ratios = np.linalg.eigvalsh(np.linalg.solve(C_a, C_p))
    ratios = np.clip(ratios.real, 1e-300, None)
    spectral = float(np.max(np.maximum(ratios, 1.0 / ratios)))

    return LoadingReading(
        jeffreys=kl_pa + kl_ap,
        bhattacharyya=bhat,
        mean_shift=mean_shift,
        spectral_ratio=spectral,
    )


def interpolate_distribution(
    probe_points: np.ndarray,
    activation_points: np.ndarray,
    alpha: float,
    *,
    rng: np.random.Generator | None = None,
    n_samples: int | None = None,
) -> np.ndarray:
    """Draw points from a Gaussian ``alpha`` of the way probe to activation.

    ``alpha = 0`` reproduces the probing distribution's moments and
    ``alpha = 1`` the activation distribution's. Sweeping ``alpha`` is how the
    calibration harness generates a loading axis with everything else held
    fixed, so that any change in recovery is attributable to loading alone.

    Covariances are interpolated along the geodesic-free but positive-definite
    path ``C(alpha) = M C_p M`` with ``M`` the matrix power of the whitening
    transform, which keeps every intermediate covariance positive definite.
    """
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must lie in [0, 1]")
    if rng is None:
        rng = np.random.default_rng(0)
    mu_p, C_p = _moments(probe_points, 1e-9)
    mu_a, C_a = _moments(activation_points, 1e-9)

    vals, vecs = np.linalg.eigh(C_p)
    root = (vecs * np.sqrt(np.clip(vals, 1e-300, None))) @ vecs.T
    inv_root = (vecs * (1.0 / np.sqrt(np.clip(vals, 1e-300, None)))) @ vecs.T

    mid = inv_root @ C_a @ inv_root
    mvals, mvecs = np.linalg.eigh(mid)
    mpow = (mvecs * np.power(np.clip(mvals, 1e-300, None), alpha)) @ mvecs.T
    C = root @ mpow @ root
    C = 0.5 * (C + C.T)

    mu = (1.0 - alpha) * mu_p + alpha * mu_a
    n = n_samples if n_samples is not None else len(probe_points)
    return rng.multivariate_normal(mu, C, size=n)
