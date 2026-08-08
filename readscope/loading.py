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
class LoadingCorrection:
    """A fitted map from measured probe loading to expected attenuation.

    A scope specifies input impedance so a reading can be *corrected*, not
    merely doubted. This is the same object: fitted on a calibration family
    where the true answer is known, it says how much a reading is expected to
    be pulled down by a given amount of loading, so a later reading can be
    divided back out.

    The model is multiplicative, ``reading ~ truth * g(loading)``, with ``g``
    interpolated piecewise-linearly in ``log(1 + loading)`` and clamped
    outside the fitted range. **Multiplicativity is an assumption, not a
    result**, and whether it survives a change of consumer is exactly what
    the calibration that fits this has to test. An uncorrected reading is the
    honest default until it does.
    """

    loadings: np.ndarray
    attenuation: np.ndarray
    source: str = ""

    def __post_init__(self):
        order = np.argsort(np.asarray(self.loadings, dtype=float))
        self.loadings = np.asarray(self.loadings, dtype=float)[order]
        self.attenuation = np.clip(
            np.asarray(self.attenuation, dtype=float)[order], 1e-6, 1.0
        )
        # enforce monotone non-increasing; more loading cannot help
        self.attenuation = np.minimum.accumulate(self.attenuation)

    def in_domain(self, loading: float) -> bool:
        """Whether ``loading`` lies inside the fitted range."""
        return bool(
            self.loadings.min() <= float(loading) <= self.loadings.max()
        )

    def expected_attenuation(self, loading: float) -> float:
        """``g(loading)``, the factor a true reading is scaled by.

        Outside the fitted range this clamps to the nearest endpoint, which
        is a guess and not a measurement. Use :meth:`in_domain` to find out.
        """
        x = np.log1p(max(float(loading), 0.0))
        xs = np.log1p(self.loadings)
        return float(np.interp(x, xs, self.attenuation))

    def correct(
        self, reading: float, loading: float, *, strict_domain: bool = True
    ) -> float:
        """Undo the expected attenuation, clipped to a valid resolution.

        **Refuses to extrapolate by default.** A correction evaluated far
        outside its fitted range clamps to the endpoint attenuation and then
        the output clip does the rest, which manufactures a confident answer
        out of nothing. C-7b produced a 92 percent apparent error reduction
        that way, with 202 of 240 corrected values pinned at exactly 1.0 and
        only 15 percent of readings inside the fitted domain, and the sweep
        reported PASS. Extrapolation is not a smaller version of
        interpolation here; it is the failure mode.

        Pass ``strict_domain=False`` to clamp anyway, and then say so
        wherever the number is reported.
        """
        if strict_domain and not self.in_domain(loading):
            raise ValueError(
                f"loading {loading:.4g} is outside the fitted range "
                f"[{self.loadings.min():.4g}, {self.loadings.max():.4g}]; "
                f"this correction cannot be evaluated there. Refit over a "
                f"range that covers it, or pass strict_domain=False and "
                f"report the reading as extrapolated"
            )
        g = self.expected_attenuation(loading)
        return float(np.clip(float(reading) / max(g, 1e-6), -1.0, 1.0))

    def to_dict(self) -> dict:
        return {
            "loadings": [float(v) for v in self.loadings],
            "attenuation": [float(v) for v in self.attenuation],
            "source": self.source,
            "model": "reading ~ truth * g(loading), g piecewise linear in "
            "log1p(loading), clamped outside the fitted range",
        }


def fit_loading_correction(
    loadings, readings, *, truth: float = 1.0, source: str = ""
) -> LoadingCorrection:
    """Fit a correction from a calibration family with a known truth.

    ``readings`` are the resolutions measured at each loading when the true
    resolution is ``truth``, so the attenuation is their ratio.
    """
    ld = np.asarray(loadings, dtype=float).ravel()
    rd = np.asarray(readings, dtype=float).ravel()
    if ld.shape != rd.shape:
        raise ValueError("loadings and readings must align")
    if ld.size < 2:
        raise ValueError("need at least two calibration points")
    if truth <= 0:
        raise ValueError("truth must be positive")
    return LoadingCorrection(
        loadings=ld, attenuation=rd / truth, source=source
    )


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


SAMPLES_PER_DIM_WARN = 5.0


def probe_loading(
    probe_points: np.ndarray,
    activation_points: np.ndarray,
    *,
    ridge: float = 1e-9,
    strict: bool = True,
) -> LoadingReading:
    """Measure the divergence between probing and activation distributions.

    Both are summarized by their first two moments, which is the honest level
    of description for a quantity meant to be a single datasheet axis. A
    heavier divergence estimator would be more faithful and much harder to
    read off a curve.

    **Sample count is not optional here.** A ``d``-dimensional covariance
    fitted from ``n <= d`` points is rank deficient, the ridge dominates its
    log determinant, and the divergence comes back enormous and meaningless:
    16 points in 128 dimensions produced readings around 1e11 nats before
    this guard existed. So ``n <= d`` raises, and ``n < 5 d`` warns.

    **Loading is a property of two distributions, not of the sample you
    happened to probe at.** Estimate it from as many draws of each as you can
    afford, independently of how many operating points the probe visits.

    Set ``strict=False`` to downgrade the hard error to a warning, which is
    only sensible when you already know the covariance is well conditioned.
    """
    import warnings

    P = np.atleast_2d(np.asarray(probe_points, dtype=float))
    A = np.atleast_2d(np.asarray(activation_points, dtype=float))
    d_p, d_a = P.shape[1], A.shape[1]
    for name, arr, dd in (("probe", P, d_p), ("activation", A, d_a)):
        n = arr.shape[0]
        if n <= dd:
            msg = (
                f"{name} sample has {n} points in {dd} dimensions; a "
                f"covariance fitted from n <= d is rank deficient and the "
                f"divergence it produces is meaningless. Supply at least "
                f"{int(SAMPLES_PER_DIM_WARN * dd)} points, or pass "
                f"strict=False if you know better"
            )
            if strict:
                raise ValueError(msg)
            warnings.warn(msg, RuntimeWarning, stacklevel=2)
        elif n < SAMPLES_PER_DIM_WARN * dd:
            warnings.warn(
                f"{name} sample has {n} points in {dd} dimensions, below "
                f"the {SAMPLES_PER_DIM_WARN:g} per dimension this estimator "
                f"wants; the reading may be inflated",
                RuntimeWarning,
                stacklevel=2,
            )

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
