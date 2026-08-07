"""The blind probe: recover a consumer's read operator from its outputs.

The read operator is ``P_C = J^T G J`` and ``J`` is a derivative, so it is
measurable. Perturb the input, watch the output move, accumulate the outer
product of the estimated gradients:

    S = E[ g g^T ],   g = grad_x C(x)

The top eigenvectors of ``S`` span the read subspace. No labels enter and no
oracle direction enters. The probe sees ``C(.)`` and nothing else.

Two estimators are provided and the choice is recorded in the result, because
it sets the instrument's sample rate and its noise floor.

``exact``
    Central differences along every coordinate. Costs ``2 d`` consumer calls
    per operating point and is exact to order ``eps^2``.

``sketch``
    Central differences along ``k`` random Gaussian directions, combined into
    the standard two-point estimator. Costs ``2 k`` calls per point and is
    unbiased for ``S`` only in expectation over the sketch, so it carries
    estimator variance that ``exact`` does not. This is the estimator that
    makes the probe affordable on a frontier model, and its variance is part
    of what the calibration sweep has to characterize.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Literal, Optional

import numpy as np

Consumer = Callable[[np.ndarray], np.ndarray]


@dataclass
class ProbeResult:
    """A recovered read operator and everything needed to judge it."""

    S: np.ndarray
    """The recovered sensitivity operator, ``d x d``, symmetric PSD."""

    n_points: int
    """Operating points probed."""

    n_calls: int
    """Consumer evaluations spent. The instrument's sample-rate axis."""

    mode: str
    """``exact`` or ``sketch``."""

    eps: float
    """Finite-difference step actually used."""

    sketch_dim: Optional[int] = None
    """``k`` for the sketch estimator, ``None`` for exact."""

    meta: dict = field(default_factory=dict)

    @property
    def dim(self) -> int:
        return int(self.S.shape[0])

    def read_subspace(self, rank: int) -> np.ndarray:
        """Top-``rank`` eigenvectors of ``S``, as columns."""
        if rank < 1 or rank > self.dim:
            raise ValueError(f"rank {rank} outside 1..{self.dim}")
        _, vecs = np.linalg.eigh(self.S)
        return np.ascontiguousarray(vecs[:, ::-1][:, :rank])


def _central_difference(
    consumer: Consumer, x: np.ndarray, direction: np.ndarray, eps: float
) -> float:
    """Directional derivative of a scalar-margin consumer."""
    plus = float(np.asarray(consumer(x + eps * direction)).reshape(()))
    minus = float(np.asarray(consumer(x - eps * direction)).reshape(()))
    return (plus - minus) / (2.0 * eps)


def blind_probe(
    consumer: Consumer,
    points: np.ndarray,
    *,
    mode: Literal["exact", "sketch"] = "exact",
    sketch_dim: Optional[int] = None,
    eps: float = 1e-3,
    rng: Optional[np.random.Generator] = None,
) -> ProbeResult:
    """Recover ``S = E[g g^T]`` from consumer outputs alone.

    Parameters
    ----------
    consumer:
        Callable mapping a point of shape ``(d,)`` to a scalar margin. This is
        the only access the probe has. It is never handed a label, a Jacobian,
        or a hint about which directions matter.
    points:
        Operating points, shape ``(n, d)``. The probe is only as
        representative as these are, which is the probe-loading question that
        :mod:`readscope.loading` exists to quantify.
    mode:
        ``exact`` for coordinate differences, ``sketch`` for a random sketch.
    sketch_dim:
        Number of random directions per point when ``mode='sketch'``.
    eps:
        Finite-difference step.
    rng:
        Generator for the sketch. Required for reproducibility when sketching.
    """
    pts = np.atleast_2d(np.asarray(points, dtype=float))
    if pts.ndim != 2:
        raise ValueError("points must have shape (n, d)")
    n, d = pts.shape
    if n == 0:
        raise ValueError("no operating points supplied")

    if mode == "exact":
        k = d
    elif mode == "sketch":
        if sketch_dim is None:
            raise ValueError("sketch mode requires sketch_dim")
        if sketch_dim < 1:
            raise ValueError("sketch_dim must be positive")
        k = int(sketch_dim)
        if rng is None:
            rng = np.random.default_rng(0)
    else:
        raise ValueError(f"unknown mode {mode!r}")

    S = np.zeros((d, d), dtype=float)
    calls = 0

    for x in pts:
        if mode == "exact":
            g = np.empty(d, dtype=float)
            for i in range(d):
                e = np.zeros(d)
                e[i] = 1.0
                g[i] = _central_difference(consumer, x, e, eps)
                calls += 2
        else:
            assert rng is not None
            U = rng.standard_normal((k, d))
            g = np.zeros(d, dtype=float)
            for u in U:
                deriv = _central_difference(consumer, x, u, eps)
                g += deriv * u
                calls += 2
            g /= k
        S += np.outer(g, g)

    S /= n
    S = 0.5 * (S + S.T)

    return ProbeResult(
        S=S,
        n_points=n,
        n_calls=calls,
        mode=mode,
        eps=eps,
        sketch_dim=None if mode == "exact" else k,
        meta={"dim": d},
    )


def retrieval_margin_gradient(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Closed-form margin gradient for a cosine-ranking consumer.

    ``g = a_hat - cos(a, b) * b_hat``. Included because this consumer's
    gradient is available in closed form, which makes it the cheapest ground
    truth the calibration harness has: the probe can be graded against it
    without a planted subspace.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0.0 or nb == 0.0:
        raise ValueError("cosine margin undefined for a zero vector")
    ahat, bhat = a / na, b / nb
    return ahat - float(ahat @ bhat) * bhat
