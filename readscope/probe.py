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
    Central differences along ``k`` iid Gaussian directions, combined into
    the standard two-point estimator. Costs ``2 k`` calls per point. Unbiased
    for the gradient, but squaring it inflates the operator, exactly:

        E[ghat ghat^T] = (1 + 1/k) g g^T + (||g||^2 / k) I

    by Isserlis on ``E[(g.u)^2 u u^T] = ||g||^2 I + 2 g g^T``. See
    :func:`debias_sketch`, which inverts this in closed form.

    **The inflation is isotropic, so it cannot rotate an eigenvector.** What
    degrades subspace recovery for this estimator is its variance, not its
    bias, and no amount of debiasing fixes that. C-2b measured the damage: a
    bandwidth of one to two directions where the exact estimator resolves
    all thirty-two.

``lstsq``
    Central differences along ``k`` unit-norm random directions, with the
    gradient recovered by least squares, ``ghat = pinv(U) y``. This is the
    estimator the `geometric-observation` program actually used to produce
    the published accuracy figures, and it is the one to reach for. At
    ``k >= d`` the system is overdetermined and the estimate is exact.

``ortho``
    Central differences along ``k`` **orthonormal** directions, recombined as
    ``ghat = U^T y``, which is exactly the orthogonal projection of ``g`` onto
    the drawn subspace. Same ``2 k`` calls as ``sketch``. It removes the
    magnitude noise of the Gaussian frame, and at ``k = d`` the projector is
    the identity so the estimate is exact. This is the variance fix that
    debiasing is not. Requires ``k <= d``.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from readscope import _xp

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

    sketch_dim: int | None = None
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
    mode: Literal["exact", "sketch", "ortho", "lstsq"] = "exact",
    sketch_dim: int | None = None,
    eps: float = 1e-3,
    rng: np.random.Generator | None = None,
    check_regime: bool = True,
    batched: bool = False,
) -> ProbeResult:
    """Recover ``S = E[g g^T]`` from consumer outputs alone.

    ``batched=True`` treats ``consumer`` as batch-shaped — one call on an
    ``(m, d)`` array returns ``m`` outputs — and reads all of a point's
    finite differences from a single invocation (calibration C-14).
    Batching collapses *invocations*, never *observations*: the same
    directions are probed from the same seed, the recovered operator is
    the serial one to rounding, and the budget cliff at ``k = d`` — a
    statement about directional observations — stands untouched. The
    batched path self-checks row-independence on sampled rows and
    refuses consumers with cross-row coupling.

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
        ``exact`` for coordinate differences, ``sketch`` for an iid Gaussian
        frame, ``ortho`` for an orthonormal frame. Prefer ``ortho`` over
        ``sketch`` at equal cost.
    sketch_dim:
        Number of directions per point when sketching. Must not exceed the
        ambient dimension in ``ortho`` mode.
    eps:
        Finite-difference step.
    rng:
        Generator for the sketch. Required for reproducibility when sketching.
    check_regime:
        Verify the consumer is one this probe applies to before spending a
        budget on it, and raise if it is not. On by default, because a probe
        pointed at a selection consumer returns a confident reading that is
        worthless rather than an error. Costs up to 128 extra consumer calls.
        Turn it off only when the regime is already established.
    """
    xp = _xp.of(points)
    pts = xp.atleast_2d(xp.asarray(points, dtype=xp.float64))
    if pts.ndim != 2:
        raise ValueError("points must have shape (n, d)")
    n, d = int(pts.shape[0]), int(pts.shape[1])
    if n == 0:
        raise ValueError("no operating points supplied")

    verdict = None
    if check_regime:
        from readscope.regimes import applicability

        # A fixed independent stream. Drawing from the caller's rng here
        # would shift the sketch's directions and silently change every
        # result recorded before this guard existed.
        gate_consumer = consumer if not batched else (
            lambda x_: np.asarray(consumer(x_[None, :])).ravel()[0])
        verdict = applicability(
            gate_consumer, pts, eps=eps, rng=np.random.default_rng(20260807)
        )
        if not verdict.probeable:
            raise ValueError(
                f"blind probe does not apply: {verdict.reason}. Pass "
                "check_regime=False to override"
            )

    if batched:
        return _blind_probe_batched(
            consumer, pts, xp, mode=mode, sketch_dim=sketch_dim, eps=eps,
            rng=rng, verdict=verdict)

    if mode == "exact":
        k = d
    elif mode in ("sketch", "ortho", "lstsq"):
        if sketch_dim is None:
            raise ValueError(f"{mode} mode requires sketch_dim")
        if sketch_dim < 1:
            raise ValueError("sketch_dim must be positive")
        k = int(sketch_dim)
        if mode == "ortho" and k > d:
            raise ValueError(
                f"ortho mode needs sketch_dim <= dim, got {k} > {d}"
            )
        if rng is None:
            rng = np.random.default_rng(0)
    else:
        raise ValueError(f"unknown mode {mode!r}")

    S = xp.zeros((d, d), dtype=xp.float64)
    calls = 0

    for x in pts:
        if mode == "exact":
            g = xp.empty(d, dtype=xp.float64)
            for i in range(d):
                e = xp.zeros(d)
                e[i] = 1.0
                g[i] = _central_difference(consumer, x, e, eps)
                calls += 2
        else:
            assert rng is not None
            # draws stay numpy, in the pure-numpy path's exact order, then
            # transfer -- same seed probes the same directions on any backend
            if mode == "ortho":
                # orthonormal rows; U^T U is the projector onto their span,
                # so the recombination below is exactly P_U g
                U = _xp.to_xp(xp, np.linalg.qr(
                    rng.standard_normal((d, k)))[0].T)
            else:
                U_np = rng.standard_normal((k, d))
                if mode == "lstsq":
                    U_np /= np.linalg.norm(U_np, axis=1, keepdims=True) + 1e-12
                U = _xp.to_xp(xp, U_np)

            if mode == "lstsq":
                y = xp.empty(k, dtype=xp.float64)
                for j in range(k):
                    y[j] = _central_difference(consumer, x, U[j], eps)
                    calls += 2
                g = xp.linalg.pinv(U) @ y
            else:
                g = xp.zeros(d, dtype=xp.float64)
                for j in range(k):
                    u = U[j]
                    deriv = _central_difference(consumer, x, u, eps)
                    g += deriv * u
                    calls += 2
                if mode == "sketch":
                    g /= k
        S += xp.outer(g, g)

    S /= n
    S = 0.5 * (S + S.T)

    return ProbeResult(
        S=S,
        n_points=n,
        n_calls=calls,
        mode=mode,
        eps=eps,
        sketch_dim=None if mode == "exact" else k,
        meta={
            "dim": d,
            "regime": None if verdict is None else verdict.to_dict(),
        },
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


def debias_sketch(S: np.ndarray, sketch_dim: int) -> np.ndarray:
    """Remove the isotropic inflation the Gaussian sketch adds.

    The sketch's operator satisfies, in expectation,

        S_hat = (1 + 1/k) S_true + (tr(S_true) / k) I

    Taking traces gives ``tr(S_true) = tr(S_hat) k / (k + d + 1)``, and
    substituting back inverts the map exactly:

        S_true = (k / (k + 1)) [ S_hat - tr(S_hat) / (k + d + 1) I ]

    **This corrects the spectrum and provably not the subspace.** Subtracting
    a multiple of the identity shifts every eigenvalue equally and leaves
    every eigenvector where it was, so a bit allocation computed from the
    debiased spectrum is right where one from the raw spectrum is flattened,
    while subspace recovery is untouched. Use it before
    :func:`~readscope.allocate.water_fill`, and do not expect it to buy
    bandwidth.

    Applies to ``sketch`` mode only. The ``ortho`` estimator has a different
    and smaller isotropic term and is not corrected here.
    """
    A = np.asarray(S, dtype=float)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("S must be square")
    d = A.shape[0]
    k = int(sketch_dim)
    if k < 1:
        raise ValueError("sketch_dim must be positive")
    shift = float(np.trace(A)) / (k + d + 1)
    return (k / (k + 1.0)) * (A - shift * np.eye(d))


def jacobian_probe(
    consumer: Callable[[np.ndarray], np.ndarray],
    points: np.ndarray,
    *,
    n_directions: int,
    eps: float = 1e-3,
    rng: np.random.Generator | None = None,
    batched: bool = False,
) -> ProbeResult:
    """Recover a read operator from a **vector-valued** consumer.

    ``batched=True``: consumer maps ``(m, d)`` to ``(m, out)``; one
    invocation per operating point; observations unchanged (C-14).

    This is the estimator the `geometric-observation` program used for the
    published Llama figures, ported here because the difference from
    :func:`blind_probe` explains most of the accuracy gap between the two.

    A scalar-margin consumer yields one number per probe direction, so the
    read operator has to be built from an outer product of estimated
    gradients. A consumer that returns a **vector**, such as the full
    attention softmax over a key set, yields ``m`` numbers per direction, and
    the object recovered is the Gram of the whole Jacobian:

        M = E_x[ J(x)^T J(x) ],   J(x) = dC/dx,  shape (m, d)

    The Jacobian is recovered per point by least squares from unit-norm
    random directions, ``J^T = pinv(U) dC``, which is exact once
    ``n_directions >= d``.

    **This is not a cheaper probe, it is a richer one.** Each direction costs
    the same two consumer calls and returns ``m`` times as much information.
    Whether that lets the direction budget fall below the ambient dimension
    is a measured question, not an assumption. See C-2e.
    """
    xp = _xp.of(points)
    pts = xp.atleast_2d(xp.asarray(points, dtype=xp.float64))
    if pts.ndim != 2:
        raise ValueError("points must have shape (n, d)")
    n, d = int(pts.shape[0]), int(pts.shape[1])
    if n == 0:
        raise ValueError("no operating points supplied")
    k = int(n_directions)
    if k < 1:
        raise ValueError("n_directions must be positive")
    if rng is None:
        rng = np.random.default_rng(0)

    if batched:
        return _jacobian_probe_batched(consumer, pts, xp, k=k, eps=eps,
                                       rng=rng)

    M = xp.zeros((d, d), dtype=xp.float64)
    calls = 0
    out_dim = None

    for x in pts:
        U_np = rng.standard_normal((k, d))
        U_np /= np.linalg.norm(U_np, axis=1, keepdims=True) + 1e-12
        U = _xp.to_xp(xp, U_np)
        rows = []
        for j in range(k):
            u = U[j]
            hi = xp.asarray(consumer(x + eps * u), dtype=xp.float64).ravel()
            lo = xp.asarray(consumer(x - eps * u), dtype=xp.float64).ravel()
            rows.append((hi - lo) / (2.0 * eps))
            calls += 2
        dC = xp.stack(rows)
        if out_dim is None:
            out_dim = int(dC.shape[1])
        Jt = xp.linalg.pinv(U) @ dC
        M += Jt @ Jt.T

    M /= n
    M = 0.5 * (M + M.T)
    return ProbeResult(
        S=M,
        n_points=n,
        n_calls=calls,
        mode="jacobian",
        eps=eps,
        sketch_dim=k,
        meta={"dim": d, "out_dim": out_dim, "regime": None},
    )


# --------------------------------------------------------------- batched

def _check_row_independence(consumer, X, xp, atol=1e-9):
    """Refuse batch-shaped consumers with cross-row coupling (C-14 B3)."""
    m = int(X.shape[0])
    for i in (0, m // 2):
        full = xp.asarray(consumer(X)).reshape(m, -1)[i]
        single = xp.asarray(consumer(X[i:i + 1])).reshape(1, -1)[0]
        dev = float(abs(full - single).max())
        scale = float(abs(single).max()) + 1e-12
        if dev > atol * max(1.0, scale):
            raise ValueError(
                "batched consumer is not row-independent: row "
                f"{i} differs by {dev:.3e} between batch and single-row "
                "evaluation. Cross-row coupling (shared normalization, "
                "cache state) corrupts every finite difference; fix the "
                "consumer or use the serial probe."
            )


def _blind_probe_batched(consumer, pts, xp, *, mode, sketch_dim, eps,
                         rng, verdict):
    n, d = int(pts.shape[0]), int(pts.shape[1])
    if mode == "exact":
        k = d
    else:
        if sketch_dim is None:
            raise ValueError(f"{mode} mode requires sketch_dim")
        k = int(sketch_dim)
        if mode == "ortho" and k > d:
            raise ValueError("ortho mode needs sketch_dim <= dim")
        if rng is None:
            rng = np.random.default_rng(0)

    S = xp.zeros((d, d), dtype=xp.float64)
    invocations = 0
    checked = False
    for x in pts:
        if mode == "exact":
            U = xp.eye(d)
        elif mode == "ortho":
            U = _xp.to_xp(xp, np.linalg.qr(
                rng.standard_normal((d, k)))[0].T)
        else:
            U_np = rng.standard_normal((k, d))
            if mode == "lstsq":
                U_np /= np.linalg.norm(U_np, axis=1, keepdims=True) + 1e-12
            U = _xp.to_xp(xp, U_np)
        X = xp.concatenate([x[None, :] + eps * U, x[None, :] - eps * U])
        if not checked:
            _check_row_independence(consumer, X, xp)
            checked = True
        y = xp.asarray(consumer(X), dtype=xp.float64).ravel()
        invocations += 1
        diffs = (y[:k] - y[k:]) / (2.0 * eps)
        if mode in ("exact", "lstsq"):
            g = (diffs if mode == "exact"
                 else xp.linalg.pinv(U) @ diffs)
        else:
            g = U.T @ diffs
            if mode == "sketch":
                g /= k
        S += xp.outer(g, g)
    S /= n
    S = 0.5 * (S + S.T)
    return ProbeResult(
        S=S, n_points=n, n_calls=invocations, mode=mode, eps=eps,
        sketch_dim=None if mode == "exact" else k,
        meta={"dim": d, "batched": True,
              "observations": k * n,
              "regime": None if verdict is None else verdict.to_dict()},
    )


def _jacobian_probe_batched(consumer, pts, xp, *, k, eps, rng):
    n, d = int(pts.shape[0]), int(pts.shape[1])
    M = xp.zeros((d, d), dtype=xp.float64)
    invocations = 0
    out_dim = None
    checked = False
    for x in pts:
        U_np = rng.standard_normal((k, d))
        U_np /= np.linalg.norm(U_np, axis=1, keepdims=True) + 1e-12
        U = _xp.to_xp(xp, U_np)
        X = xp.concatenate([x[None, :] + eps * U, x[None, :] - eps * U])
        if not checked:
            _check_row_independence(consumer, X, xp)
            checked = True
        Y = xp.asarray(consumer(X), dtype=xp.float64)
        Y = Y.reshape(2 * k, -1)
        invocations += 1
        dC = (Y[:k] - Y[k:]) / (2.0 * eps)
        if out_dim is None:
            out_dim = int(dC.shape[1])
        Jt = xp.linalg.pinv(U) @ dC
        M += Jt @ Jt.T
    M /= n
    M = 0.5 * (M + M.T)
    return ProbeResult(
        S=M, n_points=n, n_calls=invocations, mode="jacobian", eps=eps,
        sketch_dim=k,
        meta={"dim": d, "out_dim": out_dim, "batched": True,
              "observations": k * n, "regime": None},
    )
