"""Reading diagnostics: step-size response and sampling uncertainty.

Two questions every trusted reading needs answered that the probe
result alone does not carry:

**Is the finite-difference step converged?** ``mode="exact"`` means
exact *directional coverage* — central differences at ``eps`` are
still an order-``eps^2`` approximation to the differential of a
nonlinear consumer, and ``S`` is still a finite-sample average.
:func:`step_response` reruns the probe at ``eps/2``, ``eps`` and
``2*eps`` and reports the operator's movement, which is the
step-size sensitivity a specification has to state (the fp32 /
bfloat16 / curved-consumer regimes especially).

**Do these operating points determine the operator?**
:func:`split_half_overlap` probes two disjoint halves of the
operating points and compares the recovered subspaces. High overlap
says the reading is stable under resampling of points; low overlap
says the points, not the probe, are the bottleneck — the difference
between "the instrument read this accurately" and "these 32 points
do not pin down the population operator".
"""

from __future__ import annotations

import numpy as np

from readscope.metrics import subspace_overlap
from readscope.probe import blind_probe


def _rel_fro(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(b))
    return float(np.linalg.norm(a - b)) / max(denom, 1e-300)


def step_response(
    consumer,
    points: np.ndarray,
    *,
    eps: float = 1e-3,
    probe=blind_probe,
    **probe_kwargs,
) -> dict:
    """Probe at ``eps/2``, ``eps``, ``2*eps``; report operator movement.

    Returns relative Frobenius distances between the three readings and
    a ``converged`` flag: both neighbors within ``tol`` (default 1e-3
    relative) of the central reading. A non-converged flag does not say
    which step is right — it says the reading depends on the step and
    the specification must carry that.
    """
    tol = probe_kwargs.pop("tol", 1e-3)
    readings = {
        scale: probe(
            consumer,
            points,
            eps=scale * eps,
            check_regime=False,
            **probe_kwargs,
        ).S
        for scale in (0.5, 1.0, 2.0)
    }
    half = _rel_fro(readings[0.5], readings[1.0])
    double = _rel_fro(readings[2.0], readings[1.0])
    return {
        "eps": eps,
        "rel_change_half": round(half, 8),
        "rel_change_double": round(double, 8),
        "tol": tol,
        "converged": bool(half <= tol and double <= tol),
    }


def split_half_overlap(
    consumer,
    points: np.ndarray,
    rank: int,
    *,
    rng: np.random.Generator | None = None,
    probe=blind_probe,
    **probe_kwargs,
) -> dict:
    """Probe two disjoint halves of the points; compare read subspaces.

    The returned ``resolution`` is the noise-floor-corrected overlap of
    the two half-sample rank-``rank`` subspaces. Near 1: the reading is
    stable under point resampling at half the sample size. Near 0: the
    operating points do not determine the operator at this rank, and no
    per-point budget fixes that.
    """
    if rng is None:
        rng = np.random.default_rng(0)
    pts = np.atleast_2d(np.asarray(points, dtype=float))
    n = pts.shape[0]
    if n < 4:
        raise ValueError(
            f"split-half needs at least 4 operating points, got {n}"
        )
    perm = rng.permutation(n)
    a, b = pts[perm[: n // 2]], pts[perm[n // 2 : 2 * (n // 2)]]
    ra = probe(consumer, a, check_regime=False, **probe_kwargs)
    rb = probe(consumer, b, check_regime=False, **probe_kwargs)
    ov = subspace_overlap(ra.read_subspace(rank), rb.read_subspace(rank))
    return {
        "rank": rank,
        "n_per_half": int(n // 2),
        "overlap": round(float(ov.overlap), 6),
        "chance": round(float(ov.chance), 6),
        "resolution": round(float(ov.resolution), 6),
    }
