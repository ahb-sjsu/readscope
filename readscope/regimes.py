"""Consumer regimes, and where the blind probe does not apply.

Inherited from `turboquant_pro.operator_sensitivity` and
`turboquant_pro.operator_trace`, which established this before readscope
existed. Ported rather than imported so this package stays numpy-only, with
the debt stated here.

**Why an instrument needs this.** The blind probe assumes the consumer is a
differentiable scalar margin, so that ``grad_x C(x)`` exists and its outer
product means something. Two consumer regimes that matter in practice break
that assumption, and a probe pointed at either returns a confident reading
that is worthless. A scope that reads 0 V on a circuit it cannot couple to is
not a working scope, and neither is this.

``SCALAR_MARGIN``
    A logit, a ranking score, an attention weight. The probe applies. This is
    the only regime the calibrations in `SPEC.md` cover.

``SELECTION``
    Top-k routers and argmax gates. The consumer reads the *order* of its
    logits, not their values, so it is invariant to a common-mode shift and
    its derivative is zero almost everywhere and undefined on the decision
    boundary. Finite differencing returns zero, which the probe would report
    as a consumer that reads nothing.

    The right sensitivity here is the **routing margin**, the gap between the
    k-th and the (k+1)-th logit, and the right error statistic is the
    **differential fraction**, the share of a perturbation that is not
    common-mode. Both are provided below.

``RECURRENCE``
    Per-channel linear recurrences, ``h_t = a h_{t-1} + b_t``. A pointwise
    Jacobian is well defined but misleading, because error in the decay
    compounds over the sequence: sensitivity of the accumulated state goes as
    ``1 / (1 - a)^2``, so slow channels over long sequences dominate and a
    single-step probe sees none of it.

Use :func:`applicability` before trusting a reading, and
:func:`assert_probeable` when you would rather fail than be misled.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np


class Regime(str, Enum):
    SCALAR_MARGIN = "scalar_margin"
    SELECTION = "selection"
    RECURRENCE = "recurrence"
    UNKNOWN = "unknown"


@dataclass
class Applicability:
    """Whether the blind probe means anything for this consumer."""

    regime: Regime
    probeable: bool
    reason: str
    evidence: dict

    def to_dict(self) -> dict:
        return {
            "regime": self.regime.value,
            "probeable": self.probeable,
            "reason": self.reason,
            "evidence": self.evidence,
        }


# --------------------------------------------------------------- selection


def routing_margins(logits: np.ndarray, k: int = 1) -> np.ndarray:
    """Per-token top-k routing margin, the gap between rank k and k+1.

    A small margin means the selection is fragile: any perturbation larger
    than it flips the routing.
    """
    x = np.asarray(logits, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError("logits must be (n_tokens, n_experts)")
    if not 1 <= k < x.shape[1]:
        raise ValueError("k must satisfy 1 <= k < n_experts")
    part = np.sort(x, axis=1)[:, ::-1]
    return part[:, k - 1] - part[:, k]


def differential_fraction(delta_logits: np.ndarray) -> np.ndarray:
    """Share of a logit perturbation that can affect selection.

    Selection is invariant to a common-mode shift, so only the component
    orthogonal to the all-ones direction matters. Near zero means the error
    is common-mode and routing-safe; near one means it is fully differential.
    """
    d = np.asarray(delta_logits, dtype=np.float64)
    if d.ndim != 2:
        raise ValueError("delta_logits must be (n_tokens, n_experts)")
    total = (d**2).sum(axis=1)
    centered = d - d.mean(axis=1, keepdims=True)
    diff = (centered**2).sum(axis=1)
    out = np.divide(
        diff, total, out=np.full_like(total, np.nan), where=total > 0
    )
    return np.clip(out, 0.0, 1.0)


# -------------------------------------------------------------- recurrence


def decay_sensitivity(
    decay: np.ndarray, seq_len: int | None = None
) -> np.ndarray:
    """Sensitivity of an accumulated state to an error in its decay.

    At steady state this is ``1 / (1 - a)^2``. It grows sharply as ``a``
    approaches one and with sequence length, which is why slow channels over
    long sequences are the fragile ones and why a single-step Jacobian
    understates them.
    """
    a = np.clip(np.asarray(decay, dtype=np.float64), 0.0, 1.0 - 1e-9)
    if seq_len is None:
        return 1.0 / (1.0 - a) ** 2
    T = int(seq_len)
    one_minus = 1.0 - a
    return (1.0 - a**T * (1.0 + T * one_minus)) / one_minus**2


# ------------------------------------------------------------ applicability


def applicability(
    consumer,
    points: np.ndarray,
    *,
    eps: float = 1e-3,
    flat_fraction: float = 0.9,
    min_trials: int = 16,
    rng: np.random.Generator | None = None,
) -> Applicability:
    """Decide empirically whether the blind probe applies to this consumer.

    The test is the one that catches the regime the probe silently fails on.
    Perturb along random directions and count how often the consumer's output
    does not move at all. A selection consumer is piecewise constant, so a
    large majority of perturbations produce an exactly zero difference, while
    a scalar-margin consumer almost never does.

    The gate's trial count is **decoupled from the number of operating
    points** (issue #1): with fewer points than ``min_trials``, points are
    revisited with fresh directions, so a single-point smooth consumer gets
    a real trial set instead of an automatic rejection from one
    perturbation pair.

    ``flat_fraction`` is the share of zero responses above which the consumer
    is reported as SELECTION. It is a declared threshold and not a measured
    one, and no calibration in this repository has swept it yet.
    """
    if rng is None:
        rng = np.random.default_rng(0)
    pts = np.atleast_2d(np.asarray(points, dtype=float))
    n, d = pts.shape
    trials = min(64, max(int(min_trials), n))
    if n >= trials:
        idx = rng.choice(n, size=trials, replace=False)
    else:
        idx = rng.choice(n, size=trials, replace=True)

    zero = 0
    total = 0
    values = []
    for i in idx:
        x = pts[i]
        u = rng.standard_normal(d)
        u /= np.linalg.norm(u)
        if type(x).__module__.split(".")[0] == "cupy":
            import cupy

            u = cupy.asarray(u)
        hi = float(np.asarray(consumer(x + eps * u)).reshape(()))
        lo = float(np.asarray(consumer(x - eps * u)).reshape(()))
        values.extend([hi, lo])
        total += 1
        if hi == lo:
            zero += 1

    frac = zero / max(total, 1)
    distinct = len(set(np.round(values, 12)))
    evidence = {
        "zero_response_fraction": frac,
        "distinct_output_values": distinct,
        "trials": total,
        "flat_fraction_threshold": flat_fraction,
    }

    if frac >= flat_fraction:
        return Applicability(
            regime=Regime.SELECTION,
            probeable=False,
            reason=(
                "the consumer did not move under most perturbations, which is "
                "the signature of a selection or argmax consumer. Its "
                "derivative is zero almost everywhere, so a blind probe would "
                "report that it reads nothing. Use routing_margins and "
                "differential_fraction instead"
            ),
            evidence=evidence,
        )
    if distinct <= 2:
        return Applicability(
            regime=Regime.UNKNOWN,
            probeable=False,
            reason=(
                "the consumer took at most two distinct values over the "
                "probe, so it is closer to an indicator than to a margin and "
                "a recovered operator would be meaningless"
            ),
            evidence=evidence,
        )
    return Applicability(
        regime=Regime.SCALAR_MARGIN,
        probeable=True,
        reason="the consumer responds smoothly to perturbation",
        evidence=evidence,
    )


def assert_probeable(consumer, points: np.ndarray, **kwargs) -> Applicability:
    """Raise unless the blind probe applies to this consumer."""
    a = applicability(consumer, points, **kwargs)
    if not a.probeable:
        raise ValueError(f"blind probe does not apply: {a.reason}")
    return a
