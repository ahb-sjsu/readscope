"""Instrument-neutral sealed-run statistics, extracted from five
hand-rolled copies across the readscope calibrations, the
geometric-observation crucible runners, and the network-governor
audit runners. Numpy-only, like everything in this package.

Nothing here knows about operators, episodes, or certificates — that
is the point. A sealed prereg that uses these MUST pin the readscope
version it ran under (record it in the prereg and the result JSON):
sealed runners are as-executed artifacts and never float with a
dependency.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np


@dataclass
class BootstrapCI:
    """A percentile bootstrap interval with its provenance."""

    point: float
    lo: float
    hi: float
    n_boot: int
    ci: float

    def as_list(self) -> list[float]:
        return [round(self.lo, 6), round(self.hi, 6)]


def bootstrap_ci(
    n_items: int,
    statistic: Callable[[np.ndarray], float],
    *,
    n_boot: int = 1000,
    ci: float = 95.0,
    rng: np.random.Generator | None = None,
) -> BootstrapCI:
    """Percentile bootstrap over item indices.

    ``statistic`` receives an integer index array of length
    ``n_items`` (resampled with replacement) and returns the
    statistic computed on that resample; the point estimate uses the
    identity indexing. This is the resample-the-units shape every
    sealed runner in the program uses — episodes, certificates,
    codec pairs — with the unit definition staying in the caller.
    """
    if rng is None:
        rng = np.random.default_rng(0)
    if n_items < 1:
        raise ValueError("bootstrap needs at least one item")
    point = float(statistic(np.arange(n_items)))
    boots = np.empty(n_boot)
    for b in range(n_boot):
        boots[b] = statistic(rng.integers(0, n_items, n_items))
    alpha = (100.0 - ci) / 2.0
    return BootstrapCI(
        point=point,
        lo=float(np.percentile(boots, alpha)),
        hi=float(np.percentile(boots, 100.0 - alpha)),
        n_boot=n_boot,
        ci=ci,
    )


def _rank_average_ties(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x, kind="stable")
    ranks = np.empty(len(x))
    sx = x[order]
    i = 0
    while i < len(sx):
        j = i
        while j + 1 < len(sx) and sx[j + 1] == sx[i]:
            j += 1
        ranks[order[i : j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return ranks


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman rank correlation, average ties, pure numpy.

    The trend statistic the program's decay/tracking bars use
    (OT-14, OT-18, the RPKI cell). Returns nan for degenerate
    (constant) inputs rather than raising — a constant series has no
    trend to report, and the caller's bar should treat nan as a
    failure to exhibit one.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.shape != y.shape or x.ndim != 1:
        raise ValueError("spearman expects two equal-length 1-d arrays")
    rx, ry = _rank_average_ties(x), _rank_average_ties(y)
    sx, sy = rx.std(), ry.std()
    if sx == 0 or sy == 0:
        return float("nan")
    return float(((rx - rx.mean()) * (ry - ry.mean())).mean() / (sx * sy))
