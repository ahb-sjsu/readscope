"""The instrument's displays.

A scope has a screen. These are the readscope equivalents, one per thing the
calibration program actually measured.

Every display carries the caveat that belongs with it. A spectrum drawn
without its effective rank invites you to read structure into a tail that
carries no sensitivity. A recovery curve drawn without the chance floor
invites you to read skill into a number that a random subspace would also
score. A drift curve drawn without its null is the specific mistake C-11b
made and C-11c had to undo. So the annotations here are not decoration; they
are the part that stops the picture from lying.

matplotlib is an optional dependency, deliberately: the measuring code stays
numpy-only.

    pip install 'readscope[viz]'

Import this module explicitly. It is not pulled in by ``import readscope``,
so the core never acquires a plotting dependency.
"""

from __future__ import annotations

import numpy as np

_MISSING = (
    "readscope.viz needs matplotlib, which the core deliberately does not "
    "require.\n    pip install 'readscope[viz]'"
)


def _plt():
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError(_MISSING) from exc
    return plt


def _ax(ax, figsize):
    plt = _plt()
    if ax is not None:
        return ax.figure, ax
    fig, axes = plt.subplots(figsize=figsize)
    return fig, axes


def plot_spectrum(spectrum, ax=None, budget_k=None, title=None):
    """The spectrum analyzer trace: where a consumer's sensitivity sits.

    Normalised eigenvalues on a log axis, largest first. The effective rank
    is drawn because the tail of a concentrated spectrum is not small
    structure, it is no structure, and a linear-looking tail on a log axis is
    the easiest thing in this instrument to over-read.

    ``budget_k`` marks the probe's direction budget. Below ``k = d`` the
    recovery is a cliff, so anything past that mark is unresolved rather than
    measured, and the display says so instead of drawing it as data.
    """
    fig, ax = _ax(ax, (7.0, 4.2))
    lam = np.asarray(spectrum.normalized, dtype=float)
    d = lam.size
    idx = np.arange(1, d + 1)
    floor = max(float(lam[lam > 0].min()) if np.any(lam > 0) else 1e-12, 1e-12)
    ax.step(idx, np.maximum(lam, floor), where="mid", lw=1.6, color="#1f77b4")
    ax.fill_between(
        idx,
        floor,
        np.maximum(lam, floor),
        step="mid",
        alpha=0.15,
        color="#1f77b4",
    )
    er = float(spectrum.effective_rank)
    ax.axvline(er, color="#d62728", ls="--", lw=1.3)
    ax.annotate(
        f"effective rank {er:.2f}",
        xy=(er, lam.max()),
        xytext=(6, -2),
        textcoords="offset points",
        color="#d62728",
        fontsize=9,
        va="top",
    )
    e90 = spectrum.energy_rank(0.9)
    ax.axvline(e90, color="#7f7f7f", ls=":", lw=1.1)
    ax.annotate(
        f"90% of sensitivity by direction {e90}",
        xy=(e90, floor),
        xytext=(6, 10),
        textcoords="offset points",
        color="#7f7f7f",
        fontsize=8,
    )
    if budget_k is not None and budget_k < d:
        ax.axvspan(budget_k, d, color="#999999", alpha=0.18)
        ax.annotate(
            f"probe budget k={budget_k} < d={d}\nunresolved, not measured",
            xy=((budget_k + d) / 2, floor),
            xytext=(0, 22),
            textcoords="offset points",
            ha="center",
            fontsize=8,
            color="#444444",
        )
    ax.set_yscale("log")
    ax.set_xlabel("eigendirection, most sensitive first")
    ax.set_ylabel("share of total sensitivity")
    ax.set_title(title or "read operator spectrum")
    ax.grid(alpha=0.25, which="both")
    fig.tight_layout()
    return fig, ax


def plot_allocation(sensitivity, allocation, ax=None, title=None):
    """Reverse water-filling, drawn as the water-filling picture it is.

    Bars are per-direction sensitivity, the line is the water level. Every
    direction above the line gets bits in proportion to how far above it
    sits; everything below gets none and is simply accepted as distortion.
    This is the same optimisation as power allocation across frequency bins,
    with a task's sensitivity where a signal's power would be, so it is drawn
    the way a spectrum analyzer would draw it.
    """
    fig, ax = _ax(ax, (7.0, 4.2))
    lam = np.asarray(sensitivity, dtype=float)
    idx = np.arange(1, lam.size + 1)
    theta = float(allocation.water_level)
    live = lam > theta
    ax.bar(idx[live], lam[live], color="#2ca02c", width=0.9, label="funded")
    ax.bar(
        idx[~live],
        lam[~live],
        color="#c7c7c7",
        width=0.9,
        label="below the water, no bits",
    )
    ax.axhline(
        theta,
        color="#1f77b4",
        lw=1.6,
        label=f"water level {theta:.3g}",
    )
    ax.set_yscale("log")
    ax.set_xlabel("eigendirection")
    ax.set_ylabel("sensitivity")
    starved = int(allocation.n_starved)
    ax.set_title(
        title
        or (
            f"bit allocation at {allocation.budget:g} bits "
            f"({starved} of {lam.size} directions starved)"
        )
    )
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(alpha=0.25, axis="y", which="both")
    fig.tight_layout()
    return fig, ax


def plot_budget_curve(k_over_d, resolution, labels=None, ax=None, title=None):
    """Recovery against the probe's direction budget.

    ``resolution`` may be one curve or several. The measured shape is a cliff
    at ``k = d`` rather than a gentle trade, and it does not move with the
    operator's rank, which is why several ranks are worth overlaying on one
    axis: they land on top of each other.
    """
    fig, ax = _ax(ax, (7.0, 4.2))
    x = np.asarray(k_over_d, dtype=float)
    curves = np.atleast_2d(np.asarray(resolution, dtype=float))
    for i, y in enumerate(curves):
        lab = None if labels is None else labels[i]
        ax.plot(x, y, marker="o", ms=4, lw=1.5, label=lab)
    ax.axvline(1.0, color="#d62728", ls="--", lw=1.4)
    ax.annotate(
        "k = d",
        xy=(1.0, 0.5),
        xytext=(6, 0),
        textcoords="offset points",
        color="#d62728",
        fontsize=9,
    )
    ax.axhline(0.0, color="#7f7f7f", lw=0.8)
    ax.annotate(
        "0 = chance; a random subspace scores here",
        xy=(x.min(), 0.0),
        xytext=(4, 4),
        textcoords="offset points",
        color="#7f7f7f",
        fontsize=8,
    )
    ax.set_xlabel("direction budget k / d")
    ax.set_ylabel("resolution  (overlap - chance) / (1 - chance)")
    ax.set_title(title or "recovery against direction budget")
    if labels is not None:
        ax.legend(fontsize=8)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    return fig, ax


def plot_loading_curve(
    loading, resolution, null_floor=None, ax=None, title=None
):
    """Probe loading: recovery against probe/activation mismatch.

    The instrument's known error term, and the analogue of a scope's input
    impedance. The null floor is drawn because this estimator reads a nonzero
    mismatch at finite sample size when there is no mismatch at all, so
    anything left of that mark is the estimator, not the distributions.
    """
    fig, ax = _ax(ax, (7.0, 4.2))
    x = np.asarray(loading, dtype=float)
    y = np.asarray(resolution, dtype=float)
    order = np.argsort(x)
    ax.plot(x[order], y[order], marker="o", ms=4, lw=1.5, color="#1f77b4")
    if null_floor is not None and null_floor > 0:
        ax.axvspan(x.min(), float(null_floor), color="#d62728", alpha=0.12)
        ax.annotate(
            "below the null floor:\nthe estimator, not the mismatch",
            xy=(float(null_floor), y.max()),
            xytext=(-6, -4),
            textcoords="offset points",
            ha="right",
            va="top",
            fontsize=8,
            color="#a03020",
        )
    ax.set_xlabel("dimensionless loading")
    ax.set_ylabel("resolution")
    ax.set_title(title or "probe loading")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    return fig, ax


def plot_drift(
    positional, null_draws=None, ax=None, title=None, xlabel=None, x=None
):
    """Operator agreement across windows, against its null.

    The null is the whole point. Two disjoint samples of one distribution do
    not give identical operators either, so the positional curve alone means
    nothing and the gap between the curves is the effect. C-11b compared
    against 1.0 and overstated the result by more than a factor of two; this
    display makes that mistake hard to repeat.
    """
    fig, ax = _ax(ax, (7.0, 4.2))
    y = np.asarray(positional, dtype=float)
    idx = np.arange(1, y.size + 1) if x is None else np.asarray(x, float)
    ax.plot(
        idx, y, marker="o", ms=5, lw=1.8, color="#d62728", label="by position"
    )
    if null_draws is not None:
        nd = np.atleast_2d(np.asarray(null_draws, dtype=float))
        mu, sd = nd.mean(axis=0), nd.std(axis=0)
        ax.plot(
            idx,
            mu,
            marker="s",
            ms=4,
            lw=1.5,
            color="#7f7f7f",
            label="null: random split",
        )
        if nd.shape[0] > 1:
            ax.fill_between(idx, mu - sd, mu + sd, color="#7f7f7f", alpha=0.20)
        ax.fill_between(idx, y, mu, color="#d62728", alpha=0.12, label="drift")
    ax.set_ylim(0.0, 1.02)
    ax.set_xlabel(xlabel or "window")
    ax.set_ylabel("agreement with the first window (resolution)")
    ax.set_title(title or "read operator drift, against its null")
    ax.legend(fontsize=8, loc="lower left")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    return fig, ax


def plot_overlap_matrix(overlaps, ax=None, title=None, labels=None):
    """Pairwise agreement between recovered operators.

    Useful for asking whether a family of consumers reads one shared
    subspace or several. The diagonal is one by construction and carries no
    information, so it is masked rather than drawn as evidence.
    """
    fig, ax = _ax(ax, (5.6, 4.8))
    M = np.array(overlaps, dtype=float, copy=True)
    np.fill_diagonal(M, np.nan)
    im = ax.imshow(M, vmin=0.0, vmax=1.0, cmap="viridis")
    fig.colorbar(im, ax=ax, label="subspace overlap")
    if labels is not None:
        ax.set_xticks(range(len(labels)))
        ax.set_yticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
        ax.set_yticklabels(labels, fontsize=8)
    ax.set_title(title or "pairwise operator agreement (diagonal masked)")
    fig.tight_layout()
    return fig, ax
