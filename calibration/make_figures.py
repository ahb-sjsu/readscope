#!/usr/bin/env python3
"""Regenerate the spec sheet figures from the committed calibration records.

Nothing here invents a curve. Every figure is driven either by a record in
``calibration/records/`` or by the stored activations the records were
computed from, and the script says which. A figure whose data is missing is
skipped and reported, rather than filled in with something illustrative.

    python calibration/make_figures.py [--acts /archive/readscope/source_match]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "calibration"))

RECORDS = ROOT / "calibration" / "records"
OUT = ROOT / "docs" / "figures"


def _load(name):
    p = RECORDS / f"{name}.json"
    if not p.exists():
        return None
    return json.loads(p.read_text())


def fig_budget_cliff(viz):
    """C-6: recovery against direction budget, one curve per operator rank."""
    rec = _load("c6-rank-budget-surface")
    if rec is None:
        return "c6 record missing"
    surf = rec["surface_mean_resolution"]
    ranks = sorted(surf, key=int)
    budgets = sorted({float(b) for r in ranks for b in surf[r]})
    curves = [[surf[r][f"{b:g}"] for b in budgets] for r in ranks]
    fig, ax = viz.plot_budget_curve(
        budgets,
        np.array(curves),
        labels=[f"operator rank {r}" for r in ranks],
        title="C-6: recovery is a cliff at k = d, wherever the rank sits",
    )
    fig.savefig(OUT / "budget_cliff.png", dpi=160)
    return f"budget_cliff.png  ({len(ranks)} ranks, {len(budgets)} budgets)"


def fig_drift(viz):
    """C-11c: positional agreement against the random-split null."""
    rec = _load("c11c-operator-drift")
    if rec is None:
        return "c11c record missing"
    by_rank = rec["summary"]["by_rank"]
    ranks = sorted(by_rank, key=int)
    pos = [by_rank[r]["positional"] for r in ranks]
    null = [by_rank[r]["null"] for r in ranks]
    fig, ax = viz.plot_drift(
        pos,
        null_draws=[null],
        x=[int(r) for r in ranks],
        xlabel="graded rank",
        title="C-11c: drift is the gap, and the null is over half the effect",
    )
    ax.set_xscale("log", base=2)
    ax.set_xticks([int(r) for r in ranks])
    ax.set_xticklabels(ranks)
    ax.set_ylabel("first-to-last agreement (resolution)")
    fig.savefig(OUT / "operator_drift.png", dpi=160)
    return f"operator_drift.png  ({len(ranks)} ranks)"


def fig_loading(viz):
    """C-8: recovery against the dimensionless loading axis."""
    rec = _load("c8-dimensionless-loading")
    if rec is None:
        return "c8 record missing"
    xs, ys = [], []
    for block in rec["axis"]:
        for pt in block["points"]:
            xs.append(pt["loading"])
            ys.append(pt["mismatch"])
    fig, ax = viz.plot_loading_curve(
        xs, ys, title="C-8: the dimensionless loading axis"
    )
    ax.set_ylabel("distribution mismatch that produced it")
    fig.savefig(OUT / "loading_axis.png", dpi=160)
    return f"loading_axis.png  ({len(xs)} points)"


def fig_spectrum_and_allocation(viz, acts):
    """A real read operator from the stored source-matched activations.

    No record commits an eigenvalue array, so these two figures are computed
    from the activations the records were derived from. If those are not
    mounted the figures are skipped, not faked.
    """
    import c11c_operator_drift as C11C

    from readscope import spectrum_of, water_fill

    man = Path(acts) / "manifest.json"
    if not man.exists():
        return "activations not mounted; spectrum and allocation skipped"
    cells = json.loads(man.read_text())
    cell = cells[0]
    z = np.load(Path(acts) / cell["file"])
    K = z["K"].astype(np.float64)
    Q = z["Q"].astype(np.float64)
    d = K.shape[1]
    rng = np.random.default_rng(0)
    probe = rng.choice(K.shape[0], size=min(24, K.shape[0]), replace=False)
    M = C11C.operator(K, Q, probe, d)
    sp = spectrum_of(M)

    fig, _ = viz.plot_spectrum(
        sp,
        title=(
            f"read operator spectrum, {cell['tag']} "
            f"layer {cell['layer']} kv-head {cell['kv_head']}"
        ),
    )
    fig.savefig(OUT / "spectrum.png", dpi=160)

    lam = np.maximum(sp.eigenvalues, 0.0)
    alloc = water_fill(lam, budget=2.0 * lam.size)
    fig, _ = viz.plot_allocation(lam, alloc)
    fig.savefig(OUT / "allocation.png", dpi=160)

    (OUT / "spectrum_source.json").write_text(
        json.dumps(
            {
                "provenance": "operator recovered from the stored "
                "source-matched activations with the C-11c estimator; "
                "committed so the figure can be redrawn without them",
                "cell": cell,
                "eigenvalues": [float(v) for v in sp.eigenvalues],
                "effective_rank": sp.effective_rank,
            },
            indent=2,
        )
    )
    return (
        f"spectrum.png + allocation.png  (effective rank "
        f"{sp.effective_rank:.2f} of {d})"
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--acts", default="/archive/readscope/source_match")
    args = ap.parse_args()

    import matplotlib

    matplotlib.use("Agg")
    from readscope import viz

    OUT.mkdir(parents=True, exist_ok=True)
    for fn in (fig_budget_cliff, fig_drift, fig_loading):
        print(" ", fn(viz), flush=True)
    print(" ", fig_spectrum_and_allocation(viz, args.acts), flush=True)
    print(f"figures in {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
