"""The displays draw what was measured, and say what was not measured.

These tests are mostly about the annotations. A plotting bug that shifts a
line is annoying; a display that omits the chance floor, the effective rank
or the null is the thing that turns a picture into a wrong claim, and that
is what is asserted here.
"""

from __future__ import annotations

import numpy as np
import pytest

from readscope import spectrum_of, water_fill

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")

from readscope import viz  # noqa: E402


def _operator(d=16, rank=3, seed=0):
    rng = np.random.default_rng(seed)
    B = rng.standard_normal((d, rank))
    return B @ B.T + 1e-3 * np.eye(d)


def _texts(ax):
    out = [ax.get_title(), ax.get_xlabel(), ax.get_ylabel()]
    out += [t.get_text() for t in ax.texts]
    leg = ax.get_legend()
    if leg is not None:
        out += [t.get_text() for t in leg.get_texts()]
    return " | ".join(out)


def test_spectrum_marks_effective_rank():
    sp = spectrum_of(_operator())
    fig, ax = viz.plot_spectrum(sp)
    assert "effective rank" in _texts(ax)
    assert ax.get_yscale() == "log"
    assert any(
        abs(ln.get_xdata()[0] - sp.effective_rank) < 1e-6
        for ln in ax.get_lines()
        if len(set(ln.get_xdata())) == 1
    )
    fig.clf()


def test_spectrum_flags_a_sub_dimensional_probe():
    """Past k=d the tail is unresolved, and the display has to say so."""
    sp = spectrum_of(_operator(d=16))
    fig, ax = viz.plot_spectrum(sp, budget_k=4)
    assert "unresolved" in _texts(ax)
    fig.clf()

    fig, ax = viz.plot_spectrum(sp, budget_k=16)
    assert "unresolved" not in _texts(ax)
    fig.clf()


def test_allocation_shows_the_water_level_and_the_starved():
    lam = np.array([8.0, 4.0, 1.0, 0.05, 0.01])
    alloc = water_fill(lam, budget=4.0)
    fig, ax = viz.plot_allocation(lam, alloc)
    txt = _texts(ax)
    assert "water level" in txt
    assert "starved" in txt
    assert str(alloc.n_starved) in txt
    fig.clf()


def test_budget_curve_draws_chance_and_the_cliff():
    x = np.array([0.25, 0.5, 1.0, 1.25])
    y = np.array([0.02, 0.05, 0.98, 1.0])
    fig, ax = viz.plot_budget_curve(x, y)
    txt = _texts(ax)
    assert "k = d" in txt
    assert "chance" in txt
    fig.clf()


def test_budget_curve_overlays_several_ranks():
    x = np.array([0.5, 1.0, 1.25])
    ys = np.array([[0.05, 0.97, 1.0], [0.04, 0.98, 1.0]])
    fig, ax = viz.plot_budget_curve(x, ys, labels=["rank 2", "rank 8"])
    assert len([ln for ln in ax.get_lines() if ln.get_label()[0] != "_"]) == 2
    fig.clf()


def test_loading_curve_marks_the_null_floor():
    x = np.array([0.01, 0.1, 0.4, 0.9])
    y = np.array([1.0, 0.9, 0.6, 0.3])
    fig, ax = viz.plot_loading_curve(x, y, null_floor=0.05)
    assert "null floor" in _texts(ax)
    fig.clf()


def test_drift_always_draws_its_null():
    """The null is the finding. A drift plot without it overstates by 2x."""
    pos = np.array([1.0, 0.62, 0.45, 0.39])
    null = np.array([[1.0, 0.80, 0.72, 0.70], [1.0, 0.78, 0.70, 0.68]])
    fig, ax = viz.plot_drift(pos, null_draws=null)
    txt = _texts(ax)
    assert "null" in txt
    assert "drift" in txt
    fig.clf()


def test_overlap_matrix_masks_its_diagonal():
    M = np.array([[1.0, 0.4], [0.4, 1.0]])
    fig, ax = viz.plot_overlap_matrix(M, labels=["a", "b"])
    # imshow hands back a masked array, so a masked cell is the blank one
    drawn = np.ma.filled(
        np.ma.asarray(ax.images[0].get_array(), dtype=float), np.nan
    )
    assert np.isnan(drawn[0, 0]) and np.isnan(drawn[1, 1])
    assert not np.isnan(drawn[0, 1])
    assert M[0, 0] == 1.0, "the caller's array must not be mutated"
    fig.clf()


def test_displays_accept_a_caller_supplied_axis():
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    _, got = viz.plot_spectrum(spectrum_of(_operator()), ax=ax)
    assert got is ax
    fig.clf()
