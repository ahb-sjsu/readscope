"""C-0, the instrument layer: exact controls with closed forms.

Every check here has a known answer that does not depend on the instrument
being any good. They gate everything else.
"""

import numpy as np
import pytest

from readscope import (
    blind_probe,
    chance_overlap,
    consumer_distortion,
    interpolate_distribution,
    probe_loading,
    retrieval_margin_gradient,
    spectrum_of,
    subspace_overlap,
    uniform_allocation,
    water_fill,
)


def linear_consumer(w):
    def C(x):
        return float(w @ x)

    return C


def quadratic_consumer(A):
    def C(x):
        return float(0.5 * x @ A @ x)

    return C


# --------------------------------------------------------------- the probe


def test_linear_consumer_recovered_exactly():
    """A linear read has a constant gradient, so S must be w w^T."""
    rng = np.random.default_rng(0)
    d = 8
    w = rng.standard_normal(d)
    pts = rng.standard_normal((16, d))

    res = blind_probe(linear_consumer(w), pts, eps=1e-4)

    assert np.allclose(res.S, np.outer(w, w), atol=1e-6)
    assert res.n_calls == 16 * 2 * d
    assert res.mode == "exact"


def test_single_direction_consumer_has_effective_rank_one():
    rng = np.random.default_rng(1)
    d = 12
    w = rng.standard_normal(d)
    pts = rng.standard_normal((8, d))

    spec = spectrum_of(blind_probe(linear_consumer(w), pts, eps=1e-4).S)

    assert spec.effective_rank == pytest.approx(1.0, abs=1e-6)
    assert spec.energy_rank(0.9) == 1


def test_isotropic_consumer_has_full_effective_rank():
    """A quadratic read with identity curvature, probed on isotropic
    points, has S = E[x x^T] = I, so every direction is read equally."""
    rng = np.random.default_rng(2)
    d = 10
    pts = rng.standard_normal((20000, d))

    S = pts.T @ pts / len(pts)
    spec = spectrum_of(S)

    assert spec.effective_rank == pytest.approx(d, rel=0.05)


def test_quadratic_consumer_gradient_is_recovered():
    """grad of 0.5 x^T A x is A x, so S = A E[x x^T] A = A A for white x."""
    rng = np.random.default_rng(3)
    d = 6
    B = rng.standard_normal((d, d))
    A = B @ B.T
    pts = rng.standard_normal((4000, d))

    res = blind_probe(quadratic_consumer(A), pts, eps=1e-4)
    expected = A @ (pts.T @ pts / len(pts)) @ A

    assert np.allclose(res.S, expected, atol=1e-2)


def test_sketch_estimator_approaches_exact():
    rng = np.random.default_rng(4)
    d = 6
    w = rng.standard_normal(d)
    pts = rng.standard_normal((200, d))

    exact = blind_probe(linear_consumer(w), pts, eps=1e-4)
    sketch = blind_probe(
        linear_consumer(w),
        pts,
        mode="sketch",
        sketch_dim=256,
        eps=1e-4,
        rng=np.random.default_rng(5),
    )

    top_e = exact.read_subspace(1)
    top_s = sketch.read_subspace(1)
    assert subspace_overlap(top_s, top_e).overlap > 0.9
    assert sketch.n_calls == 200 * 2 * 256


def test_sketch_requires_its_dimension():
    pts = np.random.default_rng(6).standard_normal((4, 3))
    with pytest.raises(ValueError):
        blind_probe(linear_consumer(np.ones(3)), pts, mode="sketch")


def test_retrieval_margin_gradient_is_orthogonal_to_the_key():
    rng = np.random.default_rng(7)
    a, b = rng.standard_normal(9), rng.standard_normal(9)
    g = retrieval_margin_gradient(a, b)
    assert abs(float(g @ (b / np.linalg.norm(b)))) < 1e-12


# ------------------------------------------------------------- the metrics


def test_overlap_is_one_for_the_same_subspace_and_reports_chance():
    rng = np.random.default_rng(8)
    U = np.linalg.qr(rng.standard_normal((20, 4)))[0]
    r = subspace_overlap(U, U)
    assert r.overlap == pytest.approx(1.0)
    assert r.chance == pytest.approx(4 / 20)
    assert r.ratio == pytest.approx(5.0)


def test_random_subspaces_sit_near_the_chance_floor():
    rng = np.random.default_rng(9)
    d, r, trials = 64, 8, 200
    vals = []
    for _ in range(trials):
        A = np.linalg.qr(rng.standard_normal((d, r)))[0]
        B = np.linalg.qr(rng.standard_normal((d, r)))[0]
        vals.append(subspace_overlap(A, B).overlap)
    assert np.mean(vals) == pytest.approx(chance_overlap(r, d), abs=0.02)


def test_orthogonal_subspaces_have_zero_overlap():
    d = 10
    U = np.eye(d)[:, :3]
    V = np.eye(d)[:, 3:6]
    assert subspace_overlap(U, V).overlap == pytest.approx(0.0, abs=1e-12)


def test_consumer_distortion_is_the_trace():
    rng = np.random.default_rng(10)
    P = np.diag([3.0, 1.0, 0.0])
    Sd = np.diag([1.0, 2.0, 100.0])
    assert consumer_distortion(P, Sd) == pytest.approx(5.0)
    del rng


def test_distortion_ignores_error_the_consumer_cannot_read():
    """Error piled entirely into a null direction of P costs nothing."""
    P = np.diag([1.0, 0.0])
    assert consumer_distortion(P, np.diag([0.0, 1e6])) == pytest.approx(0.0)


# ---------------------------------------------------------- the allocation


def test_flat_spectrum_reproduces_uniform_allocation():
    d, budget = 8, 32.0
    alloc = water_fill(np.ones(d), budget=budget)
    assert np.allclose(alloc.bits, uniform_allocation(d, budget), atol=1e-6)
    assert alloc.n_starved == 0


def test_rank_one_spectrum_starves_every_other_direction():
    lam = np.array([1.0, 0.0, 0.0, 0.0])
    alloc = water_fill(lam, budget=8.0)
    assert alloc.bits[0] == pytest.approx(8.0)
    assert np.allclose(alloc.bits[1:], 0.0)
    assert alloc.n_starved == 3


def test_allocation_respects_its_budget():
    rng = np.random.default_rng(11)
    lam = np.abs(rng.standard_normal(32))
    alloc = water_fill(lam, budget=64.0)
    assert alloc.bits.sum() == pytest.approx(64.0, abs=1e-6)
    assert np.all(alloc.bits >= 0.0)


def test_water_filling_beats_uniform_on_a_skewed_spectrum():
    lam = np.array([100.0, 10.0, 1.0, 0.1])
    budget = 8.0
    wf = water_fill(lam, budget=budget)
    uni = uniform_allocation(4, budget)
    d_uni = float(np.sum(lam * np.power(2.0, -2.0 * uni)))
    assert wf.distortion < d_uni


def test_zero_budget_spends_nothing():
    alloc = water_fill(np.array([1.0, 2.0]), budget=0.0)
    assert np.allclose(alloc.bits, 0.0)


def test_max_bits_ceiling_is_respected():
    lam = np.array([1e6, 1.0, 1.0])
    alloc = water_fill(lam, budget=12.0, max_bits=4.0)
    assert np.all(alloc.bits <= 4.0 + 1e-9)


# --------------------------------------------------------------- the loading


def test_identical_distributions_have_zero_loading():
    rng = np.random.default_rng(12)
    X = rng.standard_normal((4000, 5))
    r = probe_loading(X, X)
    assert r.jeffreys == pytest.approx(0.0, abs=1e-9)
    assert r.bhattacharyya == pytest.approx(0.0, abs=1e-9)
    assert r.mean_shift == pytest.approx(0.0, abs=1e-9)
    assert r.spectral_ratio == pytest.approx(1.0, abs=1e-6)


def test_loading_rises_with_a_mean_shift():
    rng = np.random.default_rng(13)
    A = rng.standard_normal((4000, 4))
    near = A + 0.1
    far = A + 2.0
    assert probe_loading(far, A).jeffreys > probe_loading(near, A).jeffreys


def test_loading_rises_with_a_covariance_mismatch():
    rng = np.random.default_rng(14)
    A = rng.standard_normal((4000, 4))
    mild = A * np.array([1.2, 1.0, 1.0, 1.0])
    harsh = A * np.array([6.0, 1.0, 1.0, 1.0])
    assert probe_loading(harsh, A).jeffreys > probe_loading(mild, A).jeffreys
    assert probe_loading(harsh, A).spectral_ratio > 4.0


def test_interpolation_ends_match_their_endpoints():
    rng = np.random.default_rng(15)
    P = rng.standard_normal((6000, 3)) * 3.0 + 5.0
    A = rng.standard_normal((6000, 3))

    at_zero = interpolate_distribution(
        P, A, 0.0, rng=np.random.default_rng(16), n_samples=6000
    )
    at_one = interpolate_distribution(
        P, A, 1.0, rng=np.random.default_rng(17), n_samples=6000
    )

    assert probe_loading(at_zero, P).jeffreys < 0.2
    assert probe_loading(at_one, A).jeffreys < 0.2


def test_interpolation_is_monotone_in_loading():
    rng = np.random.default_rng(18)
    P = rng.standard_normal((6000, 3)) * 4.0 + 3.0
    A = rng.standard_normal((6000, 3))

    loads = [
        probe_loading(
            interpolate_distribution(
                P, A, a, rng=np.random.default_rng(19), n_samples=6000
            ),
            A,
        ).jeffreys
        for a in (0.0, 0.25, 0.5, 0.75, 1.0)
    ]
    assert all(loads[i] > loads[i + 1] for i in range(len(loads) - 1))


def test_interpolation_rejects_alpha_outside_the_unit_interval():
    rng = np.random.default_rng(20)
    X = rng.standard_normal((100, 3))
    with pytest.raises(ValueError):
        interpolate_distribution(X, X, 1.5)


# ----------------------------------------------- the loading effect itself


def test_loading_degrades_recovery():
    """The reason the calibration curve is worth measuring.

    A consumer that reads one direction is probed at operating points drawn
    increasingly far from the activation distribution. Recovery of the read
    direction must not be indifferent to that, or probe loading is not the
    error term this instrument thinks it is.

    This is a smoke test on one configuration and is not the calibration
    curve. C-1 is.
    """
    rng = np.random.default_rng(21)
    d = 16
    w = np.zeros(d)
    w[0] = 1.0

    def consumer(x):
        return float(np.tanh(w @ x))

    activation = rng.standard_normal((400, d)) @ np.diag(
        np.concatenate([[3.0], np.ones(d - 1) * 0.2])
    )
    truth = w.reshape(-1, 1)

    on_dist = blind_probe(consumer, activation, eps=1e-3).read_subspace(1)
    assert subspace_overlap(on_dist, truth).overlap > 0.9
