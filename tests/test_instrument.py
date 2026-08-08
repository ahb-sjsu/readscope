"""C-0, the instrument layer: exact controls with closed forms.

Every check here has a known answer that does not depend on the instrument
being any good. They gate everything else.
"""

import numpy as np
import pytest

from readscope import (
    OverlapReading,
    Regime,
    applicability,
    assert_probeable,
    blind_probe,
    chance_overlap,
    consumer_distortion,
    debias_sketch,
    decay_sensitivity,
    differential_fraction,
    displacement_decomposition,
    fit_loading_correction,
    interpolate_distribution,
    jacobian_probe,
    loading_null_floor,
    probe_loading,
    retrieval_margin_gradient,
    routing_margins,
    spectrum_of,
    subspace_overlap,
    tangential_fraction,
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


def test_resolution_is_zero_at_chance_and_one_at_perfect():
    rng = np.random.default_rng(80)
    U = np.linalg.qr(rng.standard_normal((32, 4)))[0]
    perfect = subspace_overlap(U, U)
    assert perfect.resolution == pytest.approx(1.0)

    floor = OverlapReading(overlap=4 / 32, chance=4 / 32, rank=4, dim=32)
    assert floor.resolution == pytest.approx(0.0)


def test_resolution_compares_across_shapes_where_overlap_cannot():
    """0.5 raw overlap means very different things at different ranks."""
    easy = OverlapReading(overlap=0.5, chance=1 / 64, rank=1, dim=64)
    hard = OverlapReading(overlap=0.5, chance=32 / 64, rank=32, dim=64)
    assert easy.resolution > 0.49
    assert hard.resolution == pytest.approx(0.0)


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


# ------------------------------------------- regimes, inherited from tqp


def test_selection_consumer_is_refused():
    """The regime the blind probe silently fails on.

    A top-k router is piecewise constant, so finite differencing returns
    zero and an unguarded probe would report a consumer that reads nothing.
    """
    rng = np.random.default_rng(90)
    d, e = 12, 6
    W = rng.standard_normal((e, d))

    def router(x):
        return float(np.argmax(W @ x))

    pts = rng.standard_normal((48, d))
    verdict = applicability(router, pts, rng=np.random.default_rng(91))

    assert verdict.regime is Regime.SELECTION
    assert not verdict.probeable
    assert verdict.evidence["zero_response_fraction"] >= 0.9
    with pytest.raises(ValueError):
        assert_probeable(router, pts, rng=np.random.default_rng(91))


def test_smooth_consumer_is_admitted():
    rng = np.random.default_rng(92)
    d = 12
    w = rng.standard_normal(d)
    pts = rng.standard_normal((48, d))
    verdict = applicability(
        lambda x: float(np.tanh(w @ x)), pts, rng=np.random.default_rng(93)
    )
    assert verdict.regime is Regime.SCALAR_MARGIN
    assert verdict.probeable


def test_indicator_consumer_is_refused():
    rng = np.random.default_rng(94)
    pts = rng.standard_normal((48, 8))
    verdict = applicability(
        lambda x: float(x[0] > 0), pts, rng=np.random.default_rng(95)
    )
    assert not verdict.probeable


def test_routing_margin_is_the_top_two_gap():
    logits = np.array([[3.0, 1.0, 0.5], [2.0, 1.9, 0.0]])
    m = routing_margins(logits, k=1)
    assert m[0] == pytest.approx(2.0)
    assert m[1] == pytest.approx(0.1)


def test_common_mode_logit_error_is_routing_safe():
    """Selection is invariant to a shift shared by every expert."""
    common = np.ones((4, 5)) * 0.7
    assert np.allclose(differential_fraction(common), 0.0, atol=1e-12)
    differential = np.tile(np.array([1.0, -1.0, 0.0, 0.0, 0.0]), (4, 1))
    assert np.allclose(differential_fraction(differential), 1.0, atol=1e-12)


def test_decay_sensitivity_blows_up_for_slow_channels():
    fast, slow = decay_sensitivity(np.array([0.1])), decay_sensitivity(
        np.array([0.99])
    )
    assert slow[0] > 100.0 * fast[0]
    assert (
        decay_sensitivity(np.array([0.9]), seq_len=4)[0]
        < decay_sensitivity(np.array([0.9]))[0]
    )


# ------------------------------------------ quotient, inherited from tqp


def test_pure_direction_change_is_fully_tangential():
    x = np.array([1.0, 0.0])
    y = np.array([0.0, 1.0])
    assert tangential_fraction(x, y) == pytest.approx(1.0)


def test_pure_scaling_is_fully_radial():
    x = np.array([1.0, 2.0, 3.0])
    assert tangential_fraction(x, 2.5 * x) == pytest.approx(0.0, abs=1e-12)


def test_coincident_vectors_are_undefined():
    x = np.array([1.0, 1.0])
    assert np.isnan(tangential_fraction(x, x))


def test_displacement_decomposition_reports_pairs():
    rng = np.random.default_rng(96)
    batch = rng.standard_normal((200, 16))
    d = displacement_decomposition(batch, n_pairs=500, seed=1)
    assert d["n_pairs"] > 0
    assert 0.0 <= d["median_tangential_fraction"] <= 1.0
    assert d["median_radial_fraction"] == pytest.approx(
        1.0 - d["median_tangential_fraction"]
    )


def test_blind_probe_refuses_a_selection_consumer_by_default():
    rng = np.random.default_rng(97)
    W = rng.standard_normal((6, 10))
    pts = rng.standard_normal((32, 10))
    with pytest.raises(ValueError, match="does not apply"):
        blind_probe(lambda x: float(np.argmax(W @ x)), pts)


def test_the_guard_can_be_overridden_and_records_nothing():
    rng = np.random.default_rng(98)
    w = rng.standard_normal(6)
    pts = rng.standard_normal((8, 6))
    res = blind_probe(linear_consumer(w), pts, check_regime=False)
    assert res.meta["regime"] is None


def test_the_guard_records_its_verdict_when_it_runs():
    rng = np.random.default_rng(99)
    w = rng.standard_normal(6)
    pts = rng.standard_normal((8, 6))
    res = blind_probe(linear_consumer(w), pts)
    assert res.meta["regime"]["regime"] == Regime.SCALAR_MARGIN.value


def test_the_guard_does_not_disturb_the_sketch_stream():
    """The guard must not consume from the caller's generator, or every
    record taken before it existed becomes irreproducible."""
    rng = np.random.default_rng(100)
    w = rng.standard_normal(8)
    pts = rng.standard_normal((16, 8))

    guarded = blind_probe(
        linear_consumer(w),
        pts,
        mode="sketch",
        sketch_dim=32,
        rng=np.random.default_rng(7),
    )
    unguarded = blind_probe(
        linear_consumer(w),
        pts,
        mode="sketch",
        sketch_dim=32,
        rng=np.random.default_rng(7),
        check_regime=False,
    )
    assert np.allclose(guarded.S, unguarded.S)


# --------------------------------- the sketch's bias, and what it is not


def test_sketch_bias_matches_the_closed_form():
    """E[ghat ghat^T] = (1 + 1/k) g g^T + (||g||^2 / k) I."""
    rng = np.random.default_rng(110)
    d, k = 5, 3
    g = rng.standard_normal(d)
    acc = np.zeros((d, d))
    trials = 60000
    for _ in range(trials):
        U = rng.standard_normal((k, d))
        ghat = (U @ g) @ U / k
        acc += np.outer(ghat, ghat)
    acc /= trials
    predicted = (1 + 1 / k) * np.outer(g, g) + (g @ g) / k * np.eye(d)
    assert np.abs(acc - predicted).max() < 0.05 * np.abs(predicted).max()


def test_debias_inverts_the_bias_exactly():
    rng = np.random.default_rng(111)
    d, k = 6, 4
    g = rng.standard_normal(d)
    S_true = np.outer(g, g)
    S_hat = (1 + 1 / k) * S_true + np.trace(S_true) / k * np.eye(d)
    assert np.allclose(debias_sketch(S_hat, k), S_true, atol=1e-12)


def test_debias_cannot_rotate_an_eigenvector():
    """The sharp prediction. The inflation is isotropic, so debiasing fixes
    the spectrum and provably not the subspace."""
    rng = np.random.default_rng(112)
    d, k = 10, 4
    A = rng.standard_normal((d, d))
    S_hat = A @ A.T
    before = spectrum_of(S_hat).eigenvectors
    after = spectrum_of(debias_sketch(S_hat, k)).eigenvectors
    for i in range(d):
        assert abs(abs(float(before[:, i] @ after[:, i])) - 1.0) < 1e-9


def test_debias_preserves_the_recovered_subspace_end_to_end():
    rng = np.random.default_rng(113)
    d, r = 24, 3
    basis = np.linalg.qr(rng.standard_normal((d, r)))[0]
    w = 0.8 ** np.arange(r)

    def consumer(x):
        return float(np.tanh(basis.T @ x) @ w)

    pts = rng.standard_normal((128, d)) * 0.35
    res = blind_probe(
        consumer,
        pts,
        mode="sketch",
        sketch_dim=8,
        rng=np.random.default_rng(114),
    )
    raw = subspace_overlap(res.read_subspace(r), basis).overlap
    deb = spectrum_of(debias_sketch(res.S, 8)).eigenvectors[:, :r]
    assert subspace_overlap(deb, basis).overlap == pytest.approx(raw, abs=1e-9)


# --------------------------------------------- the orthonormal estimator


def test_ortho_at_full_rank_is_exact():
    """At k = d the projector is the identity, so the estimate is exact."""
    rng = np.random.default_rng(115)
    d = 8
    w = rng.standard_normal(d)
    pts = rng.standard_normal((12, d))
    res = blind_probe(
        linear_consumer(w),
        pts,
        mode="ortho",
        sketch_dim=d,
        rng=np.random.default_rng(116),
    )
    assert np.allclose(res.S, np.outer(w, w), atol=1e-6)


def test_ortho_rejects_more_directions_than_dimensions():
    pts = np.random.default_rng(117).standard_normal((4, 5))
    with pytest.raises(ValueError, match="sketch_dim <= dim"):
        blind_probe(
            linear_consumer(np.ones(5)), pts, mode="ortho", sketch_dim=9
        )


def test_ortho_beats_the_gaussian_sketch_at_equal_cost():
    rng = np.random.default_rng(118)
    d, r, k = 24, 3, 8
    basis = np.linalg.qr(rng.standard_normal((d, r)))[0]
    w = 0.8 ** np.arange(r)

    def consumer(x):
        return float(np.tanh(basis.T @ x) @ w)

    pts = rng.standard_normal((128, d)) * 0.35
    common = {"sketch_dim": k, "eps": 1e-3}
    a = blind_probe(
        consumer, pts, mode="sketch", rng=np.random.default_rng(119), **common
    )
    b = blind_probe(
        consumer, pts, mode="ortho", rng=np.random.default_rng(119), **common
    )
    assert a.n_calls == b.n_calls
    assert (
        subspace_overlap(b.read_subspace(r), basis).overlap
        > subspace_overlap(a.read_subspace(r), basis).overlap
    )


# ------------------- the estimator the published figures actually used


def test_lstsq_at_full_rank_is_exact():
    rng = np.random.default_rng(120)
    d = 8
    w = rng.standard_normal(d)
    pts = rng.standard_normal((10, d))
    res = blind_probe(
        linear_consumer(w),
        pts,
        mode="lstsq",
        sketch_dim=d,
        rng=np.random.default_rng(121),
    )
    assert np.allclose(res.S, np.outer(w, w), atol=1e-6)


def test_lstsq_is_exact_when_overdetermined():
    """n_directions > d is the regime the source program actually ran."""
    rng = np.random.default_rng(122)
    d = 8
    w = rng.standard_normal(d)
    pts = rng.standard_normal((10, d))
    res = blind_probe(
        linear_consumer(w),
        pts,
        mode="lstsq",
        sketch_dim=2 * d,
        rng=np.random.default_rng(123),
    )
    assert np.allclose(res.S, np.outer(w, w), atol=1e-6)
    assert res.n_calls == 10 * 2 * 2 * d


def test_jacobian_probe_recovers_a_linear_vector_consumer():
    """For C(x) = A x the Jacobian is A, so M = A^T A exactly."""
    rng = np.random.default_rng(124)
    d, m = 7, 4
    A = rng.standard_normal((m, d))
    pts = rng.standard_normal((6, d))
    res = jacobian_probe(
        lambda x: A @ x,
        pts,
        n_directions=d,
        rng=np.random.default_rng(125),
    )
    assert np.allclose(res.S, A.T @ A, atol=1e-6)
    assert res.meta["out_dim"] == m
    assert res.n_calls == 6 * 2 * d


def test_jacobian_probe_reduces_to_the_scalar_case():
    rng = np.random.default_rng(126)
    d = 6
    w = rng.standard_normal(d)
    pts = rng.standard_normal((8, d))
    vec = jacobian_probe(
        lambda x: np.array([w @ x]),
        pts,
        n_directions=d,
        rng=np.random.default_rng(127),
    )
    assert np.allclose(vec.S, np.outer(w, w), atol=1e-6)


def test_vector_output_carries_more_directions_than_a_scalar():
    """A scalar margin reveals a rank-one operator per point; a vector
    consumer reveals up to m."""
    rng = np.random.default_rng(128)
    d, m = 12, 5
    A = rng.standard_normal((m, d))
    pts = rng.standard_normal((4, d))
    res = jacobian_probe(
        lambda x: A @ x, pts, n_directions=d, rng=np.random.default_rng(129)
    )
    assert np.linalg.matrix_rank(res.S, tol=1e-8) == m


# ------------------------------------------ the loading correction itself


def test_correction_is_identity_at_zero_loading():
    c = fit_loading_correction([0.0, 10.0, 100.0], [1.0, 0.9, 0.4])
    assert c.expected_attenuation(0.0) == pytest.approx(1.0)
    assert c.correct(0.8, 0.0) == pytest.approx(0.8)


def test_correction_undoes_a_known_attenuation():
    c = fit_loading_correction([0.0, 50.0], [1.0, 0.5])
    assert c.correct(0.25, 50.0) == pytest.approx(0.5, abs=1e-9)


def test_correction_is_monotone_non_increasing():
    """More loading can never be reported as helping."""
    c = fit_loading_correction([0.0, 5.0, 10.0, 20.0], [1.0, 0.7, 0.8, 0.3])
    xs = [0.0, 2.0, 5.0, 10.0, 15.0, 20.0, 40.0]
    g = [c.expected_attenuation(x) for x in xs]
    assert all(g[i] >= g[i + 1] - 1e-12 for i in range(len(g) - 1))


def test_correction_clamps_outside_the_fitted_range():
    c = fit_loading_correction([1.0, 10.0], [1.0, 0.5])
    assert c.expected_attenuation(0.0) == pytest.approx(1.0)
    assert c.expected_attenuation(1e6) == pytest.approx(0.5)


def test_correction_output_stays_a_valid_resolution():
    c = fit_loading_correction([0.0, 100.0], [1.0, 0.01])
    assert c.correct(0.9, 100.0) <= 1.0


def test_correction_rejects_bad_calibration():
    with pytest.raises(ValueError):
        fit_loading_correction([1.0], [1.0])
    with pytest.raises(ValueError):
        fit_loading_correction([1.0, 2.0], [1.0])


def test_correction_round_trips_its_own_calibration():
    load = [0.9, 3.2, 10.7, 25.3, 50.5, 91.6]
    read = [0.992, 0.989, 0.975, 0.852, 0.491, 0.437]
    c = fit_loading_correction(load, read, source="c1b")
    for x, y in zip(load, read, strict=False):
        assert c.correct(y, x) == pytest.approx(1.0, abs=1e-6)
    assert c.to_dict()["source"] == "c1b"


def test_loading_refuses_a_rank_deficient_sample():
    """16 points in 128 dimensions produced 1e11 nats before this guard."""
    rng = np.random.default_rng(130)
    activation = rng.standard_normal((2000, 128))
    too_few = rng.standard_normal((16, 128))
    with pytest.raises(ValueError, match="rank deficient"):
        probe_loading(too_few, activation)


def test_loading_warns_below_five_samples_per_dimension():
    rng = np.random.default_rng(131)
    activation = rng.standard_normal((2000, 16))
    thin = rng.standard_normal((40, 16))
    with pytest.warns(RuntimeWarning, match="per dimension"):
        probe_loading(thin, activation)


def test_loading_is_quiet_when_well_sampled():
    import warnings

    rng = np.random.default_rng(132)
    a = rng.standard_normal((2000, 8))
    b = rng.standard_normal((2000, 8))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        probe_loading(a, b)


def test_strict_false_downgrades_to_a_warning():
    rng = np.random.default_rng(133)
    activation = rng.standard_normal((300, 32))
    too_few = rng.standard_normal((8, 32))
    with pytest.warns(RuntimeWarning, match="rank deficient"):
        r = probe_loading(too_few, activation, strict=False)
    assert np.isfinite(r.jeffreys)


def test_correction_refuses_to_extrapolate():
    """The failure mode that manufactured C-7b's 92% error reduction."""
    c = fit_loading_correction([1.0, 90.0], [1.0, 0.44])
    assert c.in_domain(50.0)
    assert not c.in_domain(1e11)
    with pytest.raises(ValueError, match="outside the fitted range"):
        c.correct(0.78, 1e11)


def test_extrapolation_can_be_forced_but_is_the_documented_failure():
    c = fit_loading_correction([1.0, 90.0], [1.0, 0.44])
    forced = c.correct(0.78, 1e11, strict_domain=False)
    assert forced == pytest.approx(1.0)


def test_in_domain_correction_still_works():
    c = fit_loading_correction([1.0, 90.0], [1.0, 0.44])
    assert c.correct(0.44, 90.0) == pytest.approx(1.0, abs=1e-9)


# --------------------------------------------- the dimensionless axis


def test_null_floor_is_positive_and_grows_as_samples_thin():
    """Two samples of one law still read a positive divergence."""
    thick = loading_null_floor(4000, 4000, 8, trials=8)
    thin = loading_null_floor(200, 200, 8, trials=8)
    assert thick > 0.0
    assert thin > thick


def test_null_floor_is_distribution_free_in_scale():
    """It depends on shape only, which is why it can be cached."""
    a = loading_null_floor(500, 500, 16, trials=8)
    b = loading_null_floor(500, 500, 16, trials=8)
    assert a == b


def test_identical_distributions_read_zero_loading():
    rng = np.random.default_rng(140)
    X = rng.standard_normal((3000, 8))
    assert probe_loading(X, X).loading == pytest.approx(0.0)


def test_independent_samples_of_one_law_read_near_zero_loading():
    """The null correction is what makes this true; raw Jeffreys is not 0."""
    rng = np.random.default_rng(141)
    a = rng.standard_normal((3000, 8))
    b = rng.standard_normal((3000, 8))
    r = probe_loading(a, b)
    assert r.jeffreys > 0.0
    assert r.loading < 0.02


def test_loading_is_comparable_across_dimensions():
    """The whole point. A per-direction mismatch of the same size must read
    the same number at any dimension, which raw Jeffreys does not."""
    readings, raws = [], []
    for d in (8, 16, 32, 64):
        rng = np.random.default_rng(200 + d)
        n = 200 * d
        a = rng.standard_normal((n, d))
        b = rng.standard_normal((n, d)) * 2.0
        r = probe_loading(a, b)
        readings.append(r.loading)
        raws.append(r.jeffreys)

    # comparative, not absolute: "dimensionless" means the normalised
    # spread is far smaller than the raw one, and a fixed tolerance on the
    # normalised value would be a threshold guess rather than the property
    def rel_spread(v):
        return (max(v) - min(v)) / max(abs(np.mean(v)), 1e-12)

    assert rel_spread(raws) > 1.0
    assert rel_spread(readings) < 0.2
    assert rel_spread(readings) < rel_spread(raws) / 5.0


def test_loading_is_monotone_in_mismatch():
    rng = np.random.default_rng(142)
    d, n = 16, 4000
    a = rng.standard_normal((n, d))
    out = [
        probe_loading(rng.standard_normal((n, d)) * s, a).loading
        for s in (1.2, 2.0, 4.0)
    ]
    assert out[0] < out[1] < out[2]
