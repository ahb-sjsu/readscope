"""Reviewer-driven fixes: gate decoupling (issue #1), resource
accounting axes, and the output metric G in P_C = E[J^T G J]."""

import numpy as np
import pytest

from readscope.probe import blind_probe, jacobian_probe
from readscope.regimes import Regime, applicability

D = 12
RNG = np.random.default_rng(7)
W = RNG.standard_normal(D)


def smooth(x):
    return float(np.tanh(W @ np.asarray(x, float)))


def argmaxer(x):
    return float(np.argmax(np.asarray(x, float)))


class TestGateDecoupling:
    def test_single_point_smooth_consumer_passes(self):
        # the issue #1 failure: one point used to mean one perturbation
        # pair, two values, automatic rejection
        a = applicability(smooth, RNG.standard_normal((1, D)))
        assert a.probeable
        assert a.regime is Regime.SCALAR_MARGIN
        assert a.evidence["trials"] >= 16

    def test_single_point_selection_still_rejected(self):
        a = applicability(argmaxer, RNG.standard_normal((1, D)))
        assert not a.probeable

    def test_many_points_still_capped(self):
        a = applicability(smooth, RNG.standard_normal((200, D)))
        assert a.probeable
        assert a.evidence["trials"] == 64


class TestAccountingAxes:
    def test_batched_and_unbatched_axes_agree(self):
        def vec(x):
            return np.tanh(np.asarray(x, float) @ RNG_MAT)

        def vec_batched(X):
            return np.tanh(np.asarray(X, float) @ RNG_MAT)

        pts = RNG.standard_normal((5, D))
        k = 8
        a = jacobian_probe(
            vec, pts, n_directions=k, rng=np.random.default_rng(0)
        )
        b = jacobian_probe(
            vec_batched,
            pts,
            n_directions=k,
            batched=True,
            rng=np.random.default_rng(0),
        )
        # the estimate-defining axis is identical across modes
        assert (
            a.meta["n_directional_observations"]
            == b.meta["n_directional_observations"]
            == k * 5
        )
        # invocations differ by construction, and n_calls means invocations
        assert a.n_calls == a.meta["n_invocations"] == 2 * k * 5
        assert b.n_calls == b.meta["n_invocations"] == 5
        assert b.meta["n_row_evaluations"] == 2 * k * 5

    def test_blind_probe_reports_axes(self):
        r = blind_probe(smooth, RNG.standard_normal((3, D)), mode="exact")
        assert r.meta["n_invocations"] == r.n_calls
        assert r.meta["n_directional_observations"] == D * 3


RNG_MAT = np.random.default_rng(11).standard_normal((D, 4))


class TestOutputMetric:
    def test_identity_default_matches_explicit_identity(self):
        def vec(x):
            return np.asarray(x, float) @ RNG_MAT

        pts = RNG.standard_normal((4, D))
        a = jacobian_probe(
            vec, pts, n_directions=D + 2, rng=np.random.default_rng(1)
        )
        b = jacobian_probe(
            vec,
            pts,
            n_directions=D + 2,
            rng=np.random.default_rng(1),
            output_metric=np.eye(4),
        )
        np.testing.assert_allclose(a.S, b.S, atol=1e-10)
        assert a.meta["output_metric"] == "I"
        assert b.meta["output_metric"] == "custom"

    def test_linear_consumer_recovers_AtGA(self):
        # C(x) = A x has constant Jacobian A, so P_C = A^T G A exactly
        A = RNG_MAT.T  # (4, D)

        def vec(x):
            return A @ np.asarray(x, float)

        g = np.array([2.0, 0.5, 1.0, 3.0])
        r = jacobian_probe(
            vec,
            RNG.standard_normal((3, D)),
            n_directions=D + 4,
            rng=np.random.default_rng(2),
            output_metric=g,
        )
        expected = A.T @ np.diag(g) @ A
        np.testing.assert_allclose(r.S, expected, atol=1e-6)

    def test_batched_metric_matches_unbatched(self):
        def vec(x):
            return np.tanh(np.asarray(x, float) @ RNG_MAT)

        def vec_batched(X):
            return np.tanh(np.asarray(X, float) @ RNG_MAT)

        G = np.diag([1.0, 2.0, 3.0, 4.0])
        pts = RNG.standard_normal((4, D))
        a = jacobian_probe(
            vec,
            pts,
            n_directions=D + 2,
            rng=np.random.default_rng(3),
            output_metric=G,
        )
        b = jacobian_probe(
            vec_batched,
            pts,
            n_directions=D + 2,
            batched=True,
            rng=np.random.default_rng(3),
            output_metric=G,
        )
        np.testing.assert_allclose(a.S, b.S, atol=1e-8)

    def test_bad_metric_shapes_raise(self):
        def vec(x):
            return np.asarray(x, float) @ RNG_MAT

        pts = RNG.standard_normal((2, D))
        with pytest.raises(ValueError, match="does not match"):
            jacobian_probe(vec, pts, n_directions=D, output_metric=np.eye(3))
        with pytest.raises(ValueError, match="symmetric"):
            jacobian_probe(
                vec,
                pts,
                n_directions=D,
                output_metric=np.triu(np.ones((4, 4))),
            )
