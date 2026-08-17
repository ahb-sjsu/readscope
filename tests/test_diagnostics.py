"""step_response and split_half_overlap (reviewer items 1 and 2)."""

import numpy as np
import pytest

from readscope.diagnostics import split_half_overlap, step_response

D = 16
RNG = np.random.default_rng(3)
BASIS = np.linalg.qr(RNG.standard_normal((D, 4)))[0]
W = 0.75 ** np.arange(4)


def smooth(x):
    return float(np.tanh(BASIS.T @ np.asarray(x, float)) @ W)


def curved(x):
    # strong curvature so the step response is visibly eps-dependent
    v = float(BASIS[:, 0] @ np.asarray(x, float))
    return float(np.sin(40.0 * v))


class TestStepResponse:
    def test_smooth_consumer_converged_at_default_eps(self):
        r = step_response(smooth, RNG.standard_normal((8, D)) * 0.3)
        assert r["converged"]
        assert r["rel_change_half"] < 1e-3

    def test_curved_consumer_flags_step_dependence(self):
        r = step_response(curved, RNG.standard_normal((8, D)) * 0.3, eps=5e-2)
        assert not r["converged"]
        assert r["rel_change_double"] > r["tol"]


class TestSplitHalf:
    def test_determined_operator_high_resolution(self):
        r = split_half_overlap(
            smooth, RNG.standard_normal((64, D)) * 0.3, rank=4
        )
        assert r["resolution"] > 0.8
        assert r["n_per_half"] == 32

    def test_too_few_points_raises(self):
        with pytest.raises(ValueError, match="at least 4"):
            split_half_overlap(smooth, RNG.standard_normal((3, D)), rank=2)
