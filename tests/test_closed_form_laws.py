"""Machine verification of readscope's closed-form claims, CI-resident:
the Isserlis sketch expectation and its debias inversion (exact, via
the identity's own algebra), the subspace-overlap chance value, and
water_fill's KKT optimality against an independent optimizer."""

import numpy as np
import pytest

from readscope import subspace_overlap, water_fill
from readscope.probe import debias_sketch

RNG = np.random.default_rng(5)


class TestIsserlisChain:
    def test_sketch_expectation_montecarlo(self):
        # E[ghat ghat^T] = (1+1/k) g g^T + (||g||^2/k) I, checked at
        # high replication; the symbolic Wick proof lives in
        # geometric-observation/crucible/verify_theorems.py
        d, k, n = 6, 4, 400_000
        g = RNG.standard_normal(d)
        u = RNG.standard_normal((n, k, d))
        ghat = np.einsum("nkd,nke->nde", u, u) @ g / k
        emp = np.einsum("nd,ne->de", ghat, ghat) / n
        theory = (1 + 1 / k) * np.outer(g, g) + (g @ g / k) * np.eye(d)
        assert np.linalg.norm(emp - theory) / np.linalg.norm(theory) < 0.02

    def test_debias_inverts_the_forward_map_exactly(self):
        d, k = 8, 5
        s = RNG.standard_normal((d, d))
        s = s @ s.T
        inflated = (1 + 1 / k) * s + (np.trace(s) / k) * np.eye(d)
        recovered = debias_sketch(inflated, sketch_dim=k)
        np.testing.assert_allclose(recovered, s, atol=1e-10)


class TestOverlapChance:
    def test_chance_is_rank_over_dim(self):
        # E[overlap(U_true, U_haar)] = r/d by E[P_haar] = (r/d) I
        d, r, trials = 24, 5, 4000
        u_true = np.linalg.qr(RNG.standard_normal((d, r)))[0]
        vals = []
        for _ in range(trials):
            u = np.linalg.qr(RNG.standard_normal((d, r)))[0]
            vals.append(subspace_overlap(u, u_true).overlap)
        assert abs(np.mean(vals) - r / d) < 0.01
        assert abs(subspace_overlap(u_true, u_true).chance - r / d) < 1e-12


class TestWaterFillKKT:
    def test_kkt_conditions_hold(self):
        # D = sum w_i 2^(-2 b_i): at optimum, active directions share a
        # common marginal (water level) and starved ones sit below it
        w = np.sort(RNG.uniform(0.01, 5.0, 12))[::-1]
        budget = 14.0
        alloc = water_fill(w, budget=budget)
        b = alloc.bits
        assert abs(b.sum() - budget) < 1e-9
        marg = w * 2.0 ** (-2 * b)  # ∝ -dD/db_i up to constant
        active = b > 1e-12
        assert active.any()
        lv = marg[active]
        assert lv.max() / lv.min() < 1 + 1e-6
        if (~active).any():
            assert marg[~active].max() <= lv.min() * (1 + 1e-9)

    def test_beats_or_matches_random_feasible_allocations(self):
        w = RNG.uniform(0.05, 3.0, 8)
        budget = 10.0
        alloc = water_fill(w, budget=budget)

        def distortion(bits):
            return float(np.sum(w * 2.0 ** (-2 * bits)))

        best = distortion(alloc.bits)
        for _ in range(300):
            raw = RNG.uniform(0, 1, 8)
            cand = raw / raw.sum() * budget
            assert best <= distortion(cand) + 1e-9


if __name__ == "__main__":
    pytest.main([__file__, "-q"])
