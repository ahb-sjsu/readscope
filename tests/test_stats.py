"""readscope.stats: the extracted sealed-run statistics."""

import numpy as np
import pytest

from readscope.stats import BootstrapCI, bootstrap_ci, spearman

RNG = np.random.default_rng(9)


class TestBootstrap:
    def test_covers_the_mean_of_a_known_distribution(self):
        data = RNG.normal(3.0, 1.0, 400)
        r = bootstrap_ci(
            len(data),
            lambda idx: float(data[idx].mean()),
            rng=np.random.default_rng(1),
        )
        assert isinstance(r, BootstrapCI)
        assert r.lo < 3.0 < r.hi
        assert r.hi - r.lo < 0.5
        assert abs(r.point - data.mean()) < 1e-12

    def test_matches_the_hand_rolled_pattern(self):
        # the exact resample-the-units shape the sealed runners used
        vals = RNG.uniform(0, 1, 200) < 0.3
        r = bootstrap_ci(
            len(vals),
            lambda i: float(vals[i].mean()),
            n_boot=2000,
            rng=np.random.default_rng(2),
        )
        assert r.lo < vals.mean() < r.hi

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            bootstrap_ci(0, lambda i: 0.0)


class TestSpearman:
    def test_perfect_monotone(self):
        x = np.arange(10.0)
        assert spearman(x, x**3) == pytest.approx(1.0)
        assert spearman(x, -np.exp(x)) == pytest.approx(-1.0)

    def test_known_value_with_ties(self):
        # hand-computed: ranks with average ties
        x = np.array([1.0, 2.0, 2.0, 3.0])
        y = np.array([10.0, 20.0, 30.0, 40.0])
        # rx = [1, 2.5, 2.5, 4], ry = [1, 2, 3, 4]
        rx = np.array([1, 2.5, 2.5, 4.0])
        ry = np.array([1, 2, 3, 4.0])
        expected = float(np.corrcoef(rx, ry)[0, 1])
        assert spearman(x, y) == pytest.approx(expected)

    def test_constant_input_is_nan_not_crash(self):
        assert np.isnan(spearman(np.ones(5), np.arange(5.0)))

    def test_agrees_with_scipy_when_available(self):
        scipy_stats = pytest.importorskip("scipy.stats")
        for _ in range(20):
            x = RNG.normal(size=30)
            y = RNG.normal(size=30) + 0.5 * x
            ours = spearman(x, y)
            theirs = float(scipy_stats.spearmanr(x, y).statistic)
            assert ours == pytest.approx(theirs, abs=1e-12)
