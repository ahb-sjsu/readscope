"""top_spectrum: agreement with the full decomposition, exact aggregates,
and refusal to extrapolate beyond computed directions."""

import numpy as np
import pytest

from readscope import spectrum_of, top_spectrum


def make_psd(d=96, r=6, seed=3):
    rng = np.random.default_rng(seed)
    v = np.linalg.qr(rng.standard_normal((d, d)))[0][:, :r]
    lam = np.array([10.0, 5.0, 2.0, 1.0, 0.5, 0.2])
    return v @ np.diag(lam) @ v.T


def test_matches_full_decomposition():
    S = make_psd()
    full = spectrum_of(S)
    top = top_spectrum(S, 4, seed=1)
    assert np.allclose(top.eigenvalues, full.eigenvalues[:4], rtol=1e-8)
    for i in range(4):
        overlap = abs(top.eigenvectors[:, i] @ full.eigenvectors[:, i])
        assert overlap > 1 - 1e-8


def test_aggregates_are_exact_without_full_decomposition():
    S = make_psd()
    full = spectrum_of(S)
    top = top_spectrum(S, 2)
    assert top.trace == pytest.approx(float(full.eigenvalues.sum()))
    assert top.effective_rank == pytest.approx(full.effective_rank)


def test_energy_rank_refuses_beyond_coverage():
    S = make_psd()
    top = top_spectrum(S, 2)  # top-2 carry 15/18.7 ~ 0.80
    assert top.energy_rank(0.5) >= 1  # answerable inside coverage
    with pytest.raises(ValueError, match="more directions"):
        top.energy_rank(0.99)


def test_input_validation():
    with pytest.raises(ValueError):
        top_spectrum(np.eye(4), 0)
    with pytest.raises(ValueError):
        top_spectrum(np.ones((3, 4)), 1)
