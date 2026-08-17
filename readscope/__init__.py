"""readscope: an oscilloscope and spectrum analyzer for model consumers.

Point it at a consumer, get back what that consumer actually reads.

The instrument recovers the read operator from the consumer's outputs alone,
shows you where its sensitivity sits, and spends a bit budget against that
spectrum. You do not have to accept any account of what a consumer *is* to
use it, in the same way nobody has to accept an account of what voltage is to
read a trace.

Status: this instrument does not yet have a specification. See SPEC.md for
what is measured, what is not, and what the calibration program has to
produce before any accuracy claim here is worth the name.
"""

from readscope.allocate import Allocation, uniform_allocation, water_fill
from readscope.diagnostics import split_half_overlap, step_response
from readscope.loading import (
    LoadingCorrection,
    LoadingReading,
    fit_loading_correction,
    interpolate_distribution,
    loading_null_floor,
    probe_loading,
)
from readscope.metrics import (
    OverlapReading,
    chance_overlap,
    consumer_distortion,
    subspace_overlap,
)
from readscope.probe import (
    ProbeResult,
    blind_probe,
    debias_sketch,
    jacobian_probe,
    retrieval_margin_gradient,
)
from readscope.quotient import (
    displacement_decomposition,
    tangential_fraction,
    tangential_fractions,
)
from readscope.regimes import (
    Applicability,
    Regime,
    applicability,
    assert_probeable,
    decay_sensitivity,
    differential_fraction,
    routing_margins,
)
from readscope.spectrum import Spectrum, TopSpectrum, spectrum_of, top_spectrum
from readscope.stats import BootstrapCI, bootstrap_ci, spearman

__version__ = "0.2.0"

__all__ = [
    "Allocation",
    "Applicability",
    "LoadingCorrection",
    "LoadingReading",
    "OverlapReading",
    "ProbeResult",
    "Regime",
    "Spectrum",
    "__version__",
    "applicability",
    "assert_probeable",
    "blind_probe",
    "chance_overlap",
    "consumer_distortion",
    "debias_sketch",
    "decay_sensitivity",
    "differential_fraction",
    "displacement_decomposition",
    "fit_loading_correction",
    "interpolate_distribution",
    "loading_null_floor",
    "jacobian_probe",
    "probe_loading",
    "retrieval_margin_gradient",
    "routing_margins",
    "spectrum_of",
    "top_spectrum",
    "TopSpectrum",
    "BootstrapCI",
    "bootstrap_ci",
    "spearman",
    "split_half_overlap",
    "step_response",
    "subspace_overlap",
    "tangential_fraction",
    "tangential_fractions",
    "uniform_allocation",
    "water_fill",
]
