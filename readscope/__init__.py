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
from readscope.loading import (
    LoadingReading,
    interpolate_distribution,
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
    retrieval_margin_gradient,
)
from readscope.spectrum import Spectrum, spectrum_of

__version__ = "0.0.1"

__all__ = [
    "Allocation",
    "LoadingReading",
    "OverlapReading",
    "ProbeResult",
    "Spectrum",
    "__version__",
    "blind_probe",
    "chance_overlap",
    "consumer_distortion",
    "interpolate_distribution",
    "probe_loading",
    "retrieval_margin_gradient",
    "spectrum_of",
    "subspace_overlap",
    "uniform_allocation",
    "water_fill",
]
