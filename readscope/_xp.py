"""Backend dispatch: numpy by default, CuPy when the caller's arrays are
CuPy.

The instrument remains numpy-only in every default path — no import of
anything beyond numpy happens unless the *caller* hands in GPU arrays,
in which case the linear algebra runs where the data already lives
(pinv, QR, eigh at large d are the costs worth moving; see
``top_spectrum`` for the iterative option).

Reproducibility rule: every random draw is made with numpy, in the same
order as the pure-numpy path, and transferred afterward. A GPU run and a
CPU run of the same seed therefore probe the same directions exactly.

The budget law is untouched by any of this: the cliff at ``k = d`` is a
property of consumer calls, not FLOPs (it is a theorem — see
PRINCIPLES.md P3), and a faster backend buys speed, never admission.
"""

from __future__ import annotations

import numpy as np


def of(x):
    """The array namespace of ``x``: cupy for CuPy arrays, else numpy."""
    if type(x).__module__.split(".")[0] == "cupy":
        import cupy

        return cupy
    return np


def to_xp(xp, a):
    """Move a numpy array to ``xp`` (no-op when ``xp`` is numpy)."""
    return a if xp is np else xp.asarray(a)
