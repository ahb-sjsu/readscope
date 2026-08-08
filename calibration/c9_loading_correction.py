#!/usr/bin/env python3
"""C-9, the loading correction, third attempt. Declared before it runs.

Two previous attempts produced confident numbers out of nothing. C-7b's
correction saturated against its output clip and reported a 92 percent error
reduction. C-8's D5 graded a consumer whose read subspace cannot move, so
every reading was exactly 1.000 and an identity correction scored a perfect
zero. Both passed. Neither measured anything.

The rule those two produced is the first thing this sweep obeys:
**before believing a bar of the form "X predicts Y", require a prior bar that
Y varies.** E0 below is that bar and it runs before anything else is
believed.

**The consumer is chosen so that loading can bite.** F-1 established that
probe loading cannot degrade subspace recovery when the read subspace is the
same everywhere. Here the consumer is
``C(x) = sum_j w_j tanh(b_j . x / tau)`` over a planted basis of rank 12,
graded at rank 4. The gradient is ``sum_j w_j sech^2(b_j . x / tau) b_j``, so
the *weighting* of the planted directions depends on where ``x`` sits: probe
points that saturate some gates drop those directions out of the recovered
operator entirely. Grading at 4 of 12 means which directions dominate is a
property of the probing distribution, which is exactly the mechanism a
loading correction has to predict.

**The axis is the dimensionless one from C-8**, since C-7b showed a
correction fitted against raw Jeffreys cannot transfer across dimensions.

**Both estimators are exact**, at ``k = d`` coordinate differences, so the
only thing varying between the truth and a reading is *where* the probe
looked. Estimator noise is removed from the question deliberately.

Declared bars, in the order they must be believed:

  E0  separation, first   in the fit family, mean resolution at the highest
                          loading is below that at the lowest by at least
                          0.15. **If this fails the sweep is void and no
                          other bar means anything**, because there is no
                          degradation for a correction to predict. This is
                          the bar C-7b and C-8 both lacked.
  E1  monotone            in the fit family, resolution is non-increasing as
                          loading rises, allowing one inversion.
  E2  test family varies  every test dimension also shows at least 0.15 of
                          separation, so the transfer is evaluated where
                          something is happening.
  E3  the question        the correction fitted at dimension 32 predicts the
                          reading at 16, 64 and 128 with mean absolute error
                          below 0.10, evaluated only on readings inside its
                          fitted domain, with at least 70 percent in domain.
  E4  beats the null      the correction reduces mean absolute error by at
                          least 50 percent against reporting the raw reading,
                          **and fewer than half its outputs are pinned at
                          1.0**, so it cannot pass by saturating the way
                          C-7b did.
  E5  census              every declared cell computed and reported.

Runs on Atlas. Numpy only.
"""

from __future__ import annotations

import json
import os
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from readscope import (  # noqa: E402
    blind_probe,
    fit_loading_correction,
    interpolate_distribution,
    probe_loading,
    subspace_overlap,
)

DIMS = [16, 32, 64, 128]
FIT_DIM = 32
PLANTED_RANK = 12
GRADED_RANK = 4
ALPHAS = [0.0, 0.25, 0.5, 0.75, 1.0]
SEEDS = [0, 1, 2]
N_PROBE_POINTS = 96
N_DIST_SAMPLES_PER_DIM = 120
TAU = 1.0
DECAY = 0.85
EPS = 1e-3

SEPARATION_BAR = 0.15
MAE_BAR = 0.10
IN_DOMAIN_BAR = 0.70
REDUCTION_BAR = 0.50
SATURATION_BAR = 0.50


def planted_consumer(basis, weights):
    """Read subspace of rank 12 whose *weighting* moves with position."""

    def C(x):
        return float(np.tanh((basis.T @ x) / TAU) @ weights)

    return C


def cell(dim, seed):
    """One consumer, its truth, and its loading ladder."""
    rng = np.random.default_rng(seed * 9173 + dim)
    basis = np.linalg.qr(rng.standard_normal((dim, PLANTED_RANK)))[0]
    weights = DECAY ** np.arange(PLANTED_RANK)
    cons = planted_consumer(basis, weights)

    n = N_DIST_SAMPLES_PER_DIM * dim
    # activations are anisotropic, so different regions saturate different
    # gates and the top-4 subspace genuinely depends on where one probes
    scale = np.concatenate(
        [np.full(dim // 2, 2.5), np.full(dim - dim // 2, 0.4)]
    )
    activation = rng.standard_normal((n, dim)) * scale
    far = rng.standard_normal((n, dim)) * scale[::-1] + 2.0

    truth = blind_probe(
        cons, activation[:N_PROBE_POINTS], eps=EPS, check_regime=False
    ).read_subspace(GRADED_RANK)

    rungs = []
    for alpha in ALPHAS:
        pts = interpolate_distribution(
            far,
            activation,
            alpha,
            rng=np.random.default_rng(seed * 31 + 7),
            n_samples=n,
        )
        load = probe_loading(pts, activation).loading
        got = blind_probe(
            cons, pts[:N_PROBE_POINTS], eps=EPS, check_regime=False
        ).read_subspace(GRADED_RANK)
        rungs.append(
            {
                "alpha": alpha,
                "loading": float(load),
                "reading": subspace_overlap(got, truth).resolution,
            }
        )
    return rungs


def mean_rungs(dim):
    per_seed = [cell(dim, s) for s in SEEDS]
    out = []
    for i, alpha in enumerate(ALPHAS):
        out.append(
            {
                "alpha": alpha,
                "loading": float(np.mean([r[i]["loading"] for r in per_seed])),
                "reading": float(np.mean([r[i]["reading"] for r in per_seed])),
            }
        )
    return out


def separation(rungs):
    """Reading at the loosest loading minus reading at the tightest."""
    hi = max(rungs, key=lambda r: r["loading"])
    lo = min(rungs, key=lambda r: r["loading"])
    return lo["reading"] - hi["reading"]


def main() -> int:
    per_dim = {}
    for d in DIMS:
        per_dim[str(d)] = mean_rungs(d)
        sep = separation(per_dim[str(d)])
        print(
            f"  d={d:<5} "
            + "  ".join(
                f"L{r['loading']:6.3f} r{r['reading']:.3f}"
                for r in per_dim[str(d)]
            )
            + f"   separation {sep:+.3f}",
            flush=True,
        )

    fit = per_dim[str(FIT_DIM)]
    fit_sep = separation(fit)
    e0 = bool(fit_sep >= SEPARATION_BAR)

    inversions = 0
    ordered = sorted(fit, key=lambda r: r["loading"])
    for i in range(len(ordered) - 1):
        if ordered[i]["reading"] < ordered[i + 1]["reading"] - 1e-9:
            inversions += 1
    e1 = bool(inversions <= 1)

    test_dims = [d for d in DIMS if d != FIT_DIM]
    seps = {str(d): separation(per_dim[str(d)]) for d in test_dims}
    e2 = bool(all(v >= SEPARATION_BAR for v in seps.values()))

    corr = fit_loading_correction(
        [r["loading"] for r in fit],
        [r["reading"] for r in fit],
        truth=1.0,
        source=f"c9 fit at dim {FIT_DIM}",
    )

    entries, errs, raw_errs, total, inside_n, pinned = [], [], [], 0, 0, 0
    for d in test_dims:
        for r in per_dim[str(d)]:
            total += 1
            inside = corr.in_domain(r["loading"])
            inside_n += int(inside)
            e = {
                "dim": d,
                "alpha": r["alpha"],
                "loading": r["loading"],
                "reading": r["reading"],
                "in_domain": inside,
            }
            if inside:
                pred = corr.expected_attenuation(r["loading"])
                corrected = corr.correct(r["reading"], r["loading"])
                e["predicted_attenuation"] = pred
                e["corrected"] = corrected
                e["abs_error"] = abs(r["reading"] - pred)
                e["abs_error_raw"] = abs(1.0 - r["reading"])
                errs.append(e["abs_error"])
                raw_errs.append(e["abs_error_raw"])
                if abs(corrected - 1.0) < 1e-9:
                    pinned += 1
            entries.append(e)

    frac_in = inside_n / max(total, 1)
    mae = float(np.mean(errs)) if errs else float("nan")
    mae_raw = float(np.mean(raw_errs)) if raw_errs else float("nan")
    reduction = (mae_raw - mae) / mae_raw if mae_raw > 0 else 0.0
    frac_pinned = pinned / max(len(errs), 1)

    e3 = bool(errs and mae < MAE_BAR and frac_in >= IN_DOMAIN_BAR)
    e4 = bool(reduction >= REDUCTION_BAR and frac_pinned < SATURATION_BAR)
    e5 = bool(
        len(per_dim) == len(DIMS) and total == len(test_dims) * len(ALPHAS)
    )

    bars = {
        "E0_separation_in_fit_family": e0,
        "E1_monotone": e1,
        "E2_test_family_varies": e2,
        "E3_correction_predicts": e3,
        "E4_beats_null_without_saturating": e4,
        "E5_census": e5,
    }
    verdict = "PASS" if all(bars.values()) else "FAIL"
    if not e0:
        verdict = "VOID"

    record = {
        "schema": "readscope-c9-loading-correction-v1",
        "declared": {
            "dims": DIMS,
            "fit_dim": FIT_DIM,
            "planted_rank": PLANTED_RANK,
            "graded_rank": GRADED_RANK,
            "alphas": ALPHAS,
            "seeds": SEEDS,
            "n_probe_points": N_PROBE_POINTS,
            "tau": TAU,
            "decay": DECAY,
            "eps": EPS,
            "separation_bar": SEPARATION_BAR,
            "mae_bar": MAE_BAR,
            "in_domain_bar": IN_DOMAIN_BAR,
            "reduction_bar": REDUCTION_BAR,
            "saturation_bar": SATURATION_BAR,
            "axis": "dimensionless loading from C-8",
            "estimator": "exact, so only the probing region varies",
        },
        "per_dim": per_dim,
        "fit_separation": fit_sep,
        "test_separations": seps,
        "fit_inversions": inversions,
        "correction": corr.to_dict(),
        "transfer": entries,
        "summary": {
            "in_domain_fraction": frac_in,
            "mae_corrected": mae,
            "mae_raw": mae_raw,
            "error_reduction": reduction,
            "fraction_pinned_at_one": frac_pinned,
        },
        "bars": bars,
        "verdict": {"value": verdict, "computed_from": sorted(bars)},
        "runtime": {
            "generated_utc": datetime.now(timezone.utc).isoformat(),
            "python": sys.version,
            "numpy": np.__version__,
            "platform": platform.platform(),
            "hostname": platform.node(),
            "code_commit": os.environ.get("CODE_COMMIT", "unknown"),
        },
    }
    out = (
        Path(__file__).resolve().parent
        / "records"
        / "c9-loading-correction.json"
    )
    out.write_text(json.dumps(record, indent=2, sort_keys=True))

    print(f"\nfit separation {fit_sep:+.3f}   test {seps}")
    print(
        f"in domain {inside_n}/{total} ({frac_in * 100:.0f}%)   "
        f"MAE {mae:.4f} vs raw {mae_raw:.4f}   "
        f"reduction {reduction * 100:.1f}%   pinned {frac_pinned * 100:.0f}%"
    )
    for k in sorted(bars):
        print(f"{k:<34} {'PASS' if bars[k] else 'FAIL'}")
    print("VERDICT", verdict)
    print(out)
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
