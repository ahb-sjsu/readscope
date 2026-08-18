#!/usr/bin/env python3
"""C-16, the integer-codebook departure. Shakedown mode until the
declaration seals.

`water_fill` optimizes the continuous surrogate
``D = sum_i w_i 2^(-2 b_i)`` with real-valued per-direction bits. Real
codecs cannot spend 2.7 bits on a direction — they spend an integer
number, or draw from a finite codebook. The `water_fill` scope note
(2026-08-17) named this departure as uncalibrated: *how much distortion
does the integer constraint cost over the continuous optimum, and does
a smarter integer allocation recover it?*

This calibration measures the **excess distortion ratio**
``D_integer / D_continuous >= 1`` for four rounding strategies against
the continuous water-fill floor, over planted read spectra:

- **round**   nearest-integer of the continuous bits (naive);
- **floor**   floor the continuous bits, drop the freed budget
              (a fixed-budget lower bound on cleverness);
- **greedy**  spend the integer budget one bit at a time onto the
              direction with the steepest marginal distortion drop
              (the reverse-water-filling integer optimum for the
              per-direction exponential model);
- **ceil-topk** give ceil to the highest-sensitivity directions until
              the integer budget is met (a common codec heuristic).

Greedy is the integer optimum for this separable convex surrogate;
the others bound how much a real codec leaves on the table by not
being greedy. The interesting quantities: how the excess decays as
the budget grows (more bits -> integer grain matters less), and
whether greedy's excess ever exceeds a small constant.

The decision rule and any bars live in DECLARATION-C16.md and bind
only once it seals; this script runs with --shakedown and writes a
record with no evidential weight.

    python calibration/c16_integer_codebook.py --shakedown
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from readscope import water_fill  # noqa: E402

DIMS = [16, 32, 64]
SPECTRA = {
    "geom": lambda d: 0.7 ** np.arange(d),  # geometric decay
    "flat": lambda d: np.ones(d),  # whitened, no structure
    "lowrank": lambda d: np.concatenate(  # a few strong directions
        [np.ones(max(1, d // 8)), 0.02 * np.ones(d - max(1, d // 8))]
    ),
}
BITS_PER_DIM = [0.25, 0.5, 1.0, 2.0, 4.0]
SEEDS = [0, 1, 2, 3, 4]
HERE = Path(__file__).resolve().parent
OUT = HERE / "records" / "c16-shakedown.json"


def distortion(w: np.ndarray, bits: np.ndarray) -> float:
    return float(np.sum(w * np.power(2.0, -2.0 * bits)))


def greedy_integer(w: np.ndarray, budget_bits: int) -> np.ndarray:
    """The integer optimum for the separable surrogate: repeatedly add
    one bit where it drops distortion most. A bit on direction i cuts
    its term from w_i 2^(-2b) to w_i 2^(-2b-2), a drop of
    (3/4) w_i 2^(-2b); pick the max each step."""
    bits = np.zeros_like(w, dtype=float)
    for _ in range(int(budget_bits)):
        drop = 0.75 * w * np.power(2.0, -2.0 * bits)
        bits[int(np.argmax(drop))] += 1.0
    return bits


def ceil_topk(cont: np.ndarray, w: np.ndarray, budget_bits: int) -> np.ndarray:
    """Ceil the continuous bits on the highest-sensitivity directions
    until the integer budget is spent -- a common codec heuristic."""
    order = np.argsort(-w)
    bits = np.floor(cont).astype(float)
    spent = int(bits.sum())
    for i in order:
        if spent >= budget_bits:
            break
        if cont[i] > bits[i]:
            bits[i] += 1.0
            spent += 1
    return bits


def cell(w: np.ndarray, bpd: float):
    d = w.size
    budget = bpd * d
    cont = water_fill(w, budget=budget)
    d_cont = cont.distortion
    if d_cont <= 0:
        return None
    ib = int(round(budget))
    strategies = {
        "round": np.round(cont.bits),
        "floor": np.floor(cont.bits),
        "greedy": greedy_integer(w, ib),
        "ceil_topk": ceil_topk(cont.bits, w, ib),
    }
    return {
        "d": d,
        "bits_per_dim": bpd,
        "excess": {
            name: round(distortion(w, b) / d_cont, 4)
            for name, b in strategies.items()
        },
        "int_budget_used": {
            name: int(b.sum()) for name, b in strategies.items()
        },
        # a strategy is budget-feasible iff it spends no more than the
        # integer budget; naive round overspends and is NOT comparable
        # on excess (it cheats the constraint).
        "feasible": {
            name: bool(int(b.sum()) <= ib) for name, b in strategies.items()
        },
        "int_budget": ib,
        "continuous_budget": round(budget, 2),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shakedown", action="store_true")
    args = ap.parse_args()
    dims = [32] if args.shakedown else DIMS
    seeds = [0] if args.shakedown else SEEDS

    rows = []
    for d in dims:
        for sname, spec in SPECTRA.items():
            base = spec(d)
            for seed in seeds:
                rng = np.random.default_rng(seed * 131 + d)
                w = base * rng.uniform(
                    0.5, 1.5, d
                )  # jitter the planted spectrum
                for bpd in BITS_PER_DIM:
                    c = cell(w, bpd)
                    if c is None:
                        continue
                    c.update({"spectrum": sname, "seed": seed})
                    rows.append(c)

    # interior evidence, printed
    print("EXCESS DISTORTION RATIO (D_integer / D_continuous), d=32 geom")
    print(
        "  round is NOT budget-feasible (overspends); shown for the "
        "record, not comparable"
    )
    print(
        f"{'bits/dim':>9}  {'round*':>7} {'floor':>7} {'greedy':>7} "
        f"{'ceil_topk':>9}   round-overspend"
    )
    for bpd in BITS_PER_DIM:
        cs = [
            r
            for r in rows
            if r["d"] == 32
            and r["spectrum"] == "geom"
            and r["bits_per_dim"] == bpd
        ]
        if not cs:
            continue
        med = {
            k: float(np.median([c["excess"][k] for c in cs]))
            for k in ("round", "floor", "greedy", "ceil_topk")
        }
        over = float(
            np.median(
                [c["int_budget_used"]["round"] - c["int_budget"] for c in cs]
            )
        )
        print(
            f"{bpd:>9}  {med['round']:>7.3f} {med['floor']:>7.3f} "
            f"{med['greedy']:>7.3f} {med['ceil_topk']:>9.3f}   "
            f"+{over:.0f} bits"
        )

    record = {
        "calibration": "C-16" + ("-shakedown" if args.shakedown else ""),
        "sealed": not args.shakedown,
        "generated": datetime.now(timezone.utc).isoformat(),
        "host": platform.node(),
        "constants": {
            "dims": dims,
            "spectra": list(SPECTRA),
            "bits_per_dim": BITS_PER_DIM,
            "seeds": seeds,
        },
        "rows": rows,
    }
    name = (
        "c16-shakedown.json" if args.shakedown else "c16-integer-codebook.json"
    )
    dest = HERE / "records" / name
    json.dump(record, open(dest, "w"), indent=1)
    print(f"-> {dest.relative_to(HERE.parent)}  (shakedown, no verdict)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
