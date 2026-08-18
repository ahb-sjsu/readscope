#!/usr/bin/env python3
"""CI census check: SPEC.md must not drift from calibration/records/.

Specification discipline is this project's central claim, so drift in
the summary document is a build failure, not a doc chore. Three checks:

1. **Citation coverage** — every record JSON in `calibration/records/`
   must be cited in SPEC.md by its calibration tag (``c3b-...json`` →
   ``C-3b``) or listed in EXEMPT below with a reason.
2. **No duplicated section headings** — the stale-merge signature that
   shipped a doubled "Accuracy" section.
3. **Stale-phrase tripwires** — phrases that were once wrong and must
   not return (each was individually shipped and individually fixed).

    python tools/check_spec_census.py
"""

from __future__ import annotations

import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

EXEMPT = {
    "README.md": "records index, not a record",
    "rightsize-extraction.json": "infra sizing note, not a calibration",
    "c15-shakedown.json": "shakedown, no evidential weight until C-15 seals",
    "c16-shakedown.json": "shakedown, no evidential weight until C-16 seals",
    "op3-shakedown.json": "OP3 v1-line shakedown, no weight until PREREG-OP3 seals",
}

TRIPWIRES = [
    "Three real-model points",
    "That is the entire real-model evidence base",
    "It claims three measurements",
]


def tag_of(name: str) -> str | None:
    m = re.match(r"c(\d+[a-z]?)-", name)
    return f"C-{m.group(1)}" if m else None


def main() -> int:
    spec = (ROOT / "SPEC.md").read_text(encoding="utf-8")
    failures = []

    for rec in sorted((ROOT / "calibration" / "records").glob("*")):
        if rec.name in EXEMPT:
            continue
        tag = tag_of(rec.name)
        cited = (tag is not None and tag in spec) or rec.name in spec
        if not cited:
            failures.append(
                f"record {rec.name} ({tag}) is never cited in SPEC.md"
            )

    heads = Counter(
        ln.strip() for ln in spec.splitlines() if ln.startswith("## ")
    )
    for h, c in heads.items():
        if c > 1:
            failures.append(f"duplicated heading ({c}x): {h}")

    for phrase in TRIPWIRES:
        if phrase in spec:
            failures.append(f"stale phrase returned: {phrase!r}")

    if failures:
        print("SPEC census check FAILED:")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("SPEC census check passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
