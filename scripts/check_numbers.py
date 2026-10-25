"""Cross-check every number in the write-ups against results/ (Definition of Done item 4).

For README.md, paper/*.tex, RESUME_BULLETS.md, FINAL_REVIEW.md and DEMO.md, every *decimal* number
or percentage (e.g. 0.73, -2.1, 12.5 %) outside code blocks must equal, at its displayed precision,
some numeric value found in results/** (JSON, CSV, MD). Integers are treated as configuration
values (budgets, sample rates, counts) and not checked. Lines may opt out with
``<!-- nocheck -->`` / ``% nocheck`` (used only for citations' years/volumes and the like).

Exit code 1 lists every unbacked number.
"""

from __future__ import annotations

import bisect
import csv
import json
import math
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DOCS = ["README.md", "RESUME_BULLETS.md", "FINAL_REVIEW.md", "DEMO.md", *sorted(str(p.relative_to(REPO))
        for p in (REPO / "paper").glob("*.tex"))]
NUM = re.compile(r"(?<![\w.])[-−]?\d+\.\d+(?![\w.])|(?<![\w.])[-−]?\d+(?:\.\d+)?(?=\s?%)")


def _walk(o, out: list[float]) -> None:
    if isinstance(o, dict):
        for v in o.values():
            _walk(v, out)
    elif isinstance(o, list):
        for v in o:
            _walk(v, out)
    elif isinstance(o, bool):
        return
    elif isinstance(o, (int, float)) and math.isfinite(o):
        out.append(float(o))


def result_values() -> list[float]:
    vals: list[float] = []
    for p in (REPO / "results").rglob("*"):
        if p.suffix == ".json":
            try:
                _walk(json.loads(p.read_text()), vals)
            except Exception:
                pass
        elif p.suffix == ".csv":
            with open(p) as f:
                for row in csv.reader(f):
                    for c in row:
                        try:
                            x = float(c)
                            if math.isfinite(x):
                                vals.append(x)
                        except ValueError:
                            pass
        elif p.suffix == ".md":
            vals += [float(m.replace("−", "-")) for m in NUM.findall(p.read_text())]
    return vals


def main() -> int:
    vals = sorted(set(result_values()))

    def near(x: float, tol: float) -> bool:
        i = bisect.bisect_left(vals, x - tol)
        return i < len(vals) and vals[i] <= x + tol

    bad = []
    for d in DOCS:
        p = REPO / d
        if not p.exists():
            continue
        in_code = False
        for ln, line in enumerate(p.read_text().splitlines(), 1):
            if line.strip().startswith("```"):
                in_code = not in_code
                continue
            if in_code or "nocheck" in line:
                continue
            line = re.sub(r"`[^`]*`|\]\([^)]*\)|\\cite\{[^}]*\}|\\ref\{[^}]*\}|\\label\{[^}]*\}", "", line)
            for m in NUM.finditer(line):
                tok = m.group(0)
                pct = line[m.end():m.end() + 2].lstrip().startswith("%")
                x = float(tok.replace("−", "-"))
                nd = len(tok.split(".")[1]) if "." in tok else 0
                ok = False
                for c, extra in ((x, 0), (x / 100, 2)) if pct else ((x, 0),):
                    tol = 0.5 * 10 ** -(nd + extra) + 1e-9
                    if near(c, tol) or near(-c, tol):
                        ok = True
                if not ok:
                    bad.append(f"{d}:{ln}: {tok}{'%' if pct else ''}  | {line.strip()[:110]}")
    if bad:
        print("UNBACKED numbers (not found in results/):")
        print("\n".join(bad))
        return 1
    print(f"check-numbers OK ({len(vals)} result values indexed)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
