#!/usr/bin/env python3
"""Strict suite-coverage validator for LIBERO result progress files.

A result file is valid only if the set of (suite, task_id) keys exactly
matches the expected 4 suites x 10 tasks grid: duplicates and missing tasks
are both hard failures. Counting rows is explicitly not enough.

Usage:
  python scripts/validate_suite_coverage.py <progress.txt> [more.txt ...]

Exit 0 = all files pass; exit 1 = any failure (details printed).
"""
import re
import sys
from pathlib import Path

SUITES = ("spatial", "object", "goal", "10")
EXPECTED = {f"{s}_t{t}" for s in SUITES for t in range(10)}


def validate(path: Path) -> bool:
    keys = []
    for line in path.read_text(encoding="utf-8").splitlines():
        m = re.search(r"^(?:\S+\s+\S+\s+)?\S+\s+(spatial|object|goal|10)\s+(t\d+)\s+Success rate:", line)
        if m:
            keys.append(f"{m.group(1)}_{m.group(2)}")
    got = set(keys)
    dupes = sorted({k for k in keys if keys.count(k) > 1})
    missing = sorted(EXPECTED - got)
    extra = sorted(got - EXPECTED)
    ok = not dupes and not missing and not extra
    status = "PASS" if ok else "FAIL"
    print(f"[{status}] {path.name}: {len(keys)} rows, {len(got)} unique keys")
    if dupes:
        print(f"   duplicates: {dupes}")
    if missing:
        print(f"   missing:    {missing}")
    if extra:
        print(f"   unexpected: {extra}")
    return ok


def main() -> int:
    paths = [Path(a) for a in sys.argv[1:]]
    if not paths:
        print(__doc__)
        return 2
    ok = all(validate(p) for p in paths)
    print("ALL PASS" if ok else "COVERAGE FAILURES PRESENT")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
