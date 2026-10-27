"""Verify that no forbidden patterns are tracked in git."""

from __future__ import annotations

import re
import subprocess
import sys

PROHIBITED_PATTERNS = [
    re.compile(r"__pycache__"),
    re.compile(r"\.egg-info"),
    re.compile(r"\.ruff_cache"),
    re.compile(r"^runs/"),
    re.compile(r"^\.env$"),
]


def check_git_tracked_files() -> bool:
    try:
        proc = subprocess.run(
            ["git", "ls-files"],
            capture_output=True,
            text=True,
            check=True,
        )
    except Exception as exc:
        print(f"Error running git ls-files: {exc}", file=sys.stderr)
        return False

    files = [f.strip() for f in proc.stdout.splitlines() if f.strip()]
    violations = []
    for f in files:
        normalized = f.replace("\\", "/")
        for pattern in PROHIBITED_PATTERNS:
            if pattern.search(normalized):
                violations.append((normalized, pattern.pattern))

    if violations:
        print("ERROR: Forbidden files are tracked in git:", file=sys.stderr)
        for f, pat in violations:
            print(f"  - {f} (matched prohibited pattern: {pat})", file=sys.stderr)
        return False

    print(f"Clean git repository: all {len(files)} tracked files verified.")
    return True


if __name__ == "__main__":
    if not check_git_tracked_files():
        sys.exit(1)
    sys.exit(0)
