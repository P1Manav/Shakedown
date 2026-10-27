"""Check GitHub Pages deployment status and update SUBMISSION_CHECKLIST.md."""

from __future__ import annotations

import re
import sys
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

PAGES_URL = "https://p1manav.github.io/Shakedown/"
CHECKLIST_PATH = Path(__file__).resolve().parent.parent / "SUBMISSION_CHECKLIST.md"


def check_pages_url() -> tuple[int, str]:
    try:
        req = urllib.request.Request(
            PAGES_URL,
            headers={"User-Agent": "Shakedown-Pages-Check/0.4.0"},
        )
        with urllib.request.urlopen(req, timeout=15) as resp:
            return resp.status, "OK"
    except urllib.error.HTTPError as err:
        return err.code, str(err.reason)
    except Exception as exc:
        return 0, str(exc)


def update_checklist(status_code: int, reason: str) -> None:
    if not CHECKLIST_PATH.exists():
        print(f"Error: {CHECKLIST_PATH} not found", file=sys.stderr)
        return

    content = CHECKLIST_PATH.read_text(encoding="utf-8")
    now_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ")

    is_live = (status_code == 200)
    box = "[x]" if is_live else "[ ]"
    status_note = (
        f"(HTTP {status_code} verified live at {now_utc})"
        if is_live
        else f"(HTTP {status_code} {reason}; pending GitHub Pages activation in repo settings: Settings -> Pages -> Deploy from branch -> master /docs)"
    )

    pattern = re.compile(
        r"- \[[ x]\] \*\*Working live report link:\*\* .*?(?=\n-|\Z)",
        re.DOTALL,
    )
    replacement = (
        f"- {box} **Working live report link:** [{PAGES_URL}]({PAGES_URL}) {status_note}"
    )

    if pattern.search(content):
        new_content = pattern.sub(replacement, content)
    else:
        new_content = content + f"\n{replacement}\n"

    CHECKLIST_PATH.write_text(new_content, encoding="utf-8")
    print(f"Updated SUBMISSION_CHECKLIST.md with Pages status: {status_code} ({reason})")


def main() -> int:
    code, reason = check_pages_url()
    print(f"Checked {PAGES_URL} -> Status {code} ({reason})")
    update_checklist(code, reason)
    return 0 if code == 200 else 1


if __name__ == "__main__":
    sys.exit(main())
