"""Rule-based failure taxonomy. No model judges another model."""

from __future__ import annotations

import re

FAILURE_CLASSES = ["loop", "call-cap-hit", "malformed-tool-call", "gave-up",
                   "false-success", "timeout", "tampering", "invalid", "other"]

CLASS_LABELS = {
    "loop": "Loop",
    "call-cap-hit": "Call cap",
    "malformed-tool-call": "Malformed call",
    "gave-up": "Gave up",
    "false-success": "False success",
    "timeout": "Timeout",
    "tampering": "Test tampering",
    "invalid": "Invalid run",
    "other": "Other",
    "none": "Passed",
}


def _event_names(run: dict) -> list[str]:
    return [str(e.get("event", "")) for e in run.get("events", []) or []]


def _has_repeated_action(actions: list[str]) -> bool:
    for i in range(len(actions) - 2):
        if actions[i] == actions[i + 1] == actions[i + 2]:
            return True
    return False


def classify_run(run: dict, max_calls: int | None = None) -> str:
    if run.get("tamper_detected"):
        return "tampering"
    if run.get("invalid"):
        return "invalid"
    if run.get("success"):
        return "none"
    if run.get("timed_out"):
        return "timeout"

    text = ((run.get("error") or "") + " " + (run.get("test_output") or "")).lower()
    events = _event_names(run)
    if "call cap" in text or "budget" in text and "cap" in text:
        return "call-cap-hit"
    if max_calls and run.get("model_calls", 0) >= max_calls:
        return "call-cap-hit"

    if "tool_error" in events:
        error_events = [e for e in run.get("events", []) if e.get("event") == "tool_error"]
        if any(re.search(r"malformed|json|argument", str(e.get("error", "")).lower())
               for e in error_events):
            return "malformed-tool-call"

    if "loop_detected" in events or _has_repeated_action(run.get("actions") or []):
        return "loop"

    if run.get("claimed_success"):
        return "false-success"

    if run.get("final_message") and re.search(
            r"\b(unable|cannot|can't|could not|failed|stuck|give up|no such file)\b",
            run["final_message"], re.IGNORECASE):
        return "gave-up"

    if not run.get("actions"):
        return "gave-up"
    return "other"


def aggregate_taxonomy(runs: list[dict]) -> dict:
    counts = {cls: 0 for cls in FAILURE_CLASSES}
    for run in runs:
        cls = run.get("failure_class") or classify_run(run)
        if cls in counts:
            counts[cls] += 1
    failed = sum(1 for r in runs if r.get("invalid") or not r.get("success"))
    return {"counts": counts, "failed": failed, "total": len(runs),
            "other_share": counts["other"] / failed if failed else 0.0}