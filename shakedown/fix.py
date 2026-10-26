"""Suggest a concrete fix for the top fragility, then re-measure it."""

from __future__ import annotations

import os
from pathlib import Path

from shakedown.claims import FAILURE_HINT
from shakedown.stats import two_proportion_test, wilson_interval

CHAOS_FIXES = {
    "truncate_output": (
        "Tool output may be truncated. Never infer pass or fail from incomplete output. "
        "After any truncated result, re-run the tests and check the final exit code before claiming success."),
    "empty_result": (
        "A tool may return an empty result. Treat empty output as missing evidence, not as success. "
        "Retry the tool or inspect the file another way before concluding."),
    "malformed_json": (
        "Tool output may be malformed or not valid JSON. Do not assume the operation succeeded. "
        "Re-run the tool, or verify the artifact directly, before claiming success."),
    "error_once": (
        "A tool can fail transiently once. Retry a failed tool call once before giving up, "
        "and never claim success on the basis of a failed call."),
    "stale_file": (
        "File reads may be stale. After writing a file, verify the new content directly "
        "or re-run the tests instead of trusting an older read."),
    "renamed_path": (
        "A path may be missing or renamed. List the directory and locate the real file "
        "before concluding the task is impossible."),
}

CHUNK_FIXES = {
    "verification": (
        "Verification is mandatory. Before you finish, run the test suite and read the final exit code. "
        "Claim success only when it is 0; otherwise keep working or say plainly that you are stuck."),
    "workflow": (
        "Follow a strict loop: list the directory, read the source and the tests, write the smallest fix, "
        "then run the tests. Only write_file changes the repository."),
    "identity": "Keep edits minimal and explain each action in one short sentence.",
}

TOOL_FIXES = {
    "run_tests": (
        "run_tests is the only source of truth. Call it after every edit, read the whole output, "
        "and check the exit code before claiming the task is complete."),
    "read_file": "Read the actual file before editing; never reconstruct contents from memory.",
    "write_file": "write_file replaces the whole file, so include the complete corrected contents.",
    "list_dir": "List the directory first to discover the real filenames before reading.",
}


def suggest_for_cell(cell: dict, task_chunks: dict) -> dict:
    params = cell.get("params") or {}
    if cell.get("kind") == "chaos":
        fault = params.get("fault", "")
        text = CHAOS_FIXES.get(fault, CHAOS_FIXES["error_once"])
        return {"target": "system prompt", "title": f"Handle {fault}",
                "text": text, "fault": fault}
    if cell.get("kind") == "ablation":
        chunk = params.get("chunk", "")
        info = task_chunks.get(chunk, {})
        if info.get("kind") == "tool_description":
            text = TOOL_FIXES.get(info.get("tool", ""), TOOL_FIXES["run_tests"])
            return {"target": f"tool description: {info.get('tool')}",
                    "title": f"Strengthen {info.get('tool')} description", "text": text}
        text = CHUNK_FIXES.get(chunk, CHUNK_FIXES["verification"])
        return {"target": "system prompt", "title": f"Reinforce {chunk}", "text": text}
    return {"target": "system prompt", "title": "Verify before claiming success",
            "text": CHAOS_FIXES["truncate_output"]}


def rank_fix_first(cells: list[dict], baseline_rate: float, task_chunks: dict) -> list[dict]:
    ranked = []
    for cell in cells:
        if not cell.get("n"):
            continue
        verdict = cell.get("verdict")
        delta_points = cell.get("delta_points", 0.0)
        false_rate = (cell.get("claims") or {}).get("false_claim_rate", 0.0)
        score = 0.0
        reasons = []
        if verdict == "critical":
            score += max(0.0, -delta_points) + 15
            reasons.append(f"{abs(delta_points):.0f} pt drop")
        if false_rate >= 0.10 and cell.get("significant"):
            score += false_rate * 45
            reasons.append(f"{false_rate * 100:.0f}% false claims")
        if (cell.get("cost") or {}).get("cost_risk"):
            score += 6
            reasons.append("cost per success up")
        if score <= 0:
            continue
        suggestion = suggest_for_cell(cell, task_chunks)
        ranked.append({
            "cell_id": cell["id"], "label": cell["label"], "kind": cell.get("kind"),
            "verdict": verdict,
            "score": round(score, 2), "delta_points": delta_points,
            "p_adjusted": cell.get("p_adjusted"),
            "false_claim_rate": false_rate,
            "reasons": reasons,
            "suggestion": suggestion,
        })
    ranked.sort(key=lambda item: item["score"], reverse=True)
    return ranked[:6]


def _verdict(original: dict, fixed: dict, alpha: float = 0.05) -> dict:
    o_ci = wilson_interval(original["k"], original["n"])
    f_ci = wilson_interval(fixed["k"], fixed["n"])
    cmp = two_proportion_test(fixed["k"], fixed["n"], original["k"], original["n"])
    non_overlap = bool(f_ci.low > o_ci.high or o_ci.low > f_ci.high)
    lift = (fixed["k"] / fixed["n"] if fixed["n"] else 0.0) - \
           (original["k"] / original["n"] if original["n"] else 0.0)
    return {
        "original": {"n": original["n"], "k": original["k"],
                     "rate": original["k"] / original["n"] if original["n"] else 0.0,
                     "wilson": [o_ci.low, o_ci.high]},
        "fixed": {"n": fixed["n"], "k": fixed["k"],
                  "rate": fixed["k"] / fixed["n"] if fixed["n"] else 0.0,
                  "wilson": [f_ci.low, f_ci.high]},
        "lift_points": round(lift * 100, 1),
        "p_value": cmp.p_value,
        "significant": cmp.p_value < alpha,
        "non_overlapping": non_overlap,
        "verdict": ("measured lift, intervals do not overlap" if non_overlap and cmp.p_value < alpha
                    else "measured lift, intervals overlap" if cmp.p_value < alpha
                    else "no statistically significant lift"),
    }


def run_fix(store_dir, cells_to_run: list[str] | None, baseline_n: int, max_minutes: float,
            workers: int, keep_workspaces: bool = False) -> dict:
    """Re-run the baseline and the top fragility cell with a fix injected."""
    from shakedown.analysis import build_analysis
    from shakedown.cost import BudgetTracker, Pricing
    from shakedown.proxy import ProxyServer
    from shakedown.runner import Runner
    from shakedown.scheduler import CellSpec, GridRunner
    from shakedown.spec import discover_tasks, load_dotenv
    from shakedown.store import Store, new_experiment_id, utc_now

    load_dotenv()
    original_path = Path(store_dir)
    original_store = Store(original_path.name, original_path.parent)
    meta = original_store.load_meta()
    analysis = build_analysis(original_store, meta)
    if not analysis["fix_first"]:
        return {"status": "no-fragility", "analysis": analysis,
                "message": "No critical or false-success fragility to fix."}

    top = analysis["fix_first"][0]
    target_cell = next((c for c in meta.get("cells", []) if c.get("id") == top["cell_id"]), None)
    if not target_cell:
        raise SystemExit(f"cannot find cell {top['cell_id']} in experiment metadata")
    target = CellSpec(id=target_cell["id"], label=target_cell["label"],
                      kind=target_cell["kind"], config=target_cell.get("config", {}))

    task_ids = meta.get("task_ids") or []
    tasks = discover_tasks(names=task_ids or None)
    budget_override = {
        "baseline_runs": baseline_n,
        "min_runs_per_cell": min(12, baseline_n),
        "max_runs_per_cell": baseline_n,
        "stop_rule": "fixed",
        "max_workers": workers,
        "max_minutes": max_minutes,
        "batch_size": max(1, min(4, baseline_n)),
    }
    tasks = discover_tasks(names=task_ids or None, budget_override=budget_override)

    original_runs = original_store.load_runs()
    prior_spend = max(
        float(meta.get("spend_usd", 0.0)),
        sum(float(run.get("cost_usd", 0.0)) for run in original_runs),
    )
    prior_calls = max(
        int(meta.get("calls", 0)),
        sum(int(run.get("model_calls", 0)) for run in original_runs),
    )
    fix_experiments = list(dict.fromkeys(meta.get("fix_experiments", [])))
    for previous_id in fix_experiments:
        previous_store = Store(previous_id, original_store.dir.parent)
        previous_meta = previous_store.load_meta()
        previous_runs = previous_store.load_runs()
        prior_spend += max(
            float(previous_meta.get("spend_usd", 0.0)),
            sum(float(run.get("cost_usd", 0.0)) for run in previous_runs),
        )
        prior_calls += max(
            int(previous_meta.get("calls", 0)),
            sum(int(run.get("model_calls", 0)) for run in previous_runs),
        )

    experiment_id = new_experiment_id("fix")
    new_store = Store(experiment_id, original_store.dir.parent)
    fix_experiments.append(experiment_id)
    meta["fix_experiments"] = fix_experiments
    original_store.save_meta(meta)
    tracker = BudgetTracker(meta.get("budget", {}).get("max_total_calls", 6000),
                            meta.get("budget", {}).get("max_usd", 15),
                            initial_calls=prior_calls, initial_usd=prior_spend)
    model = tasks[0].model
    proxy = ProxyServer(model.base_url, new_store, tracker, Pricing.from_model(model),
                        upstream_api_key=os.environ.get("MODEL_API_KEY", "local"))
    proxy.preflight()
    proxy.start()
    try:
        runner = Runner(new_store, proxy, keep_workspaces=keep_workspaces,
                        fix_line=top["suggestion"]["text"])
        cells = [CellSpec(id="baseline", label="Baseline + fix", kind="baseline"), target]
        grid = GridRunner(runner, tasks, cells, tasks[0].budget, tracker)
        grid.run()
    finally:
        proxy.stop()

    new_store.save_meta({
        "experiment_id": experiment_id,
        "status": "complete" if not grid.stopped_reason else "stopped",
        "budget": meta.get("budget", {}),
        "spend_usd": round(tracker.usd - prior_spend, 6),
        "calls": tracker.calls - prior_calls,
        "campaign_spend_usd": round(tracker.usd, 6),
        "campaign_calls": tracker.calls,
        "finished_at": utc_now(),
        "stopped_reason": grid.stopped_reason,
    })

    fixed_analysis = build_analysis(new_store, {
        "budget": tasks[0].budget.__dict__, "cells": [
            {"id": c.id, "label": c.label, "kind": c.kind, "config": c.config} for c in cells],
        "task_ids": [t.id for t in tasks], "model": model.__dict__,
        "finished_at": utc_now(), "elapsed_minutes": round(grid.elapsed_minutes, 1),
        "stopped_reason": grid.stopped_reason, "experiment_id": experiment_id,
    })

    orig_base = {"n": analysis["baseline"]["n"], "k": analysis["baseline"]["k"]}
    orig_cell_row = next((c for c in analysis["cells"] if c["id"] == target.id), None)
    fixed_cell_row = next((c for c in fixed_analysis["cells"] if c["id"] == target.id), None)
    fix_payload = {
        "status": "ok",
        "experiment_id": experiment_id,
        "treating": top,
        "fix_line": top["suggestion"]["text"],
        "target_cell": target.id,
        "baseline": _verdict(orig_base, {"n": fixed_analysis["baseline"]["n"],
                                         "k": fixed_analysis["baseline"]["k"]}),
        "target": None,
        "spend_usd": fixed_analysis["summary"]["spend_usd"],
        "campaign_spend_usd": round(tracker.usd, 6),
        "runs": fixed_analysis["summary"]["runs"],
    }
    if orig_cell_row and fixed_cell_row and orig_cell_row["n"] and fixed_cell_row["n"]:
        fix_payload["target"] = _verdict(
            {"n": orig_cell_row["n"], "k": orig_cell_row["k"]},
            {"n": fixed_cell_row["n"], "k": fixed_cell_row["k"]})
    fixed_analysis["fix"] = fix_payload
    new_store.write_json("analysis.json", fixed_analysis)
    return fix_payload | {"analysis": fixed_analysis}