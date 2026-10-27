"""Suggest a concrete fix for the top fragility, then re-measure it."""

from __future__ import annotations

import json
import os
from pathlib import Path

import httpx
from shakedown.cost import BudgetExceeded, Pricing, estimate_usd, request_cost_ceiling
from shakedown.scheduler import CellSpec
from shakedown.stats import (
    minimum_detectable_effect,
    two_proportion_test,
    wilson_interval,
)

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


def _fix_experiment_cells(fix_text: str, runs: int) -> list[CellSpec]:
    if runs < 1:
        raise ValueError("fix experiment runs must be at least one")
    return [
        CellSpec(
            id="baseline", label="No-fix control", kind="baseline",
            config={"run_count": runs},
        ),
        CellSpec(
            id="fix:baseline", label="Fix arm", kind="fix",
            config={"fix_line": fix_text, "run_count": runs},
        ),
    ]


def _fix_prompt(fallback_text: str, failing_traces: list[dict]) -> str:
    evidence = []
    for trace in failing_traces[:6]:
        run_id = str(trace.get("run_id") or "unknown")
        events = []
        for event in trace.get("events", []):
            if event.get("event") == "request":
                body = event.get("body") or {}
                messages = body.get("messages") or []
                system = [
                    str(message.get("content") or "")[:1200]
                    for message in messages
                    if message.get("role") == "system"
                ]
                calls = []
                for message in messages:
                    for call in message.get("tool_calls") or []:
                        function = call.get("function") or {}
                        calls.append({
                            "tool": function.get("name"),
                            "arguments": str(function.get("arguments") or "")[:500],
                        })
                events.append({"system": system, "tool_calls": calls})
            elif event.get("event") == "response":
                body = event.get("body") or {}
                choices = body.get("choices") or []
                if choices:
                    message = choices[0].get("message") or {}
                    content = str(message.get("content") or "")
                    if content:
                        events.append({"assistant": content[:1200]})
                    calls = []
                    for call in message.get("tool_calls") or []:
                        function = call.get("function") or {}
                        calls.append({
                            "tool": function.get("name"),
                            "arguments": str(function.get("arguments") or "")[:500],
                        })
                    if calls:
                        events.append({"tool_calls": calls})
        evidence.append({"run_id": run_id, "trace": events})
    return (
        "You are improving a coding agent's system guidance using recorded failures. "
        "Return one concise, directly usable instruction (no analysis or claims). "
        "Use only the trace evidence; do not claim you reran tests. Do not instruct "
        "the agent to bypass tests or alter tests.\n\n"
        f"Canned fallback suggestion:\n{fallback_text[:1500]}\n\n"
        "Failing trace evidence (JSON):\n"
        + json.dumps(evidence, ensure_ascii=True)[:7000]
    )


def _generate_fix_text(prompt: str, fallback: str, model, tracker, api_key: str,
                       max_usd: float = 0.02) -> dict:
    payload = {
        "model": model.name,
        "temperature": 0.0,
        "max_tokens": 384,
        "messages": [
            {"role": "system", "content": (
                "Write one concise instruction grounded in the supplied evidence."
            )},
            {"role": "user", "content": prompt},
        ],
    }
    pricing = Pricing.from_model(model)
    ceiling = request_cost_ceiling(payload, pricing, max_output_tokens=384)
    available = max(0.0, tracker.max_usd - tracker.usd)
    cap = min(max_usd, available)
    if ceiling > cap + 1e-12:
        return {
            "text": fallback,
            "fallback_text": fallback,
            "generated_text": "",
            "source": "canned fallback",
            "status": "generation budget cap",
            "model": model.name,
            "cost_usd": 0.0,
        }

    try:
        reservation = tracker.reserve(ceiling)
    except BudgetExceeded:
        return _fallback_guidance(fallback, model.name, "generation budget cap")
    response = None
    try:
        response = httpx.post(
            model.base_url.rstrip("/") + "/chat/completions",
            json=payload,
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=15.0,
        )
        response.raise_for_status()
        body = response.json()
        generated = str(
            body.get("choices", [{}])[0].get("message", {}).get("content") or ""
        ).strip()
        if not generated or len(generated) > 2000:
            raise ValueError("model returned empty or oversized fix guidance")
        usage = body.get("usage") or {}
        actual = (
            estimate_usd(
                int(usage.get("prompt_tokens", 0) or 0),
                int(usage.get("completion_tokens", 0) or 0),
                pricing,
            )
            if "prompt_tokens" in usage or "completion_tokens" in usage
            else ceiling
        )
    except Exception as exc:
        charged = ceiling if response is not None else 0.0
        tracker.finish(reservation, charged)
        reason = str(exc)
        if api_key:
            reason = reason.replace(api_key, "[redacted]")
        return {
            "text": fallback,
            "fallback_text": fallback,
            "generated_text": "",
            "source": "canned fallback",
            "status": f"generation failed: {type(exc).__name__}: {reason[:300]}",
            "model": model.name,
            "cost_usd": round(charged, 6),
        }
    tracker.finish(reservation, actual)
    return {
        "text": generated,
        "fallback_text": fallback,
        "generated_text": generated,
        "source": "model",
        "status": "generated",
        "model": model.name,
        "cost_usd": round(actual, 6),
    }


def _fallback_guidance(text: str, model_name: str, status: str) -> dict:
    return {
        "text": text,
        "fallback_text": text,
        "generated_text": "",
        "source": "canned fallback",
        "status": status,
        "model": model_name,
        "cost_usd": 0.0,
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
        "verdict": (
            "measured lift, intervals do not overlap"
            if lift > 0 and non_overlap and cmp.p_value < alpha
            else "measured lift, intervals overlap"
            if lift > 0 and cmp.p_value < alpha
            else "significant decline"
            if lift < 0 and cmp.p_value < alpha
            else "no statistically significant lift"
        ),
    }


def run_fix(store_dir, cells_to_run: list[str] | None, baseline_n: int, max_minutes: float,
            workers: int, keep_workspaces: bool = False) -> dict:
    """Interleave equal-N no-fix control and fix arms on fresh task workspaces."""
    from shakedown.analysis import build_analysis
    from shakedown.cost import BudgetTracker, Pricing
    from shakedown.proxy import ProxyServer
    from shakedown.runner import Runner
    from shakedown.scheduler import GridRunner
    from shakedown.spec import discover_tasks, load_dotenv
    from shakedown.store import Store, new_experiment_id, utc_now

    if baseline_n < 1:
        raise ValueError("fix experiment runs must be at least one")
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
    budget_override = {
        "baseline_runs": baseline_n,
        "min_runs_per_cell": baseline_n,
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
    fix_started_at = utc_now()
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
    failing_traces = []
    for row in original_runs:
        if (row.get("cell_id") != top["cell_id"] or row.get("success")
                or row.get("invalid")):
            continue
        trace = original_store.load_trace(row["run_id"])
        if trace:
            failing_traces.append({"run_id": row["run_id"], "events": trace})
        if len(failing_traces) == 6:
            break
    prompt = _fix_prompt(top["suggestion"]["text"], failing_traces)
    fix_generation = (
        _generate_fix_text(
            prompt,
            top["suggestion"]["text"],
            model,
            tracker,
            os.environ.get("MODEL_API_KEY", "local"),
        )
        if failing_traces
        else _fallback_guidance(
            top["suggestion"]["text"], model.name, "no failing traces available"
        )
    )
    cells = _fix_experiment_cells(fix_generation["text"], baseline_n)
    proxy.start()
    try:
        runner = Runner(new_store, proxy, keep_workspaces=keep_workspaces)
        grid = GridRunner(runner, tasks, cells, tasks[0].budget, tracker)
        grid.run()
    finally:
        proxy.stop()

    fixed_runs = new_store.load_runs()
    run_spend = sum(float(run.get("cost_usd", 0.0)) for run in fixed_runs)
    generation_spend = fix_generation["cost_usd"]
    new_store.save_meta({
        "experiment_id": experiment_id,
        "status": "complete" if not grid.stopped_reason else "stopped",
        "started_at": fix_started_at,
        "finished_at": utc_now(),
        "budget": tasks[0].budget.__dict__,
        "campaign_budget": meta.get("budget", {}),
        "model": model.__dict__,
        "tasks": [{"id": task.id, "name": task.name} for task in tasks],
        "task_ids": [task.id for task in tasks],
        "task_chunks": meta.get("task_chunks", {}),
        "cells": [
            {"id": cell.id, "label": cell.label, "kind": cell.kind,
             "config": cell.config}
            for cell in cells
        ],
        "fix_generation": fix_generation,
        "spend_usd": round(run_spend + generation_spend, 6),
        "calls": tracker.calls - prior_calls,
        "campaign_spend_usd": round(tracker.usd, 6),
        "campaign_calls": tracker.calls,
        "runs": len(fixed_runs),
        "stopped_reason": grid.stopped_reason,
    })

    fixed_analysis = build_analysis(new_store, {
        "budget": tasks[0].budget.__dict__, "cells": [
            {"id": c.id, "label": c.label, "kind": c.kind, "config": c.config} for c in cells],
        "task_ids": [t.id for t in tasks], "model": model.__dict__,
        "task_chunks": meta.get("task_chunks", {}),
        "finished_at": utc_now(), "elapsed_minutes": round(grid.elapsed_minutes, 1),
        "stopped_reason": grid.stopped_reason, "experiment_id": experiment_id,
    })

    control = {
        "n": fixed_analysis["baseline"]["n"],
        "k": fixed_analysis["baseline"]["k"],
    }
    fixed_cell_row = next(
        (cell for cell in fixed_analysis["cells"] if cell["id"] == "fix:baseline"),
        None,
    )
    fixed = {
        "n": fixed_cell_row["n"] if fixed_cell_row else 0,
        "k": fixed_cell_row["k"] if fixed_cell_row else 0,
    }
    verdict = _verdict(control, fixed)
    mde = minimum_detectable_effect(
        control["k"] / control["n"] if control["n"] else 0.5,
        fixed["n"], control["n"],
    )
    fix_payload = {
        "status": "ok",
        "experiment_id": experiment_id,
        "treating": top,
        "fix_line": fix_generation["text"],
        "fallback_text": fix_generation["fallback_text"],
        "generated_text": fix_generation["generated_text"],
        "generation_source": fix_generation["source"],
        "generation_status": fix_generation["status"],
        "generation_model": fix_generation["model"],
        "generation_cost_usd": generation_spend,
        "target_cell": target.id,
        "control": verdict["original"],
        "fixed": verdict["fixed"],
        "baseline": verdict,
        "target": {"cell_id": target.id, "original_observation": {
            "n": next((cell["n"] for cell in analysis["cells"]
                       if cell["id"] == target.id), 0),
            "k": next((cell["k"] for cell in analysis["cells"]
                       if cell["id"] == target.id), 0),
        }},
        "mde_points": round(mde * 100, 1),
        "p_value": verdict["p_value"],
        "lift_points": verdict["lift_points"],
        "non_overlapping": verdict["non_overlapping"],
        "verdict": verdict["verdict"],
        "spend_usd": round(run_spend + generation_spend, 6),
        "campaign_spend_usd": round(tracker.usd, 6),
        "runs": len(fixed_runs),
        "control_runs": control["n"],
        "fixed_runs": fixed["n"],
    }
    fixed_analysis["fix"] = fix_payload
    new_store.write_json("analysis.json", fixed_analysis)
    return fix_payload | {"analysis": fixed_analysis}