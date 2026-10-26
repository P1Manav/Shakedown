import asyncio

import httpx
import pytest

from shakedown.cost import BudgetTracker, Pricing
from shakedown.perturb.ablation import AblationPerturbation
from shakedown.perturb.chaos import ChaosPerturbation
from shakedown.perturb.base import Baseline
from shakedown.proxy import ProxyServer, RunContext
from shakedown.store import RunResult, Store


class FakeUpstream:
    async def post(self, url, json, headers):
        return httpx.Response(
            200,
            json={"choices": [{"message": {"role": "assistant", "content": "ok"}}]},
            request=httpx.Request("POST", url),
        )


def _post(store, perturbation, payload):
    proxy = ProxyServer(
        "https://model.example/v1",
        store,
        BudgetTracker(max_calls=5, max_usd=0.01),
        Pricing(input_per_1m=1.0, output_per_1m=2.0),
    )
    proxy.app.state.client = FakeUpstream()
    context = RunContext(
        run_id="run-exposure",
        task_id="task",
        cell_id=perturbation.id,
        perturbation=perturbation,
        max_calls=5,
        store=store,
        pricing=Pricing(input_per_1m=1.0, output_per_1m=2.0),
    )
    proxy.register(context)

    async def call():
        transport = httpx.ASGITransport(app=proxy.app)
        async with httpx.AsyncClient(transport=transport, base_url="http://proxy") as client:
            return await client.post(
                "/run/run-exposure/v1/chat/completions",
                json=payload,
            )

    return asyncio.run(call()), context, store.load_trace("run-exposure")


def test_ablation_fails_if_target_chunk_is_missing_from_outgoing_prompt():
    perturbation = AblationPerturbation(
        "rules", "remove", "Keep the tests passing.", "system", None
    )

    with pytest.raises(ValueError, match="not found in the outgoing system prompt"):
        perturbation.apply_request(
            {"messages": [{"role": "system", "content": "Do the requested work."}]},
            type("Context", (), {"state": type("State", (), {"rng": None})()})(),
        )


def test_request_perturbation_records_applied_diff_in_trace_and_result(tmp_path):
    store = Store("request-exposure", tmp_path)
    perturbation = AblationPerturbation(
        "rules", "remove", "Keep the tests passing.", "system", None
    )

    response, context, trace = _post(
        store,
        perturbation,
        {"messages": [{"role": "system", "content": "Keep the tests passing."}]},
    )

    assert response.status_code == 200
    assert context.state.perturbation_applied is True
    event = next(item for item in trace if item["event"] == "perturbation")
    assert event["applied"] is True
    assert len(event["diff_hash"]) == 64
    result = RunResult(
        run_id="run-exposure",
        experiment_id="request-exposure",
        task_id="task",
        task_name="Task",
        cell_id=perturbation.id,
        cell_label=perturbation.label,
        cell_kind="ablation",
        perturbation_applied=context.state.perturbation_applied,
        perturbation_records=context.state.perturbation_records,
    )
    assert result.to_dict()["perturbation_records"][0]["diff_hash"] == event["diff_hash"]


def test_tool_fault_short_output_is_not_exposed_and_is_traced(tmp_path):
    store = Store("no-exposure", tmp_path)
    perturbation = ChaosPerturbation("truncate_output", "read_file", {"keep_chars": 20})
    payload = {
        "messages": [
            {"role": "assistant", "tool_calls": [{
                "id": "call-1",
                "function": {"name": "read_file", "arguments": '{"path":"a.py"}'},
            }]},
            {"role": "tool", "tool_call_id": "call-1", "content": "short"},
        ]
    }

    response, context, trace = _post(store, perturbation, payload)

    assert response.status_code == 200
    assert context.state.perturbation_applied is False
    record = context.state.perturbation_records[-1]
    assert record["applied"] is False
    assert record["reason"] == "tool output is shorter than keep_chars"
    assert trace[-2]["event"] == "perturbation"
    assert trace[-2]["applied"] is False


def test_tool_fault_target_never_called_is_not_exposed(tmp_path):
    store = Store("tool-not-called", tmp_path)
    perturbation = ChaosPerturbation("truncate_output", "read_file", {"keep_chars": 10})

    response, context, trace = _post(
        store, perturbation, {"messages": [{"role": "user", "content": "hello"}]}
    )

    assert response.status_code == 200
    assert context.state.perturbation_applied is False
    assert context.state.perturbation_records[-1]["reason"] == "target tool was never called"
    assert any(
        event["event"] == "perturbation" and event["reason"] == "target tool was never called"
        for event in trace
    )


def test_single_sentence_tool_description_shuffle_is_not_exposed(tmp_path):
    store = Store("short-description", tmp_path)
    perturbation = AblationPerturbation(
        "read_rule", "shuffle", "The tool reads a file.", "tool_description", "read_file"
    )
    response, context, trace = _post(
        store,
        perturbation,
        {
            "messages": [{"role": "user", "content": "read it"}],
            "tools": [{"type": "function", "function": {
                "name": "read_file",
                "description": "The tool reads a file.",
            }}],
        },
    )

    assert response.status_code == 200
    record = context.state.perturbation_records[0]
    assert record["applied"] is False
    assert record["reason"] == "tool description has fewer than two sentences"
    assert trace[1]["event"] == "perturbation"
    assert trace[1]["applied"] is False


def test_analysis_reports_exposure_and_exposed_run_effect(tmp_path):
    store = Store("exposure-analysis", tmp_path)
    meta = {
        "budget": {},
        "cells": [
            {"id": "baseline", "label": "Baseline", "kind": "baseline", "config": {}},
            {"id": "chaos:truncate", "label": "Truncate", "kind": "chaos", "config": {}},
        ],
        "task_chunks": {},
    }
    store.save_meta(meta)
    fixtures = [
        ("base-1", "baseline", True, False, True),
        ("base-2", "baseline", False, False, True),
        ("cell-1", "chaos:truncate", False, True, True),
        ("cell-2", "chaos:truncate", True, False, False),
    ]
    for run_id, cell_id, success, applied, _ in fixtures:
        store.append_run(RunResult(
            run_id=run_id,
            experiment_id="exposure-analysis",
            task_id="task",
            task_name="Task",
            cell_id=cell_id,
            cell_label="Baseline" if cell_id == "baseline" else "Truncate",
            cell_kind="baseline" if cell_id == "baseline" else "chaos",
            success=success,
            perturbation_applied=applied,
        ))

    from shakedown.analysis import build_analysis

    analysis = build_analysis(store, meta)
    cell = analysis["cells"][0]

    assert cell["exposure"]["exposure_rate"] == 0.5
    assert cell["exposure"]["n"] == 1
    assert cell["exposure"]["k"] == 0
    assert cell["exposure"]["rate"] == 0
    assert cell["exposure"]["comparison"]["delta"] == -0.5
    assert "intent-to-treat" in analysis["methods"]["effect_populations"]["comparison"]
