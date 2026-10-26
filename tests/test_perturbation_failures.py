import asyncio
from types import SimpleNamespace

import httpx

from shakedown.cost import BudgetTracker, Pricing
from shakedown.perturb.ablation import AblationPerturbation
from shakedown.perturb.base import Perturbation, RunState
from shakedown.perturb.noise import NoisePerturbation
from shakedown.proxy import ProxyServer, RunContext
from shakedown.store import RunResult, Store


def test_ablation_and_noise_use_the_run_state_rng():
    ctx = SimpleNamespace(state=RunState("rng-regression"))
    payload = {
        "messages": [{"role": "system", "content": "First chunk\n\nSecond chunk"}],
        "tools": [{"function": {"name": "edit", "description": "First sentence. Second sentence."}}],
    }
    ablation = AblationPerturbation(
        chunk_id="first",
        mode="shuffle",
        chunk_text="First chunk",
        kind="system",
        tool=None,
        all_system_chunks=[("first", "First chunk"), ("second", "Second chunk")],
    )

    assert ablation.apply_request(payload, ctx)["messages"][0]["content"]
    assert NoisePerturbation("typos").apply_request(
        {"messages": [{"role": "user", "content": "A sufficiently lengthy description"}]},
        ctx,
    )["messages"][0]["content"]


class RaisingPerturbation(Perturbation):
    def __init__(self):
        super().__init__(id="raising", kind="noise", label="Raising perturbation")

    def apply_request(self, payload, ctx):
        raise RuntimeError("broken perturbation")


class FakeUpstream:
    def __init__(self):
        self.calls = []

    async def post(self, url, json, headers):
        self.calls.append(url)
        return httpx.Response(
            200,
            json={"choices": [{"message": {"role": "assistant", "content": "ok"}}]},
            request=httpx.Request("POST", url),
        )


def test_perturbation_failure_marks_run_invalid_and_never_passes_through(tmp_path):
    store = Store("invalid-perturbation", tmp_path)
    proxy = ProxyServer(
        "https://model.example/v1",
        store,
        BudgetTracker(max_calls=5, max_usd=0.01),
        Pricing(input_per_1m=1.0, output_per_1m=2.0),
    )
    upstream = FakeUpstream()
    proxy.app.state.client = upstream
    ctx = RunContext(
        run_id="run-invalid",
        task_id="task",
        cell_id="raising",
        perturbation=RaisingPerturbation(),
        max_calls=2,
        store=store,
        pricing=Pricing(input_per_1m=1.0, output_per_1m=2.0),
    )
    proxy.register(ctx)

    async def post_request():
        transport = httpx.ASGITransport(app=proxy.app)
        async with httpx.AsyncClient(transport=transport, base_url="http://proxy") as client:
            return await client.post(
                "/run/run-invalid/v1/chat/completions",
                json={"messages": [{"role": "user", "content": "hello"}]},
            )

    response = asyncio.run(post_request())

    assert response.status_code == 500
    assert ctx.state.invalid is True
    assert upstream.calls == []
    assert store.load_trace("run-invalid")[-1]["event"] == "perturbation_error"


def test_analysis_excludes_invalid_runs_from_pass_rate_and_reports_them(tmp_path):
    store = Store("invalid-analysis", tmp_path)
    meta = {
        "budget": {},
        "cells": [
            {"id": "baseline", "label": "Baseline", "kind": "baseline", "config": {}},
            {"id": "noise:test", "label": "Noise", "kind": "noise", "config": {}},
        ],
        "task_chunks": {},
    }
    store.save_meta(meta)
    for run_id, cell_id, invalid, success in (
        ("baseline-ok", "baseline", False, True),
        ("baseline-invalid", "baseline", True, False),
        ("cell-ok", "noise:test", False, False),
        ("cell-invalid", "noise:test", True, True),
    ):
        store.append_run(RunResult(
            run_id=run_id,
            experiment_id="invalid-analysis",
            task_id="task",
            task_name="Task",
            cell_id=cell_id,
            cell_label="Baseline" if cell_id == "baseline" else "Noise",
            cell_kind="baseline" if cell_id == "baseline" else "noise",
            invalid=invalid,
            success=success,
        ))

    from shakedown.analysis import build_analysis

    analysis = build_analysis(store, meta)

    assert analysis["baseline"]["n"] == 1
    assert analysis["baseline"]["invalid_runs"] == 1
    cell = analysis["cells"][0]
    assert cell["n"] == 1
    assert cell["invalid_runs"] == 1
    assert cell["taxonomy"]["counts"]["invalid"] == 1
    assert analysis["summary"]["invalid_runs"] == 2
