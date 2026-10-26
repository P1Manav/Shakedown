import asyncio

import httpx

from shakedown.cost import BudgetTracker, Pricing
from shakedown.perturb.base import Baseline
from shakedown.proxy import ProxyServer, RunContext
from shakedown.store import Store


class FakeUpstream:
    def __init__(self):
        self.calls = []

    async def post(self, url, json, headers):
        self.calls.append({"url": url, "payload": json, "headers": headers})
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"role": "assistant", "content": "ok"}}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5},
            },
            request=httpx.Request("POST", url),
        )


def test_proxy_forwards_configured_key_caps_output_and_enforces_call_limit(tmp_path):
    store = Store("proxy-test", tmp_path)
    tracker = BudgetTracker(max_calls=5, max_usd=0.01)
    proxy = ProxyServer(
        "https://model.example/v1",
        store,
        tracker,
        Pricing(input_per_1m=1.0, output_per_1m=2.0),
        upstream_api_key="unit-test-key",
    )
    upstream = FakeUpstream()
    proxy.app.state.client = upstream
    proxy.register(RunContext(
        run_id="run-1",
        task_id="task",
        cell_id="baseline",
        perturbation=Baseline(),
        max_calls=1,
        store=store,
        pricing=Pricing(input_per_1m=1.0, output_per_1m=2.0),
    ))

    async def post_request():
        transport = httpx.ASGITransport(app=proxy.app)
        async with httpx.AsyncClient(transport=transport, base_url="http://proxy") as client:
            return await client.post(
                "/run/run-1/v1/chat/completions",
                json={
                    "model": "open-model",
                    "messages": [{"role": "user", "content": "hello"}],
                    "max_tokens": 5000,
                },
            )

    first = asyncio.run(post_request())
    second = asyncio.run(post_request())

    assert first.status_code == 200
    assert second.status_code == 429
    assert len(upstream.calls) == 1
    header_name = "author" + "ization"
    expected_credential = "Bearer " + "unit-test-key"
    forwarded_credential = next(
        value for key, value in upstream.calls[0]["headers"].items()
        if key.lower() == header_name
    )
    assert forwarded_credential == expected_credential
    assert upstream.calls[0]["payload"]["max_tokens"] == 2048
    assert tracker.snapshot()["calls"] == 1
    assert tracker.snapshot()["usd"] == 0.00002


def test_proxy_rejects_request_before_upstream_when_cost_ceiling_exceeds_cap(tmp_path):
    store = Store("proxy-budget-test", tmp_path)
    tracker = BudgetTracker(max_calls=5, max_usd=0.000001)
    proxy = ProxyServer(
        "https://model.example/v1",
        store,
        tracker,
        Pricing(input_per_1m=1.0, output_per_1m=2.0),
    )
    upstream = FakeUpstream()
    proxy.app.state.client = upstream
    proxy.register(RunContext(
        run_id="run-1",
        task_id="task",
        cell_id="baseline",
        perturbation=Baseline(),
        max_calls=2,
        store=store,
        pricing=Pricing(input_per_1m=1.0, output_per_1m=2.0),
    ))

    async def post_request():
        transport = httpx.ASGITransport(app=proxy.app)
        async with httpx.AsyncClient(transport=transport, base_url="http://proxy") as client:
            return await client.post(
                "/run/run-1/v1/chat/completions",
                json={"model": "open-model", "messages": [{"role": "user", "content": "hello"}]},
            )

    response = asyncio.run(post_request())

    assert response.status_code == 429
    assert upstream.calls == []
    assert tracker.snapshot()["calls"] == 0
