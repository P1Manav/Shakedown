import asyncio

import httpx
import pytest

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


class UpstreamFailure:
    async def post(self, url, json, headers):
        return httpx.Response(
            503,
            json={"error": {"message": "temporarily unavailable"}},
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


def test_proxy_marks_upstream_failures_as_invalid(tmp_path):
    store = Store("proxy-upstream-failure", tmp_path)
    proxy = ProxyServer(
        "https://model.example/v1",
        store,
        BudgetTracker(max_calls=5, max_usd=0.01),
        Pricing(input_per_1m=1.0, output_per_1m=2.0),
    )
    proxy.app.state.client = UpstreamFailure()
    context = RunContext(
        run_id="run-upstream-failure",
        task_id="task",
        cell_id="baseline",
        perturbation=Baseline(),
        max_calls=2,
        store=store,
        pricing=Pricing(input_per_1m=1.0, output_per_1m=2.0),
    )
    proxy.register(context)

    async def post_request():
        transport = httpx.ASGITransport(app=proxy.app)
        async with httpx.AsyncClient(transport=transport, base_url="http://proxy") as client:
            return await client.post(
                "/run/run-upstream-failure/v1/chat/completions",
                json={"model": "open-model", "messages": [{"role": "user", "content": "hi"}]},
            )

    response = asyncio.run(post_request())

    assert response.status_code == 502
    assert context.state.invalid is True


def test_proxy_forces_configured_model_temperature_and_supported_seed(tmp_path):
    store = Store("proxy-config", tmp_path)
    proxy = ProxyServer(
        "https://model.example/v1",
        store,
        BudgetTracker(max_calls=5, max_usd=0.01),
        Pricing(input_per_1m=1.0, output_per_1m=2.0),
    )
    upstream = FakeUpstream()
    proxy.app.state.client = upstream
    context = RunContext(
        run_id="configured-run",
        task_id="task",
        cell_id="baseline",
        perturbation=Baseline(),
        max_calls=2,
        store=store,
        pricing=Pricing(input_per_1m=1.0, output_per_1m=2.0),
        model_name="configured-model",
        temperature=0.25,
        supports_seed=True,
    )
    proxy.register(context)

    async def post_request():
        transport = httpx.ASGITransport(app=proxy.app)
        async with httpx.AsyncClient(transport=transport, base_url="http://proxy") as client:
            return await client.post(
                "/run/configured-run/v1/chat/completions",
                json={
                    "model": "agent-selected-model",
                    "temperature": 0.9,
                    "seed": 123,
                    "messages": [{"role": "user", "content": "hello"}],
                },
            )

    response = asyncio.run(post_request())

    assert response.status_code == 200
    sent = upstream.calls[0]["payload"]
    assert sent["model"] == "configured-model"
    assert sent["temperature"] == 0.25
    assert sent["seed"] == context.state.seed
    assert store.load_trace("configured-run")[-1]["seed"] == context.state.seed


def test_proxy_cache_is_per_run_and_disabled_for_nonzero_or_missing_temperature(tmp_path):
    store = Store("proxy-cache-isolation", tmp_path)
    proxy = ProxyServer(
        "https://model.example/v1",
        store,
        BudgetTracker(max_calls=20, max_usd=0.01),
        Pricing(input_per_1m=1.0, output_per_1m=2.0),
    )
    upstream = FakeUpstream()
    proxy.app.state.client = upstream
    contexts = [
        RunContext(
            run_id=run_id,
            task_id="task",
            cell_id="baseline",
            perturbation=Baseline(),
            max_calls=4,
            store=store,
            pricing=Pricing(input_per_1m=1.0, output_per_1m=2.0),
            temperature=temperature,
        )
        for run_id, temperature in (
            ("zero-1", 0.0),
            ("zero-2", 0.0),
            ("warm", 0.7),
            ("unset", None),
        )
    ]
    for context in contexts:
        proxy.register(context)

    async def post_request(run_id, temperature):
        transport = httpx.ASGITransport(app=proxy.app)
        async with httpx.AsyncClient(transport=transport, base_url="http://proxy") as client:
            payload = {"model": "model", "messages": [{"role": "user", "content": "same"}]}
            if temperature is not None:
                payload["temperature"] = temperature
            return await client.post(f"/run/{run_id}/v1/chat/completions", json=payload)

    async def send_all():
        for run_id, temperature in (
            ("zero-1", 0.0), ("zero-1", 0.0),
            ("zero-2", 0.0),
            ("warm", 0.7), ("warm", 0.7),
            ("unset", None), ("unset", None),
        ):
            assert (await post_request(run_id, temperature)).status_code == 200

    asyncio.run(send_all())

    assert len(upstream.calls) == 6
    assert contexts[0].state.cached_calls == 1
    assert contexts[1].state.cached_calls == 0
    assert contexts[2].state.cached_calls == 0
    assert contexts[3].state.cached_calls == 0


def test_preflight_rejects_unreachable_or_unhealthy_model_endpoint(monkeypatch, tmp_path):
    proxy = ProxyServer(
        "http://127.0.0.1:11434/v1",
        Store("preflight", tmp_path),
        BudgetTracker(max_calls=1, max_usd=1.0),
        Pricing(),
    )

    class FailedResponse:
        def raise_for_status(self):
            request = httpx.Request("GET", "http://127.0.0.1:11434/v1/models")
            response = httpx.Response(503, request=request)
            raise httpx.HTTPStatusError("unavailable", request=request, response=response)

    class FakeClient:
        def __init__(self, timeout):
            pass

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            return False

        def get(self, url, headers):
            assert url.endswith("/models")
            return FailedResponse()

    monkeypatch.setattr(httpx, "Client", FakeClient)

    with pytest.raises(RuntimeError, match="model endpoint preflight failed"):
        proxy.preflight()


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
