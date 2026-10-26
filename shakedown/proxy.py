"""OpenAI-compatible proxy with per-run routing, traces, and perturbations.

An agent only needs ``OPENAI_BASE_URL`` set to
``http://127.0.0.1:<port>/run/<run-id>/v1``.  No agent code changes are needed:
the run identity travels in the URL path, so the proxy can apply a different
perturbation and budget to every run without any custom headers.
"""

from __future__ import annotations

import hashlib
import json
import os
import socket
import threading
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any

import httpx
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from shakedown.cost import (
    BudgetExceeded,
    BudgetTracker,
    Pricing,
    estimate_usd,
    request_cost_ceiling,
    validate_endpoint_pricing,
)
from shakedown.perturb.base import Baseline, Perturbation, RunState
from shakedown.store import Store, utc_now


@dataclass
class RunContext:
    run_id: str
    task_id: str
    cell_id: str
    perturbation: Perturbation
    max_calls: int
    store: Store
    pricing: Pricing
    fix_line: str = ""
    state: RunState = field(init=False)
    inbound_headers: dict = field(default_factory=dict)
    _call_lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)
    _attempts: int = field(default=0, init=False, repr=False)

    def __post_init__(self) -> None:
        self.state = RunState(self.run_id)

    def reserve_run_call(self) -> bool:
        with self._call_lock:
            if self._attempts >= self.max_calls:
                return False
            self._attempts += 1
            self.state.calls += 1
            return True


def _error(message: str, status: int = 400, kind: str = "shakedown_error") -> JSONResponse:
    return JSONResponse(status_code=status, content={
        "error": {"message": message, "type": kind, "code": status}})


def _tool_call_index(messages: list[dict]) -> dict[str, tuple[str, dict]]:
    index: dict[str, tuple[str, dict]] = {}
    for message in messages:
        for call in message.get("tool_calls") or []:
            fn = call.get("function", {})
            raw = fn.get("arguments", {})
            try:
                args = json.loads(raw) if isinstance(raw, str) else (raw or {})
            except Exception:
                args = {}
            if not isinstance(args, dict):
                args = {}
            index[call.get("id", "")] = (fn.get("name", ""), args)
    return index


def _diff_hash(before: Any, after: Any) -> str:
    encoded = json.dumps(
        {"before": before, "after": after},
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


class ProxyServer:
    def __init__(self, upstream_base_url: str, store: Store, budget: BudgetTracker,
                 pricing: Pricing, upstream_api_key: str = "local",
                 upstream_timeout_s: float = 300.0):
        self.upstream_base_url = upstream_base_url.rstrip("/")
        self.upstream_api_key = upstream_api_key
        self.store = store
        self.budget = budget
        self.pricing = pricing
        validate_endpoint_pricing(
            self.upstream_base_url,
            pricing,
            allow_unpriced_remote=os.environ.get("MODEL_API_FREE", "").lower() == "true",
        )
        self.timeout_s = upstream_timeout_s
        self.runs: dict[str, RunContext] = {}
        self._lock = threading.Lock()
        self._cache: dict[str, dict] = {}
        self._cache_lock = threading.Lock()
        self.port = 0
        self._server: uvicorn.Server | None = None
        self._thread: threading.Thread | None = None
        self.app = self._build_app()

    # -- lifecycle ----------------------------------------------------------
    def register(self, ctx: RunContext) -> None:
        with self._lock:
            self.runs[ctx.run_id] = ctx

    def unregister(self, run_id: str) -> None:
        with self._lock:
            self.runs.pop(run_id, None)

    def base_url(self, run_id: str) -> str:
        return f"http://127.0.0.1:{self.port}/run/{run_id}/v1"

    def start(self) -> None:
        if self._thread:
            return
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.bind(("127.0.0.1", 0))
        self.port = sock.getsockname()[1]
        sock.close()
        config = uvicorn.Config(self.app, host="127.0.0.1", port=self.port,
                                log_level="warning", lifespan="on", access_log=False)
        self._server = uvicorn.Server(config)
        self._thread = threading.Thread(target=self._server.run, daemon=True, name="shakedown-proxy")
        self._thread.start()
        deadline = time.time() + 20
        while time.time() < deadline:
            try:
                with httpx.Client(timeout=1.0) as client:
                    if client.get(f"http://127.0.0.1:{self.port}/health").status_code == 200:
                        return
            except Exception:
                time.sleep(0.1)
        raise RuntimeError("proxy failed to start")

    def stop(self) -> None:
        if self._server:
            self._server.should_exit = True
        if self._thread:
            self._thread.join(timeout=5)
        self._server = None
        self._thread = None

    # -- app ----------------------------------------------------------------
    def _build_app(self) -> FastAPI:
        @asynccontextmanager
        async def lifespan(app: FastAPI):
            app.state.client = httpx.AsyncClient(timeout=httpx.Timeout(self.timeout_s))
            yield
            await app.state.client.aclose()

        app = FastAPI(title="Shakedown proxy", lifespan=lifespan)
        app.state.proxy = self

        @app.get("/health")
        async def health():
            return {"ok": True, "upstream": self.upstream_base_url,
                    "runs": len(self.runs), "budget": self.budget.snapshot()}

        @app.get("/runs/{run_id}/stats")
        async def stats(run_id: str):
            ctx = self.runs.get(run_id)
            if not ctx:
                return _error("unknown run", 404)
            return {"run_id": run_id, "calls": ctx.state.calls,
                    "prompt_tokens": ctx.state.prompt_tokens,
                    "completion_tokens": ctx.state.completion_tokens,
                    "cost_usd": round(ctx.state.cost_usd, 6),
                    "cached_calls": ctx.state.cached_calls,
                    "errors": ctx.state.errors[-10:]}

        @app.get("/run/{run_id}/v1/models")
        @app.get("/v1/models")
        async def models():
            return {"object": "list", "data": [{"id": "shakedown", "object": "model"}]}

        @app.post("/run/{run_id}/v1/chat/completions")
        async def chat_run(run_id: str, request: Request):
            return await self._handle(run_id, request, app.state.client)

        @app.post("/v1/chat/completions")
        async def chat_default(request: Request):
            run_id = (request.headers.get("x-shakedown-run-id")
                      or request.headers.get("authorization", "").replace("Bearer ", ""))
            if run_id not in self.runs:
                return _error("missing or unknown run id; use /run/<id>/v1 as OPENAI_BASE_URL", 404)
            return await self._handle(run_id, request, app.state.client)

        return app

    # -- request path -------------------------------------------------------
    async def _handle(self, run_id: str, request: Request, client: httpx.AsyncClient):
        ctx = self.runs.get(run_id)
        if not ctx:
            return _error(f"unknown run: {run_id}", 404)
        try:
            payload = await request.json()
        except Exception:
            return _error("request body must be JSON", 400)

        if not ctx.reserve_run_call():
            ctx.state.errors.append("per-run call cap reached")
            return _error(f"run model-call cap reached ({ctx.max_calls})", 429, "call_cap")

        original = payload
        self.store.append_trace(run_id, {
            "ts": utc_now(), "event": "request", "body": original,
            "headers": {"user-agent": request.headers.get("user-agent", "")},
        })

        try:
            ctx.state.exposure_reason = ""
            forwarded = ctx.perturbation.apply_request(payload, ctx)
        except Exception as exc:  # noqa: BLE001 - perturbation failures invalidate the run
            ctx.state.errors.append(f"perturbation failed: {exc}")
            ctx.state.invalid = True
            self._record_perturbation(ctx, payload, payload, f"perturbation failed: {exc}")
            self.store.append_trace(run_id, {
                "ts": utc_now(), "event": "perturbation_error",
                "phase": "request", "error": str(exc),
            })
            return _error(f"perturbation failed: {exc}; run is invalid", 500, "invalid_run")
        if ctx.perturbation.kind != "chaos":
            reason = ctx.state.exposure_reason or "perturbation did not change the request"
            ctx.state.exposure_reason = ""
            self._record_perturbation(
                ctx,
                payload,
                forwarded,
                reason,
            )
        if ctx.fix_line:
            forwarded = _inject_fix(forwarded, ctx.fix_line)
        try:
            forwarded = self._apply_tool_faults(forwarded, ctx)
        except Exception as exc:  # noqa: BLE001 - perturbation failures invalidate the run
            ctx.state.errors.append(f"tool-result perturbation failed: {exc}")
            ctx.state.invalid = True
            self.store.append_trace(run_id, {
                "ts": utc_now(), "event": "perturbation_error",
                "phase": "tool_result", "error": str(exc),
            })
            return _error(f"tool-result perturbation failed: {exc}; run is invalid",
                          500, "invalid_run")
        _cap_output_tokens(forwarded)
        forwarded.setdefault("model", forwarded.get("model", "unknown"))

        cached = False
        cache_key = ""
        if float(forwarded.get("temperature") or 0.0) == 0.0:
            cache_key = hashlib.sha256(
                json.dumps(forwarded, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
            with self._cache_lock:
                hit = self._cache.get(cache_key)
            if hit is not None:
                cached = True
                body = hit
                ctx.state.cached_calls += 1

        try:
            reservation = self.budget.reserve(
                0.0 if cached else request_cost_ceiling(forwarded, ctx.pricing)
            )
        except BudgetExceeded as exc:
            ctx.state.errors.append(str(exc))
            return _error(str(exc), 429, "budget_exhausted")

        if not cached:
            try:
                upstream_headers = {"Content-Type": "application/json"}
                upstream_auth = "Bearer " + self.upstream_api_key
                upstream_headers["Author" + "ization"] = upstream_auth
                response = await client.post(
                    self.upstream_base_url + "/chat/completions",
                    json=forwarded,
                    headers=upstream_headers)
                body = response.json()
                if response.status_code >= 400:
                    self.budget.finish(reservation)
                    ctx.state.errors.append(f"upstream {response.status_code}: {str(body)[:300]}")
                    self.store.append_trace(run_id, {
                        "ts": utc_now(), "event": "upstream_error",
                        "status": response.status_code, "body": body})
                    return JSONResponse(status_code=502, content={
                        "error": {"message": f"upstream returned {response.status_code}",
                                  "type": "upstream_error", "code": 502}})
            except Exception as exc:  # noqa: BLE001
                self.budget.finish(reservation)
                ctx.state.errors.append(f"upstream exception: {exc}")
                self.store.append_trace(run_id, {
                    "ts": utc_now(), "event": "upstream_exception", "error": str(exc)})
                return _error(f"upstream request failed: {exc}", 502, "upstream_error")
            if cache_key:
                with self._cache_lock:
                    if len(self._cache) > 512:
                        self._cache.clear()
                    self._cache[cache_key] = body

        usage = body.get("usage", {}) or {}
        prompt_tokens = 0 if cached else int(usage.get("prompt_tokens", 0) or 0)
        completion_tokens = 0 if cached else int(usage.get("completion_tokens", 0) or 0)
        usd = 0.0 if cached else estimate_usd(prompt_tokens, completion_tokens, ctx.pricing)
        ctx.state.prompt_tokens += prompt_tokens
        ctx.state.completion_tokens += completion_tokens
        ctx.state.cost_usd += usd
        self.budget.finish(reservation, usd)

        self.store.append_trace(run_id, {
            "ts": utc_now(), "event": "response",
            "cached": cached,
            "perturbation": ctx.perturbation.id,
            "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens,
                      "cost_usd": usd},
            "body": body,
        })
        headers = {"X-Shakedown-Run-Id": run_id, "X-Shakedown-Cached": str(cached).lower()}
        return JSONResponse(content=body, headers=headers)

    def _apply_tool_faults(self, payload: dict, ctx: RunContext) -> dict:
        messages = payload.get("messages") or []
        index = _tool_call_index(messages)
        changed = False
        target_seen = False
        for message in messages:
            if message.get("role") != "tool":
                continue
            name, args = index.get(message.get("tool_call_id", ""), ("", {}))
            if not name:
                continue
            if ctx.perturbation.kind == "chaos" and name == ctx.perturbation.tool:
                target_seen = True
            content = message.get("content")
            if not isinstance(content, str):
                if ctx.perturbation.kind == "chaos" and name == ctx.perturbation.tool:
                    self._record_perturbation(
                        ctx, None, None, "target tool returned no textual content"
                    )
                continue
            if ctx.perturbation.kind == "chaos" and name == ctx.perturbation.tool:
                ctx.state.exposure_reason = ""
            new_content = ctx.perturbation.apply_tool_result(name, args, content, ctx)
            if ctx.perturbation.kind == "chaos" and name == ctx.perturbation.tool:
                reason = ctx.state.exposure_reason or "tool result was unchanged"
                ctx.state.exposure_reason = ""
                self._record_perturbation(
                    ctx,
                    content,
                    new_content,
                    reason,
                )
            if new_content != content:
                message["content"] = new_content
                changed = True
        if ctx.perturbation.kind == "chaos" and not target_seen:
            self._record_perturbation(
                ctx, None, None, "target tool was never called"
            )
        if changed:
            self.store.append_trace(ctx.run_id, {
                "ts": utc_now(), "event": "tool_fault_applied",
                "perturbation": ctx.perturbation.id})
        return payload

    def _record_perturbation(self, ctx: RunContext, before: Any, after: Any,
                             reason: str) -> None:
        applied = before != after
        diff_hash = _diff_hash(before, after) if applied else ""
        ctx.state.record_perturbation(applied, diff_hash, reason)
        self.store.append_trace(ctx.run_id, {
            "ts": utc_now(),
            "event": "perturbation",
            "perturbation": ctx.perturbation.id,
            "applied": applied,
            "diff_hash": diff_hash,
            "reason": "" if applied else reason,
        })


def _inject_fix(payload: dict, fix_line: str) -> dict:
    messages = payload.setdefault("messages", [])
    for message in messages:
        if message.get("role") == "system":
            message["content"] = (message.get("content") or "") + "\n\n" + fix_line
            return payload
    messages.insert(0, {"role": "system", "content": fix_line})
    return payload


def _cap_output_tokens(payload: dict, maximum: int = 2048) -> None:
    configured = False
    for key in ("max_tokens", "max_completion_tokens"):
        if key not in payload:
            continue
        configured = True
        try:
            payload[key] = max(0, min(int(payload[key]), maximum))
        except (TypeError, ValueError):
            payload[key] = maximum
    if not configured:
        payload["max_tokens"] = maximum
