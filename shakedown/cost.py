"""Token accounting, USD estimates, and hard spend/call caps."""

from __future__ import annotations

import threading
import math
import json
import ipaddress
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlsplit

from shakedown.spec import ModelSpec


@dataclass(frozen=True)
class Pricing:
    input_per_1m: float = 0.0
    output_per_1m: float = 0.0

    @classmethod
    def from_model(cls, model: ModelSpec) -> "Pricing":
        return cls(model.price_input_per_1m, model.price_output_per_1m)

    @property
    def is_free(self) -> bool:
        return self.input_per_1m == 0 and self.output_per_1m == 0


def validate_endpoint_pricing(base_url: str, pricing: Pricing,
                              allow_unpriced_remote: bool = False) -> None:
    host = urlsplit(base_url).hostname
    local = host == "localhost"
    if host:
        try:
            local = local or ipaddress.ip_address(host).is_loopback
        except ValueError:
            pass
    if pricing.is_free and not local and not allow_unpriced_remote:
        raise ValueError(
            "Remote model endpoint has no configured token prices; set "
            "PRICE_INPUT_PER_1M and PRICE_OUTPUT_PER_1M, or explicitly set "
            "MODEL_API_FREE=true only when the endpoint is genuinely free."
        )


def estimate_usd(prompt_tokens: int, completion_tokens: int, pricing: Pricing) -> float:
    return (prompt_tokens * pricing.input_per_1m + completion_tokens * pricing.output_per_1m) / 1_000_000


def request_cost_ceiling(payload: dict[str, Any], pricing: Pricing,
                         max_output_tokens: int = 2048) -> float:
    """Conservatively bound a request before forwarding it upstream."""
    messages = payload.get("messages") or []
    prompt_bytes = len(json.dumps(payload, ensure_ascii=False, separators=(",", ":"),
                                  default=str).encode("utf-8"))
    prompt_token_ceiling = prompt_bytes + 64 * len(messages) + 64
    output_tokens = payload.get("max_completion_tokens", payload.get("max_tokens", max_output_tokens))
    try:
        output_tokens = max(0, min(int(output_tokens), max_output_tokens))
    except (TypeError, ValueError):
        output_tokens = max_output_tokens
    return estimate_usd(prompt_token_ceiling, output_tokens, pricing)


def estimate_grid(tasks: int, cells: int, baseline_runs: int, runs_per_cell: int,
                  calls_per_run: float, prompt_tokens_per_call: float,
                  completion_tokens_per_call: float, pricing: Pricing,
                  seconds_per_run: float, workers: int) -> dict:
    total_runs = baseline_runs + cells * runs_per_cell
    total_calls = total_runs * calls_per_run
    usd = total_calls * (
        prompt_tokens_per_call * pricing.input_per_1m +
        completion_tokens_per_call * pricing.output_per_1m) / 1_000_000
    wall_minutes = total_runs * seconds_per_run / max(1, workers) / 60.0
    return {
        "total_runs": int(total_runs),
        "model_calls": int(total_calls),
        "usd": round(usd, 4),
        "wall_minutes": round(wall_minutes, 1),
        "free": pricing.is_free,
    }


def shrink_plan(cells: int, baseline_runs: int, runs_per_cell: int, min_runs: int,
                usd: float, max_usd: float, minutes: float, max_minutes: float,
                tasks: int) -> dict:
    """Shrink N (then the task set) until estimates fit the caps.

    This is intentionally conservative: it leaves 20% headroom on spend.
    """
    budget_usd = max_usd * 0.8
    plan = {"tasks": tasks, "baseline_runs": baseline_runs,
            "runs_per_cell": runs_per_cell, "notes": []}
    if usd <= budget_usd and minutes <= max_minutes:
        return plan

    while plan["runs_per_cell"] > min_runs and (
            usd > budget_usd or minutes > max_minutes):
        plan["runs_per_cell"] = max(min_runs, int(plan["runs_per_cell"] * 0.75))
        scale = plan["runs_per_cell"] / max(1, runs_per_cell)
        usd *= scale
        minutes *= scale
        runs_per_cell = plan["runs_per_cell"]
        plan["notes"].append(f"reduced runs/cell to {plan['runs_per_cell']}")

    while plan["tasks"] > 1 and (usd > budget_usd or minutes > max_minutes):
        plan["tasks"] -= 1
        usd *= plan["tasks"] / (plan["tasks"] + 1)
        minutes *= plan["tasks"] / (plan["tasks"] + 1)
        plan["notes"].append(f"reduced task set to {plan['tasks']}")
    return plan


class BudgetExceeded(RuntimeError):
    pass


@dataclass(frozen=True)
class BudgetReservation:
    ceiling_usd: float


class BudgetTracker:
    """Thread-safe global cap shared by every run through the proxy."""

    def __init__(self, max_calls: int = 6000, max_usd: float = 15.0):
        if max_calls < 0 or not math.isfinite(max_usd) or max_usd < 0:
            raise ValueError("budget limits must be non-negative and finite")
        self.max_calls = max_calls
        self.max_usd = max_usd
        self.calls = 0
        self.usd = 0.0
        self._reserved_calls = 0
        self._reserved_usd = 0.0
        self._lock = threading.Lock()
        self.stopped_reason = ""

    def reserve(self, ceiling_usd: float) -> BudgetReservation:
        if not math.isfinite(ceiling_usd) or ceiling_usd < 0:
            raise ValueError("request cost ceiling must be non-negative and finite")
        with self._lock:
            if self.calls + self._reserved_calls >= self.max_calls:
                self.stopped_reason = f"global call cap reached ({self.max_calls})"
                raise BudgetExceeded(self.stopped_reason)
            if self.stopped_reason:
                raise BudgetExceeded(self.stopped_reason)
            if self.usd + self._reserved_usd + ceiling_usd > self.max_usd + 1e-12:
                self.stopped_reason = f"global USD cap would be exceeded (${self.max_usd:.2f})"
                raise BudgetExceeded(self.stopped_reason)
            self._reserved_calls += 1
            self._reserved_usd += ceiling_usd
        return BudgetReservation(ceiling_usd)

    def finish(self, reservation: BudgetReservation, actual_usd: float | None = None) -> None:
        charged_usd = reservation.ceiling_usd if actual_usd is None else actual_usd
        if not math.isfinite(charged_usd) or charged_usd < 0:
            raise ValueError("actual request cost must be non-negative and finite")
        with self._lock:
            self._reserved_calls -= 1
            self._reserved_usd -= reservation.ceiling_usd
            self.calls += 1
            self.usd += charged_usd
            if charged_usd > reservation.ceiling_usd + 1e-12:
                self.stopped_reason = "provider usage exceeded the reserved request-cost ceiling"
            if self.usd > self.max_usd + 1e-12:
                self.stopped_reason = f"global USD cap exceeded (${self.max_usd:.2f})"

    def snapshot(self) -> dict:
        with self._lock:
            return {"calls": self.calls, "usd": round(self.usd, 6),
                    "max_calls": self.max_calls, "max_usd": self.max_usd,
                    "reserved_calls": self._reserved_calls,
                    "reserved_usd": round(self._reserved_usd, 6),
                    "stopped_reason": self.stopped_reason}