"""Token accounting, USD estimates, and hard spend/call caps."""

from __future__ import annotations

import threading
from dataclasses import dataclass

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


def estimate_usd(prompt_tokens: int, completion_tokens: int, pricing: Pricing) -> float:
    return (prompt_tokens * pricing.input_per_1m + completion_tokens * pricing.output_per_1m) / 1_000_000


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


class BudgetTracker:
    """Thread-safe global cap shared by every run through the proxy."""

    def __init__(self, max_calls: int = 6000, max_usd: float = 15.0):
        self.max_calls = max_calls
        self.max_usd = max_usd
        self.calls = 0
        self.usd = 0.0
        self._lock = threading.Lock()
        self.stopped_reason = ""

    def check(self) -> None:
        if self.calls >= self.max_calls:
            self.stopped_reason = f"global call cap reached ({self.max_calls})"
            raise BudgetExceeded(self.stopped_reason)
        if self.usd >= self.max_usd:
            self.stopped_reason = f"global USD cap reached (${self.max_usd:.2f})"
            raise BudgetExceeded(self.stopped_reason)

    def add(self, usd: float) -> None:
        with self._lock:
            self.calls += 1
            self.usd += usd

    def snapshot(self) -> dict:
        return {"calls": self.calls, "usd": round(self.usd, 6),
                "max_calls": self.max_calls, "max_usd": self.max_usd,
                "stopped_reason": self.stopped_reason}