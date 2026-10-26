"""Perturbation interface.

A perturbation sees the outgoing OpenAI-compatible request and every tool
result on its way back to the model.  It must not mutate its inputs in place;
the proxy keeps the original request for traces.
"""

from __future__ import annotations

import copy
import random
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Perturbation:
    id: str
    kind: str                    # baseline | ablation | chaos | noise
    label: str
    params: dict = field(default_factory=dict)

    def apply_request(self, payload: dict, ctx: Any) -> dict:
        return copy.deepcopy(payload)

    def apply_tool_result(self, tool_name: str, args: dict, content: str, ctx: Any) -> str:
        return content

    def describe(self) -> dict:
        return {"id": self.id, "kind": self.kind, "label": self.label, "params": self.params}


class Baseline(Perturbation):
    def __init__(self) -> None:
        super().__init__(id="baseline", kind="baseline", label="Baseline (no perturbation)")

    def apply_request(self, payload: dict, ctx: Any) -> dict:
        return copy.deepcopy(payload)


class RunState:
    """Per-run mutable state shared by perturbations and the proxy."""

    def __init__(self, run_id: str):
        self.run_id = run_id
        self.fired: set[str] = set()
        self.read_cache: dict[str, str] = {}
        self.calls = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.cost_usd = 0.0
        self.cached_calls = 0
        self.errors: list[str] = []
        self.invalid = False
        self.perturbation_records: list[dict[str, Any]] = []
        self.exposure_reason = ""
        self.rng = random.Random(run_id)

    @property
    def perturbation_applied(self) -> bool:
        return any(record["applied"] for record in self.perturbation_records)

    def record_perturbation(self, applied: bool, diff_hash: str = "",
                            reason: str = "") -> None:
        self.perturbation_records.append({
            "applied": applied,
            "diff_hash": diff_hash,
            "reason": "" if applied else reason,
        })
        if applied:
            self.exposure_reason = ""
        elif reason and not self.exposure_reason:
            self.exposure_reason = reason