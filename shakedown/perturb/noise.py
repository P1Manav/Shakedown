"""Input-noise perturbations (Tier 3)."""

from __future__ import annotations

import copy
import re

from shakedown.perturb.base import Perturbation


class NoisePerturbation(Perturbation):
    def __init__(self, mode: str):
        super().__init__(id=f"noise:{mode}", kind="noise", label=f"noise: {mode}",
                         params={"mode": mode})
        self.mode = mode

    def apply_request(self, payload: dict, ctx) -> dict:
        out = copy.deepcopy(payload)
        if self.mode == "reorder_tools":
            if out.get("tools"):
                out["tools"] = list(reversed(out["tools"]))
        elif self.mode == "typos":
            for message in out.get("messages", []):
                if message.get("role") == "user" and isinstance(message.get("content"), str):
                    message["content"] = _typo(message["content"], ctx)
        return out


def _typo(text: str, ctx) -> str:
    """Swap two adjacent letters in a few long words, deterministically."""
    words = text.split(" ")
    for i, word in enumerate(words):
        if len(word) > 6 and ctx.rng.random() < 0.08:
            j = ctx.rng.randrange(1, len(word) - 2)
            words[i] = word[:j] + word[j + 1] + word[j] + word[j + 2:]
    return " ".join(words)