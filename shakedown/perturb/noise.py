"""Input-noise perturbations (Tier 3)."""

from __future__ import annotations

import copy
import re

from shakedown.perturb.base import Perturbation

_PARAPHRASE_TERMS = {
    "please": "kindly",
    "fix": "repair",
    "bug": "defect",
    "function": "routine",
    "tests": "test suite",
}


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
        elif self.mode == "paraphrase":
            for message in out.get("messages", []):
                if message.get("role") == "user" and isinstance(message.get("content"), str):
                    message["content"] = _paraphrase(message["content"])
        return out


def _paraphrase(text: str) -> str:
    """Apply a small, deterministic synonym set without external model calls."""
    for source, replacement in _PARAPHRASE_TERMS.items():
        text = re.sub(
            rf"\b{source}\b",
            lambda match: (
                replacement.upper() if match.group().isupper()
                else replacement.capitalize() if match.group()[0].isupper()
                else replacement
            ),
            text,
            flags=re.IGNORECASE,
        )
    return text


def _typo(text: str, ctx) -> str:
    """Swap two adjacent letters in a few long words, deterministically."""
    words = text.split(" ")
    for i, word in enumerate(words):
        if len(word) > 6 and ctx.state.rng.random() < 0.08:
            j = ctx.state.rng.randrange(1, len(word) - 2)
            words[i] = word[:j] + word[j + 1] + word[j] + word[j + 2:]
    return " ".join(words)