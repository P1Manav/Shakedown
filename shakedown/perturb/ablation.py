"""Prompt-chunk ablation: remove or shuffle one named chunk at a time."""

from __future__ import annotations

import copy
import re

from shakedown.perturb.base import Perturbation


class AblationPerturbation(Perturbation):
    def __init__(self, chunk_id: str, mode: str, chunk_text: str, kind: str,
                 tool: str | None, all_system_chunks: list[tuple[str, str]] | None = None):
        super().__init__(
            id=f"ablation:{mode}:{chunk_id}",
            kind="ablation",
            label=f"{mode} {chunk_id}",
            params={"chunk": chunk_id, "mode": mode, "target_kind": kind, "tool": tool},
        )
        self.chunk_id = chunk_id
        self.mode = mode
        self.chunk_text = chunk_text
        self.target_kind = kind
        self.tool = tool
        self.all_system_chunks = all_system_chunks or [(chunk_id, chunk_text)]

    def apply_request(self, payload: dict, ctx) -> dict:
        out = copy.deepcopy(payload)
        if self.target_kind == "system":
            system_messages = [
                message for message in out.get("messages", [])
                if message.get("role") == "system" and isinstance(message.get("content"), str)
            ]
            if not any(self.chunk_text in message["content"] for message in system_messages):
                raise ValueError(
                    f"chunk {self.chunk_id!r} not found in the outgoing system prompt"
                )
            for message in out.get("messages", []):
                if message.get("role") == "system" and isinstance(message.get("content"), str):
                    message["content"] = self._transform_system(message["content"], ctx)
        elif self.target_kind == "tool_description" and self.tool:
            matched_tool = False
            for tool in out.get("tools", []) or []:
                fn = tool.get("function", {})
                if fn.get("name") == self.tool:
                    matched_tool = True
                    desc = fn.get("description", "") or ""
                    shuffled = _shuffle_sentences(desc, ctx)
                    if self.mode != "remove" and shuffled == desc and len(
                            re.findall(r"(?<=[.!?])\s+", desc.strip())) == 0:
                        ctx.state.exposure_reason = (
                            "tool description has fewer than two sentences"
                        )
                    fn["description"] = "" if self.mode == "remove" else shuffled
            if not matched_tool:
                ctx.state.exposure_reason = "target tool description was not provided"
        return out

    def _transform_system(self, content: str, ctx) -> str:
        if self.mode == "remove":
            new = content.replace(self.chunk_text, "")
            new = re.sub(r"\n{3,}", "\n\n", new).strip()
            return new
        # Shuffle: rebuild the system message from the chunk blocks that are
        # present, placing any residual text (e.g. an injected fix line) last.
        present = [(cid, text) for cid, text in self.all_system_chunks if text and text in content]
        if len(present) < 2:
            return content
        residual = content
        for _, text in present:
            residual = residual.replace(text, "")
        residual = re.sub(r"\n{3,}", "\n\n", residual).strip()
        ctx.state.rng.shuffle(present)
        blocks = [text for _, text in present]
        if residual:
            blocks.append(residual)
        return "\n\n".join(blocks)


def _shuffle_sentences(text: str, ctx) -> str:
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
    if len(sentences) < 2:
        return text
    ctx.state.rng.shuffle(sentences)
    return " ".join(sentences)