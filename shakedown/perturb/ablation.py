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
                    if self.mode != "remove" and len(_sentences(desc)) < 2:
                        ctx.state.exposure_reason = (
                            "tool description shuffle skipped: fewer than two sentences"
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
        present = [
            (cid, text) for cid, text in self.all_system_chunks
            if text and text in content
        ]
        if len(present) < 2 or self.chunk_id not in {cid for cid, _ in present}:
            return content
        present.sort(key=lambda item: content.index(item[1]))
        original_index = next(i for i, (cid, _) in enumerate(present) if cid == self.chunk_id)
        target = present.pop(original_index)
        destination = ctx.state.rng.choice(
            [i for i in range(len(present) + 1) if i != original_index]
        )
        present.insert(destination, target)

        # Rebuild only the known chunk blocks; residual text (such as a fix
        # line) remains untouched and in its original trailing position.
        residual = content
        for _, text in self.all_system_chunks:
            if text:
                residual = residual.replace(text, "", 1)
        residual = re.sub(r"\n{3,}", "\n\n", residual).strip()
        blocks = [text for _, text in present]
        if residual:
            blocks.append(residual)
        return "\n\n".join(blocks)


def _sentences(text: str) -> list[str]:
    return [sentence.strip() for sentence in re.split(r"(?<=[.!?])\s+", text) if sentence.strip()]


def _shuffle_sentences(text: str, ctx) -> str:
    sentences = _sentences(text)
    if len(sentences) < 2:
        return text
    original = list(sentences)
    ctx.state.rng.shuffle(sentences)
    if sentences == original:
        sentences = sentences[1:] + sentences[:1]
    return " ".join(sentences)