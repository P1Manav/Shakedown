"""Shared parsing helpers for text-encoded and JSON tool calls."""

from __future__ import annotations

import ast
import json
import re


def parse_text_tool_call(content: str, step: int) -> dict | None:
    """Some open models emit a tool call as JSON text instead of tool_calls.

    Accept the common shapes rather than failing the run:
      {"name": "list_dir", "arguments": {"path": "."}}
      {"tool": "list_dir", "args": {...}}
    """
    text = (content or "").strip()
    if not text:
        return None
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.IGNORECASE).strip()
    candidates = [text]
    for block in re.findall(r"```(?:json)?\s*(.*?)```", text, flags=re.IGNORECASE | re.DOTALL):
        try:
            candidates.append(ast.literal_eval(block.strip()))
        except (SyntaxError, ValueError):
            continue
    # Scan for embedded objects too: models sometimes wrap a call in prose.
    decoder = json.JSONDecoder()
    for start in [i for i, ch in enumerate(text) if ch == "{"]:
        try:
            obj, _ = decoder.raw_decode(text[start:])
            candidates.append(obj)
        except Exception:
            continue
    # Some coding models emit Python dict literals with triple-quoted strings.
    if "{" in text and "}" in text:
        try:
            candidates.append(ast.literal_eval(text[text.find("{"):text.rfind("}") + 1]))
        except Exception:
            pass
    for candidate in candidates:
        try:
            obj = json.loads(candidate) if isinstance(candidate, str) else candidate
        except Exception:
            continue
        if not isinstance(obj, dict):
            continue
        name = obj.get("name") or obj.get("tool") or obj.get("function")
        args = obj.get("arguments", obj.get("args", obj.get("parameters", {})))
        if isinstance(name, dict):  # {"function": {"name": ..., "arguments": ...}}
            args = name.get("arguments", args)
            name = name.get("name")
        if isinstance(name, str) and name in {"list_dir", "read_file", "write_file", "run_tests"}:
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except Exception:
                    args = {}
            if not isinstance(args, dict):
                args = {}
            content = args.get("content")
            if name == "write_file" and isinstance(content, str) and "\n" not in content:
                args["content"] = content.replace("\\r\\n", "\n").replace("\\n", "\n")
            return {
                "id": f"textcall_{step}",
                "type": "function",
                "function": {"name": name, "arguments": json.dumps(args)},
            }
    return None
