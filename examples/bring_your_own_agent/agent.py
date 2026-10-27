"""Minimal third-party tool-calling agent using OPENAI_BASE_URL.

Connects to Shakedown's proxy (or any OpenAI-compatible server), receives tool
calls, executes file operations locally, and returns results in a loop.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from urllib.request import Request, urlopen


def execute_tool(name: str, args: dict) -> str:
    if name == "read_file":
        p = Path(args.get("path", ""))
        return p.read_text(encoding="utf-8") if p.exists() else f"FileNotFound: {p}"
    if name == "write_file":
        p = Path(args.get("path", ""))
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(args.get("content", ""), encoding="utf-8")
        return f"Wrote {len(args.get('content', ''))} characters to {p}"
    return f"Unknown tool: {name}"


def main() -> None:
    base_url = os.environ.get("OPENAI_BASE_URL", "http://127.0.0.1:8000/v1").rstrip("/")
    api_key = os.environ.get("OPENAI_API_KEY", "dummy")
    model = os.environ.get("MODEL_NAME", "qwen2.5-coder:7b")

    tools = [
        {"type": "function", "function": {"name": "read_file", "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}},
        {"type": "function", "function": {"name": "write_file", "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}}, "required": ["path", "content"]}}},
    ]
    messages = [
        {"role": "system", "content": "You are a coding assistant. Use write_file to edit or create files."},
        {"role": "user", "content": os.environ.get("AGENT_PROMPT", "Create a file named hello.txt with content 'Hello from BYOA'.")},
    ]

    for step in range(int(os.environ.get("SHAKEDOWN_MAX_STEPS", "10"))):
        req = Request(
            f"{base_url}/chat/completions",
            data=json.dumps({"model": model, "messages": messages, "tools": tools, "temperature": 0}).encode("utf-8"),
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"},
            method="POST",
        )
        with urlopen(req, timeout=30) as resp:
            choice = json.load(resp)["choices"][0]["message"]
        messages.append(choice)
        tool_calls = choice.get("tool_calls") or []
        if not tool_calls:
            print(f"Agent finished: {choice.get('content', '')}")
            break
        for call in tool_calls:
            fn = call["function"]
            res = execute_tool(fn["name"], json.loads(fn.get("arguments", "{}")))
            messages.append({"role": "tool", "tool_call_id": call["id"], "content": res})


if __name__ == "__main__":
    main()
