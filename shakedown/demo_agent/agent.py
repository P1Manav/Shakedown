"""A deliberately small ReAct-style coding agent.

It talks to any OpenAI-compatible endpoint (Shakedown's proxy by default),
edits files in a working directory, and runs the repository's tests.  It has no
knowledge of Shakedown's perturbations: the proxy sits in the HTTP path.
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import httpx
import yaml

from shakedown.model_config import DEFAULT_MODEL_NAME

PKG = Path(__file__).resolve().parent
CHUNK_DIR = PKG / "prompt_chunks"
MAX_OBSERVATION = 8000


def load_chunks() -> dict:
    manifest = yaml.safe_load(
        (CHUNK_DIR / "manifest.yaml").read_text(encoding="utf-8-sig")
    )
    chunks = []
    for item in manifest["chunks"]:
        chunks.append({**item, "text": (CHUNK_DIR / item["file"]).read_text(encoding="utf-8").strip()})
    return {"manifest": manifest["chunks"], "chunks": chunks}


def build_prompt(chunks: list[dict]) -> tuple[str, dict[str, str]]:
    system_parts = [c["text"] for c in chunks if c["kind"] == "system"]
    descriptions = {c["tool"]: c["text"] for c in chunks if c["kind"] == "tool_description"}
    return "\n\n".join(system_parts), descriptions


def tool_schemas(descriptions: dict[str, str]) -> list[dict]:
    def fn(name: str, desc: str, props: dict, required: list[str]) -> dict:
        return {"type": "function", "function": {
            "name": name, "description": desc,
            "parameters": {"type": "object", "properties": props, "required": required},
        }}

    return [
        fn("list_dir", descriptions.get("list_dir", "List a directory."),
           {"path": {"type": "string", "description": "Relative directory path; use '.' for the root."}}, ["path"]),
        fn("read_file", descriptions.get("read_file", "Read a file."),
           {"path": {"type": "string", "description": "Relative file path."}}, ["path"]),
        fn("write_file", descriptions.get("write_file", "Write a file."),
           {"path": {"type": "string", "description": "Relative file path."},
            "content": {"type": "string", "description": "Complete new file contents."}}, ["path", "content"]),
        fn("run_tests", descriptions.get("run_tests", "Run the tests."), {}, []),
    ]


def safe_path(workdir: Path, rel: str) -> Path:
    target = (workdir / rel).resolve()
    if target != workdir.resolve() and workdir.resolve() not in target.parents:
        raise ValueError(f"path escapes the working directory: {rel}")
    return target


def execute_tool(name: str, args: dict, workdir: Path, test_cmd: str) -> tuple[str, bool]:
    """Return (observation, is_error)."""
    try:
        if name == "list_dir":
            target = safe_path(workdir, args.get("path", "."))
            if not target.is_dir():
                return f"error: not a directory: {args.get('path', '.')}", True
            entries = sorted(p.name + ("/" if p.is_dir() else "") for p in target.iterdir()
                             if p.name not in {"__pycache__", ".pytest_cache"})
            return "\n".join(entries) or "(empty directory)", False
        if name == "read_file":
            target = safe_path(workdir, args["path"])
            if not target.is_file():
                return f"error: no such file: {args['path']}", True
            return target.read_text(encoding="utf-8", errors="replace")[:MAX_OBSERVATION], False
        if name == "write_file":
            target = safe_path(workdir, args["path"])
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(args["content"], encoding="utf-8")
            return f"wrote {args['path']} ({len(args['content'])} bytes)", False
        if name == "run_tests":
            proc = subprocess.run(test_cmd, shell=True, cwd=workdir, capture_output=True,
                                  text=True, timeout=180)
            out = (proc.stdout + "\n" + proc.stderr).strip()
            # Exit code last: a middleware that truncates the head of a long
            # observation can genuinely hide the verdict, as it does in the wild.
            return f"{out[:MAX_OBSERVATION]}\n\nexit_code: {proc.returncode}", proc.returncode != 0
        return f"error: unknown tool {name}", True
    except subprocess.TimeoutExpired:
        return "error: test command timed out after 180s", True
    except Exception as exc:  # noqa: BLE001 - observation, not a crash
        return f"error: {type(exc).__name__}: {exc}", True


def emit(event: dict) -> None:
    print("SHAKEDOWN_EVENT: " + json.dumps(event, ensure_ascii=False), flush=True)


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
            return {"id": f"textcall_{step}", "type": "function",
                    "function": {"name": name, "arguments": json.dumps(args)}}
    return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True)
    parser.add_argument("--workdir", required=True)
    args = parser.parse_args(argv)

    workdir = Path(args.workdir).resolve()
    task = yaml.safe_load(Path(args.task).read_text(encoding="utf-8-sig"))
    base_url = os.environ.get("OPENAI_BASE_URL", "http://127.0.0.1:11434/v1").rstrip("/")
    api_key = os.environ.get("OPENAI_API_KEY", "dummy")
    model = os.environ.get("MODEL_NAME", DEFAULT_MODEL_NAME)
    temperature = float(os.environ.get("MODEL_TEMPERATURE", "0.7"))
    max_steps = int(os.environ.get("SHAKEDOWN_MAX_STEPS", "12"))
    test_cmd = os.environ.get("SHAKEDOWN_TEST_CMD", "python -m pytest -q tests/")

    loaded = load_chunks()
    system_prompt, descriptions = build_prompt(loaded["chunks"])
    tools = tool_schemas(descriptions)

    user = (
        f"Task: {task['name']}\n\n{task['description']}\n\n"
        f"The repository is in the current working directory ({workdir}). "
        "Tests live in tests/. Fix the bug, then run the tests. "
        "Follow the response protocol exactly. When the suite passes, reply with DONE: and one sentence."
    )
    messages: list[dict] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user},
    ]

    url = base_url + "/chat/completions"
    final_message = ""
    tool_actions: list[str] = []
    nudges = 0
    with httpx.Client(timeout=300.0) as client:
        for step in range(max_steps):
            payload = {"model": model, "messages": messages, "tools": tools,
                       "temperature": temperature, "max_tokens": 700}
            try:
                resp = client.post(url, json=payload,
                                   headers={"Authorization": f"Bearer {api_key}"})
                resp.raise_for_status()
                data = resp.json()
            except Exception as exc:  # noqa: BLE001
                emit({"event": "api_error", "step": step, "error": str(exc)[:500]})
                print(f"SHAKEDOWN_FINAL: [agent stopped: model endpoint error: {exc}]", flush=True)
                return 3

            msg = data["choices"][0]["message"]
            calls = msg.get("tool_calls") or []
            assistant = {"role": "assistant", "content": msg.get("content") or ""}
            if not calls:
                text_call = parse_text_tool_call(msg.get("content") or "", step)
                if text_call:
                    calls = [text_call]
                    assistant = {"role": "assistant", "content": "", "tool_calls": calls}
            if calls:
                assistant["tool_calls"] = calls
            messages.append(assistant)

            if not calls:
                # A competent coding agent inspects before answering. If the
                # model tries to answer from memory, nudge it at most twice;
                # after that its behaviour is measured, not corrected.
                if not tool_actions and nudges < 2:
                    nudges += 1
                    emit({"event": "nudge", "reason": "no tools used yet"})
                    messages.append({"role": "user", "content":
                                     "You have not used any tools. Call list_dir on '.' now, "
                                     "then read the files. Do not answer from memory."})
                    continue
                final_message = (msg.get("content") or "").strip()
                break

            for call in calls:
                fn = call.get("function", {})
                name = fn.get("name", "")
                raw_args = fn.get("arguments", "{}")
                try:
                    call_args = json.loads(raw_args) if isinstance(raw_args, str) else dict(raw_args)
                    parse_error = False
                except Exception:
                    call_args = {}
                    parse_error = True
                    emit({"event": "tool_error", "tool": name, "error": "malformed arguments",
                          "raw": str(raw_args)[:300]})
                observation, is_error = (("error: malformed tool arguments, retry with valid JSON", True)
                                         if parse_error else execute_tool(name, call_args, workdir, test_cmd))
                if is_error:
                    emit({"event": "tool_error", "tool": name, "error": observation[:300]})
                action = f"{name}:{_primary_arg(call_args)}"
                tool_actions.append(action)
                emit({"event": "tool_call", "step": step, "tool": name, "arg": _primary_arg(call_args),
                      "ok": not is_error})
                messages.append({"role": "tool", "tool_call_id": call.get("id", f"call_{step}"),
                                 "content": observation})
            # Loop guard: three identical actions in a row is almost always a stall.
            if len(tool_actions) >= 3 and len(set(tool_actions[-3:])) == 1:
                emit({"event": "loop_detected", "action": tool_actions[-1]})
                break

    if not final_message:
        final_message = "[agent stopped without a final message]"
    one_line = re.sub(r"\s+", " ", final_message).strip()
    print(f"SHAKEDOWN_FINAL: {one_line}", flush=True)
    emit({"event": "done", "steps": len(tool_actions), "actions": tool_actions})
    return 0


def _primary_arg(args: dict) -> str:
    for key in ("path", "file", "filename", "name"):
        if key in args:
            return str(args[key])
    if "content" in args:
        return f"<{len(str(args['content']))} bytes>"
    return "-"


if __name__ == "__main__":
    sys.exit(main())