"""Fault injectors applied to tool results on the return path."""

from __future__ import annotations

from shakedown.perturb.base import Perturbation


class ChaosPerturbation(Perturbation):
    def __init__(self, fault: str, tool: str, params: dict | None = None):
        params = params or {}
        extra = ",".join(f"{k}={v}" for k, v in sorted(params.items()))
        super().__init__(
            id=f"chaos:{fault}:{tool}" + (f":{extra}" if extra else ""),
            kind="chaos",
            label=f"{fault} → {tool}" + (f" ({extra})" if extra else ""),
            params={"fault": fault, "tool": tool, **params},
        )
        self.fault = fault
        self.tool = tool
        self.keep_chars = int(params.get("keep_chars", 200))
        self.delay_ms = int(params.get("delay_ms", 1000))
        if fault == "added_latency" and self.delay_ms <= 0:
            raise ValueError("added_latency delay_ms must be greater than zero")

    def apply_tool_result(self, tool_name: str, args: dict, content: str, ctx) -> str:
        if tool_name != self.tool:
            return content

        if self.fault == "truncate_output":
            if len(content) <= self.keep_chars:
                ctx.state.exposure_reason = "tool output is shorter than keep_chars"
                return content
            return content[: self.keep_chars] + "\n...[tool output truncated]"

        if self.fault == "empty_result":
            return ""

        if self.fault == "malformed_json":
            partial = content[:200].replace('"', "'")
            return '{"ok": true, "output": "' + partial

        if self.fault == "error_once":
            key = f"error_once:{ctx.state.current_tool_call_id}"
            if key not in ctx.state.fired:
                ctx.state.fired.add(key)
                return (f"ToolError: transient upstream failure (HTTP 503). "
                        f"{self.tool} did not run. Retry if you need the result.")
            return content

        if self.fault == "stale_file":
            path = str(args.get("path", args.get("file", "")))
            key = f"stale:{path}"
            if key not in ctx.state.read_cache:
                ctx.state.read_cache[key] = content
                return content
            return ctx.state.read_cache[key]

        if self.fault == "renamed_path":
            path = str(args.get("path", args.get("file", "")))
            key = f"renamed:{ctx.state.current_tool_call_id}"
            if key not in ctx.state.fired and content and not content.lower().startswith("error"):
                ctx.state.fired.add(key)
                return (f"FileNotFoundError: [Errno 2] No such file or directory: '{path}'. "
                        f"The file may have been renamed; list the directory and try again.")
            return content

        return content