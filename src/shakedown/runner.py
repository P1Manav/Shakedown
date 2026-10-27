"""Sandboxed single-run execution.

Each run gets a fresh copy of the task workspace, a unique proxy route, a
subprocess with a wall-clock timeout, a model-call cap enforced by the proxy,
and an executable success check that runs *after* the agent exits.
"""

from __future__ import annotations

import json
import hashlib
import os
import re
import signal
import shutil
import subprocess
import tempfile
import threading
import time
import uuid
from pathlib import Path

from shakedown.claims import detect_claim
from shakedown.perturb.base import Perturbation
from shakedown.proxy import ProxyServer, RunContext
from shakedown.spec import TaskSpec
from shakedown.store import RunResult, Store, utc_now
from shakedown.taxonomy import classify_run

ROOT = Path(__file__).resolve().parents[1]
_PROTECTED_CONFIGS = {
    "pytest.ini", "tox.ini", "setup.cfg", "setup.py", "pyproject.toml",
    "conftest.py", "sitecustomize.py", "pytest.py",
}
_SECRET_ENV_NAME = re.compile(
    r"(?:API[_-]?KEY|SECRET|TOKEN|PASSWORD|PASSWD|CREDENTIAL|AUTHORIZATION)",
    re.IGNORECASE,
)


def _is_protected(relative: Path) -> bool:
    if relative.suffix == ".pyc" or any(
        part in {"__pycache__", ".pytest_cache"} for part in relative.parts
    ):
        return False
    return bool(relative.parts and (
        relative.parts[0] == "tests"
        or relative.name in _PROTECTED_CONFIGS
        or relative.name == "pytest"
    ))


def _protected_snapshot(source: Path) -> dict[Path, bytes]:
    return {
        path.relative_to(source): path.read_bytes()
        for path in source.rglob("*")
        if path.is_file() and _is_protected(path.relative_to(source))
    }


def _restore_verification_workspace(workdir: Path, verification: Path,
                                    pristine: dict[Path, bytes]) -> bool:
    shutil.copytree(workdir, verification)
    actual = {
        path.relative_to(verification): path.read_bytes()
        for path in verification.rglob("*")
        if path.is_file() and _is_protected(path.relative_to(verification))
    }
    tampered = actual != pristine
    tests_dir = verification / "tests"
    if tests_dir.exists():
        shutil.rmtree(tests_dir)

    for path in sorted(
        (p for p in verification.rglob("*") if p.is_dir() and p.name == "pytest"),
        key=lambda item: len(item.parts),
        reverse=True,
    ):
        relative = path.relative_to(verification)
        if not any(relative in expected.parents for expected in pristine):
            tampered = True
            shutil.rmtree(path)

    candidates = [
        path for path in verification.rglob("*")
        if path.is_file() and _is_protected(path.relative_to(verification))
    ]
    for path in candidates:
        relative = path.relative_to(verification)
        if relative not in pristine:
            path.unlink()
    for relative, content in pristine.items():
        destination = verification / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(content)
    return tampered


def _kill_tree(proc: subprocess.Popen) -> None:
    if os.name == "nt":
        result = subprocess.run(
            ["taskkill", "/F", "/T", "/PID", str(proc.pid)],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode and proc.poll() is None:
            raise RuntimeError(
                f"failed to terminate process tree {proc.pid}: {result.stderr.strip()}"
            )
        return
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


def _build_agent_environment(task: TaskSpec, run_id: str, workdir: Path,
                             proxy_base_url: str) -> dict[str, str]:
    env = {
        key: value for key, value in os.environ.items()
        if not _SECRET_ENV_NAME.search(key)
    }
    env.update({
        "OPENAI_BASE_URL": proxy_base_url,
        "OPENAI_API_KEY": "dummy",
        "SHAKEDOWN_RUN_ID": run_id,
        "MODEL_NAME": task.model.name,
        "MODEL_TEMPERATURE": str(task.model.temperature),
        "SHAKEDOWN_TEST_CMD": task.success_command,
        "SHAKEDOWN_MAX_STEPS": str(task.max_model_calls),
        "PYTHONPATH": str(ROOT) + (
            os.pathsep + env.get("PYTHONPATH", "") if env.get("PYTHONPATH") else ""
        ),
        "PYTHONUNBUFFERED": "1",
        "PYTHONUTF8": "1",
        "PYTHONIOENCODING": "utf-8",
    })
    env.pop("MODEL_BASE_URL", None)
    return env


def run_command(command: str, cwd: Path, env: dict, timeout_s: float) -> tuple[int, str, str, bool]:
    proc = subprocess.Popen(
        command, cwd=str(cwd), env=env, shell=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, encoding="utf-8", errors="replace",
        **({"start_new_session": True} if os.name != "nt" else {}))
    try:
        out, err = proc.communicate(timeout=timeout_s)
        return proc.returncode, out or "", err or "", False
    except subprocess.TimeoutExpired:
        _kill_tree(proc)
        try:
            out, err = proc.communicate(timeout=10)
        except Exception:
            out, err = "", ""
        return -1, out or "", err or "", True


def parse_agent_output(stdout: str) -> tuple[str, list[str], list[dict]]:
    final = ""
    actions: list[str] = []
    events: list[dict] = []
    for line in stdout.splitlines():
        if line.startswith("SHAKEDOWN_EVENT: "):
            try:
                event = json.loads(line[len("SHAKEDOWN_EVENT: "):])
                events.append(event)
                if event.get("event") == "done" and isinstance(event.get("actions"), list):
                    actions = [str(a) for a in event["actions"]]
            except json.JSONDecodeError:
                continue
        elif line.startswith("SHAKEDOWN_FINAL: "):
            final = line[len("SHAKEDOWN_FINAL: "):].strip()
    return final, actions, events


def parse_proxy_trace(trace: list[dict]) -> dict:
    actions: list[str] = []
    normalized_actions: list[str] = []
    events: list[dict] = []
    final_message = ""

    for trace_event in trace:
        if trace_event.get("event") != "response":
            continue
        body = trace_event.get("body")
        if not isinstance(body, dict):
            continue
        calls = []
        response_text = []
        if body.get("stream"):
            streamed_calls: dict[int, dict] = {}
            for chunk in body.get("events", []):
                if not isinstance(chunk, dict):
                    continue
                for choice in chunk.get("choices") or []:
                    delta = choice.get("delta") or {}
                    content = delta.get("content")
                    if isinstance(content, str):
                        response_text.append(content)
                    for index, call in enumerate(delta.get("tool_calls") or []):
                        call_index = int(call.get("index", index))
                        accumulated = streamed_calls.setdefault(
                            call_index,
                            {"id": "", "type": "function",
                             "function": {"name": "", "arguments": ""}},
                        )
                        accumulated["id"] += str(call.get("id") or "")
                        function = call.get("function") or {}
                        accumulated["function"]["name"] += str(function.get("name") or "")
                        arguments = function.get("arguments") or ""
                        if isinstance(arguments, str):
                            accumulated["function"]["arguments"] += arguments
                    message = choice.get("message")
                    if isinstance(message, dict):
                        text = _message_text(message.get("content"))
                        if text:
                            response_text.append(text)
                        calls.extend(message.get("tool_calls") or [])
            calls.extend(streamed_calls[index] for index in sorted(streamed_calls))
        else:
            for choice in body.get("choices") or []:
                message = choice.get("message") or {}
                text = _message_text(message.get("content"))
                if text:
                    response_text.append(text)
                calls.extend(message.get("tool_calls") or [])

        text = "".join(response_text).strip()
        if text and not calls:
            from shakedown.demo_agent.agent import parse_text_tool_call

            text_call = parse_text_tool_call(text, len(actions))
            if text_call:
                calls.append(text_call)
                text = ""
        if text:
            final_message = text
        for call in calls:
            function = call.get("function") or {}
            name = str(function.get("name") or "")
            raw_args = function.get("arguments") or {}
            try:
                args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
            except json.JSONDecodeError:
                args = {}
                events.append({
                    "event": "tool_error", "tool": name,
                    "error": "malformed tool arguments in proxy trace",
                })
            if not isinstance(args, dict):
                args = {}
                events.append({
                    "event": "tool_error", "tool": name,
                    "error": "tool arguments were not an object",
                })
            primary = next(
                (str(args[key]) for key in ("path", "file", "filename", "name")
                 if key in args),
                f"<{len(str(args['content']))} bytes>" if "content" in args else "-",
            )
            action = f"{name}:{primary}"
            actions.append(action)
            normalized_args = {
                key: _normalize_trace_value(value)
                for key, value in sorted(args.items())
            }
            normalized_actions.append(
                f"{name}({json.dumps(normalized_args, sort_keys=True, separators=(',', ':'))})"
            )
            events.append({
                "event": "tool_call",
                "step": len(actions) - 1,
                "tool": name,
                "arg": primary,
            })

    for index in range(len(actions) - 2):
        if actions[index] == actions[index + 1] == actions[index + 2]:
            events.append({"event": "loop_detected", "action": actions[index]})
            break
    return {
        "steps": len(actions),
        "actions": actions,
        "normalized_actions": normalized_actions,
        "final_message": final_message,
        "claimed_success": detect_claim(final_message),
        "events": events,
    }


def _normalize_trace_value(value):
    if isinstance(value, str) and len(value) > 256:
        return {
            "sha256": hashlib.sha256(value.encode("utf-8")).hexdigest(),
            "length": len(value),
        }
    if isinstance(value, list):
        return [_normalize_trace_value(item) for item in value]
    if isinstance(value, dict):
        return {
            key: _normalize_trace_value(item)
            for key, item in sorted(value.items())
        }
    return value


def _message_text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            str(part.get("text") or "")
            for part in content if isinstance(part, dict)
        )
    return ""


class Runner:
    def __init__(self, store: Store, proxy: ProxyServer, keep_workspaces: bool = False,
                 fix_line: str = ""):
        self.store = store
        self.proxy = proxy
        self.keep_workspaces = keep_workspaces
        self.fix_line = fix_line
        self._counter = 0
        self._lock = threading.Lock()
        self.work_root = Path(tempfile.gettempdir()) / "shakedown" / store.experiment_id
        self.work_root.mkdir(parents=True, exist_ok=True)

    def next_run_id(self) -> str:
        with self._lock:
            self._counter += 1
            return f"run-{uuid.uuid4().hex}"

    def run(self, task: TaskSpec, cell_id: str, cell_label: str, cell_kind: str,
            perturbation: Perturbation, params: dict | None = None,
            replicate: int = -1, fix_line: str | None = None) -> RunResult:
        effective_fix_line = self.fix_line if fix_line is None else fix_line
        run_id = self.next_run_id()
        workdir = self.work_root / run_id
        if workdir.exists():
            shutil.rmtree(workdir, ignore_errors=True)
        shutil.copytree(task.source, workdir)
        pristine_protected = _protected_snapshot(task.source)

        ctx = RunContext(run_id=run_id, task_id=task.id, cell_id=cell_id,
                         perturbation=perturbation, max_calls=task.max_model_calls,
                         store=self.store, pricing=self._pricing(task),
                         fix_line=effective_fix_line, model_name=task.model.name,
                         temperature=task.model.temperature,
                         supports_seed=task.model.supports_seed,
                         prompt_chunks=[
                             {"id": chunk.id, "kind": chunk.kind,
                              "tool": chunk.tool, "text": chunk.text}
                             for chunk in task.chunks
                         ])
        self.proxy.register(ctx)

        env = _build_agent_environment(
            task, run_id, workdir, self.proxy.base_url(run_id)
        )

        command = task.agent_command.format(
            task_file=f'"{task.path}"', workdir=f'"{workdir}"')

        started = utc_now()
        t0 = time.perf_counter()
        agent_exit, stdout, stderr, timed_out = run_command(
            command, workdir, env, task.agent_timeout_s)
        duration = time.perf_counter() - t0

        verification = self.work_root / f"{run_id}-verification"
        if verification.exists():
            shutil.rmtree(verification, ignore_errors=True)
        tamper_detected = _restore_verification_workspace(
            workdir, verification, pristine_protected
        )
        test_exit, test_out, test_err, test_timeout = run_command(
            task.success_command, verification, env, task.success_timeout_s)
        success = (not tamper_detected and not test_timeout and test_exit == 0)

        marker_final, marker_actions, marker_events = parse_agent_output(stdout)
        trace_metrics = parse_proxy_trace(self.store.load_trace(run_id))
        stats = ctx.state
        if not stats.perturbation_applied and not stats.exposure_reason:
            stats.exposure_reason = "no perturbation opportunity occurred"
        self.store.append_trace(run_id, {
            "ts": utc_now(),
            "event": "perturbation_summary",
            "perturbation": perturbation.id,
            "applied": stats.perturbation_applied,
            "diff_hashes": [
                record["diff_hash"] for record in stats.perturbation_records
                if record["diff_hash"]
            ],
            "reason": stats.exposure_reason if not stats.perturbation_applied else "",
        })
        self.proxy.unregister(run_id)

        result = RunResult(
            run_id=run_id, experiment_id=self.store.experiment_id,
            task_id=task.id, task_name=task.name, cell_id=cell_id,
            cell_label=cell_label, cell_kind=cell_kind, replicate=replicate,
            params=params or {},
            started_at=started, duration_s=round(duration, 3), success=success,
            invalid=stats.invalid,
            tamper_detected=tamper_detected,
            perturbation_applied=stats.perturbation_applied,
            perturbation_records=stats.perturbation_records,
            perturbation_diff_hashes=[
                record["diff_hash"] for record in stats.perturbation_records
                if record["diff_hash"]
            ],
            exposure_reason=stats.exposure_reason,
            exit_code=test_exit, agent_exit=agent_exit, timed_out=timed_out,
            model_calls=stats.calls, prompt_tokens=stats.prompt_tokens,
            completion_tokens=stats.completion_tokens,
            cost_usd=round(stats.cost_usd, 8),
            steps=trace_metrics["steps"], actions=trace_metrics["actions"],
            normalized_actions=trace_metrics["normalized_actions"],
            final_message=trace_metrics["final_message"],
            claimed_success=trace_metrics["claimed_success"],
            events=trace_metrics["events"] + (
                [{"event": "stderr", "text": stderr[-800:]}] if stderr.strip() else []
            ),
            agent_markers={
                "final": marker_final,
                "actions": marker_actions,
                "events": marker_events,
            },
            test_output=(test_out + "\n" + test_err).strip()[-4000:],
            error="; ".join(stats.errors[-5:]),
            trace_path=str(self.store.trace_path(run_id)),
            cached_calls=stats.cached_calls,
        )
        result.failure_class = classify_run(result.to_dict(), task.max_model_calls)
        self.store.append_run(result)
        self.store.append_trace(run_id, {
            "ts": utc_now(),
            "event": "verification",
            "tamper_detected": tamper_detected,
            "test_exit": test_exit,
            "test_timeout": test_timeout,
        })

        if not self.keep_workspaces:
            shutil.rmtree(workdir, ignore_errors=True)
            shutil.rmtree(verification, ignore_errors=True)
        return result

    @staticmethod
    def _pricing(task: TaskSpec):
        from shakedown.cost import Pricing
        return Pricing(task.model.price_input_per_1m, task.model.price_output_per_1m)