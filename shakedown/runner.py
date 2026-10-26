"""Sandboxed single-run execution.

Each run gets a fresh copy of the task workspace, a unique proxy route, a
subprocess with a wall-clock timeout, a model-call cap enforced by the proxy,
and an executable success check that runs *after* the agent exits.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

from shakedown.claims import detect_claim
from shakedown.perturb.base import Perturbation
from shakedown.proxy import ProxyServer, RunContext
from shakedown.spec import TaskSpec
from shakedown.store import RunResult, Store, utc_now
from shakedown.taxonomy import classify_run

ROOT = Path(__file__).resolve().parents[1]


def _kill_tree(proc: subprocess.Popen) -> None:
    try:
        if os.name == "nt":
            subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                           capture_output=True, timeout=10)
        else:
            proc.kill()
    except Exception:
        pass


def run_command(command: str, cwd: Path, env: dict, timeout_s: float) -> tuple[int, str, str, bool]:
    proc = subprocess.Popen(
        command, cwd=str(cwd), env=env, shell=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, encoding="utf-8", errors="replace")
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
            return f"run-{self._counter:05d}"

    def run(self, task: TaskSpec, cell_id: str, cell_label: str, cell_kind: str,
            perturbation: Perturbation, params: dict | None = None) -> RunResult:
        run_id = self.next_run_id()
        workdir = self.work_root / run_id
        if workdir.exists():
            shutil.rmtree(workdir, ignore_errors=True)
        shutil.copytree(task.source, workdir)

        ctx = RunContext(run_id=run_id, task_id=task.id, cell_id=cell_id,
                         perturbation=perturbation, max_calls=task.max_model_calls,
                         store=self.store, pricing=self._pricing(task),
                         fix_line=self.fix_line)
        self.proxy.register(ctx)

        env = dict(os.environ)
        env.update({
            "OPENAI_BASE_URL": self.proxy.base_url(run_id),
            "OPENAI_API_KEY": "dummy",
            "SHAKEDOWN_RUN_ID": run_id,
            "MODEL_NAME": task.model.name,
            "MODEL_TEMPERATURE": str(task.model.temperature),
            "SHAKEDOWN_TEST_CMD": task.success_command,
            "SHAKEDOWN_MAX_STEPS": str(task.max_model_calls),
            "PYTHONPATH": str(ROOT) + (os.pathsep + env.get("PYTHONPATH", "") if env.get("PYTHONPATH") else ""),
            "PYTHONUNBUFFERED": "1",
        })
        env.pop("MODEL_BASE_URL", None)

        command = task.agent_command.format(
            task_file=f'"{task.path}"', workdir=f'"{workdir}"')

        started = utc_now()
        t0 = time.perf_counter()
        agent_exit, stdout, stderr, timed_out = run_command(
            command, workdir, env, task.agent_timeout_s)
        duration = time.perf_counter() - t0

        test_exit, test_out, test_err, test_timeout = run_command(
            task.success_command, workdir, env, task.success_timeout_s)
        success = (not test_timeout) and test_exit == 0

        final, actions, events = parse_agent_output(stdout)
        claimed = detect_claim(final)
        stats = ctx.state
        self.proxy.unregister(run_id)

        result = RunResult(
            run_id=run_id, experiment_id=self.store.experiment_id,
            task_id=task.id, task_name=task.name, cell_id=cell_id,
            cell_label=cell_label, cell_kind=cell_kind, params=params or {},
            started_at=started, duration_s=round(duration, 3), success=success,
            exit_code=test_exit, agent_exit=agent_exit, timed_out=timed_out,
            model_calls=stats.calls, prompt_tokens=stats.prompt_tokens,
            completion_tokens=stats.completion_tokens,
            cost_usd=round(stats.cost_usd, 8),
            steps=len(actions), actions=actions, final_message=final,
            claimed_success=claimed,
            events=events + ([{"event": "stderr", "text": stderr[-800:]}] if stderr.strip() else []),
            test_output=(test_out + "\n" + test_err).strip()[-4000:],
            error="; ".join(stats.errors[-5:]),
            trace_path=str(self.store.trace_path(run_id)),
            cached_calls=stats.cached_calls,
        )
        result.failure_class = classify_run(result.to_dict(), task.max_model_calls)
        self.store.append_run(result)

        if not self.keep_workspaces:
            shutil.rmtree(workdir, ignore_errors=True)
        return result

    @staticmethod
    def _pricing(task: TaskSpec):
        from shakedown.cost import Pricing
        return Pricing(task.model.price_input_per_1m, task.model.price_output_per_1m)