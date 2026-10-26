import os
import subprocess
import sys
import time
from types import SimpleNamespace

from shakedown import cli, runner


def test_agent_environment_scrubs_secrets_and_enables_utf8(monkeypatch, tmp_path):
    monkeypatch.setenv("MODEL_API_KEY", "model-secret")
    monkeypatch.setenv("GITHUB_TOKEN", "github-secret")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "aws-secret")
    task = SimpleNamespace(
        success_command="python -m pytest",
        max_model_calls=8,
        model=SimpleNamespace(name="open-model", temperature=0.2),
    )

    env = runner._build_agent_environment(
        task, "run-1", tmp_path, "http://127.0.0.1:1234/run/run-1/v1"
    )

    assert env["PYTHONUTF8"] == "1"
    assert env["PYTHONIOENCODING"].lower() == "utf-8"
    assert env["OPENAI_API_KEY"] == "dummy"
    assert "MODEL_API_KEY" not in env
    assert "GITHUB_TOKEN" not in env
    assert "AWS_SECRET_ACCESS_KEY" not in env


def test_run_command_uses_utf8_and_platform_process_group(monkeypatch, tmp_path):
    captured = {}

    class Process:
        pid = 123
        returncode = 0

        def communicate(self, timeout=None):
            return "ok", ""

    def fake_popen(command, **kwargs):
        captured.update(kwargs)
        return Process()

    monkeypatch.setattr(runner.subprocess, "Popen", fake_popen)

    assert runner.run_command("agent", tmp_path, {}, 1) == (0, "ok", "", False)
    assert captured["encoding"] == "utf-8"
    assert captured["errors"] == "replace"
    assert captured.get("start_new_session", False) is (os.name != "nt")


def test_kill_tree_uses_platform_tree_termination(monkeypatch):
    calls = []

    class Process:
        pid = 321

    if os.name == "nt":
        def fake_run(args, **kwargs):
            calls.append((args, kwargs))
            return subprocess.CompletedProcess(args, 0)

        monkeypatch.setattr(runner.subprocess, "run", fake_run)
        runner._kill_tree(Process())
        assert calls[0][0] == ["taskkill", "/F", "/T", "/PID", "321"]
    else:
        monkeypatch.setattr(runner.os, "killpg", lambda pid, sig: calls.append((pid, sig)))
        runner._kill_tree(Process())
        assert calls == [(321, runner.signal.SIGKILL)]


def test_cli_output_reconfigures_both_streams_as_utf8(monkeypatch):
    configured = []

    class Stream:
        def reconfigure(self, **kwargs):
            configured.append(kwargs)

    monkeypatch.setattr(cli.sys, "stdout", Stream())
    monkeypatch.setattr(cli.sys, "stderr", Stream())

    cli._configure_cli_output()

    assert configured == [
        {"encoding": "utf-8", "errors": "replace"},
        {"encoding": "utf-8", "errors": "replace"},
    ]


def test_timeout_terminates_spawned_child_process(tmp_path):
    script = tmp_path / "spawn_child.py"
    pid_file = tmp_path / "child.pid"
    script.write_text(
        "import subprocess, sys, time\n"
        "child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])\n"
        "open(sys.argv[1], 'w').write(str(child.pid))\n"
        "time.sleep(60)\n",
        encoding="utf-8",
    )
    command = f'"{sys.executable}" "{script}" "{pid_file}"'

    exit_code, _, _, timed_out = runner.run_command(command, tmp_path, {}, 1)

    assert timed_out is True
    assert exit_code == -1
    child_pid = int(pid_file.read_text(encoding="utf-8"))
    deadline = time.monotonic() + 2
    while time.monotonic() < deadline:
        if os.name == "nt":
            result = subprocess.run(
                ["tasklist", "/FI", f"PID eq {child_pid}"],
                capture_output=True, text=True,
            )
            alive = str(child_pid) in result.stdout
        else:
            try:
                os.kill(child_pid, 0)
                alive = True
            except ProcessLookupError:
                alive = False
        if not alive:
            break
        time.sleep(0.05)
    assert not alive
