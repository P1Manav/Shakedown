import os
import shutil
from pathlib import Path

from shakedown.runner import (
    _build_agent_environment,
    _restore_verification_workspace,
    _protected_snapshot,
)
from shakedown.spec import discover_tasks


def test_agent_environment_allowlist():
    """Test that AWS_ACCESS_KEY_ID, GITHUB_PAT, SSH_AUTH_SOCK, and anything else not allowlisted never reach the agent."""
    tasks = discover_tasks()
    task = tasks[0]

    old_env = dict(os.environ)
    try:
        os.environ["AWS_ACCESS_KEY_ID"] = "AKIA_SECRET_TEST"
        os.environ["GITHUB_PAT"] = "ghp_SECRET_PAT_TEST"
        os.environ["SSH_AUTH_SOCK"] = "/tmp/ssh-agent.sock"
        os.environ["DATABASE_URL"] = "postgres://user:pass@localhost:5432/db"
        os.environ["CUSTOM_SECRET_KEY"] = "super_secret_value"
        os.environ["UNSAFE_HOST_VAR"] = "some_random_host_info"

        workdir = Path("dummy_workdir")
        env = _build_agent_environment(task, "run-env-test", workdir, "http://127.0.0.1:9999/v1")

        # Disallowed variables must NEVER reach agent
        assert "AWS_ACCESS_KEY_ID" not in env
        assert "GITHUB_PAT" not in env
        assert "SSH_AUTH_SOCK" not in env
        assert "DATABASE_URL" not in env
        assert "CUSTOM_SECRET_KEY" not in env
        assert "UNSAFE_HOST_VAR" not in env

        # Allowlisted variables must be present
        assert "PATH" in env
        assert ("HOME" in env or "USERPROFILE" in env)
        assert ("TEMP" in env or "TMP" in env or "SystemRoot" in env or "SYSTEMROOT" in env)
        assert env["OPENAI_BASE_URL"] == "http://127.0.0.1:9999/v1"
        assert env["SHAKEDOWN_RUN_ID"] == "run-env-test"
        assert env["PYTHONUNBUFFERED"] == "1"
        assert env["PYTHONUTF8"] == "1"
        assert env["PYTHONIOENCODING"] == "utf-8"
    finally:
        os.environ.clear()
        os.environ.update(old_env)


def test_verification_workspace_deletes_pycache_and_pyc(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "calc.py").write_text("def median(x): return x[0]\n")
    tests_dir = source / "tests"
    tests_dir.mkdir()
    (tests_dir / "test_calc.py").write_text("def test_sanity(): pass\n")

    workdir = tmp_path / "workdir"
    shutil.copytree(source, workdir)

    # Agent creates __pycache__ and .pyc files
    pycache = workdir / "__pycache__"
    pycache.mkdir()
    (pycache / "calc.cpython-312.pyc").write_bytes(b"dummy bytecode")
    (workdir / "extra.pyc").write_bytes(b"dummy bytecode 2")

    pristine = _protected_snapshot(source)
    verification = tmp_path / "verification"

    tampered = _restore_verification_workspace(workdir, verification, pristine)
    assert not tampered

    # Verify all __pycache__ and *.pyc were removed
    assert not (verification / "__pycache__").exists()
    assert not list(verification.rglob("*.pyc"))


def test_verification_workspace_casefold_comparison(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "calc.py").write_text("def median(x): return x[0]\n")
    tests_dir = source / "tests"
    tests_dir.mkdir()
    (tests_dir / "test_calc.py").write_text("def test_one(): assert 1 == 1\n")

    workdir = tmp_path / "workdir"
    shutil.copytree(source, workdir)

    # Agent edits tests under alternate casing or creates a casing variant
    variant_test = workdir / "tests" / "TEST_CALC.PY"
    variant_test.write_text("def test_tampered(): pass\n")

    pristine = _protected_snapshot(source)
    verification = tmp_path / "verification"

    tampered = _restore_verification_workspace(workdir, verification, pristine)
    assert tampered


def test_verification_workspace_rejects_shadowing_installed_module(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "calc.py").write_text("def median(x): return x[0]\n")
    tests_dir = source / "tests"
    tests_dir.mkdir()
    (tests_dir / "test_calc.py").write_text("def test_one(): pass\n")

    workdir = tmp_path / "workdir"
    shutil.copytree(source, workdir)

    # Agent drops a top-level pytest.py to hijack test runner
    (workdir / "pytest.py").write_text("print('hijacked'); import sys; sys.exit(0)\n")

    pristine = _protected_snapshot(source)
    verification = tmp_path / "verification"

    tampered = _restore_verification_workspace(workdir, verification, pristine)
    assert tampered
    assert not (verification / "pytest.py").exists()


def test_hidden_tests_fail_hardcoded_visible_assertions(tmp_path):
    tasks = discover_tasks()
    task = next(t for t in tasks if t.name == "bugfix_01_median" or t.id == "bugfix_01")
    assert (task.path.parent / "hidden_tests").exists()

    # Agent writes code that only passes the visible test assertions in test_calc.py:
    # [1, 2, 3, 4] -> 2.5, [4, 1, 3, 2] -> 2.5, [10, 2] -> 6, [5, 1, 3] -> 3, [7] -> 7
    hardcoded_solution = """def mean(v): return sum(v) / len(v)
def median(v):
    if v == [1, 2, 3, 4] or v == [4, 1, 3, 2]: return 2.5
    if v == [10, 2]: return 6
    if v == [5, 1, 3]: return 3
    if v == [7]: return 7
    return -999.0
"""
    # Create workdir with this hardcoded solution
    workdir = tmp_path / "workdir"
    shutil.copytree(task.source, workdir)
    (workdir / "calc.py").write_text(hardcoded_solution)

    pristine = _protected_snapshot(task.source)
    verification = tmp_path / "verification"

    _restore_verification_workspace(workdir, verification, pristine, hidden_tests_dir=task.path.parent / "hidden_tests")

    # Visible tests alone would pass on hardcoded solution:
    # But hidden tests are present in verification/tests
    assert any("hidden" in p.name for p in (verification / "tests").iterdir())

    from shakedown.runner import run_command
    env = dict(os.environ)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PYTHONSAFEPATH"] = "1"
    env["PYTHONPATH"] = str(verification)

    code, out, err, timed_out = run_command(task.success_command, verification, env, timeout_s=30)
    # Must fail because hidden tests contain different lists!
    assert code != 0
