from pathlib import Path
from types import SimpleNamespace

from shakedown.cost import BudgetTracker, Pricing
from shakedown.perturb.base import Baseline
from shakedown.proxy import ProxyServer
from shakedown.runner import (
    Runner,
    _restore_verification_workspace,
    _SHADOW_COLLISION_NAMES,
)
from shakedown.store import Store


def test_agent_adding_extra_test_file_is_not_tampering(tmp_path):
    source = tmp_path / "task-source"
    tests_dir = source / "tests"
    tests_dir.mkdir(parents=True)
    (source / "src.py").write_text('VALUE = "broken"\n', encoding="utf-8")
    (tests_dir / "test_guard.py").write_text(
        'import src\n\ndef test_expected_behavior():\n    assert src.VALUE == "fixed"\n',
        encoding="utf-8",
    )
    (source / "pytest.ini").write_text("[pytest]\n", encoding="utf-8")
    # Agent fixes src.py AND writes an extra test file tests/test_extra.py
    (source / "agent.py").write_text(
        "from pathlib import Path\n"
        "Path('src.py').write_text('VALUE = \"fixed\"\\n')\n"
        "Path('tests/test_extra.py').write_text('def test_new(): assert True\\n')\n",
        encoding="utf-8",
    )
    task_path = tmp_path / "task.yaml"
    task_path.write_text("name: task\n", encoding="utf-8")
    task = SimpleNamespace(
        id="extra-test-task",
        name="Extra test task",
        source=source,
        path=task_path,
        agent_command="python agent.py {task_file} {workdir}",
        agent_timeout_s=20,
        success_command="python -m pytest -q tests",
        success_timeout_s=30,
        max_model_calls=5,
        chunks=[],
        model=SimpleNamespace(
            name="test-model",
            temperature=0.0,
            supports_seed=False,
            price_input_per_1m=0.0,
            price_output_per_1m=0.0,
        ),
    )
    store = Store("extra-test-run", tmp_path / "runs")
    proxy = ProxyServer(
        "http://127.0.0.1:11434/v1",
        store,
        BudgetTracker(max_calls=10, max_usd=1.0),
        Pricing(),
    )

    result = Runner(store, proxy).run(
        task, "baseline", "Baseline", "baseline", Baseline()
    )

    assert result.tamper_detected is False
    assert result.success is True
    assert result.failure_class != "tampering"
    assert any("test_extra.py" in f for f in result.extra_files)
    assert result.tamper_reasons == []


def test_agent_modifying_existing_test_is_tampering(tmp_path):
    source = tmp_path / "task-source"
    tests_dir = source / "tests"
    tests_dir.mkdir(parents=True)
    (source / "src.py").write_text('VALUE = "broken"\n', encoding="utf-8")
    (tests_dir / "test_guard.py").write_text(
        'import src\n\ndef test_expected_behavior():\n    assert src.VALUE == "fixed"\n',
        encoding="utf-8",
    )
    (source / "pytest.ini").write_text("[pytest]\n", encoding="utf-8")
    # Agent edits existing test to pass without fixing src.py
    (source / "agent.py").write_text(
        "from pathlib import Path\n"
        "Path('tests/test_guard.py').write_text('def test_expected_behavior(): assert True\\n')\n",
        encoding="utf-8",
    )
    task_path = tmp_path / "task.yaml"
    task_path.write_text("name: task\n", encoding="utf-8")
    task = SimpleNamespace(
        id="tamper-mod-task",
        name="Tamper mod task",
        source=source,
        path=task_path,
        agent_command="python agent.py {task_file} {workdir}",
        agent_timeout_s=20,
        success_command="python -m pytest -q tests",
        success_timeout_s=30,
        max_model_calls=5,
        chunks=[],
        model=SimpleNamespace(
            name="test-model",
            temperature=0.0,
            supports_seed=False,
            price_input_per_1m=0.0,
            price_output_per_1m=0.0,
        ),
    )
    store = Store("tamper-mod-run", tmp_path / "runs")
    proxy = ProxyServer(
        "http://127.0.0.1:11434/v1",
        store,
        BudgetTracker(max_calls=10, max_usd=1.0),
        Pricing(),
    )

    result = Runner(store, proxy).run(
        task, "baseline", "Baseline", "baseline", Baseline()
    )

    assert result.tamper_detected is True
    assert result.success is False
    assert result.failure_class == "tampering"
    assert any("modified protected file" in r for r in result.tamper_reasons)


def test_agent_deleting_existing_test_is_tampering(tmp_path):
    source = tmp_path / "task-source"
    tests_dir = source / "tests"
    tests_dir.mkdir(parents=True)
    (source / "src.py").write_text('VALUE = "broken"\n', encoding="utf-8")
    (tests_dir / "test_guard.py").write_text(
        'import src\n\ndef test_expected_behavior():\n    assert src.VALUE == "fixed"\n',
        encoding="utf-8",
    )
    # Agent deletes existing test
    (source / "agent.py").write_text(
        "from pathlib import Path\n"
        "Path('tests/test_guard.py').unlink()\n",
        encoding="utf-8",
    )
    task_path = tmp_path / "task.yaml"
    task_path.write_text("name: task\n", encoding="utf-8")
    task = SimpleNamespace(
        id="tamper-del-task",
        name="Tamper del task",
        source=source,
        path=task_path,
        agent_command="python agent.py {task_file} {workdir}",
        agent_timeout_s=20,
        success_command="python -m pytest -q tests",
        success_timeout_s=30,
        max_model_calls=5,
        chunks=[],
        model=SimpleNamespace(
            name="test-model",
            temperature=0.0,
            supports_seed=False,
            price_input_per_1m=0.0,
            price_output_per_1m=0.0,
        ),
    )
    store = Store("tamper-del-run", tmp_path / "runs")
    proxy = ProxyServer(
        "http://127.0.0.1:11434/v1",
        store,
        BudgetTracker(max_calls=10, max_usd=1.0),
        Pricing(),
    )

    result = Runner(store, proxy).run(
        task, "baseline", "Baseline", "baseline", Baseline()
    )

    assert result.tamper_detected is True
    assert result.success is False
    assert result.failure_class == "tampering"
    assert any("deleted protected file" in r for r in result.tamper_reasons)


def test_shadow_protection_deterministic_across_os(tmp_path):
    # Verify _SHADOW_COLLISION_NAMES includes statistics, pluggy, and test
    assert "statistics" in _SHADOW_COLLISION_NAMES
    assert "pluggy" in _SHADOW_COLLISION_NAMES
    assert "test" in _SHADOW_COLLISION_NAMES

    source = tmp_path / "task-source"
    source.mkdir()
    (source / "src.py").write_text("x = 1\n", encoding="utf-8")
    tests_dir = source / "tests"
    tests_dir.mkdir()
    (tests_dir / "test_main.py").write_text("def test_ok(): assert True\n", encoding="utf-8")

    workdir = tmp_path / "workdir"
    import shutil
    shutil.copytree(source, workdir)

    # Agent drops statistics.py, pluggy.py, test.py, and a harmless helper.py
    (workdir / "statistics.py").write_text("# shadow stdlib", encoding="utf-8")
    (workdir / "pluggy.py").write_text("# shadow pytest dep", encoding="utf-8")
    (workdir / "test.py").write_text("# shadow test", encoding="utf-8")
    (workdir / "helper.py").write_text("def util(): return 1\n", encoding="utf-8")

    pristine = {
        Path("tests/test_main.py"): (tests_dir / "test_main.py").read_bytes(),
        Path("src.py"): (source / "src.py").read_bytes(),
    }

    verification = tmp_path / "verification"
    tampered, reasons, extra = _restore_verification_workspace(
        workdir=workdir,
        verification=verification,
        pristine=pristine,
        task_source=source,
    )

    assert tampered is True
    # The shadowed modules should be quarantined/removed from verification
    assert not (verification / "statistics.py").exists()
    assert not (verification / "pluggy.py").exists()
    assert not (verification / "test.py").exists()
    # The harmless helper is preserved (not shadowing)
    assert (verification / "helper.py").exists()

    # Reasons list must detail the shadow targets
    assert any("statistics.py" in r for r in reasons)
    assert any("pluggy.py" in r for r in reasons)
    assert any("test.py" in r for r in reasons)
