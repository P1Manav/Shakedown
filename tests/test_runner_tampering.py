import tempfile
from types import SimpleNamespace
from pathlib import Path

from shakedown.cost import BudgetTracker, Pricing
from shakedown.perturb.base import Baseline
from shakedown.proxy import ProxyServer
from shakedown.runner import Runner
from shakedown.store import Store
from shakedown.analysis import build_analysis


def test_runner_restores_pristine_tests_and_fails_tampered_run(tmp_path):
    source = tmp_path / "task-source"
    tests_dir = source / "tests"
    tests_dir.mkdir(parents=True)
    (source / "src.py").write_text('VALUE = "broken"\n', encoding="utf-8")
    (tests_dir / "test_guard.py").write_text(
        'import src\n\ndef test_expected_behavior():\n    assert src.VALUE == "fixed"\n',
        encoding="utf-8",
    )
    (source / "pytest.ini").write_text("[pytest]\n", encoding="utf-8")
    (source / "agent.py").write_text(
        "from pathlib import Path\n"
        "Path('src.py').write_text('VALUE = \"fixed\"\\n')\n"
        "Path('tests/test_guard.py').write_text('assert True\\n')\n"
        "Path('sitecustomize.py').write_text('raise RuntimeError(\"shadowing\")\\n')\n",
        encoding="utf-8",
    )
    task_path = tmp_path / "task.yaml"
    task_path.write_text("name: task\n", encoding="utf-8")
    task = SimpleNamespace(
        id="tamper-task",
        name="Tamper task",
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
    store = Store("tamper-run", tmp_path / "runs")
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
    assert result.to_dict()["tamper_detected"] is True
    assert "1 passed" in result.test_output
    saved = store.load_runs()[0]
    assert saved["tamper_detected"] is True
    assert saved["failure_class"] == "tampering"
    analysis = build_analysis(store, {
        "budget": {},
        "cells": [
            {"id": "baseline", "label": "Baseline", "kind": "baseline", "config": {}},
        ],
        "task_chunks": {},
    })
    assert analysis["summary"]["tampering_runs"] == 1
    verification = (Path(tempfile.gettempdir()) / "shakedown" / store.experiment_id
                    / f"{result.run_id}-verification")
    assert not verification.exists()
    assert store.load_trace(result.run_id)[-1]["tamper_detected"] is True
