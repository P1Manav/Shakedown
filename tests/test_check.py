import json
from unittest.mock import patch

from shakedown import cli
from shakedown.store import Store


def test_cmd_check_detects_regression_and_returns_exit_code_two(tmp_path, capsys):
    baseline_analysis = {
        "baseline": {
            "n": 50,
            "k": 40,
            "rate": 0.8,
            "wilson": [0.67, 0.89],
        }
    }
    baseline_path = tmp_path / "baseline_analysis.json"
    baseline_path.write_text(json.dumps(baseline_analysis), encoding="utf-8")

    experiment_id = "test-check-regression"
    runs_dir = tmp_path / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)

    def fake_execute_run(args, cells, exp_id):
        store = Store(exp_id, runs_dir)
        meta = {
            "experiment_id": exp_id,
            "model": {"name": "test-model", "temperature": 0.0},
            "budget": {"alpha": 0.05},
            "tasks": [{"id": "t1", "name": "task 1", "description": ""}],
            "cells": [{"id": "baseline", "label": "Baseline", "kind": "baseline"}],
            "status": "complete",
        }
        store.save_meta(meta)
        return store, meta

    def fake_build_analysis(store, meta):
        return {
            "baseline": {
                "n": 50,
                "k": 10,
                "rate": 0.2,
                "wilson": [0.11, 0.33],
            }
        }

    parser = cli.build_parser()
    args = parser.parse_args([
        "check",
        "--baseline", str(baseline_path),
        "-n", "50",
        "--experiment", experiment_id,
    ])

    with patch("shakedown.cli._execute_run", side_effect=fake_execute_run), \
         patch("shakedown.analysis.build_analysis", side_effect=fake_build_analysis), \
         patch("shakedown.cli.Store", lambda name, root=None: Store(name, runs_dir)):
        exit_code = cli.cmd_check(args)

    captured = capsys.readouterr()
    assert exit_code == 2
    assert "REGRESSION" in captured.out
    assert "badge: 20% [11-33]" in captured.out

    result_file = runs_dir / experiment_id / "check_result.json"
    assert result_file.is_file()
    payload = json.loads(result_file.read_text(encoding="utf-8"))
    assert payload["regressed"] is True
    assert payload["badge"] == "20% [11-33]"
    assert payload["current"]["k"] == 10


def test_cmd_check_passes_when_not_regressed(tmp_path, capsys):
    baseline_analysis = {
        "baseline": {
            "n": 50,
            "k": 30,
            "rate": 0.6,
            "wilson": [0.46, 0.72],
        }
    }
    baseline_path = tmp_path / "baseline_analysis.json"
    baseline_path.write_text(json.dumps(baseline_analysis), encoding="utf-8")

    experiment_id = "test-check-pass"
    runs_dir = tmp_path / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)

    def fake_execute_run(args, cells, exp_id):
        store = Store(exp_id, runs_dir)
        meta = {
            "experiment_id": exp_id,
            "model": {"name": "test-model", "temperature": 0.0},
            "budget": {"alpha": 0.05},
            "tasks": [{"id": "t1", "name": "task 1", "description": ""}],
            "cells": [{"id": "baseline", "label": "Baseline", "kind": "baseline"}],
            "status": "complete",
        }
        store.save_meta(meta)
        return store, meta

    def fake_build_analysis(store, meta):
        return {
            "baseline": {
                "n": 50,
                "k": 35,
                "rate": 0.7,
                "wilson": [0.56, 0.81],
            }
        }

    parser = cli.build_parser()
    args = parser.parse_args([
        "check",
        "--baseline", str(baseline_path),
        "-n", "50",
        "--experiment", experiment_id,
    ])

    with patch("shakedown.cli._execute_run", side_effect=fake_execute_run), \
         patch("shakedown.analysis.build_analysis", side_effect=fake_build_analysis), \
         patch("shakedown.cli.Store", lambda name, root=None: Store(name, runs_dir)):
        exit_code = cli.cmd_check(args)

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "ok | badge: 70% [56-81]" in captured.out

    result_file = runs_dir / experiment_id / "check_result.json"
    assert result_file.is_file()
    payload = json.loads(result_file.read_text(encoding="utf-8"))
    assert payload["regressed"] is False
    assert payload["badge"] == "70% [56-81]"


def test_cmd_check_defaults_to_configured_runs_when_omitted(tmp_path):
    baseline_analysis = {
        "baseline": {
            "n": 30,
            "k": 20,
            "rate": 0.67,
            "wilson": [0.49, 0.81],
        }
    }
    baseline_path = tmp_path / "baseline_analysis.json"
    baseline_path.write_text(json.dumps(baseline_analysis), encoding="utf-8")

    experiment_id = "test-check-default-runs"
    runs_dir = tmp_path / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)

    def fake_execute_run(args, cells, exp_id):
        store = Store(exp_id, runs_dir)
        meta = {
            "experiment_id": exp_id,
            "model": {"name": "test-model", "temperature": 0.0},
            "budget": {"alpha": 0.05},
            "tasks": [{"id": "t1", "name": "task 1", "description": ""}],
            "cells": [{"id": "baseline", "label": "Baseline", "kind": "baseline"}],
            "status": "complete",
        }
        store.save_meta(meta)
        return store, meta

    def fake_build_analysis(store, meta):
        return {
            "baseline": {
                "n": 30,
                "k": 20,
                "rate": 0.67,
                "wilson": [0.49, 0.81],
            }
        }

    from types import SimpleNamespace
    args = SimpleNamespace(
        command="check",
        baseline=str(baseline_path),
        tasks=None,
        runs=None,
        workers=None,
        max_minutes=None,
        max_usd=None,
        alpha=0.05,
        correction=None,
        stop_rule=None,
        min_runs=None,
        max_runs=None,
        meaningful_drop=None,
        experiment=experiment_id,
        keep_workdirs=False,
        no_ablation=False,
        no_chaos=False,
        noise=False,
    )

    with patch("shakedown.cli._execute_run", side_effect=fake_execute_run), \
         patch("shakedown.analysis.build_analysis", side_effect=fake_build_analysis), \
         patch("shakedown.cli.Store", lambda name, root=None: Store(name, runs_dir)):
        exit_code = cli.cmd_check(args)

    assert exit_code == 0

