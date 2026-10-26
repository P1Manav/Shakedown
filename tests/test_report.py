from shakedown.report.build import build_report
from shakedown.store import RunResult, Store


def test_report_is_self_contained_and_escapes_embedded_run_data(tmp_path):
    store = Store("report-test", tmp_path)
    hostile_label = "</script><img src=x onerror=alert(1)>"
    meta = {
        "experiment_id": "report-test",
        "started_at": "2026-01-01T00:00:00Z",
        "finished_at": "2026-01-01T00:01:00Z",
        "elapsed_minutes": 1,
        "budget": {},
        "model": {"name": "local-open-model"},
        "cells": [
            {"id": "baseline", "label": "Baseline", "kind": "baseline", "config": {}},
            {"id": "chaos:truncate", "label": hostile_label, "kind": "chaos", "config": {}},
        ],
        "task_chunks": {},
    }
    store.save_meta(meta)
    for run_id, cell_id, cell_label, success in (
        ("run-1", "baseline", "Baseline", True),
        ("run-2", "chaos:truncate", hostile_label, False),
    ):
        store.append_run(RunResult(
            run_id=run_id,
            experiment_id="report-test",
            task_id="bugfix_01",
            task_name="Median",
            cell_id=cell_id,
            cell_label=cell_label,
            cell_kind="baseline" if cell_id == "baseline" else "chaos",
            success=success,
            invalid=run_id == "run-2",
            claimed_success=not success,
            final_message="DONE: tests passed",
            actions=["read_file:calc.py"],
        ))

    report = build_report(store)
    html = report.read_text(encoding="utf-8")

    assert "Sensitivity heatmap" in html
    assert "<span>Invalid runs</span><strong>1</strong>" in html
    assert "Failure taxonomy" in html
    assert "prefers-color-scheme" in html
    assert "https://" not in html
    assert hostile_label not in html
    assert r"\u003c/script\u003e" in html
