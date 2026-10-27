from types import SimpleNamespace

from shakedown.cost import BudgetTracker
from shakedown.fix import (
    _fix_experiment_cells,
    _fix_prompt,
    _generate_fix_text,
    _verdict,
)
from shakedown.cli import build_parser
from shakedown.report.build import build_report
from shakedown.scheduler import GridRunner
from shakedown.stats import minimum_detectable_effect
from shakedown.store import RunResult, Store
from shakedown.spec import BudgetSpec


def test_fix_verdict_requires_positive_lift_and_reports_intervals_and_p_value():
    zero_lift = _verdict({"n": 10, "k": 5}, {"n": 10, "k": 5})
    negative_lift = _verdict({"n": 10, "k": 9}, {"n": 10, "k": 1})
    positive_lift = _verdict({"n": 10, "k": 1}, {"n": 10, "k": 9})

    assert zero_lift["verdict"] == "no statistically significant lift"
    assert negative_lift["verdict"] == "significant decline"
    assert positive_lift["lift_points"] > 0
    assert positive_lift["p_value"] < 0.05
    assert positive_lift["non_overlapping"] is True
    assert positive_lift["verdict"] == "measured lift, intervals do not overlap"
    assert "wilson" in positive_lift["original"]
    assert "wilson" in positive_lift["fixed"]


def test_fix_arms_are_fresh_equal_n_interleaved_and_have_a_control():
    cells = _fix_experiment_cells("Fix text", 60)

    assert [cell.id for cell in cells] == ["baseline", "fix:baseline"]
    assert [cell.kind for cell in cells] == ["baseline", "fix"]
    assert cells[0].config == {"run_count": 60}
    assert cells[1].config == {"fix_line": "Fix text", "run_count": 60}
    assert cells[0].id != "chaos:the-original-worst-cell"
    assert cells[0].config["run_count"] == cells[1].config["run_count"]
    mde = minimum_detectable_effect(0.5, 60, 60)
    assert 0 < mde < 0.3


def test_fix_prompt_uses_failing_trace_evidence_not_only_canned_advice():
    prompt = _fix_prompt(
        "Retry test runs after tool errors.",
        [
            {
                "run_id": "fail-run",
                "events": [
                    {"event": "request", "body": {"messages": [
                        {"role": "system", "content": "Verify all edits with tests."},
                    ]}},
                    {"event": "response", "body": {"choices": [{
                        "message": {"tool_calls": [{
                            "function": {"name": "run_tests", "arguments": "{}"},
                        }]}
                    }]}},
                ],
            }
        ],
    )

    assert "fail-run" in prompt
    assert "Verify all edits with tests." in prompt
    assert "run_tests" in prompt
    assert "Retry test runs after tool errors." in prompt
    assert "do not claim you reran tests" in prompt


def test_fix_text_generation_uses_configured_model_under_a_cost_cap(monkeypatch):
    observed = {}

    class Response:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                "choices": [{"message": {"content": "Retry the failed test call."}}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 8},
            }

    def fake_post(url, **kwargs):
        observed.update({"url": url, **kwargs})
        return Response()

    monkeypatch.setattr("shakedown.fix.httpx.post", fake_post)
    model = SimpleNamespace(
        name="qwen2.5-coder:7b",
        base_url="http://127.0.0.1:11434/v1",
        price_input_per_1m=0.0,
        price_output_per_1m=0.0,
    )
    tracker = BudgetTracker(max_calls=5, max_usd=0.02)

    result = _generate_fix_text(
        "Use the trace evidence.", "Use tests.", model, tracker, "local"
    )

    assert result["source"] == "model"
    assert result["text"] == "Retry the failed test call."
    assert result["fallback_text"] == "Use tests."
    assert observed["url"].endswith("/chat/completions")
    assert observed["json"]["model"] == model.name
    assert observed["json"]["temperature"] == 0.0
    assert tracker.snapshot()["calls"] == 1


def test_fix_text_generation_uses_canned_fallback_when_budget_is_too_small(monkeypatch):
    monkeypatch.setattr(
        "shakedown.fix.httpx.post",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("must not call model")),
    )
    model = SimpleNamespace(
        name="paid-open-model",
        base_url="https://models.example/v1",
        price_input_per_1m=100.0,
        price_output_per_1m=100.0,
    )
    tracker = BudgetTracker(max_calls=5, max_usd=1e-12)

    result = _generate_fix_text(
        "Large trace evidence " * 100, "Canned guidance.", model, tracker, "secret"
    )

    assert result["source"] == "canned fallback"
    assert result["text"] == "Canned guidance."
    assert result["status"] == "generation budget cap"
    assert tracker.snapshot()["calls"] == 0


def test_fix_and_demo_default_to_sixty_control_and_fix_runs():
    parser = build_parser()

    fix_args = parser.parse_args(["fix", "runs/example"])
    demo_args = parser.parse_args(["demo"])

    assert fix_args.runs == 60
    assert demo_args.fix_runs == 60


def test_fix_runner_uses_equal_fresh_arms_and_interleaves_control_and_fix(tmp_path):
    store = Store("fix-grid", tmp_path)
    calls = []

    class FakeRunner:
        def __init__(self):
            self.store = store
            self.counter = 0

        def next_run_id(self):
            self.counter += 1
            return f"run-{self.counter}"

        def run(self, task, cell_id, cell_label, cell_kind, perturbation,
                params=None, replicate=-1, fix_line=None):
            run = RunResult(
                run_id=self.next_run_id(),
                experiment_id=store.experiment_id,
                task_id=task.id,
                task_name=task.name,
                cell_id=cell_id,
                cell_label=cell_label,
                cell_kind=cell_kind,
                replicate=replicate,
                params=params or {},
                success=(cell_id == "fix:baseline"),
            )
            calls.append((cell_id, task.id, replicate, fix_line))
            store.append_run(run)
            return run

    tasks = [
        SimpleNamespace(id="task-a", name="A"),
        SimpleNamespace(id="task-b", name="B"),
    ]
    budget = BudgetSpec(
        baseline_runs=4,
        min_runs_per_cell=4,
        max_runs_per_cell=4,
        max_workers=1,
        max_minutes=1,
        batch_size=1,
        stop_rule="fixed",
    )
    grid = GridRunner(
        FakeRunner(), tasks, _fix_experiment_cells("Fix text", 4),
        budget, BudgetTracker(max_calls=20, max_usd=1),
    )

    grid.run()
    rows = store.load_runs()
    control = [row for row in rows if row["cell_id"] == "baseline"]
    fixed = [row for row in rows if row["cell_id"] == "fix:baseline"]

    assert len(control) == len(fixed) == 4
    assert {(row["task_id"], row["replicate"]) for row in control} == {
        (row["task_id"], row["replicate"]) for row in fixed
    }
    assert all(call[3] is None for call in calls if call[0] == "baseline")
    assert all(call[3] == "Fix text" for call in calls if call[0] == "fix:baseline")
    assert len({row["run_id"] for row in rows}) == 8
    for offset in range(0, len(calls), 2):
        assert {call[0] for call in calls[offset:offset + 2]} == {
            "baseline", "fix:baseline",
        }


def test_fix_payload_is_rendered_from_analysis_with_control_and_fixed_arms(tmp_path):
    store = Store("fix-report", tmp_path)
    cells = [
        {"id": "baseline", "label": "No-fix control", "kind": "baseline", "config": {}},
        {"id": "fix:baseline", "label": "Fix arm", "kind": "fix",
         "config": {"fix_line": "Retry failed test calls."}},
    ]
    meta = {
        "experiment_id": "fix-report",
        "budget": {"min_runs_per_cell": 1},
        "cells": cells,
        "task_chunks": {},
        "model": {"name": "local-open-model"},
    }
    store.save_meta(meta)
    for arm, cell_id, passed in (
        ("control", "baseline", [True, False, False, False]),
        ("fixed", "fix:baseline", [True, True, True, False]),
    ):
        for index, success in enumerate(passed):
            store.append_run(RunResult(
                run_id=f"{arm}-{index}",
                experiment_id="fix-report",
                task_id="task",
                task_name="Task",
                cell_id=cell_id,
                cell_label=cells[0 if arm == "control" else 1]["label"],
                cell_kind="baseline" if arm == "control" else "fix",
                success=success,
            ))
    from shakedown.analysis import build_analysis

    analysis = build_analysis(store, meta)
    analysis["fix"] = {
        "control": {"rate": 0.25, "n": 4, "k": 1, "wilson": [0.05, 0.70]},
        "fixed": {"rate": 0.75, "n": 4, "k": 3, "wilson": [0.30, 0.95]},
        "lift_points": 50,
        "mde_points": 80,
        "p_value": 0.2,
        "fix_line": "Retry failed test calls.",
        "fallback_text": "Canned fallback text.",
        "generation_source": "canned fallback",
        "generation_cost_usd": 0.0,
        "non_overlapping": False,
        "verdict": "no statistically significant lift",
    }

    path = build_report(store, analysis)
    report = path.read_text(encoding="utf-8")

    assert "Fix experiment" in report
    assert "No-fix control" in report
    assert "Fix arm" in report
    assert "Canned fallback text." in report
    assert "MDE" in report
