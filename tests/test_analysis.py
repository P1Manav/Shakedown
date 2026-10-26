from shakedown.analysis import _label, build_analysis
from shakedown.stats import holm, minimum_detectable_effect, two_proportion_test
from shakedown.store import RunResult, Store


def test_nonsignificant_positive_result_is_not_labeled_dead_weight():
    comparison = two_proportion_test(1, 1, 0, 1)

    label, _ = _label(
        comparison,
        adjusted_p=1.0,
        alpha=0.05,
        meaningful_drop=0.25,
        n=1,
        mde=minimum_detectable_effect(0.0, 1, 1),
    )

    assert label == "inconclusive"


def test_analysis_preserves_cell_name_separately_from_verdict(tmp_path):
    store = Store("labels", tmp_path)
    meta = {
        "experiment_id": "labels",
        "budget": {},
        "cells": [
            {"id": "baseline", "label": "Baseline", "kind": "baseline", "config": {}},
            {"id": "chaos:truncate", "label": "Truncated test output", "kind": "chaos",
             "config": {}},
        ],
        "task_chunks": {},
    }
    store.save_meta(meta)
    for run_id, cell_id, cell_label in (
        ("run-1", "baseline", "Baseline"),
        ("run-2", "chaos:truncate", "Truncated test output"),
    ):
        store.append_run(RunResult(
            run_id=run_id,
            experiment_id="labels",
            task_id="bugfix_01",
            task_name="Median",
            cell_id=cell_id,
            cell_label=cell_label,
            cell_kind="baseline" if cell_id == "baseline" else "chaos",
            success=True,
        ))

    analysis = build_analysis(store, meta)

    assert analysis["cells"][0]["label"] == "Truncated test output"
    assert analysis["cells"][0]["verdict"] == "inconclusive"


def test_dead_weight_requires_familywise_interval_to_rule_out_both_directions(tmp_path):
    store = Store("familywise", tmp_path)
    cells = [
        {"id": "baseline", "label": "Baseline", "kind": "baseline", "config": {}},
        *[
            {"id": f"chaos:{i}", "label": f"Fault {i}", "kind": "chaos", "config": {}}
            for i in range(20)
        ],
    ]
    meta = {
        "experiment_id": "familywise",
        "budget": {"meaningful_drop": 0.25, "min_runs_per_cell": 8},
        "cells": cells,
        "task_chunks": {},
    }
    store.save_meta(meta)
    run_number = 0
    for cell in cells:
        for i in range(40):
            run_number += 1
            store.append_run(RunResult(
                run_id=f"run-{run_number}",
                experiment_id="familywise",
                task_id="bugfix_01",
                task_name="Median",
                cell_id=cell["id"],
                cell_label=cell["label"],
                cell_kind=cell["kind"],
                success=i < 20,
            ))

    analysis = build_analysis(store, meta)

    assert analysis["methods"]["per_look_confidence_by_family"] == {"chaos": 0.99975}
    assert analysis["methods"]["corrections_applied_by_family"] == {"chaos": 20}
    assert analysis["cells"][0]["verdict"] == "inconclusive"
    assert analysis["cells"][0]["mde"] > minimum_detectable_effect(0.5, 40, 40, 0.05)


def test_one_run_cost_spike_does_not_create_fix_first_finding(tmp_path):
    store = Store("cost-smoke", tmp_path)
    cells = [
        {"id": "baseline", "label": "Baseline", "kind": "baseline", "config": {}},
        {"id": "chaos:cost", "label": "Cost test", "kind": "chaos", "config": {}},
    ]
    meta = {
        "experiment_id": "cost-smoke",
        "budget": {"min_runs_per_cell": 1},
        "cells": cells,
        "task_chunks": {},
    }
    store.save_meta(meta)
    for run_id, cell, tokens in (
        ("base", cells[0], 100),
        ("cost", cells[1], 300),
    ):
        store.append_run(RunResult(
            run_id=run_id,
            experiment_id="cost-smoke",
            task_id="bugfix_01",
            task_name="Median",
            cell_id=cell["id"],
            cell_label=cell["label"],
            cell_kind=cell["kind"],
            success=True,
            prompt_tokens=tokens,
        ))

    analysis = build_analysis(store, meta)

    assert analysis["cells"][0]["cost"]["cost_ratio"] == 3.0
    assert analysis["cells"][0]["cost"]["cost_risk"] is False
    assert analysis["fix_first"] == []


def test_analysis_reports_pooled_and_task_stratified_comparisons(tmp_path):
    store = Store("stratified", tmp_path)
    cells = [
        {"id": "baseline", "label": "Baseline", "kind": "baseline", "config": {}},
        {"id": "noise:test", "label": "Noise", "kind": "noise", "config": {}},
    ]
    meta = {"budget": {}, "cells": cells, "task_chunks": {}}
    store.save_meta(meta)
    outcomes = {
        ("easy", "baseline"): 9,
        ("easy", "noise:test"): 8,
        ("hard", "baseline"): 1,
        ("hard", "noise:test"): 2,
    }
    run_number = 0
    for (task_id, cell_id), successes in outcomes.items():
        for index in range(10):
            run_number += 1
            store.append_run(RunResult(
                run_id=f"run-{run_number}",
                experiment_id="stratified",
                task_id=task_id,
                task_name=task_id.title(),
                cell_id=cell_id,
                cell_label="Baseline" if cell_id == "baseline" else "Noise",
                cell_kind="baseline" if cell_id == "baseline" else "noise",
                success=index < successes,
            ))

    analysis = build_analysis(store, meta)
    cell = analysis["cells"][0]

    assert cell["comparison"]["delta"] == 0.0
    assert abs(cell["stratified_comparison"]["delta"]) < 1e-12
    assert cell["label_driven_by"] == "stratified"
    assert cell["per_task"]["easy"]["rate"] == 0.8
    assert cell["per_task"]["hard"]["rate"] == 0.2
    assert cell["assigned_per_task"] == {"easy": 10, "hard": 10}
    assert cell["p_raw_pooled"] == cell["comparison"]["p_value"]
    assert "Cochran-Mantel-Haenszel" in analysis["methods"]["label_driven_by"]


def test_multiple_testing_and_mde_are_adjusted_within_cell_families(tmp_path):
    store = Store("family-corrections", tmp_path)
    cells = [
        {"id": "baseline", "label": "Baseline", "kind": "baseline", "config": {}},
        {"id": "ablation:remove:a", "label": "Remove a", "kind": "ablation", "config": {}},
        {"id": "ablation:remove:b", "label": "Remove b", "kind": "ablation", "config": {}},
        {"id": "chaos:error", "label": "Error", "kind": "chaos", "config": {}},
    ]
    meta = {
        "budget": {"alpha": 0.05, "correction": "holm"},
        "cells": cells,
        "task_chunks": {},
    }
    store.save_meta(meta)
    outcomes = {
        "baseline": 50,
        "ablation:remove:a": 20,
        "ablation:remove:b": 50,
        "chaos:error": 30,
    }
    run_number = 0
    for cell in cells:
        for index in range(100):
            run_number += 1
            store.append_run(RunResult(
                run_id=f"run-{run_number}",
                experiment_id="family-corrections",
                task_id="task",
                task_name="Task",
                cell_id=cell["id"],
                cell_label=cell["label"],
                cell_kind=cell["kind"],
                success=index < outcomes[cell["id"]],
            ))

    analysis = build_analysis(store, meta)

    assert analysis["methods"]["family_sizes"] == {"ablation": 2, "chaos": 1}
    assert analysis["methods"]["family_alpha"] == {"ablation": 0.025, "chaos": 0.05}
    assert analysis["methods"]["mde_alpha_by_family"] == {
        "ablation": 0.0025, "chaos": 0.005
    }
    cells_by_kind = {
        kind: [cell for cell in analysis["cells"] if cell["kind"] == kind]
        for kind in ("ablation", "chaos")
    }
    for kind, family_cells in cells_by_kind.items():
        expected, _ = holm([cell["p_raw"] for cell in family_cells], alpha=0.05)
        assert [cell["p_adjusted"] for cell in family_cells] == expected
        assert all(cell["family_alpha"] == analysis["methods"]["family_alpha"][kind]
                   for cell in family_cells)
        assert all(cell["p_adjustment_family"] == kind for cell in family_cells)
    assert all(
        cell["mde"] == minimum_detectable_effect(0.5, 100, 100, alpha=0.0025)
        for cell in cells_by_kind["ablation"]
    )
