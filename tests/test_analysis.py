from shakedown.analysis import _label, build_analysis
from shakedown.stats import minimum_detectable_effect, two_proportion_test
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

    assert analysis["methods"]["family_confidence"] == 0.9975
    assert analysis["methods"]["corrections_applied_across"] == 20
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
