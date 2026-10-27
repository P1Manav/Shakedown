from types import SimpleNamespace

from shakedown.perturb.base import RunState
from shakedown.perturb.chaos import ChaosPerturbation
from shakedown.scheduler import build_cells
from shakedown.spec import discover_tasks


def _make_context():
    state = RunState("test-run-1")
    return SimpleNamespace(state=state)


def test_error_once_fires_once_per_tool_and_recovers_on_retry():
    perturbation = ChaosPerturbation("error_once", "write_file")
    ctx = _make_context()

    # Call 1 on write_file
    ctx.state.current_tool_call_id = "call_1"
    res1 = perturbation.apply_tool_result("write_file", {"path": "calc.py"}, "original", ctx)
    ctx.state.tool_result_cache["call_1"] = res1
    assert "ToolError: transient upstream failure" in res1

    # Call 2 (retry) on write_file with a new tool_call_id
    ctx.state.current_tool_call_id = "call_2"
    res2 = perturbation.apply_tool_result("write_file", {"path": "calc.py"}, "original", ctx)
    ctx.state.tool_result_cache["call_2"] = res2
    assert res2 == "original"

    # Replay of call 1 from cache is stable
    assert ctx.state.tool_result_cache["call_1"] == res1
    assert "ToolError" in ctx.state.tool_result_cache["call_1"]


def test_empty_result_transient_by_default_and_recovers_on_retry():
    perturbation = ChaosPerturbation("empty_result", "read_file")
    ctx = _make_context()

    # Call 1 for path a.py
    ctx.state.current_tool_call_id = "call_1"
    res1 = perturbation.apply_tool_result("read_file", {"path": "a.py"}, "content_a", ctx)
    ctx.state.tool_result_cache["call_1"] = res1
    assert res1 == ""

    # Call 2 (retry) for path a.py -> recovers
    ctx.state.current_tool_call_id = "call_2"
    res2 = perturbation.apply_tool_result("read_file", {"path": "a.py"}, "content_a", ctx)
    ctx.state.tool_result_cache["call_2"] = res2
    assert res2 == "content_a"

    # Call 3 for path b.py -> first call on new path empties
    ctx.state.current_tool_call_id = "call_3"
    res3 = perturbation.apply_tool_result("read_file", {"path": "b.py"}, "content_b", ctx)
    ctx.state.tool_result_cache["call_3"] = res3
    assert res3 == ""

    # Call 4 (retry) for path b.py -> recovers
    ctx.state.current_tool_call_id = "call_4"
    res4 = perturbation.apply_tool_result("read_file", {"path": "b.py"}, "content_b", ctx)
    ctx.state.tool_result_cache["call_4"] = res4
    assert res4 == "content_b"

    # Replay stability
    assert ctx.state.tool_result_cache["call_1"] == ""
    assert ctx.state.tool_result_cache["call_2"] == "content_a"


def test_empty_result_persistent_variant_stays_broken():
    perturbation = ChaosPerturbation("empty_result", "read_file", {"persist": True})
    ctx = _make_context()

    # Call 1
    ctx.state.current_tool_call_id = "call_1"
    res1 = perturbation.apply_tool_result("read_file", {"path": "a.py"}, "content_a", ctx)
    assert res1 == ""

    # Call 2 (retry) stays broken
    ctx.state.current_tool_call_id = "call_2"
    res2 = perturbation.apply_tool_result("read_file", {"path": "a.py"}, "content_a", ctx)
    assert res2 == ""


def test_renamed_path_transient_by_default_and_recovers_on_retry():
    perturbation = ChaosPerturbation("renamed_path", "read_file")
    ctx = _make_context()

    # Call 1 for path a.py
    ctx.state.current_tool_call_id = "call_1"
    res1 = perturbation.apply_tool_result("read_file", {"path": "a.py"}, "content_a", ctx)
    ctx.state.tool_result_cache["call_1"] = res1
    assert "FileNotFoundError" in res1

    # Call 2 (retry) for path a.py -> recovers
    ctx.state.current_tool_call_id = "call_2"
    res2 = perturbation.apply_tool_result("read_file", {"path": "a.py"}, "content_a", ctx)
    ctx.state.tool_result_cache["call_2"] = res2
    assert res2 == "content_a"

    # Call 3 for path b.py -> first call on new path fails
    ctx.state.current_tool_call_id = "call_3"
    res3 = perturbation.apply_tool_result("read_file", {"path": "b.py"}, "content_b", ctx)
    assert "FileNotFoundError" in res3

    # Replay stability
    assert "FileNotFoundError" in ctx.state.tool_result_cache["call_1"]
    assert ctx.state.tool_result_cache["call_2"] == "content_a"


def test_renamed_path_persistent_variant_stays_broken():
    perturbation = ChaosPerturbation("renamed_path", "read_file", {"persist": True})
    ctx = _make_context()

    # Call 1
    ctx.state.current_tool_call_id = "call_1"
    res1 = perturbation.apply_tool_result("read_file", {"path": "a.py"}, "content_a", ctx)
    assert "FileNotFoundError" in res1

    # Call 2 (retry) stays broken
    ctx.state.current_tool_call_id = "call_2"
    res2 = perturbation.apply_tool_result("read_file", {"path": "a.py"}, "content_a", ctx)
    assert "FileNotFoundError" in res2


def test_build_cells_includes_added_latency_and_noise_and_persistent_variants():
    tasks = discover_tasks()
    cells = build_cells(tasks, ablation=True, chaos=True, noise=True)
    cell_ids = [c.id for c in cells]

    assert any("added_latency" in cid for cid in cell_ids), f"Missing added_latency in {cell_ids}"
    assert "noise:reorder_tools" in cell_ids
    assert "noise:typos" in cell_ids
    assert "noise:paraphrase" in cell_ids
    assert "chaos:empty_result:read_file" in cell_ids
    assert "chaos:empty_result:read_file:persist=True" in cell_ids
    assert "chaos:renamed_path:read_file" in cell_ids
    assert "chaos:renamed_path:read_file:persist=True" in cell_ids
