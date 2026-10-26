from types import SimpleNamespace

from shakedown.cost import BudgetTracker
from shakedown.scheduler import CellSpec, GridRunner
from shakedown.spec import BudgetSpec
from shakedown.store import Store


class RaisingRunner:
    def __init__(self, store):
        self.store = store
        self._next = 0

    def next_run_id(self):
        self._next += 1
        return f"run-{self._next}"

    def run(self, *args, **kwargs):
        raise RuntimeError("sandbox unavailable")


def test_runner_exceptions_are_invalid_and_abort_after_invalid_rate_exceeds_limit(tmp_path):
    store = Store("scheduler-invalid", tmp_path)
    runner = RaisingRunner(store)
    task = SimpleNamespace(id="task", name="Task")
    budget = BudgetSpec(
        baseline_runs=4,
        min_runs_per_cell=1,
        max_runs_per_cell=1,
        batch_size=4,
        max_workers=1,
        max_minutes=1,
        max_total_calls=100,
        max_usd=1,
    )
    grid = GridRunner(
        runner,
        [task],
        [
            CellSpec("baseline", "Baseline", "baseline"),
            CellSpec("chaos:test", "Fault", "chaos", {"fault": "empty_result", "tool": "read"}),
        ],
        budget,
        BudgetTracker(max_calls=100, max_usd=1),
    )

    summaries = grid.run()

    assert grid.stopped_reason == "invalid run rate exceeded 20% (2/2)"
    assert summaries[0].n == 0
    assert summaries[0].invalid_n == 1
    assert summaries[1].invalid_n == 1
    assert all(run["invalid"] and run["failure_class"] == "invalid"
               for run in store.load_runs())
