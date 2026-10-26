from types import SimpleNamespace

from shakedown.cost import BudgetTracker
from shakedown.perturb.base import Baseline
from shakedown.runner import Runner
from shakedown.scheduler import CellSpec, GridRunner
from shakedown.spec import BudgetSpec
from shakedown.store import RunResult, Store


class RecordingRunner:
    def __init__(self, store):
        self.store = store
        self.calls = []
        self._next = 0

    def next_run_id(self):
        self._next += 1
        return f"recording-{self._next}"

    def run(self, task, cell_id, cell_label, cell_kind, perturbation,
            params=None, replicate=-1):
        self.calls.append((cell_id, task.id, replicate))
        result = RunResult(
            run_id=self.next_run_id(),
            experiment_id=self.store.experiment_id,
            task_id=task.id,
            task_name=task.name,
            cell_id=cell_id,
            cell_label=cell_label,
            cell_kind=cell_kind,
            replicate=replicate,
            success=True,
        )
        self.store.append_run(result)
        return result


def _grid(store, runner, task_count=1):
    tasks = [SimpleNamespace(id=f"task-{i}", name=f"Task {i}") for i in range(task_count)]
    budget = BudgetSpec(
        baseline_runs=4,
        min_runs_per_cell=4,
        max_runs_per_cell=4,
        batch_size=1,
        max_workers=1,
        max_minutes=1,
        max_total_calls=100,
        max_usd=1,
        stop_rule="fixed",
    )
    grid = GridRunner(
        runner,
        tasks,
        [
            CellSpec("baseline", "Baseline", "baseline"),
            CellSpec("noise:typos", "Typos", "noise", {"mode": "typos"}),
        ],
        budget,
        BudgetTracker(max_calls=100, max_usd=1),
    )
    return grid


def test_scheduler_interleaves_randomized_baseline_and_perturbed_runs(tmp_path):
    store = Store("interleave", tmp_path)
    runner = RecordingRunner(store)

    summaries = _grid(store, runner, task_count=2).run()

    for block in (runner.calls[0:2], runner.calls[2:4],
                  runner.calls[4:6], runner.calls[6:8]):
        assert {cell_id for cell_id, _, _ in block} == {"baseline", "noise:typos"}
    baseline_tasks = [task_id for cell_id, task_id, _ in runner.calls if cell_id == "baseline"]
    perturbed_tasks = [task_id for cell_id, task_id, _ in runner.calls if cell_id == "noise:typos"]
    assert baseline_tasks == perturbed_tasks
    assert summaries[0].n == summaries[1].n == 4
    assert summaries[0].task_counts == summaries[1].task_counts


def test_scheduler_resume_skips_completed_cell_task_replicate_tuples(tmp_path):
    store = Store("resume", tmp_path)
    for run_id, cell_id in (
        ("baseline-complete", "baseline"),
        ("perturbed-complete", "noise:typos"),
    ):
        store.append_run(RunResult(
            run_id=run_id,
            experiment_id="resume",
            task_id="task-0",
            task_name="Task 0",
            cell_id=cell_id,
            cell_label=cell_id,
            cell_kind="baseline" if cell_id == "baseline" else "noise",
            replicate=0,
            success=True,
        ))
    runner = RecordingRunner(store)

    _grid(store, runner).run()

    assert len(runner.calls) == 6
    assert all(replicate > 0 for _, _, replicate in runner.calls)
    assert len({row["run_id"] for row in store.load_runs()}) == 8


def test_runner_ids_are_unique_across_resumed_runner_instances(tmp_path):
    store = Store("unique-run-ids", tmp_path)

    first = Runner(store, proxy=SimpleNamespace())
    second = Runner(store, proxy=SimpleNamespace())

    assert first.next_run_id() != second.next_run_id()
