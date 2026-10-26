"""Cell construction, fixed-N fallback, and the adaptive sequential scheduler."""

from __future__ import annotations

import math
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Callable

from shakedown.cost import BudgetTracker
from shakedown.perturb import AblationPerturbation, ChaosPerturbation, NoisePerturbation
from shakedown.perturb.base import Baseline, Perturbation
from shakedown.runner import Runner
from shakedown.spec import BudgetSpec, TaskSpec
from shakedown.stats import sequential_decision
from shakedown.store import RunResult, utc_now


@dataclass(frozen=True)
class CellSpec:
    id: str
    label: str
    kind: str
    config: dict = field(default_factory=dict)


def build_cells(tasks: list[TaskSpec], ablation: bool = True, chaos: bool = True,
                noise: bool = False) -> list[CellSpec]:
    first = tasks[0]
    cells = [CellSpec(id="baseline", label="Baseline (no perturbation)", kind="baseline")]
    if ablation:
        seen: set[str] = set()
        for config in first.ablation:
            chunk_ids = [c.id for c in first.chunks] if "all" in config.chunks else config.chunks
            for chunk_id in chunk_ids:
                for mode in config.modes:
                    cid = f"ablation:{mode}:{chunk_id}"
                    if cid in seen:
                        continue
                    seen.add(cid)
                    cells.append(CellSpec(
                        id=cid, label=f"{mode} {chunk_id}", kind="ablation",
                        config={"chunk": chunk_id, "mode": mode}))
    if chaos:
        seen = set()
        for config in first.chaos:
            if config.id in seen:
                continue
            seen.add(config.id)
            cells.append(CellSpec(id=config.id, label=config.label, kind="chaos",
                                  config={"fault": config.fault, "tool": config.tool,
                                          **config.params}))
    if noise:
        for mode in ("reorder_tools", "typos"):
            cells.append(CellSpec(id=f"noise:{mode}", label=f"noise: {mode}", kind="noise",
                                  config={"mode": mode}))
    return cells


def make_perturbation(cell: CellSpec, task: TaskSpec) -> Perturbation:
    if cell.kind == "baseline":
        return Baseline()
    if cell.kind == "ablation":
        chunk = task.chunk(cell.config["chunk"])
        system_chunks = [(c.id, c.text) for c in task.chunks if c.kind == "system"]
        return AblationPerturbation(chunk.id, cell.config["mode"], chunk.text,
                                    chunk.kind, chunk.tool, system_chunks)
    if cell.kind == "chaos":
        return ChaosPerturbation(cell.config["fault"], cell.config["tool"],
                                 {k: v for k, v in cell.config.items()
                                  if k not in {"fault", "tool"}})
    if cell.kind == "noise":
        return NoisePerturbation(cell.config["mode"])
    raise ValueError(cell.kind)


@dataclass
class CellSummary:
    id: str
    label: str
    kind: str
    n: int = 0
    k: int = 0
    invalid_n: int = 0
    decision: str = "fixed"
    reason: str = ""
    task_counts: dict = field(default_factory=dict)
    skipped: bool = False


class GridRunner:
    def __init__(self, runner: Runner, tasks: list[TaskSpec], cells: list[CellSpec],
                 budget: BudgetSpec, tracker: BudgetTracker,
                 on_run: Callable[[RunResult, int, int], None] | None = None,
                 on_cell: Callable[[list[CellSummary]], None] | None = None):
        self.runner = runner
        self.tasks = tasks
        self.cells = cells
        self.budget = budget
        self.tracker = tracker
        self.on_run = on_run
        self.on_cell = on_cell
        self.started = time.time()
        self.stopped_reason = ""
        self.summaries: list[CellSummary] = []
        self._executor = ThreadPoolExecutor(max_workers=max(1, budget.max_workers))
        self._run_no = 0
        self._total_runs = 0
        self._invalid_runs = 0
        self.invalid_rate_limit = 0.20

    def _time_exceeded(self) -> bool:
        if (time.time() - self.started) / 60.0 >= self.budget.max_minutes:
            self.stopped_reason = f"wall-clock cap reached ({self.budget.max_minutes:.0f} min)"
            return True
        return False

    def _budget_exceeded(self) -> bool:
        if self.tracker.calls >= self.tracker.max_calls:
            self.stopped_reason = f"global call cap reached ({self.tracker.max_calls})"
            return True
        if self.tracker.usd >= self.tracker.max_usd:
            self.stopped_reason = f"global USD cap reached (${self.tracker.max_usd:.2f})"
            return True
        return False

    def _distribute(self, count: int, cell: CellSpec,
                    counts: dict[str, int]) -> list[tuple[TaskSpec, int]]:
        if cell.kind == "baseline":
            base, rem = divmod(count, len(self.tasks))
            plan = []
            for i, task in enumerate(self.tasks):
                n = base + (1 if i < rem else 0)
                if n:
                    plan.append((task, n))
            return plan
        plan = []
        for _ in range(count):
            task = min(self.tasks, key=lambda t: counts.get(t.id, 0))
            counts[task.id] = counts.get(task.id, 0) + 1
            plan.append((task, 1))
        # Coalesce identical tasks so the executor gets sensible chunks.
        merged: dict[str, tuple[TaskSpec, int]] = {}
        for task, n in plan:
            if task.id in merged:
                merged[task.id] = (task, merged[task.id][1] + n)
            else:
                merged[task.id] = (task, n)
        return list(merged.values())

    def _execute(self, task: TaskSpec, cell: CellSpec, replicate: int = -1) -> RunResult:
        try:
            perturbation = make_perturbation(cell, task)
            return self.runner.run(task, cell.id, cell.label, cell.kind,
                                   perturbation, params=cell.config, replicate=replicate)
        except Exception as exc:  # noqa: BLE001
            result = RunResult(
                run_id=self.runner.next_run_id(), experiment_id=self.runner.store.experiment_id,
                task_id=task.id, task_name=task.name, cell_id=cell.id,
                cell_label=cell.label, cell_kind=cell.kind, params=cell.config,
                replicate=replicate,
                started_at=utc_now(), success=False, error=f"runner exception: {exc}",
                invalid=True, failure_class="invalid")
            self.runner.store.append_run(result)
            return result

    def _run_batch(self, cell: CellSpec, count: int, counts: dict[str, int],
                   collected: list[RunResult]) -> None:
        plan = self._distribute(count, cell, counts)
        futures = []
        for task, n in plan:
            for _ in range(n):
                self._run_no += 1
                futures.append(self._executor.submit(self._execute, task, cell))
        for future in as_completed(futures):
            result = future.result()
            collected.append(result)
            self._total_runs += 1
            if result.invalid:
                self._invalid_runs += 1
            if self.on_run:
                self.on_run(result, self._total_runs, self._total_runs)
        if self._total_runs and self._invalid_runs / self._total_runs > self.invalid_rate_limit:
            self.stopped_reason = (
                f"invalid run rate exceeded {self.invalid_rate_limit:.0%} "
                f"({self._invalid_runs}/{self._total_runs})"
            )

    def run_baseline(self) -> CellSummary:
        cell = self.cells[0]
        summary = CellSummary(id=cell.id, label=cell.label, kind=cell.kind)
        collected: list[RunResult] = []
        counts: dict[str, int] = {}
        remaining = self.budget.baseline_runs
        while remaining > 0 and not self.stopped_reason:
            if self._time_exceeded() or self._budget_exceeded():
                break
            batch = min(self.budget.batch_size, remaining)
            self._run_batch(cell, batch, counts, collected)
            remaining -= batch
        for r in collected:
            if r.invalid:
                summary.invalid_n += 1
            else:
                summary.n += 1
                summary.k += int(r.success)
                summary.task_counts[r.task_id] = summary.task_counts.get(r.task_id, 0) + 1
        summary.reason = self.stopped_reason or "fixed baseline sample"
        self.summaries.append(summary)
        if self.on_cell:
            self.on_cell(self.summaries)
        return summary

    def run_cells(self) -> list[CellSummary]:
        baseline = self.summaries[0]
        k0, n0 = baseline.k, baseline.n
        for cell in self.cells[1:]:
            summary = CellSummary(id=cell.id, label=cell.label, kind=cell.kind)
            if self.stopped_reason or self._time_exceeded() or self._budget_exceeded():
                summary.skipped = True
                summary.reason = self.stopped_reason or "not run"
                self.summaries.append(summary)
                continue

            collected: list[RunResult] = []
            counts: dict[str, int] = {}
            attempted = n = k = 0
            max_n = self.budget.max_runs_per_cell
            min_n = min(self.budget.min_runs_per_cell, max_n)
            while attempted < max_n and not self.stopped_reason:
                if self._time_exceeded() or self._budget_exceeded():
                    break
                batch = min(self.budget.batch_size, max_n - attempted)
                self._run_batch(cell, batch, counts, collected)
                attempted = len(collected)
                valid = [r for r in collected if not r.invalid]
                n = len(valid)
                k = sum(1 for r in valid if r.success)
                if self.budget.stop_rule == "sequential" and n >= min_n:
                    looks = max(2, math.ceil(max_n / max(1, self.budget.batch_size)))
                    decision = sequential_decision(
                        k, n, k0, n0, alpha=self.budget.alpha, max_looks=looks,
                        meaningful_drop=self.budget.meaningful_drop,
                        min_runs=min_n, look=n // max(1, self.budget.batch_size))
                    summary.decision = decision.decision
                    summary.reason = decision.reason
                    if decision.decision.startswith("stop"):
                        break
            summary.n, summary.k = n, k
            for r in collected:
                if r.invalid:
                    summary.invalid_n += 1
                else:
                    summary.task_counts[r.task_id] = summary.task_counts.get(r.task_id, 0) + 1
            if summary.decision == "fixed":
                summary.reason = summary.reason or self.stopped_reason or "reached max N"
            self.summaries.append(summary)
            if self.on_cell:
                self.on_cell(self.summaries)
        return self.summaries

    def run(self) -> list[CellSummary]:
        try:
            self._run_interleaved()
        finally:
            self._executor.shutdown(wait=True)
        return self.summaries

    def _run_interleaved(self) -> None:
        baseline = self.cells[0]
        perturbation_cells = self.cells[1:]
        existing = self.runner.store.load_runs()
        by_cell: dict[str, list[dict]] = {cell.id: [] for cell in self.cells}
        completed: set[tuple[str, str, int]] = set()
        legacy_replicates: dict[tuple[str, str], int] = {}
        for row in existing:
            cell_id = row.get("cell_id")
            if cell_id not in by_cell:
                continue
            task_id = row.get("task_id", "")
            replicate = row.get("replicate", -1)
            if not isinstance(replicate, int) or replicate < 0:
                key = (cell_id, task_id)
                replicate = legacy_replicates.get(key, 0)
                while (cell_id, task_id, replicate) in completed:
                    replicate += 1
                legacy_replicates[key] = replicate + 1
            completed.add((cell_id, task_id, replicate))
            by_cell[cell_id].append(row)
        self._total_runs = len(existing)
        self._invalid_runs = sum(1 for row in existing if row.get("invalid", False))
        rng = random.Random(self.runner.store.experiment_id)
        stopped_cells: set[str] = set()
        decisions: dict[str, tuple[str, str]] = {}
        max_replicates = max(
            self.budget.baseline_runs,
            self.budget.max_runs_per_cell if perturbation_cells else 0,
        )
        max_looks = max(
            2,
            math.ceil(self.budget.max_runs_per_cell / max(1, self.budget.batch_size)),
        )

        for replicate in range(max_replicates):
            if self.stopped_reason:
                break
            if self._time_exceeded() or self._budget_exceeded():
                break
            task = self.tasks[replicate % len(self.tasks)]
            block: list[tuple[TaskSpec, CellSpec, int]] = []
            if replicate < self.budget.baseline_runs and (
                baseline.id, task.id, replicate
            ) not in completed:
                block.append((task, baseline, replicate))
            for cell in perturbation_cells:
                if cell.id in stopped_cells or replicate >= self.budget.max_runs_per_cell:
                    continue
                if (cell.id, task.id, replicate) not in completed:
                    block.append((task, cell, replicate))
            rng.shuffle(block)
            futures = [
                self._executor.submit(self._execute, run_task, cell, run_replicate)
                for run_task, cell, run_replicate in block
            ]
            for future in as_completed(futures):
                result = future.result()
                by_cell[result.cell_id].append(result.to_dict())
                completed.add((result.cell_id, result.task_id, result.replicate))
                self._total_runs += 1
                if result.invalid:
                    self._invalid_runs += 1
                if self.on_run:
                    self.on_run(result, self._total_runs, self._total_runs)
            if self._total_runs and self._invalid_runs / self._total_runs > self.invalid_rate_limit:
                self.stopped_reason = (
                    f"invalid run rate exceeded {self.invalid_rate_limit:.0%} "
                    f"({self._invalid_runs}/{self._total_runs})"
                )
                break

            valid_baseline = [r for r in by_cell[baseline.id] if not r.get("invalid", False)]
            n0 = len(valid_baseline)
            k0 = sum(1 for row in valid_baseline if row.get("success"))
            for cell in perturbation_cells:
                runs = by_cell[cell.id]
                valid = [r for r in runs if not r.get("invalid", False)]
                if (self.budget.stop_rule == "sequential"
                        and n0 > 0
                        and len(valid) >= min(self.budget.min_runs_per_cell,
                                              self.budget.max_runs_per_cell)):
                    decision = sequential_decision(
                        sum(1 for row in valid if row.get("success")),
                        len(valid),
                        k0,
                        n0,
                        alpha=self.budget.alpha,
                        max_looks=max_looks,
                        meaningful_drop=self.budget.meaningful_drop,
                        min_runs=min(self.budget.min_runs_per_cell,
                                     self.budget.max_runs_per_cell),
                        look=max(1, len(valid) // max(1, self.budget.batch_size)),
                    )
                    decisions[cell.id] = (decision.decision, decision.reason)
                    if decision.decision.startswith("stop"):
                        stopped_cells.add(cell.id)

            self.summaries = self._build_summaries(
                by_cell, stopped_cells, decisions
            )
            if self.on_cell:
                self.on_cell(self.summaries)

        self.summaries = self._build_summaries(by_cell, stopped_cells, decisions)
        if self.on_cell:
            self.on_cell(self.summaries)

    def _build_summaries(self, by_cell: dict[str, list[dict]],
                         stopped_cells: set[str],
                         decisions: dict[str, tuple[str, str]]) -> list[CellSummary]:
        summaries = []
        for cell in self.cells:
            rows = by_cell[cell.id]
            valid = [row for row in rows if not row.get("invalid", False)]
            decision, reason = decisions.get(cell.id, ("fixed", ""))
            summary = CellSummary(
                id=cell.id,
                label=cell.label,
                kind=cell.kind,
                n=len(valid),
                k=sum(1 for row in valid if row.get("success")),
                invalid_n=len(rows) - len(valid),
                decision=decision,
                reason=reason,
                task_counts={
                    task.id: sum(1 for row in valid if row.get("task_id") == task.id)
                    for task in self.tasks
                    if any(row.get("task_id") == task.id for row in valid)
                },
            )
            target = (self.budget.baseline_runs if cell.id == "baseline"
                      else self.budget.max_runs_per_cell)
            if cell.id in stopped_cells:
                summary.reason = reason or "sequential rule stopped sampling"
            elif self.stopped_reason:
                summary.reason = self.stopped_reason
            elif len(rows) < target:
                summary.reason = "resume or schedule incomplete"
                summary.skipped = len(rows) == 0
            else:
                summary.reason = reason or "reached max N"
            summaries.append(summary)
        return summaries

    @property
    def elapsed_minutes(self) -> float:
        return (time.time() - self.started) / 60.0