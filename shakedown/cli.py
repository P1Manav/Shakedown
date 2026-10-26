"""Command line interface: run | report | fix | check | tasks | selftest | demo."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict
from pathlib import Path

from shakedown.cost import BudgetTracker, Pricing, estimate_grid, shrink_plan
from shakedown.proxy import ProxyServer
from shakedown.runner import Runner
from shakedown.scheduler import CellSpec, GridRunner, build_cells
from shakedown.spec import BudgetSpec, ROOT, discover_tasks, load_dotenv
from shakedown.store import Store, new_experiment_id, utc_now

DEFAULT_CALLS_PER_RUN = 8.0
DEFAULT_PROMPT_TOKENS = 900.0
DEFAULT_COMPLETION_TOKENS = 140.0
DEFAULT_SECONDS_PER_RUN = 12.0


def _fmt_pct(x: float) -> str:
    return f"{x * 100:.0f}%"


def _task_chunks(tasks) -> dict:
    out = {}
    for chunk in tasks[0].chunks:
        out[chunk.id] = {"kind": chunk.kind, "tool": chunk.tool,
                         "file": chunk.file, "text": chunk.text}
    return out


def _budget_from_args(args, tasks) -> BudgetSpec:
    base = tasks[0].budget
    overrides = {
        "baseline_runs": args.runs,
        "max_runs_per_cell": args.max_runs,
        "min_runs_per_cell": args.min_runs,
        "max_workers": args.workers,
        "max_minutes": args.max_minutes,
        "max_usd": args.max_usd,
        "alpha": args.alpha,
        "correction": args.correction,
        "stop_rule": args.stop_rule,
        "meaningful_drop": args.meaningful_drop,
    }
    values = {k: v for k, v in overrides.items() if v is not None}
    return BudgetSpec(**{**asdict(base), **values})


def _discover(args, budget: BudgetSpec | None = None):
    names = [t.strip() for t in args.tasks.split(",")] if args.tasks else None
    overrides = asdict(budget) if budget else None
    return discover_tasks(names=names, budget_override=overrides)


def _select_cells(cells: list[CellSpec], spec: str | None) -> list[CellSpec]:
    if not spec:
        return cells
    wanted = [s.strip() for s in spec.split(",") if s.strip()]
    selected = [cells[0]]
    for cell in cells[1:]:
        if any(cell.id == w or cell.id.startswith(w + ":") or cell.kind == w for w in wanted):
            selected.append(cell)
    return selected


def _print_estimate(tasks, cells: list[CellSpec], budget: BudgetSpec, pricing: Pricing) -> dict:
    est = estimate_grid(tasks=len(tasks), cells=max(0, len(cells) - 1),
                        baseline_runs=budget.baseline_runs,
                        runs_per_cell=budget.max_runs_per_cell,
                        calls_per_run=DEFAULT_CALLS_PER_RUN,
                        prompt_tokens_per_call=DEFAULT_PROMPT_TOKENS,
                        completion_tokens_per_call=DEFAULT_COMPLETION_TOKENS,
                        pricing=pricing, seconds_per_run=DEFAULT_SECONDS_PER_RUN,
                        workers=budget.max_workers)
    print(f"[estimate] {len(tasks)} tasks x {len(cells)} cells, "
          f"baseline {budget.baseline_runs}, <= {budget.max_runs_per_cell}/cell")
    print(f"[estimate] upper bound: {est['total_runs']} runs, {est['model_calls']} calls, "
          f"${est['usd']:.2f}, ~{est['wall_minutes']:.0f} min wall clock at "
          f"{budget.max_workers} workers")
    plan = shrink_plan(len(cells) - 1, budget.baseline_runs, budget.max_runs_per_cell,
                       budget.min_runs_per_cell, est["usd"], budget.max_usd,
                       est["wall_minutes"], budget.max_minutes, len(tasks),
                       est["model_calls"], budget.max_total_calls)
    if plan["notes"]:
        print(f"[estimate] auto-shrink: {'; '.join(plan['notes'])}")
        print(f"[estimate] planned max: {plan['model_calls']} calls, "
              f"<= {plan['runs_per_cell']}/perturbation cell")
    return plan


def _meta(tasks, cells, budget, model, started, fix_line="") -> dict:
    return {
        "experiment_id": "",
        "status": "running",
        "started_at": started,
        "finished_at": "",
        "elapsed_minutes": None,
        "model": asdict(model),
        "budget": asdict(budget),
        "tasks": [{"id": t.id, "name": t.name, "description": t.description} for t in tasks],
        "task_ids": [t.id for t in tasks],
        "task_chunks": _task_chunks(tasks),
        "cells": [{"id": c.id, "label": c.label, "kind": c.kind, "config": c.config}
                  for c in cells],
        "fix_line": fix_line,
        "spend_usd": 0.0,
        "calls": 0,
        "runs": 0,
        "stopped_reason": "",
    }


def _execute_run(args, cells: list[CellSpec], experiment_id: str, fix_line: str = "") -> tuple[Store, dict]:
    tasks = _discover(args)
    budget = _budget_from_args(args, tasks)
    pricing = Pricing.from_model(tasks[0].model)
    plan = _print_estimate(tasks, cells, budget, pricing)
    if plan["runs_per_cell"] < budget.max_runs_per_cell:
        budget = BudgetSpec(**{
            **asdict(budget),
            "max_runs_per_cell": plan["runs_per_cell"],
        })
        tasks = _discover(args, budget)

    store = Store(experiment_id)
    tracker = BudgetTracker(budget.max_total_calls, budget.max_usd)
    started = utc_now()
    meta = _meta(tasks, cells, budget, tasks[0].model, started, fix_line)
    meta["experiment_id"] = experiment_id
    store.save_meta(meta)

    proxy = ProxyServer(tasks[0].model.base_url, store, tracker, pricing,
                        upstream_api_key=os.environ.get("MODEL_API_KEY", "local"))
    proxy.preflight()
    proxy.start()
    t0 = time.time()
    try:
        runner = Runner(store, proxy, keep_workspaces=args.keep_workdirs, fix_line=fix_line)
        grid = GridRunner(runner, tasks, cells, budget, tracker,
                          on_run=lambda r, i, _n: print(
                              f"[{i:04d}] {r.cell_id:<34} {r.task_id:<11} "
                              f"{'PASS' if r.success else 'fail'} "
                              f"{r.duration_s:5.1f}s {r.model_calls:2d} calls"),
                          on_cell=lambda summaries: _checkpoint(store, meta, summaries, tracker, t0))
        grid.run()
    finally:
        proxy.stop()

    elapsed = (time.time() - t0) / 60.0
    meta.update({
        "status": "complete" if not grid.stopped_reason else "stopped",
        "finished_at": utc_now(),
        "elapsed_minutes": round(elapsed, 1),
        "spend_usd": round(tracker.usd, 6),
        "calls": tracker.calls,
        "runs": grid._total_runs,
        "stopped_reason": grid.stopped_reason,
        "cells": [{"id": c.id, "label": c.label, "kind": c.kind, "config": c.config}
                  for c in cells],
        "cell_summaries": [asdict(s) for s in grid.summaries],
    })
    store.save_meta(meta)
    _print_status(store, meta)
    return store, meta


def _checkpoint(store: Store, meta: dict, summaries, tracker: BudgetTracker, t0: float) -> None:
    meta["cell_summaries"] = [asdict(s) for s in summaries]
    meta["spend_usd"] = round(tracker.usd, 6)
    meta["calls"] = tracker.calls
    meta["elapsed_minutes"] = round((time.time() - t0) / 60.0, 1)
    store.save_meta(meta)


def _print_status(store: Store, meta: dict) -> None:
    from shakedown.analysis import build_analysis
    analysis = build_analysis(store, meta)
    store.write_json("analysis.json", analysis)
    base = analysis["baseline"]
    print("")
    print(f"[baseline] {base['k']}/{base['n']} = {_fmt_pct(base['rate'])} "
          f"(Wilson 95% {base['wilson_text']}, unadjusted MDE at this n = {base['mde_at_n']} pts)")
    print(f"[cost] {analysis['summary']['runs']} runs, {analysis['summary']['calls']} model calls, "
          f"${analysis['summary']['spend_usd']:.4f}, {analysis['summary']['elapsed_minutes']} min")
    print(f"[validity] {analysis['summary']['invalid_runs']} invalid runs "
          f"({analysis['summary']['invalid_rate']:.1%})")
    for cell in analysis["cells"]:
        if cell["n"] == 0:
            continue
        sig = "*" if cell["significant"] else " "
        print(f"  {sig} {cell['label']:<34} {cell['k']:>2}/{cell['n']:<2} "
              f"{cell['delta_points']:+6.1f} pts  p_raw={cell['p_raw']:.3f} "
              f"p_adj({cell['p_adjustment_family']})={cell['p_adjusted']:.3f}  "
              f"{cell['verdict']}: {cell['reason']}")
    if analysis["fix_first"]:
        top = analysis["fix_first"][0]
        print(f"[fix-first] {top['label']} — {', '.join(top['reasons'])}")
    stop = meta.get("stopped_reason")
    if stop:
        print(f"[stopped] {stop}")
    print(f"[report] {store.dir / 'report.html'}")


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

def cmd_run(args) -> int:
    tasks = _discover(args)
    budget = _budget_from_args(args, tasks)
    tasks = _discover(args, budget)
    cells = build_cells(tasks, ablation=not args.no_ablation, chaos=not args.no_chaos,
                        noise=args.noise)
    if args.baseline:
        cells = [cells[0]]
    cells = _select_cells(cells, args.cells)
    experiment_id = args.experiment or new_experiment_id("baseline" if args.baseline else "grid")
    store, meta = _execute_run(args, cells, experiment_id)
    from shakedown.report.build import build_report
    path = build_report(store)
    print(f"[report] wrote {path}")
    if args.open:
        _open(path)
    return 0


def cmd_report(args) -> int:
    from shakedown.analysis import build_analysis
    from shakedown.report.build import build_report
    path = Path(args.path)
    store = Store(path.name, path.parent)
    meta = store.load_meta()
    analysis = build_analysis(store, meta)
    store.write_json("analysis.json", analysis)
    out = build_report(store, analysis)
    print(f"[report] {out}")
    base = analysis["baseline"]
    print(f"[baseline] {base['k']}/{base['n']} = {_fmt_pct(base['rate'])} "
          f"(Wilson 95% {base['wilson_text']})")
    if args.open:
        _open(out)
    return 0


def cmd_fix(args) -> int:
    from shakedown.fix import run_fix
    result = run_fix(args.path, None, args.runs, args.max_minutes, args.workers,
                     keep_workspaces=args.keep_workdirs)
    if result.get("status") != "ok":
        print(result.get("message", "nothing to fix"))
        return 1
    print(f"[fix] target: {result['treating']['label']}")
    print(f"[fix] {result['fix_line']}")
    print(f"[fix] baseline: {_fmt_pct(result['baseline']['original']['rate'])} -> "
          f"{_fmt_pct(result['baseline']['fixed']['rate'])} "
          f"({result['baseline']['verdict']})")
    if result.get("target"):
        print(f"[fix] cell {result['target_cell']}: "
              f"{_fmt_pct(result['target']['original']['rate'])} -> "
              f"{_fmt_pct(result['target']['fixed']['rate'])} "
              f"({result['target']['verdict']})")
    from shakedown.report.build import build_report
    store = Store(result["experiment_id"])
    out = build_report(store)
    print(f"[report] {out}")
    if args.open:
        _open(out)
    return 0


def cmd_check(args) -> int:
    saved = json.loads(Path(args.baseline).read_text(encoding="utf-8"))
    old = saved["baseline"]
    args.baseline = True
    tasks = _discover(args)
    budget = _budget_from_args(args, tasks)
    budget = BudgetSpec(**{**asdict(budget), "baseline_runs": args.runs})
    tasks = _discover(args, budget)
    cells = [CellSpec(id="baseline", label="Baseline (no perturbation)", kind="baseline")]
    store, meta = _execute_run(args, cells, args.experiment or new_experiment_id("check"))
    from shakedown.analysis import build_analysis
    from shakedown.stats import two_proportion_test
    analysis = build_analysis(store, store.load_meta())
    new = analysis["baseline"]
    cmp = two_proportion_test(new["k"], new["n"], old["k"], old["n"], alpha=args.alpha or 0.05)
    regressed = cmp.p_value < (args.alpha or 0.05) and cmp.delta < 0
    badge = f"{_fmt_pct(new['rate'])} [{new['wilson'][0] * 100:.0f}-{new['wilson'][1] * 100:.0f}]"
    payload = {"baseline": old, "current": new, "comparison": cmp.as_dict(),
               "regressed": regressed, "badge": badge}
    store.write_json("check_result.json", payload)
    print(f"[check] saved {_fmt_pct(old['rate'])} ({old['k']}/{old['n']}) vs current "
          f"{_fmt_pct(new['rate'])} ({new['k']}/{new['n']}); "
          f"delta {cmp.delta * 100:+.1f} pts, p={cmp.p_value:.4f}")
    print(f"[check] {'REGRESSION' if regressed else 'ok'} | badge: {badge}")
    return 2 if regressed else 0


def cmd_tasks(args) -> int:
    tasks = _discover(args)
    for task in tasks:
        print(f"{task.id:<26} {task.name:<28} {task.description.splitlines()[0][:60]}")
        print(f"  chunks: {', '.join(c.id for c in task.chunks)}")
        print(f"  cells: {len(build_cells([task])) - 1} perturbations")
    return 0


def cmd_selftest(args) -> int:
    tasks = _discover(args)
    failures = 0
    for task in tasks:
        with tempfile.TemporaryDirectory() as td:
            wd = Path(td) / "repo"
            shutil.copytree(task.source, wd)
            buggy = subprocess.run(task.success_command, shell=True, cwd=wd,
                                   capture_output=True, text=True)
            for solution in (task.dir / "solution").glob("*.py"):
                shutil.copy2(solution, wd / solution.name)
            fixed = subprocess.run(task.success_command, shell=True, cwd=wd,
                                   capture_output=True, text=True)
            ok = buggy.returncode != 0 and fixed.returncode == 0
            failures += 0 if ok else 1
            print(f"[{'ok' if ok else 'FAIL'}] {task.id}: buggy exit={buggy.returncode}, "
                  f"fixed exit={fixed.returncode}")
            if not ok:
                print(buggy.stdout[-400:])
                print(fixed.stdout[-400:])
    print(f"[selftest] {len(tasks) - failures}/{len(tasks)} tasks behave correctly")
    return 1 if failures else 0


def cmd_estimate(args) -> int:
    tasks = _discover(args)
    budget = _budget_from_args(args, tasks)
    cells = build_cells(tasks, ablation=not args.no_ablation, chaos=not args.no_chaos,
                        noise=args.noise)
    if args.baseline:
        cells = [cells[0]]
    _print_estimate(tasks, cells, budget, Pricing.from_model(tasks[0].model))
    return 0


def cmd_demo(args) -> int:
    args.baseline = False
    args.cells = None
    args.no_ablation = False
    args.no_chaos = False
    if args.runs is None:
        args.runs = 60
    if args.max_runs is None:
        args.max_runs = 40
    if args.min_runs is None:
        args.min_runs = 16
    args.experiment = args.experiment or new_experiment_id("demo")
    if cmd_selftest(args) != 0:
        print("[demo] task self-test failed; aborting")
        return 1
    code = cmd_run(args)
    if code != 0:
        return code
    if not args.no_fix:
        print("[demo] re-measuring the top fragility with a suggested fix")
        from shakedown.fix import run_fix
        result = run_fix(Path("runs") / args.experiment, None, args.fix_runs or 20,
                         args.max_minutes or 60, args.workers or 2)
        if result.get("status") == "ok":
            print(f"[demo] fixed baseline: {_fmt_pct(result['baseline']['fixed']['rate'])} "
                  f"({result['baseline']['verdict']})")
            from shakedown.report.build import build_report
            build_report(Store(result["experiment_id"]))
    return 0


def _open(path: Path) -> None:
    try:
        if os.name == "nt":
            os.startfile(str(path))  # type: ignore[attr-defined]
        elif sys.platform == "darwin":
            subprocess.Popen(["open", str(path)])
        else:
            subprocess.Popen(["xdg-open", str(path)])
    except Exception:
        pass


def _configure_cli_output() -> None:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(errors="backslashreplace")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="shakedown",
                                     description="Sensitivity profiler for AI agents")
    sub = parser.add_subparsers(dest="command", required=True)

    def common(p):
        p.add_argument("--tasks", help="comma-separated task ids (default: all)")
        p.add_argument("-n", "--runs", type=int, default=None, help="baseline runs")
        p.add_argument("--max-runs", type=int, default=None, help="max runs per cell")
        p.add_argument("--min-runs", type=int, default=None, help="min runs per cell")
        p.add_argument("--workers", type=int, default=None)
        p.add_argument("--max-minutes", type=float, default=None)
        p.add_argument("--max-usd", type=float, default=None)
        p.add_argument("--alpha", type=float, default=None)
        p.add_argument("--correction", choices=["holm", "bh"], default=None)
        p.add_argument("--stop-rule", choices=["sequential", "fixed"], default=None)
        p.add_argument("--meaningful-drop", type=float, default=None,
                       help="equivalence margin as a fraction, e.g. 0.25")
        p.add_argument("--experiment", default=None)
        p.add_argument("--keep-workdirs", action="store_true")
        p.add_argument("--no-ablation", action="store_true")
        p.add_argument("--no-chaos", action="store_true")
        p.add_argument("--noise", action="store_true", help="include Tier 3 input noise")

    run = sub.add_parser("run", help="run baseline or the full grid")
    common(run)
    run.add_argument("--baseline", action="store_true", help="baseline only")
    run.add_argument("--cells", default=None, help="filter cells by id, prefix, or kind")
    run.add_argument("--open", action="store_true")
    run.set_defaults(func=cmd_run)

    report = sub.add_parser("report", help="rebuild the HTML report")
    report.add_argument("path", help="experiment directory, e.g. runs/grid-20260101-120000")
    report.add_argument("--open", action="store_true")
    report.set_defaults(func=cmd_report)

    fix = sub.add_parser("fix", help="suggest a fix for the top fragility and re-measure")
    fix.add_argument("path")
    fix.add_argument("-n", "--runs", type=int, default=20)
    fix.add_argument("--workers", type=int, default=2)
    fix.add_argument("--max-minutes", type=float, default=60)
    fix.add_argument("--keep-workdirs", action="store_true")
    fix.add_argument("--open", action="store_true")
    fix.set_defaults(func=cmd_fix)

    check = sub.add_parser("check", help="CI gate: compare a fresh baseline to a saved one")
    check.add_argument("--baseline", required=True, help="saved analysis.json")
    check.add_argument("--tasks", default=None)
    check.add_argument("-n", "--runs", type=int, default=30)
    check.add_argument("--workers", type=int, default=2)
    check.add_argument("--max-minutes", type=float, default=30)
    check.add_argument("--max-usd", type=float, default=None)
    check.add_argument("--alpha", type=float, default=0.05)
    check.add_argument("--correction", choices=["holm", "bh"], default=None)
    check.add_argument("--stop-rule", choices=["sequential", "fixed"], default=None)
    check.add_argument("--min-runs", type=int, default=None)
    check.add_argument("--max-runs", type=int, default=None)
    check.add_argument("--meaningful-drop", type=float, default=None)
    check.add_argument("--experiment", default=None)
    check.add_argument("--keep-workdirs", action="store_true")
    check.add_argument("--no-ablation", action="store_true")
    check.add_argument("--no-chaos", action="store_true")
    check.add_argument("--noise", action="store_true")
    check.set_defaults(func=cmd_check)

    tasks_cmd = sub.add_parser("tasks", help="list discovered tasks")
    tasks_cmd.add_argument("--tasks", default=None)
    tasks_cmd.set_defaults(func=cmd_tasks)

    selftest = sub.add_parser("selftest", help="verify every task fails then passes")
    selftest.add_argument("--tasks", default=None)
    selftest.set_defaults(func=cmd_selftest)

    estimate = sub.add_parser("estimate", help="estimate grid cost and runtime")
    common(estimate)
    estimate.add_argument("--baseline", action="store_true")
    estimate.set_defaults(func=cmd_estimate)

    demo = sub.add_parser("demo", help="selftest, full grid, report, and a fix re-run")
    common(demo)
    demo.add_argument("--open", action="store_true")
    demo.add_argument("--no-fix", action="store_true")
    demo.add_argument("--fix-runs", type=int, default=20)
    demo.set_defaults(func=cmd_demo)
    return parser


def main(argv: list[str] | None = None) -> int:
    _configure_cli_output()
    load_dotenv()
    parser = build_parser()
    args = parser.parse_args(argv)
    load_dotenv()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())