"""Turn raw runs into the report's statistical model."""

from __future__ import annotations

from collections import Counter, defaultdict

from shakedown.claims import aggregate_claims
from shakedown.diverge import divergence_analysis
from shakedown.spec import BudgetSpec
from shakedown.stats import (
    benjamini_hochberg,
    holm,
    is_meaningful_effect_ruled_out,
    minimum_detectable_effect,
    newcombe_difference_interval,
    norm_ppf,
    stratified_binary_comparison,
    stratified_minimum_detectable_effect,
    two_proportion_test,
    wilson_interval,
)
from shakedown.store import Store
from shakedown.taxonomy import aggregate_taxonomy


def _per_task(runs: list[dict]) -> dict:
    out = {}
    for run in runs:
        entry = out.setdefault(run["task_id"], {"n": 0, "k": 0})
        entry["n"] += 1
        entry["k"] += int(bool(run.get("success")))
    for task_id, entry in out.items():
        ci = wilson_interval(entry["k"], entry["n"])
        entry["rate"] = entry["k"] / entry["n"] if entry["n"] else 0.0
        entry["wilson"] = [ci.low, ci.high]
    return out


def _cost_block(runs: list[dict], baseline_cost_per_success: float | None,
                baseline_rate: float, enough_runs: bool) -> dict:
    n = len(runs)
    successes = sum(1 for r in runs if r.get("success"))
    tokens = sum(r.get("prompt_tokens", 0) + r.get("completion_tokens", 0) for r in runs)
    usd = sum(r.get("cost_usd", 0.0) for r in runs)
    calls = sum(r.get("model_calls", 0) for r in runs)
    steps = sum(r.get("steps", 0) for r in runs)
    durations = [r.get("duration_s", 0.0) for r in runs]
    cost_units = usd if usd > 0 else tokens
    cost_per_success = (cost_units / successes) if successes else None
    ratio = None
    risk = False
    if cost_per_success is not None and baseline_cost_per_success:
        ratio = cost_per_success / baseline_cost_per_success
        rate = successes / n if n else 0.0
        risk = bool(enough_runs and abs(rate - baseline_rate) < 0.05 and ratio > 1.5)
    return {
        "n": n,
        "tokens_per_run": round(tokens / n, 1) if n else 0,
        "calls_per_run": round(calls / n, 2) if n else 0,
        "steps_per_run": round(steps / n, 2) if n else 0,
        "usd_per_run": round(usd / n, 6) if n else 0,
        "usd_total": round(usd, 6),
        "cost_units": "usd" if usd > 0 else "tokens",
        "cost_per_success": round(cost_per_success, 2) if cost_per_success is not None else None,
        "cost_ratio": round(ratio, 2) if ratio is not None else None,
        "cost_risk": risk,
        "mean_duration_s": round(sum(durations) / n, 2) if n else 0,
    }


def _exposure_block(runs: list[dict], baseline_k: int, baseline_n: int,
                    z: float) -> dict:
    tracked = [run for run in runs if "perturbation_applied" in run]
    exposed = [run for run in tracked if run["perturbation_applied"]]
    n = len(exposed)
    k = sum(1 for run in exposed if run.get("success"))
    comparison = None
    if n:
        result = two_proportion_test(k, n, baseline_k, baseline_n)
        ci_low, ci_high = newcombe_difference_interval(
            k, n, baseline_k, baseline_n, z
        )
        comparison = result.as_dict()
        comparison["ci_low"] = ci_low
        comparison["ci_high"] = ci_high
    return {
        "tracked_runs": len(tracked),
        "exposure_rate": len(exposed) / len(tracked) if tracked else None,
        "n": n,
        "k": k,
        "rate": k / n if n else None,
        "comparison": comparison,
    }


def _label(comparison, adjusted_p: float, alpha: float, meaningful_drop: float,
           n: int, mde: float, exposure_rate: float | None = None) -> tuple[str, str]:
    if n == 0:
        return "not-run", "no runs completed"
    if adjusted_p < alpha and comparison.delta < 0:
        return "critical", f"adjusted p={adjusted_p:.4f}; {abs(comparison.delta) * 100:.1f} pt drop"
    if adjusted_p < alpha and comparison.delta > 0:
        return "helpful", f"adjusted p={adjusted_p:.4f}; {comparison.delta * 100:.1f} pt lift"
    if is_meaningful_effect_ruled_out(
            comparison.ci_low, comparison.ci_high, meaningful_drop):
        if exposure_rate is None or exposure_rate < 0.8:
            return "insufficient exposure", (
                f"meaningful effect ruled out under ITT, but exposure rate is "
                f"{f'{exposure_rate * 100:.1f}%' if exposure_rate is not None else 'unknown'} (< 80%)"
            )
        return "dead-weight", (
            f"familywise CI rules out an effect of {meaningful_drop * 100:.0f} pts or more "
            f"(interval {comparison.ci_low * 100:.1f} to {comparison.ci_high * 100:.1f} pts)")
    return "inconclusive", f"not significant; MDE {mde * 100:.0f} pts"


def build_analysis(store: Store, meta: dict | None = None) -> dict:
    meta = meta or store.load_meta()
    all_runs = store.load_runs()
    if not all_runs:
        raise SystemExit("no runs recorded in this experiment")
    invalid_runs = [r for r in all_runs if r.get("invalid", False)]
    runs = [r for r in all_runs if not r.get("invalid", False)]
    budget = BudgetSpec(**meta.get("budget", {}))
    alpha = budget.alpha
    correction = budget.correction
    meaningful_drop = budget.meaningful_drop

    baseline = [r for r in runs if r["cell_id"] == "baseline"]
    if not baseline:
        raise SystemExit("no valid baseline runs; run the baseline first")
    all_baseline = [r for r in all_runs if r["cell_id"] == "baseline"]
    k0, n0 = sum(1 for r in baseline if r["success"]), len(baseline)
    base_rate = k0 / n0
    base_ci = wilson_interval(k0, n0)
    baseline_tokens = sum(r.get("prompt_tokens", 0) + r.get("completion_tokens", 0)
                          for r in baseline)
    baseline_usd = sum(r.get("cost_usd", 0.0) for r in baseline)
    base_units = baseline_usd if baseline_usd > 0 else baseline_tokens
    base_successes = max(1, k0)
    baseline_cost_per_success = base_units / base_successes

    cell_order = [c["id"] for c in meta.get("cells", []) if c.get("id") != "baseline"]
    by_cell: dict[str, list[dict]] = defaultdict(list)
    all_by_cell: dict[str, list[dict]] = defaultdict(list)
    for run in all_runs:
        if run["cell_id"] != "baseline":
            all_by_cell[run["cell_id"]].append(run)
            if not run.get("invalid", False):
                by_cell[run["cell_id"]].append(run)
    for cid in sorted(all_by_cell):
        if cid not in cell_order:
            cell_order.append(cid)

    cell_specs = {cell["id"]: cell for cell in meta.get("cells", [])}
    family_by_cell = {}
    for cid in cell_order:
        group = by_cell.get(cid, [])
        spec = cell_specs.get(cid, {})
        family_by_cell[cid] = (
            group[0].get("cell_kind", "") if group
            else spec.get("kind", "")
        ) or "other"
    family_sizes = dict(Counter(family_by_cell.values()))
    family_alphas = {
        family: alpha / max(1, size) for family, size in family_sizes.items()
    }
    planned_looks = (
        max(2, (budget.max_runs_per_cell + max(1, budget.batch_size) - 1)
            // max(1, budget.batch_size))
        if budget.stop_rule == "sequential" else 1
    )
    look_alphas = {
        family: family_alpha / planned_looks
        for family, family_alpha in family_alphas.items()
    }
    raw_cells = []
    for cid in cell_order:
        group = by_cell.get(cid, [])
        if group:
            first = group[0]
            label, kind, params = first["cell_label"], first["cell_kind"], first.get("params", {})
        else:
            spec = cell_specs.get(cid, {})
            label, kind, params = spec.get("label", cid), spec.get("kind", ""), spec.get("config", {})
        family = family_by_cell[cid]
        family_alpha = family_alphas[family]
        cell_alpha = look_alphas[family]
        family_z = norm_ppf(1.0 - cell_alpha / 2.0)
        n = len(group)
        k = sum(1 for r in group if r["success"])
        verdict, reason = "not-run", "not run (budget or wall-clock cap)"
        comparison = None
        stratified = None
        label_comparison = None
        mde = minimum_detectable_effect(base_rate, max(1, n), n0, cell_alpha)
        comparison_data = None
        if n:
            comparison = two_proportion_test(k, n, k0, n0)
            ci_low, ci_high = newcombe_difference_interval(k, n, k0, n0, family_z)
            comparison_data = comparison.as_dict()
            comparison_data["ci_low"] = ci_low
            comparison_data["ci_high"] = ci_high
            mde = minimum_detectable_effect(base_rate, n, n0, cell_alpha)
            treatment_by_task = {}
            for run in group:
                entry = treatment_by_task.setdefault(run["task_id"], [0, 0])
                entry[0] += int(bool(run.get("success")))
                entry[1] += 1
            control_by_task = {}
            for run in baseline:
                entry = control_by_task.setdefault(run["task_id"], [0, 0])
                entry[0] += int(bool(run.get("success")))
                entry[1] += 1
            stratified = stratified_binary_comparison(
                {task_id: tuple(counts) for task_id, counts in treatment_by_task.items()},
                {task_id: tuple(counts) for task_id, counts in control_by_task.items()},
                z=family_z,
            )
            if stratified:
                label_comparison = stratified
                task_strata = []
                for tid in sorted(set(treatment_by_task) & set(control_by_task)):
                    k_c, n_c = control_by_task[tid]
                    _, n_t = treatment_by_task[tid]
                    if n_c > 0 and n_t > 0:
                        task_strata.append((k_c / n_c, n_t, n_c))
                if task_strata:
                    mde = stratified_minimum_detectable_effect(task_strata, cell_alpha)
            else:
                label_comparison = comparison
        raw_cells.append({
            "id": cid, "label": label, "kind": kind, "params": params,
            "p_adjustment_family": family,
            "family_size": family_sizes[family],
            "family_alpha": family_alpha,
            "cell_analysis_alpha": cell_alpha,
            "n": n, "k": k, "invalid_runs": sum(
                1 for r in all_by_cell.get(cid, []) if r.get("invalid", False)),
            "tamper_detected": sum(
                1 for r in all_by_cell.get(cid, []) if r.get("tamper_detected", False)),
            "tamper_reasons": [
                reason for r in all_by_cell.get(cid, [])
                for reason in r.get("tamper_reasons", [])
            ],
            "tamper_counts_by_reason": dict(Counter(
                reason for r in all_by_cell.get(cid, [])
                for reason in r.get("tamper_reasons", [])
            )),
            "rate": (k / n) if n else 0.0,
            "exposure": _exposure_block(group, k0, n0, family_z),
            "comparison": comparison_data,
            "stratified_comparison": stratified.as_dict() if stratified else None,
            "label_driven_by": (
                "stratified" if stratified else "pooled_fallback" if comparison else "none"
            ),
            "label_comparison": label_comparison.as_dict() if label_comparison else None,
            "mde": mde, "claims": aggregate_claims(group),
            "taxonomy": aggregate_taxonomy(all_by_cell.get(cid, [])),
            "per_task": _per_task(group),
            "assigned_per_task": {
                task_id: sum(1 for run in all_by_cell.get(cid, [])
                             if run["task_id"] == task_id)
                for task_id in sorted({
                    run["task_id"] for run in all_by_cell.get(cid, [])
                })
            },
            "_raw": label_comparison.p_value if label_comparison else 1.0,
        })

    correction_results = {}
    for family in family_sizes:
        family_cells = [cell for cell in raw_cells
                        if cell["p_adjustment_family"] == family]
        p_values = [cell["_raw"] for cell in family_cells]
        if correction == "bh":
            adjusted, rejected = benjamini_hochberg(p_values, alpha)
        else:
            adjusted, rejected = holm(p_values, alpha)
        correction_results.update({
            cell["id"]: (adj, rej)
            for cell, adj, rej in zip(family_cells, adjusted, rejected)
        })
    # Cells with no runs must not be "rejected" by the correction.
    for cell in raw_cells:
        adj, rej = correction_results[cell["id"]]
        if cell["n"] == 0:
            adj, rej = 1.0, False
        cell["p_raw"] = cell["_raw"]
        cell["p_raw_pooled"] = (cell["comparison"] or {}).get("p_value", 1.0)
        cell["p_adjusted"] = adj
        cell["significant_raw"] = cell["_raw"] < alpha
        cell["significant_raw_pooled"] = cell["p_raw_pooled"] < alpha
        cell["significant"] = bool(rej)
        if cell["n"] and cell["label_comparison"]:
            from shakedown.stats import Comparison
            cmp_obj = Comparison(**cell["label_comparison"])
            verdict, reason = _label(
                cmp_obj, adj, alpha, meaningful_drop, cell["n"], cell["mde"],
                cell["exposure"].get("exposure_rate"),
            )
            cell["verdict"] = verdict
            cell["reason"] = reason
        else:
            cell["verdict"] = "not-run"
            cell["reason"] = "no runs completed"
        cell["mde_points"] = round(cell["mde"] * 100, 1)
        cell["delta_points"] = round(
            (cell["label_comparison"] or {}).get("delta", 0.0) * 100, 1
        )
        cell["ci_points"] = [
            round((cell["label_comparison"] or {}).get("ci_low", -1.0) * 100, 1),
            round((cell["label_comparison"] or {}).get("ci_high", 1.0) * 100, 1),
        ]
        cell["cost"] = _cost_block(
            by_cell.get(cell["id"], []), baseline_cost_per_success, base_rate,
            cell["n"] >= max(budget.min_runs_per_cell, 8))
        cell.pop("_raw", None)

    baseline_claims = aggregate_claims(baseline)
    baseline_taxonomy = aggregate_taxonomy(all_baseline)
    baseline_cost = _cost_block(baseline, None, base_rate, False)

    # Ranked fix-first list (suggestion text lives in fix.py to avoid a cycle).
    from shakedown.fix import rank_fix_first
    fix_first = rank_fix_first(raw_cells, base_rate, meta.get("task_chunks", {}))

    divergence = divergence_analysis(runs)

    total_tokens = sum(r.get("prompt_tokens", 0) + r.get("completion_tokens", 0)
                       for r in all_runs)
    total_usd = sum(r.get("cost_usd", 0.0) for r in all_runs)
    summary = {
        "runs": len(all_runs),
        "baseline_runs": n0,
        "invalid_runs": len(invalid_runs),
        "invalid_rate": len(invalid_runs) / len(all_runs),
        "tampering_runs": sum(1 for r in all_runs if r.get("tamper_detected", False)),
        "tamper_reasons": [
            reason for r in all_runs
            for reason in r.get("tamper_reasons", [])
        ],
        "tamper_counts_by_reason": dict(Counter(
            reason for r in all_runs
            for reason in r.get("tamper_reasons", [])
        )),
        "valid_runs": len(runs),
        "cells": len([c for c in raw_cells if c["n"] > 0]),
        "tasks": len({r["task_id"] for r in runs}),
        "model": meta.get("model", {}),
        "spend_usd": round(total_usd, 6),
        "tokens": total_tokens,
        "calls": sum(r.get("model_calls", 0) for r in all_runs),
        "started_at": min((r.get("started_at", "") for r in all_runs), default=""),
        "finished_at": meta.get("finished_at", ""),
        "elapsed_minutes": meta.get("elapsed_minutes"),
        "stopped_reason": meta.get("stopped_reason", ""),
        "correction": correction,
        "alpha": alpha,
        "meaningful_drop": meaningful_drop,
    }

    analysis = {
        "experiment_id": store.experiment_id,
        "generated_at": meta.get("finished_at", ""),
        "meta": meta,
        "summary": summary,
        "baseline": {
            "n": n0, "k": k0, "rate": base_rate,
            "invalid_runs": sum(1 for r in all_baseline if r.get("invalid", False)),
            "wilson": [base_ci.low, base_ci.high],
            "wilson_text": str(base_ci),
            "mde_at_n": round(minimum_detectable_effect(base_rate, n0, n0, alpha) * 100, 1),
            "per_task": _per_task(baseline),
            "claims": baseline_claims,
            "taxonomy": baseline_taxonomy,
            "cost": baseline_cost,
        },
        "cells": raw_cells,
        "divergence": divergence,
        "fix_first": fix_first,
        "methods": {
            "interval": "Wilson score interval (95%)",
            "test": "two-proportion z-test, Fisher exact when counts are small",
            "difference_ci": "Newcombe hybrid score interval with Bonferroni adjustment within cell-kind families",
            "effect_populations": {
                "comparison": "intent-to-treat: all valid runs assigned to the cell",
                "exposure": "runs with an observed perturbation diff, compared with baseline",
            },
            "label_driven_by": (
                "Cochran-Mantel-Haenszel task-stratified comparison when matched task strata "
                "exist; pooled two-proportion result is an explicit fallback otherwise"
            ),
            "family_sizes": family_sizes,
            "family_alpha": family_alphas,
            "per_look_confidence_by_family": {
                family: 1.0 - cell_alpha
                for family, cell_alpha in look_alphas.items()
            },
            "correction": (
                ("Holm-Bonferroni" if correction == "holm" else "Benjamini-Hochberg")
                + " within each cell-kind family"
            ),
            "corrections_applied_by_family": family_sizes,
            "mde_alpha_by_family": look_alphas,
            "sequential_looks": planned_looks,
            "sequential_spending": (
                f"family alpha divided across {planned_looks} planned looks"
                if budget.stop_rule == "sequential"
                else "no sequential alpha spending"
            ),
            "baseline_mde_alpha": alpha,
            "mde_power": 0.8,
            "meaningful_drop": meaningful_drop,
            "labels": {
                "critical": "adjusted p < alpha and a negative delta",
                "helpful": "adjusted p < alpha and a positive delta",
                "dead-weight": "not significant and the full CI rules out meaningful effects in either direction",
                "insufficient exposure": "meaningful effect ruled out under ITT, but exposure rate is below 80% or unknown",
                "inconclusive": "not significant and the CI still includes a meaningful effect",
            },
            "success_check": "executable command exit code (never an LLM judge)",
        },
    }
    if meta.get("fix_experiments"):
        latest_fix_id = meta["fix_experiments"][-1]
        try:
            fix_store = Store(latest_fix_id, store.dir.parent)
            fix_analysis = fix_store.read_json("analysis.json", {}) or {}
            if "fix" in fix_analysis:
                analysis["fix"] = fix_analysis["fix"]
        except Exception:
            pass
    return analysis