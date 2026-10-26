"""Turn raw runs into the report's statistical model."""

from __future__ import annotations

from collections import defaultdict

from shakedown.claims import aggregate_claims
from shakedown.diverge import divergence_analysis
from shakedown.spec import BudgetSpec
from shakedown.stats import (
    benjamini_hochberg,
    holm,
    is_meaningful_effect_ruled_out,
    minimum_detectable_effect,
    norm_ppf,
    newcombe_difference_interval,
    two_proportion_test,
    wilson_interval,
)
from shakedown.store import Store
from shakedown.taxonomy import aggregate_taxonomy, classify_run


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
           n: int, mde: float) -> tuple[str, str]:
    if n == 0:
        return "not-run", "no runs completed"
    if adjusted_p < alpha and comparison.delta < 0:
        return "critical", f"adjusted p={adjusted_p:.4f}; {abs(comparison.delta) * 100:.1f} pt drop"
    if adjusted_p < alpha and comparison.delta > 0:
        return "helpful", f"adjusted p={adjusted_p:.4f}; {comparison.delta * 100:.1f} pt lift"
    if is_meaningful_effect_ruled_out(
            comparison.ci_low, comparison.ci_high, meaningful_drop):
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

    raw_cells = []
    family_alpha = alpha / max(1, len(cell_order))
    family_z = norm_ppf(1.0 - family_alpha / 2.0)
    for cid in cell_order:
        group = by_cell.get(cid, [])
        if group:
            first = group[0]
            label, kind, params = first["cell_label"], first["cell_kind"], first.get("params", {})
        else:
            spec = next((c for c in meta.get("cells", []) if c.get("id") == cid), {})
            label, kind, params = spec.get("label", cid), spec.get("kind", ""), spec.get("config", {})
        n = len(group)
        k = sum(1 for r in group if r["success"])
        verdict, reason = "not-run", "not run (budget or wall-clock cap)"
        comparison = None
        mde = minimum_detectable_effect(base_rate, max(1, n), n0, family_alpha)
        comparison_data = None
        if n:
            comparison = two_proportion_test(k, n, k0, n0)
            ci_low, ci_high = newcombe_difference_interval(k, n, k0, n0, family_z)
            comparison_data = comparison.as_dict()
            comparison_data["ci_low"] = ci_low
            comparison_data["ci_high"] = ci_high
            mde = minimum_detectable_effect(base_rate, n, n0, family_alpha)
        raw_cells.append({
            "id": cid, "label": label, "kind": kind, "params": params,
            "n": n, "k": k, "invalid_runs": sum(
                1 for r in all_by_cell.get(cid, []) if r.get("invalid", False)),
            "tamper_detected": sum(
                1 for r in all_by_cell.get(cid, []) if r.get("tamper_detected", False)),
            "rate": (k / n) if n else 0.0,
            "exposure": _exposure_block(group, k0, n0, family_z),
            "comparison": comparison_data,
            "mde": mde, "claims": aggregate_claims(group),
            "taxonomy": aggregate_taxonomy(all_by_cell.get(cid, [])),
            "per_task": _per_task(group),
            "_raw": comparison.p_value if comparison else 1.0,
        })

    p_values = [c["_raw"] for c in raw_cells]
    if correction == "bh":
        adjusted, rejected = benjamini_hochberg(p_values, alpha)
    else:
        adjusted, rejected = holm(p_values, alpha)
    # Cells with no runs must not be "rejected" by the correction.
    for cell, adj, rej in zip(raw_cells, adjusted, rejected):
        if cell["n"] == 0:
            adj, rej = 1.0, False
        cell["p_raw"] = cell["_raw"]
        cell["p_adjusted"] = adj
        cell["significant_raw"] = cell["_raw"] < alpha
        cell["significant"] = bool(rej)
        if cell["n"] and cell["comparison"]:
            from shakedown.stats import Comparison
            cmp_obj = Comparison(**cell["comparison"])
            verdict, reason = _label(cmp_obj, adj, alpha, meaningful_drop, cell["n"], cell["mde"])
            cell["verdict"] = verdict
            cell["reason"] = reason
        else:
            cell["verdict"] = verdict
            cell["reason"] = reason
        cell["mde_points"] = round(cell["mde"] * 100, 1)
        cell["delta_points"] = round((cell["comparison"] or {}).get("delta", 0.0) * 100, 1)
        cell["ci_points"] = [
            round((cell["comparison"] or {}).get("ci_low", -1.0) * 100, 1),
            round((cell["comparison"] or {}).get("ci_high", 1.0) * 100, 1),
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
            "difference_ci": "Newcombe hybrid score interval with Bonferroni familywise confidence adjustment",
            "effect_populations": {
                "comparison": "intent-to-treat: all valid runs assigned to the cell",
                "exposure": "runs with an observed perturbation diff, compared with baseline",
            },
            "family_confidence": 1.0 - family_alpha,
            "correction": "Holm-Bonferroni" if correction == "holm" else "Benjamini-Hochberg",
            "corrections_applied_across": len(raw_cells),
            "mde_alpha": family_alpha,
            "mde_power": 0.8,
            "meaningful_drop": meaningful_drop,
            "labels": {
                "critical": "adjusted p < alpha and a negative delta",
                "helpful": "adjusted p < alpha and a positive delta",
                "dead-weight": "not significant and the full CI rules out meaningful effects in either direction",
                "inconclusive": "not significant and the CI still includes a meaningful effect",
            },
            "success_check": "executable command exit code (never an LLM judge)",
        },
    }
    return analysis