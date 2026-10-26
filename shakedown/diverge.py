"""First-divergence analysis between passing and failing agent traces.

For each step index we compare the distribution of the next action in passing
runs against failing runs and report the total variation distance.  The first
step where that distance crosses 0.5 (with at least two runs per group) is the
divergence point.  This is descriptive, not a significance test.
"""

from __future__ import annotations

from collections import Counter

TV_THRESHOLD = 0.5
MIN_GROUP = 2


def _tv(pass_counter: Counter, fail_counter: Counter) -> float:
    keys = set(pass_counter) | set(fail_counter)
    total_p = sum(pass_counter.values())
    total_f = sum(fail_counter.values())
    if not total_p or not total_f:
        return 0.0
    return 0.5 * sum(abs(pass_counter[k] / total_p - fail_counter[k] / total_f) for k in keys)


def _analyze_group(runs: list[dict]) -> dict | None:
    passing = [r for r in runs if r.get("success")]
    failing = [r for r in runs if not r.get("success")]
    if len(passing) < MIN_GROUP or len(failing) < MIN_GROUP:
        return None

    max_len = max(len(r.get("actions") or []) for r in runs)
    timeline = []
    divergence = None
    for step in range(max_len):
        pass_actions = Counter(r["actions"][step] for r in passing
                               if len(r.get("actions") or []) > step)
        fail_actions = Counter(r["actions"][step] for r in failing
                               if len(r.get("actions") or []) > step)
        if sum(pass_actions.values()) < MIN_GROUP or sum(fail_actions.values()) < MIN_GROUP:
            continue
        tv = _tv(pass_actions, fail_actions)
        entry = {
            "step": step,
            "tv": round(tv, 3),
            "pass_n": sum(pass_actions.values()),
            "fail_n": sum(fail_actions.values()),
            "pass_top": pass_actions.most_common(1)[0][0],
            "fail_top": fail_actions.most_common(1)[0][0],
            "pass_dist": dict(pass_actions.most_common(4)),
            "fail_dist": dict(fail_actions.most_common(4)),
        }
        timeline.append(entry)
        if divergence is None and tv >= TV_THRESHOLD:
            divergence = entry

    # Where do trajectories end? A failing run that stops at step 3 while
    # passing runs reach step 7 is itself a useful divergence signal.
    pass_lengths = [len(r.get("actions") or []) for r in passing]
    fail_lengths = [len(r.get("actions") or []) for r in failing]
    return {
        "pass_n": len(passing),
        "fail_n": len(failing),
        "pass_rate": len(passing) / len(runs),
        "timeline": timeline,
        "divergence_step": divergence["step"] if divergence else None,
        "divergence": divergence,
        "mean_steps_pass": round(sum(pass_lengths) / max(1, len(pass_lengths)), 2),
        "mean_steps_fail": round(sum(fail_lengths) / max(1, len(fail_lengths)), 2),
    }


def divergence_analysis(runs: list[dict]) -> dict:
    baseline = [r for r in runs if r.get("cell_id") == "baseline"]
    by_task: dict[str, list[dict]] = {}
    for run in baseline:
        by_task.setdefault(run["task_id"], []).append(run)

    tasks = {}
    for task_id, group in sorted(by_task.items()):
        result = _analyze_group(group)
        if result:
            tasks[task_id] = result

    pooled = _analyze_group(baseline)
    if tasks:
        # Prefer the task with the clearest divergence for the headline.
        best = max(tasks.items(),
                   key=lambda kv: (kv[1]["divergence_step"] is not None,
                                   kv[1]["divergence"]["tv"] if kv[1]["divergence"] else 0))
        headline = {"task_id": best[0], **best[1]}
    else:
        headline = pooled
    return {"per_task": tasks, "pooled": pooled, "headline": headline,
            "threshold": TV_THRESHOLD}