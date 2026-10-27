"""First-divergence analysis between passing and failing agent traces.

For each step index we compare the distribution of the next action in passing
runs against failing runs and report the total variation distance.  The first
step where that distance crosses 0.5 (with at least two runs per group) is the
divergence point.  This is descriptive, not a significance test.
"""

from __future__ import annotations

import hashlib
import random
from collections import Counter

TV_THRESHOLD = 0.5
MIN_GROUP = 2
PERMUTATIONS = 999


def _tv(pass_counter: Counter, fail_counter: Counter) -> float:
    keys = set(pass_counter) | set(fail_counter)
    total_p = sum(pass_counter.values())
    total_f = sum(fail_counter.values())
    if not total_p or not total_f:
        return 0.0
    return 0.5 * sum(abs(pass_counter[k] / total_p - fail_counter[k] / total_f) for k in keys)


def _actions(run: dict) -> list[str]:
    return run.get("normalized_actions") or run.get("actions") or []


def _timeline(paths: list[list[str]], labels: list[bool]) -> list[dict]:
    max_len = max((len(path) for path in paths), default=0)
    timeline = []
    for step in range(max_len):
        pass_actions = Counter(
            path[step] for path, passed in zip(paths, labels)
            if passed and len(path) > step
        )
        fail_actions = Counter(
            path[step] for path, passed in zip(paths, labels)
            if not passed and len(path) > step
        )
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
    return timeline


def _permutation_p(runs: list[dict], observed_max_tv: float | None) -> float | None:
    if observed_max_tv is None:
        return None
    paths = [_actions(run) for run in runs]
    labels = [bool(run.get("success")) for run in runs]
    task_indices: dict[str, list[int]] = {}
    for index, run in enumerate(runs):
        task_indices.setdefault(str(run.get("task_id", "")), []).append(index)
    seed_material = "\0".join(
        str(run.get("run_id", index)) for index, run in enumerate(runs)
    )
    rng = random.Random(int.from_bytes(
        hashlib.sha256(seed_material.encode("utf-8")).digest()[:8], "big"
    ))
    exceedances = 0
    for _ in range(PERMUTATIONS):
        permuted = labels.copy()
        for indices in task_indices.values():
            shuffled = [labels[index] for index in indices]
            rng.shuffle(shuffled)
            for index, label in zip(indices, shuffled):
                permuted[index] = label
        permuted_timeline = _timeline(paths, permuted)
        permuted_max = max((entry["tv"] for entry in permuted_timeline), default=0.0)
        if permuted_max >= observed_max_tv - 1e-12:
            exceedances += 1
    return (exceedances + 1) / (PERMUTATIONS + 1)


def _analyze_group(runs: list[dict]) -> dict | None:
    labels = [bool(run.get("success")) for run in runs]
    passing = sum(labels)
    failing = len(labels) - passing
    if passing < MIN_GROUP or failing < MIN_GROUP:
        return None

    paths = [_actions(run) for run in runs]
    timeline = _timeline(paths, labels)
    divergence = next(
        (entry for entry in timeline if entry["tv"] >= TV_THRESHOLD), None
    )
    max_tv = max((entry["tv"] for entry in timeline), default=None)
    permutation_p = _permutation_p(runs, max_tv)

    # Where do trajectories end? A failing run that stops at step 3 while
    # passing runs reach step 7 is itself a useful divergence signal.
    pass_lengths = [len(path) for path, passed in zip(paths, labels) if passed]
    fail_lengths = [len(path) for path, passed in zip(paths, labels) if not passed]
    return {
        "pass_n": passing,
        "fail_n": failing,
        "pass_rate": passing / len(runs),
        "timeline": timeline,
        "divergence_step": divergence["step"] if divergence else None,
        "divergence": divergence,
        "mean_steps_pass": round(sum(pass_lengths) / max(1, len(pass_lengths)), 2),
        "mean_steps_fail": round(sum(fail_lengths) / max(1, len(fail_lengths)), 2),
        "exploratory": True,
        "exploratory_permutation_p": permutation_p,
        "permutation_statistic": "maximum total-variation distance across eligible steps",
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
            "threshold": TV_THRESHOLD, "exploratory": True}