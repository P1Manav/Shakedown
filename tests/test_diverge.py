from shakedown.diverge import divergence_analysis


def test_divergence_uses_normalized_calls_and_reports_exploratory_permutation():
    runs = []
    for index in range(20):
        passed = index < 10
        path = "pass.py" if passed else "fail.py"
        runs.append({
            "run_id": f"run-{index}",
            "task_id": "task",
            "cell_id": "baseline",
            "success": passed,
            "actions": ["read_file:file.py"],
            "normalized_actions": [f'read_file({{"path":"{path}"}})'],
        })

    analysis = divergence_analysis(runs)
    headline = analysis["headline"]

    assert headline["divergence_step"] == 0
    assert headline["divergence"]["tv"] == 1.0
    assert headline["divergence"]["pass_top"] != headline["divergence"]["fail_top"]
    assert headline["exploratory"] is True
    assert 0.0 <= headline["exploratory_permutation_p"] <= 0.05


def test_divergence_requires_minimum_passing_and_failing_groups():
    runs = [
        {"task_id": "task", "cell_id": "baseline", "success": True, "actions": ["a"]},
        {"task_id": "task", "cell_id": "baseline", "success": False, "actions": ["b"]},
    ]

    analysis = divergence_analysis(runs)

    assert analysis["headline"] is None
