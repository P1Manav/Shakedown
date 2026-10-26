from concurrent.futures import ThreadPoolExecutor

import pytest

from shakedown.cost import (
    BudgetExceeded,
    BudgetTracker,
    Pricing,
    request_cost_ceiling,
    shrink_plan,
    validate_endpoint_pricing,
)


def test_request_cost_ceiling_includes_prompt_and_output_limit():
    payload = {
        "messages": [
            {"role": "system", "content": "Follow the task."},
            {"role": "user", "content": "Fix this code."},
        ],
        "max_tokens": 700,
    }

    ceiling = request_cost_ceiling(payload, Pricing(1.0, 2.0))

    assert ceiling > 700 * 2 / 1_000_000
    assert ceiling < 0.01


def test_unpriced_remote_endpoint_fails_closed():
    with pytest.raises(ValueError, match="Remote model endpoint"):
        validate_endpoint_pricing("https://model.example/v1", Pricing())

    validate_endpoint_pricing("http://127.0.0.1:11434/v1", Pricing())
    validate_endpoint_pricing(
        "https://free-model.example/v1", Pricing(), allow_unpriced_remote=True
    )


def test_budget_reservations_prevent_concurrent_overspend():
    tracker = BudgetTracker(max_calls=4, max_usd=0.001)
    first = tracker.reserve(0.0006)

    with pytest.raises(BudgetExceeded, match="USD cap"):
        tracker.reserve(0.0005)

    tracker.finish(first, actual_usd=0.0004)
    snapshot = tracker.snapshot()
    assert snapshot["calls"] == 1
    assert snapshot["usd"] == pytest.approx(0.0004)
    assert snapshot["reserved_calls"] == 0
    assert snapshot["reserved_usd"] == 0


def test_global_call_cap_is_atomic_across_workers():
    tracker = BudgetTracker(max_calls=10, max_usd=0)

    def reserve():
        try:
            return tracker.reserve(0)
        except BudgetExceeded:
            return None

    with ThreadPoolExecutor(max_workers=20) as pool:
        reservations = list(pool.map(lambda _: reserve(), range(100)))
    accepted = [reservation for reservation in reservations if reservation is not None]

    assert len(accepted) == 10
    for reservation in accepted:
        tracker.finish(reservation, actual_usd=0)
    assert tracker.snapshot()["calls"] == 10


def test_budget_allows_zero_cost_requests_with_zero_spend_cap():
    tracker = BudgetTracker(max_calls=1, max_usd=0)
    reservation = tracker.reserve(0)
    tracker.finish(reservation, actual_usd=0)

    assert tracker.snapshot()["calls"] == 1
    with pytest.raises(BudgetExceeded, match="call cap"):
        tracker.reserve(0)


def test_budget_tracker_carries_usage_forward_between_campaign_stages():
    tracker = BudgetTracker(max_calls=3, max_usd=1.0, initial_calls=2, initial_usd=0.5)
    reservation = tracker.reserve(0.4)
    tracker.finish(reservation, actual_usd=0.4)

    assert tracker.snapshot()["calls"] == 3
    assert tracker.snapshot()["usd"] == pytest.approx(0.9)
    with pytest.raises(BudgetExceeded, match="call cap"):
        tracker.reserve(0)

    spend = BudgetTracker(max_calls=3, max_usd=1.0, initial_usd=0.5)
    spend.reserve(0.4)
    with pytest.raises(BudgetExceeded, match="USD cap"):
        spend.reserve(0.2)
    spent = BudgetTracker(max_calls=3, max_usd=0.5, initial_usd=0.6)
    with pytest.raises(BudgetExceeded, match="USD cap"):
        spent.reserve(0)
    with pytest.raises(ValueError, match="initial budget usage"):
        BudgetTracker(initial_calls=-1)


def test_shrink_plan_reduces_runs_to_fit_global_call_cap():
    plan = shrink_plan(
        cells=20,
        baseline_runs=30,
        runs_per_cell=40,
        min_runs=12,
        usd=0,
        max_usd=15,
        minutes=55.3,
        max_minutes=180,
        tasks=4,
        model_calls=6640,
        max_total_calls=6000,
    )

    assert plan["runs_per_cell"] == 30
    assert plan["model_calls"] == 5040
    assert plan["notes"] == ["reduced runs/cell to 30"]
