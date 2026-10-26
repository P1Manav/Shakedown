from concurrent.futures import ThreadPoolExecutor

import pytest

from shakedown.cost import (
    BudgetExceeded,
    BudgetTracker,
    Pricing,
    request_cost_ceiling,
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
