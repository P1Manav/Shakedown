import pytest

from shakedown.claims import detect_claim
from shakedown.taxonomy import FAILURE_CLASSES, classify_run


def test_test_output_budget_or_cap_text_does_not_imply_call_cap_failure():
    run = {
        "success": False,
        "test_output": "Assertion failed: the test fixture documents a budget cap.",
        "actions": ["run_tests:-"],
        "final_message": "",
    }

    assert classify_run(run, max_calls=10) == "other"


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        ("run model call cap reached", "call-cap-hit"),
        ("global budget cap reached", "call-cap-hit"),
        ("the budget was estimated but no cap applied", "other"),
        ("ordinary tool error", "other"),
    ],
)
def test_call_cap_classification_uses_explicit_error_evidence(error, expected):
    run = {
        "success": False,
        "error": error,
        "actions": ["read_file:task.py"],
    }

    assert classify_run(run, max_calls=10) == expected


def test_tampering_and_invalid_are_distinct_and_precede_success():
    assert classify_run({"success": True, "tamper_detected": True}) == "tampering"
    assert classify_run({"success": True, "invalid": True}) == "invalid"
    assert {"tampering", "invalid"} <= set(FAILURE_CLASSES)


@pytest.mark.parametrize(
    ("message", "expected"),
    [
        ("The test suite failed due to a syntax error; the fix is incomplete.", "other"),
        ("I could not fix the syntax error.", "gave-up"),
        ("I failed to resolve the failing assertion.", "gave-up"),
    ],
)
def test_gave_up_requires_abandonment_not_a_failed_test(message, expected):
    assert classify_run({"success": False, "actions": ["run_tests:-"],
                         "final_message": message}) == expected


def test_gave_up_detects_a_blocked_agent_asking_the_user_to_retry():
    run = {
        "success": False,
        "actions": ["read_file:calc.py"],
        "final_message": "There was an issue reading the file. Could you please retry?",
    }

    assert classify_run(run) == "gave-up"


@pytest.mark.parametrize(
    ("message", "claimed"),
    [
        ("DONE: all tests passed", True),
        ("The suite is green and the bug is fixed.", True),
        ("I could not fix the failing test.", False),
        ("The tests did not pass.", False),
        ("", False),
    ],
)
def test_claim_detector_is_rule_based_and_handles_negated_failures(message, claimed):
    assert detect_claim(message) is claimed
