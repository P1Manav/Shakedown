from shakedown.perturb.base import RunState
from shakedown.perturb.noise import NoisePerturbation


def test_paraphrase_noise_uses_fixed_local_replacements_without_mutating_input():
    perturbation = NoisePerturbation("paraphrase")
    payload = {
        "messages": [
            {"role": "system", "content": "Keep the tests passing."},
            {
                "role": "user",
                "content": "Please fix the bug in this function and keep the tests passing.",
            },
        ],
        "tools": [{"type": "function", "function": {"name": "read_file"}}],
    }
    context = type("Context", (), {"state": RunState("paraphrase-repeatable")})()

    first = perturbation.apply_request(payload, context)
    second = perturbation.apply_request(payload, context)

    assert first["messages"][1]["content"] == (
        "Kindly repair the defect in this routine and keep the test suite passing."
    )
    assert second == first
    assert payload["messages"][1]["content"].startswith("Please fix the bug")
    assert first["messages"][0] == payload["messages"][0]
    assert first["tools"] == payload["tools"]
