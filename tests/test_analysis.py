from shakedown.analysis import _label
from shakedown.stats import minimum_detectable_effect, two_proportion_test


def test_nonsignificant_positive_result_is_not_labeled_dead_weight():
    comparison = two_proportion_test(1, 1, 0, 1)

    label, _ = _label(
        comparison,
        adjusted_p=1.0,
        alpha=0.05,
        meaningful_drop=0.25,
        n=1,
        mde=minimum_detectable_effect(0.0, 1, 1),
    )

    assert label == "inconclusive"
