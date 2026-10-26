import math

import pytest

from shakedown.stats import (
    benjamini_hochberg,
    fisher_exact_two_sided,
    holm,
    minimum_detectable_effect,
    newcombe_difference_interval,
    norm_ppf,
    sequential_decision,
    two_proportion_test,
    wilson_interval,
)


def test_norm_ppf_known_quantiles():
    assert norm_ppf(0.975) == pytest.approx(1.959964, abs=1e-6)
    assert norm_ppf(0.5) == pytest.approx(0.0, abs=1e-9)
    assert norm_ppf(0.841344746) == pytest.approx(1.0, abs=1e-6)


def test_wilson_known_value():
    ci = wilson_interval(30, 50)
    assert ci.center == pytest.approx(0.6)
    assert ci.low == pytest.approx(0.4627, abs=1e-3)
    assert ci.high == pytest.approx(0.7239, abs=1e-3)


def test_wilson_extremes_stay_in_bounds():
    for k, n in [(0, 5), (5, 5), (0, 1), (1, 1), (0, 200)]:
        ci = wilson_interval(k, n)
        assert 0.0 <= ci.low <= ci.high <= 1.0
    # Zero successes still has a non-zero upper bound with finite n.
    assert wilson_interval(0, 10).high > 0


def test_fisher_exact_known_value():
    # Classic 3/10 vs 8/10 table: two-sided p = 0.0699
    assert fisher_exact_two_sided(3, 10, 8, 10) == pytest.approx(0.0699, abs=1e-3)


def test_fisher_symmetric():
    assert fisher_exact_two_sided(2, 9, 7, 9) == pytest.approx(
        fisher_exact_two_sided(7, 9, 2, 9), abs=1e-12)


def test_small_counts_fall_back_to_exact():
    cmp = two_proportion_test(1, 6, 5, 6)
    assert cmp.method == "fisher_exact"
    assert 0.0 <= cmp.p_value <= 1.0


def test_one_success_difference_in_tiny_samples_is_not_meaningful():
    cmp = two_proportion_test(3, 5, 4, 5)
    assert cmp.method == "fisher_exact"
    assert cmp.p_value == pytest.approx(1.0)
    assert not cmp.significant


def test_large_counts_use_z_test():
    cmp = two_proportion_test(70, 100, 50, 100)
    assert cmp.method == "z_pooled"
    assert cmp.p_value < 0.01
    assert cmp.delta == pytest.approx(0.20)


def test_newcombe_interval_contains_delta():
    for args in [(8, 10, 3, 10), (30, 50, 28, 50), (0, 20, 2, 20), (19, 20, 18, 20)]:
        low, high = newcombe_difference_interval(*args)
        delta = args[0] / args[1] - args[2] / args[3]
        assert low <= delta <= high
        assert -1.0 <= low <= high <= 1.0


def test_holm_known_values():
    adjusted, rejected = holm([0.01, 0.02, 0.03, 0.04])
    assert adjusted == pytest.approx([0.04, 0.06, 0.06, 0.06])
    # Threshold for rank 1 is 0.05/4 = 0.0125 -> first rejected, then stop.
    assert rejected == [True, False, False, False]


def test_holm_monotone_and_bounded():
    adjusted, _ = holm([0.001, 0.04, 0.04, 0.2, 0.9])
    assert all(a <= b for a, b in zip(adjusted, adjusted[1:]))
    assert all(0 <= a <= 1 for a in adjusted)


def test_bh_known_values():
    adjusted, rejected = benjamini_hochberg([0.01, 0.02, 0.03, 0.04])
    assert adjusted == pytest.approx([0.04, 0.04, 0.04, 0.04])
    assert rejected == [True, True, True, True]


def test_bh_less_conservative_than_holm():
    ps = [0.011, 0.02, 0.03, 0.04]
    holm_adj, _ = holm(ps)
    bh_adj, _ = benjamini_hochberg(ps)
    assert all(b <= h + 1e-12 for b, h in zip(bh_adj, holm_adj))


def test_corrections_preserve_order_and_handle_empty():
    assert holm([]) == ([], [])
    assert benjamini_hochberg([]) == ([], [])


def test_mde_matches_two_sample_formula_at_large_n():
    # For large n, MDE ~= (z_a + z_b) * sqrt(2 p (1-p) / n)
    n = 5000
    p = 0.5
    expected = (1.959964 + 0.841621) * math.sqrt(2 * p * (1 - p) / n)
    assert minimum_detectable_effect(p, n, n) == pytest.approx(expected, rel=0.02)


def test_mde_shrinks_with_more_runs():
    small = minimum_detectable_effect(0.7, 30, 30)
    large = minimum_detectable_effect(0.7, 300, 300)
    assert large < small
    assert 0.0 < large < 0.12


def test_sequential_stops_on_strong_effect():
    # 30/60 vs 50/60 is a huge difference; should stop for effect.
    result = sequential_decision(30, 60, 50, 60, min_runs=8, look=3)
    assert result.decision == "stop_effect"


def test_sequential_continues_when_ambiguous():
    result = sequential_decision(4, 10, 5, 10, min_runs=8, look=1)
    assert result.decision == "continue"


def test_sequential_respects_min_runs():
    result = sequential_decision(0, 2, 10, 10, min_runs=8)
    assert result.decision == "continue"
    assert "min runs" in result.reason


def test_sequential_negligible_when_ci_rules_out_meaningful_drop():
    # With large n and almost identical rates the CI is tight enough.
    raw = two_proportion_test(500, 1000, 502, 1000)
    result = sequential_decision(500, 1000, 502, 1000, min_runs=100,
                                 meaningful_drop=0.10, max_looks=8)
    assert result.decision == "stop_negligible"
    assert result.p_value == pytest.approx(min(1.0, raw.p_value * 8))


def test_sequential_negligible_uses_simultaneous_look_interval():
    # An ordinary 95% interval is just narrow enough to rule out a 10-point
    # drop here; the wider repeated-look interval correctly keeps sampling.
    ordinary = two_proportion_test(100, 200, 100, 200)
    result = sequential_decision(100, 200, 100, 200, min_runs=100,
                                 meaningful_drop=0.10, max_looks=8)
    assert ordinary.ci_low > -0.10
    assert result.ci_low < -0.10
    assert result.decision == "continue"


@pytest.mark.parametrize("operation", [
    lambda: wilson_interval(6, 5),
    lambda: two_proportion_test(-1, 5, 2, 5),
    lambda: two_proportion_test(1, 5, 2, 5, alpha=1.0),
    lambda: holm([0.01, float("nan")]),
    lambda: benjamini_hochberg([-0.1]),
])
def test_invalid_statistical_inputs_fail_loudly(operation):
    with pytest.raises(ValueError):
        operation()
