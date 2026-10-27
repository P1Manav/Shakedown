import math
import itertools
import pytest

from shakedown.stats import (
    exact_stratified_test,
    stratified_binary_comparison,
    stratified_minimum_detectable_effect,
    minimum_detectable_effect,
    Comparison,
)
from shakedown.analysis import _label


def test_exact_stratified_test_matches_regression_case():
    """Regression case: baseline per-task k/n = 2/3, 0/3, 2/3, 1/3 vs a cell of 0/3 each must give p = 0.040."""
    strata = [
        (0, 3, 2, 3),  # (k_t, n_t, k_c, n_c)
        (0, 3, 0, 3),
        (0, 3, 2, 3),
        (0, 3, 1, 3),
    ]
    p = exact_stratified_test(strata)
    assert p == pytest.approx(0.040, abs=1e-4)


def test_exact_stratified_test_brute_force_enumeration():
    """Verify convolution matches brute-force enumeration across all outcome combinations."""
    # 3 strata with distinct sizes and margins
    strata = [
        (1, 2, 2, 3),  # N=5, m=3, T_obs=1
        (2, 4, 1, 2),  # N=6, m=3, T_obs=2
        (0, 3, 1, 2),  # N=5, m=1, T_obs=0
    ]

    # Brute-force enumeration
    possible_tables = []
    for k_t, n_t, k_c, n_c in strata:
        m = k_t + k_c
        N = n_t + n_c
        outcomes = []
        for a in range(max(0, m - n_c), min(n_t, m) + 1):
            prob = (math.comb(n_t, a) * math.comb(n_c, m - a)) / math.comb(N, m)
            outcomes.append((a, prob))
        possible_tables.append(outcomes)

    total_dist = {}
    for combo in itertools.product(*possible_tables):
        total_t = sum(a for a, _ in combo)
        prob = 1.0
        for _, p in combo:
            prob *= p
        total_dist[total_t] = total_dist.get(total_t, 0.0) + prob

    t_obs = sum(s[0] for s in strata)
    obs_prob = total_dist[t_obs]
    expected_p = sum(p for t, p in total_dist.items() if p <= obs_prob + 1e-12)

    computed_p = exact_stratified_test(strata)
    assert computed_p == pytest.approx(expected_p, abs=1e-9)


def test_stratified_binary_comparison_uses_exact_on_small_strata():
    # Small strata (n=3 per arm)
    treatment = {"t1": (0, 3), "t2": (0, 3), "t3": (0, 3), "t4": (0, 3)}
    control = {"t1": (2, 3), "t2": (0, 3), "t3": (2, 3), "t4": (1, 3)}
    comp = stratified_binary_comparison(treatment, control)
    assert comp is not None
    assert comp.method == "exact_stratified"
    assert comp.p_value == pytest.approx(0.040, abs=1e-4)


def test_stratified_mde_shrinks_and_matches_balanced_pooled():
    # Balanced strata with identical baseline rate should match minimum_detectable_effect
    n_per_task = 25
    strata = [(0.5, n_per_task, n_per_task) for _ in range(4)]
    strat_mde = stratified_minimum_detectable_effect(strata, alpha=0.05, power=0.8)
    pooled_mde = minimum_detectable_effect(0.5, 100, 100, alpha=0.05, power=0.8)
    assert strat_mde == pytest.approx(pooled_mde, rel=1e-2)

    larger_strata = [(0.5, n_per_task * 4, n_per_task * 4) for _ in range(4)]
    larger_strat_mde = stratified_minimum_detectable_effect(larger_strata, alpha=0.05, power=0.8)
    assert larger_strat_mde < strat_mde


def test_label_insufficient_exposure_when_exposure_rate_under_80_percent():
    # Comparison whose CI rules out meaningful drop (e.g. [-0.05, 0.05], drop=0.10)
    comp = Comparison(
        k_treat=10, n_treat=20, k_ctrl=10, n_ctrl=20,
        p_treat=0.5, p_ctrl=0.5, delta=0.0,
        ci_low=-0.05, ci_high=0.05, p_value=1.0, method="test",
    )

    # 1. Exposure rate 50% (< 80%) -> "insufficient exposure"
    verdict, reason = _label(comp, adjusted_p=1.0, alpha=0.05, meaningful_drop=0.10, n=20, mde=0.08, exposure_rate=0.5)
    assert verdict == "insufficient exposure"
    assert "exposure rate is 50.0%" in reason or "50%" in reason

    # 2. Exposure rate None (unknown) -> "insufficient exposure"
    verdict, reason = _label(comp, adjusted_p=1.0, alpha=0.05, meaningful_drop=0.10, n=20, mde=0.08, exposure_rate=None)
    assert verdict == "insufficient exposure"

    # 3. Exposure rate 90% (>= 80%) -> "dead-weight"
    verdict, reason = _label(comp, adjusted_p=1.0, alpha=0.05, meaningful_drop=0.10, n=20, mde=0.08, exposure_rate=0.9)
    assert verdict == "dead-weight"
