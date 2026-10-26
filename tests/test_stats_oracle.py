import numpy as np
import pytest
from scipy.optimize import brentq

scipy_stats = pytest.importorskip("scipy.stats")
pytest.importorskip("statsmodels")
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.power import NormalIndPower
from statsmodels.stats.proportion import (
    confint_proportions_2indep,
    proportion_confint,
    proportion_effectsize,
    proportions_ztest,
)

from shakedown.stats import (
    benjamini_hochberg,
    fisher_exact_two_sided,
    holm,
    minimum_detectable_effect,
    newcombe_difference_interval,
    two_proportion_test,
    wilson_interval,
)


def test_randomized_statistics_match_scipy_and_statsmodels_oracles():
    rng = np.random.default_rng(107)
    for _ in range(150):
        n = int(rng.integers(10, 400))
        k = int(rng.integers(0, n + 1))
        expected = proportion_confint(k, n, alpha=0.05, method="wilson")
        actual = wilson_interval(k, n)
        assert actual.low == pytest.approx(expected[0], abs=1e-12)
        assert actual.high == pytest.approx(expected[1], abs=1e-12)

        n1 = int(rng.integers(5, 80))
        n2 = int(rng.integers(5, 80))
        k1 = int(rng.integers(0, n1 + 1))
        k2 = int(rng.integers(0, n2 + 1))
        assert fisher_exact_two_sided(k1, n1, k2, n2) == pytest.approx(
            scipy_stats.fisher_exact([[k1, n1 - k1], [k2, n2 - k2]]).pvalue,
            abs=1e-12,
        )
        actual_ci = newcombe_difference_interval(k1, n1, k2, n2)
        expected_ci = confint_proportions_2indep(
            k1, n1, k2, n2, method="newcomb", alpha=0.05
        )
        assert actual_ci == pytest.approx(expected_ci, abs=1e-10)

        z_n1 = int(rng.integers(50, 500))
        z_n2 = int(rng.integers(50, 500))
        z_k1 = int(rng.integers(0, z_n1 + 1))
        z_k2 = int(rng.integers(0, z_n2 + 1))
        result = two_proportion_test(z_k1, z_n1, z_k2, z_n2)
        _, expected_z_p = proportions_ztest(
            [z_k1, z_k2], [z_n1, z_n2], alternative="two-sided"
        )
        if result.method == "z_pooled":
            assert result.p_value == pytest.approx(expected_z_p, abs=1e-9)


def test_randomized_multiple_testing_matches_statsmodels():
    rng = np.random.default_rng(2718)
    for count in range(1, 100):
        p_values = rng.uniform(0.0, 1.0, size=count).tolist()
        for method, implementation in (
            ("holm", holm),
            ("fdr_bh", benjamini_hochberg),
        ):
            actual, rejected = implementation(p_values, alpha=0.05)
            expected_rejected, expected, _, _ = multipletests(
                p_values, alpha=0.05, method=method
            )
            assert actual == pytest.approx(expected, abs=1e-12)
            assert rejected == np.atleast_1d(expected_rejected).tolist()


def test_mde_has_target_power_under_statsmodels_power_oracle():
    power_oracle = NormalIndPower()
    rng = np.random.default_rng(31415)
    for _ in range(100):
        baseline = float(rng.uniform(0.1, 0.9))
        n_treat = int(rng.integers(50, 1000))
        n_ctrl = int(rng.integers(50, 1000))
        alpha = float(rng.uniform(0.01, 0.1))
        target_power = float(rng.uniform(0.65, 0.9))
        effect = minimum_detectable_effect(
            baseline, n_treat, n_ctrl, alpha=alpha, power=target_power
        )
        directions = []
        z_alpha = scipy_stats.norm.ppf(1.0 - alpha / 2.0)
        for direction, available in ((-1.0, baseline), (1.0, 1.0 - baseline)):
            def power_at(
                amount,
                baseline=baseline,
                direction=direction,
                n_treat=n_treat,
                n_ctrl=n_ctrl,
                z_alpha=z_alpha,
            ):
                p_treat = baseline + direction * amount
                se = np.sqrt(
                    p_treat * (1.0 - p_treat) / n_treat
                    + baseline * (1.0 - baseline) / n_ctrl
                )
                noncentrality = amount / se
                return (
                    scipy_stats.norm.cdf(-z_alpha - noncentrality)
                    + scipy_stats.norm.sf(z_alpha - noncentrality)
                )

            if power_at(available) >= target_power:
                directions.append(
                    brentq(
                        lambda amount, target=target_power: power_at(amount) - target,
                        0.0,
                        available,
                    )
                )
        expected = min(directions, default=1.0)
        assert effect == pytest.approx(expected, abs=1e-8)

    baseline = 0.5
    for n in (80, 160, 320, 1000):
        effect = minimum_detectable_effect(baseline, n, n, alpha=0.05, power=0.8)
        p_treat = baseline - effect
        effect_size = proportion_effectsize(p_treat, baseline)
        oracle_power = power_oracle.power(
            effect_size=effect_size,
            nobs1=n,
            alpha=0.05,
            ratio=1.0,
            alternative="two-sided",
        )
        assert oracle_power == pytest.approx(0.8, abs=0.02)


def test_z_distribution_matches_scipy_reference():
    from shakedown.stats import norm_cdf

    for value in np.linspace(-8, 8, 101):
        assert norm_cdf(float(value)) == pytest.approx(
            scipy_stats.norm.cdf(value), abs=1e-14
        )
