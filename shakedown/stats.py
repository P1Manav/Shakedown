"""Hand-rolled statistics for sensitivity experiments.

Everything here is deliberately dependency-free and conservative:

* Wilson score intervals for a single proportion.
* Two-proportion z-test, with Fisher's exact test when counts are small.
* Newcombe hybrid-score interval for a difference of proportions.
* Minimum detectable effect for a two-arm comparison at fixed n.
* Holm and Benjamini-Hochberg multiple-comparison corrections.
* A CI-based sequential stopping rule with a Bonferroni spending bound.

No p-value here is used to *rank* a cell without correction.  The report
shows raw and adjusted p-values side by side and labels are computed from the
adjusted values.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterable, Sequence

Z_95 = 1.959963984540054

__all__ = [
    "Z_95",
    "WilsonInterval",
    "Comparison",
    "SequentialResult",
    "norm_cdf",
    "norm_ppf",
    "wilson_interval",
    "two_proportion_test",
    "fisher_exact_two_sided",
    "newcombe_difference_interval",
    "minimum_detectable_effect",
    "holm",
    "benjamini_hochberg",
    "sequential_decision",
    "rate",
]


# ---------------------------------------------------------------------------
# Normal distribution helpers
# ---------------------------------------------------------------------------

def norm_cdf(x: float) -> float:
    """Standard normal CDF using the error function."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def norm_ppf(p: float) -> float:
    """Inverse standard normal CDF (Acklam's algorithm, |error| < 1.15e-9)."""
    if not 0.0 < p < 1.0:
        raise ValueError("p must be in (0, 1)")
    a = (-3.969683028665376e01, 2.209460984245205e02, -2.759285104469687e02,
         1.383577518672690e02, -3.066479806614716e01, 2.506628277459239e00)
    b = (-5.447609879822406e01, 1.615858368580409e02, -1.556989798598866e02,
         6.680131188771972e01, -1.328068155288572e01)
    c = (-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e00,
         -2.549732539343734e00, 4.374664141464968e00, 2.938163982698783e00)
    d = (7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e00,
         3.754408661907416e00)
    plow, phigh = 0.02425, 1.0 - 0.02425
    if p < plow:
        q = math.sqrt(-2.0 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
               ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
    if p > phigh:
        q = math.sqrt(-2.0 * math.log(1.0 - p))
        return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
               ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
    q = p - 0.5
    r = q * q
    return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / \
           (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0)


def rate(k: int, n: int) -> float:
    if n < 0 or k < 0 or k > n:
        raise ValueError("successes must be between 0 and the non-negative trial count")
    if n == 0:
        return 0.0
    return k / n


def _validate_binomial_count(k: int, n: int) -> None:
    if n < 0 or k < 0 or k > n:
        raise ValueError("successes must be between 0 and the non-negative trial count")


def _validate_alpha(alpha: float) -> None:
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1)")


# ---------------------------------------------------------------------------
# Single proportion
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class WilsonInterval:
    low: float
    high: float
    center: float
    n: int

    def __str__(self) -> str:
        return f"{self.center * 100:.1f}% [{self.low * 100:.1f}, {self.high * 100:.1f}]"


def wilson_interval(k: int, n: int, z: float = Z_95) -> WilsonInterval:
    """Wilson score interval for k successes out of n trials."""
    _validate_binomial_count(k, n)
    if not math.isfinite(z) or z <= 0:
        raise ValueError("z must be a finite positive value")
    if n == 0:
        return WilsonInterval(0.0, 1.0, 0.0, 0)
    p = k / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2.0 * n)) / denom
    half = (z / denom) * math.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n))
    return WilsonInterval(max(0.0, center - half), min(1.0, center + half), p, n)


# ---------------------------------------------------------------------------
# Two proportions
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Comparison:
    """Treatment (perturbed cell) versus control (baseline)."""

    k_treat: int
    n_treat: int
    k_ctrl: int
    n_ctrl: int
    p_treat: float
    p_ctrl: float
    delta: float                 # p_treat - p_ctrl, in probability units
    ci_low: float
    ci_high: float
    p_value: float
    method: str
    z: float | None = None

    @property
    def significant(self) -> bool:
        return self.p_value < 0.05

    def as_dict(self) -> dict:
        return {
            "k_treat": self.k_treat, "n_treat": self.n_treat,
            "k_ctrl": self.k_ctrl, "n_ctrl": self.n_ctrl,
            "p_treat": self.p_treat, "p_ctrl": self.p_ctrl,
            "delta": self.delta,
            "ci_low": self.ci_low, "ci_high": self.ci_high,
            "p_value": self.p_value, "method": self.method, "z": self.z,
        }


def _hypergeom_prob(x: int, row1: int, col1: int, n: int) -> float:
    return (math.comb(col1, x) * math.comb(n - col1, row1 - x)) / math.comb(n, row1)


def fisher_exact_two_sided(k1: int, n1: int, k2: int, n2: int) -> float:
    """Two-sided Fisher exact p-value for [[k1, n1-k1], [k2, n2-k2]]."""
    _validate_binomial_count(k1, n1)
    _validate_binomial_count(k2, n2)
    if n1 == 0 or n2 == 0:
        return 1.0
    total = n1 + n2
    row1 = k1 + k2          # total successes
    col1 = n1               # group 1 size
    lo = max(0, row1 + col1 - total)
    hi = min(row1, col1)
    observed = _hypergeom_prob(k1, row1, col1, total)
    p = 0.0
    for x in range(lo, hi + 1):
        prob = _hypergeom_prob(x, row1, col1, total)
        if prob <= observed + 1e-12:
            p += prob
    return min(1.0, p)


def newcombe_difference_interval(k1: int, n1: int, k2: int, n2: int,
                                 z: float = Z_95) -> tuple[float, float]:
    """Newcombe hybrid score interval for p1 - p2 (recommended in practice)."""
    _validate_binomial_count(k1, n1)
    _validate_binomial_count(k2, n2)
    if not math.isfinite(z) or z <= 0:
        raise ValueError("z must be a finite positive value")
    if n1 == 0 or n2 == 0:
        return (-1.0, 1.0)
    p1, p2 = k1 / n1, k2 / n2
    w1 = wilson_interval(k1, n1, z)
    w2 = wilson_interval(k2, n2, z)
    delta = p1 - p2
    low = delta - math.sqrt((p1 - w1.low) ** 2 + (w2.high - p2) ** 2)
    high = delta + math.sqrt((w1.high - p1) ** 2 + (p2 - w2.low) ** 2)
    return max(-1.0, low), min(1.0, high)


def two_proportion_test(k_treat: int, n_treat: int, k_ctrl: int, n_ctrl: int,
                        alpha: float = 0.05, force_exact: bool = False) -> Comparison:
    """Treatment vs control with an automatic exact-test fallback.

    The pooled z-test is used unless counts are small (which is the common case
    in early sequential looks), where Fisher's exact test is used instead.
    """
    _validate_binomial_count(k_treat, n_treat)
    _validate_binomial_count(k_ctrl, n_ctrl)
    _validate_alpha(alpha)
    if n_treat == 0 or n_ctrl == 0:
        return Comparison(k_treat, n_treat, k_ctrl, n_ctrl, rate(k_treat, n_treat),
                          rate(k_ctrl, n_ctrl), 0.0, -1.0, 1.0, 1.0, "empty")

    p1, p2 = k_treat / n_treat, k_ctrl / n_ctrl
    delta = p1 - p2
    ci_low, ci_high = newcombe_difference_interval(k_treat, n_treat, k_ctrl, n_ctrl)

    expected = [
        n_treat * (k_treat + k_ctrl) / (n_treat + n_ctrl),
        n_treat * (n_treat - k_treat + n_ctrl - k_ctrl) / (n_treat + n_ctrl),
        n_ctrl * (k_treat + k_ctrl) / (n_treat + n_ctrl),
        n_ctrl * (n_treat - k_treat + n_ctrl - k_ctrl) / (n_treat + n_ctrl),
    ]
    small = n_treat < 20 or n_ctrl < 20 or min(expected) < 5
    if force_exact or small:
        p_value = fisher_exact_two_sided(k_treat, n_treat, k_ctrl, n_ctrl)
        return Comparison(k_treat, n_treat, k_ctrl, n_ctrl, p1, p2, delta,
                          ci_low, ci_high, p_value, "fisher_exact")

    pooled = (k_treat + k_ctrl) / (n_treat + n_ctrl)
    se = math.sqrt(pooled * (1.0 - pooled) * (1.0 / n_treat + 1.0 / n_ctrl))
    if se == 0.0:
        return Comparison(k_treat, n_treat, k_ctrl, n_ctrl, p1, p2, delta,
                          ci_low, ci_high, 1.0, "z_pooled", 0.0)
    z = delta / se
    p_value = 2.0 * (1.0 - norm_cdf(abs(z)))
    return Comparison(k_treat, n_treat, k_ctrl, n_ctrl, p1, p2, delta,
                      ci_low, ci_high, min(1.0, p_value), "z_pooled", z)


# ---------------------------------------------------------------------------
# Minimum detectable effect
# ---------------------------------------------------------------------------

def _power_two_prop(p1: float, p2: float, n1: int, n2: int, alpha: float) -> float:
    se = math.sqrt(max(1e-12, p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2))
    z_alpha = norm_ppf(1.0 - alpha / 2.0)
    noncentrality = abs(p1 - p2) / se
    return norm_cdf(-z_alpha - noncentrality) + 1.0 - norm_cdf(z_alpha - noncentrality)


def minimum_detectable_effect(baseline_p: float, n_treat: int, n_ctrl: int,
                              alpha: float = 0.05, power: float = 0.8) -> float:
    """Smallest absolute drop detectable with ``power`` at two-sided ``alpha``.

    Solved numerically because the standard error depends on the effect itself.
    Returns a probability (0..1); the report multiplies by 100 for points.
    """
    _validate_alpha(alpha)
    if not 0.0 < power < 1.0:
        raise ValueError("power must be in (0, 1)")
    if not 0.0 <= baseline_p <= 1.0:
        raise ValueError("baseline_p must be in [0, 1]")
    if n_treat < 0 or n_ctrl < 0:
        raise ValueError("sample sizes must be non-negative")
    if n_treat == 0 or n_ctrl == 0:
        return 1.0
    effects = []
    for direction, available in ((-1.0, baseline_p), (1.0, 1.0 - baseline_p)):
        if available <= 0.0:
            continue
        if _power_two_prop(baseline_p + direction * available, baseline_p,
                           n_treat, n_ctrl, alpha) < power:
            continue
        lo, hi = 0.0, available
        for _ in range(80):
            mid = (lo + hi) / 2.0
            achieved = _power_two_prop(
                baseline_p + direction * mid, baseline_p, n_treat, n_ctrl, alpha
            )
            if achieved >= power:
                hi = mid
            else:
                lo = mid
        effects.append(hi)
    return min(effects, default=1.0)


def is_effect_ruled_out(ci_low: float, meaningful_drop: float) -> bool:
    """True when the CI is tight enough to exclude a meaningful drop."""
    return ci_low > -meaningful_drop


def is_meaningful_effect_ruled_out(ci_low: float, ci_high: float,
                                   meaningful_effect: float) -> bool:
    """True when the full interval excludes meaningful changes in either direction."""
    if not -1.0 <= ci_low <= ci_high <= 1.0:
        raise ValueError("difference interval must satisfy -1 <= low <= high <= 1")
    if not 0.0 <= meaningful_effect <= 1.0:
        raise ValueError("meaningful_effect must be in [0, 1]")
    return ci_low > -meaningful_effect and ci_high < meaningful_effect


# ---------------------------------------------------------------------------
# Multiple comparisons
# ---------------------------------------------------------------------------

def holm(p_values: Sequence[float], alpha: float = 0.05) -> tuple[list[float], list[bool]]:
    """Holm-Bonferroni step-down. Returns (adjusted p, rejected at alpha)."""
    _validate_alpha(alpha)
    if any(not math.isfinite(p) or not 0.0 <= p <= 1.0 for p in p_values):
        raise ValueError("p-values must be finite and in [0, 1]")
    m = len(p_values)
    if m == 0:
        return [], []
    order = sorted(range(m), key=lambda i: p_values[i])
    adjusted = [1.0] * m
    rejected = [False] * m
    running = 0.0
    still_rejecting = True
    for rank, idx in enumerate(order):
        adj = min(1.0, p_values[idx] * (m - rank))
        running = max(running, adj)
        adjusted[idx] = running
        if still_rejecting and p_values[idx] <= alpha / (m - rank):
            rejected[idx] = True
        else:
            still_rejecting = False
    return adjusted, rejected


def benjamini_hochberg(p_values: Sequence[float], alpha: float = 0.05) -> tuple[list[float], list[bool]]:
    """Benjamini-Hochberg FDR control. Returns (adjusted p, rejected at alpha)."""
    _validate_alpha(alpha)
    if any(not math.isfinite(p) or not 0.0 <= p <= 1.0 for p in p_values):
        raise ValueError("p-values must be finite and in [0, 1]")
    m = len(p_values)
    if m == 0:
        return [], []
    order = sorted(range(m), key=lambda i: p_values[i], reverse=True)
    adjusted = [1.0] * m
    running = 1.0
    for rank_from_top, idx in enumerate(order):
        rank = m - rank_from_top  # 1-based rank ascending
        adj = min(1.0, p_values[idx] * m / rank)
        running = min(running, adj)
        adjusted[idx] = running
    # Rejection set is every hypothesis with adjusted p <= alpha.
    rejected = [adjusted[i] <= alpha for i in range(m)]
    return adjusted, rejected


# ---------------------------------------------------------------------------
# Sequential stopping
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SequentialResult:
    decision: str              # "continue" | "stop_effect" | "stop_negligible"
    reason: str
    p_value: float             # Bonferroni-adjusted across the planned looks
    ci_low: float
    ci_high: float
    alpha_look: float
    looks: int
    extra: dict = field(default_factory=dict)


def sequential_decision(k_treat: int, n_treat: int, k_ctrl: int, n_ctrl: int,
                        alpha: float = 0.05, max_looks: int = 8,
                        meaningful_drop: float = 0.10,
                        min_runs: int = 8, look: int = 1) -> SequentialResult:
    """CI-based sequential rule with a Bonferroni spending bound.

    Stops early when either:
      * the raw two-sided p-value crosses a per-look Bonferroni threshold
        (an effect is present), or
      * the difference CI rules out a drop of ``meaningful_drop`` or more
        (the perturbation clearly does not matter).

    The per-look alpha is alpha / max_looks, which keeps the family-wise error
    rate bounded under optional stopping.  Final reported p-values are still
    Holm/BH-corrected across all cells, so early stopping is belt-and-braces.
    """
    _validate_binomial_count(k_treat, n_treat)
    _validate_binomial_count(k_ctrl, n_ctrl)
    _validate_alpha(alpha)
    if max_looks < 1:
        raise ValueError("max_looks must be positive")
    if min_runs < 0:
        raise ValueError("min_runs must be non-negative")
    if not 0.0 <= meaningful_drop <= 1.0:
        raise ValueError("meaningful_drop must be in [0, 1]")
    if n_treat < min_runs:
        return SequentialResult("continue", f"below min runs ({n_treat}/{min_runs})",
                                1.0, -1.0, 1.0, alpha / max_looks, look)
    cmp = two_proportion_test(k_treat, n_treat, k_ctrl, n_ctrl)
    alpha_look = alpha / max_looks
    look_z = norm_ppf(1.0 - alpha_look / 2.0)
    ci_low, ci_high = newcombe_difference_interval(
        k_treat, n_treat, k_ctrl, n_ctrl, z=look_z
    )
    adjusted_p = min(1.0, cmp.p_value * max_looks)
    if cmp.p_value < alpha_look:
        return SequentialResult("stop_effect",
                                f"p={cmp.p_value:.4f} < {alpha_look:.4f} at look {look}",
                                adjusted_p, ci_low, ci_high, alpha_look, look)
    if is_meaningful_effect_ruled_out(ci_low, ci_high, meaningful_drop):
        return SequentialResult("stop_negligible",
                                f"simultaneous CI lower {ci_low * 100:.1f} pts rules out a {meaningful_drop * 100:.0f} pt drop",
                                adjusted_p, ci_low, ci_high, alpha_look, look)
    return SequentialResult("continue", "ambiguous; keep sampling",
                            adjusted_p, ci_low, ci_high, alpha_look, look)