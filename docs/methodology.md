# Statistical methodology

Shakedown measures outcomes using the executable task check. All runtime inference is implemented in `shakedown/stats.py` with Python's standard library; SciPy and statsmodels are optional development-only reference implementations and are not runtime dependencies.

## Outcome populations

Each valid run has a binary outcome \(Y\): 1 when the task's success command exits with status 0, otherwise 0. Infrastructure-invalid runs are reported and excluded from pass-rate estimates, tests, confidence intervals, sequential decisions, and task rates. Test-tampered runs are failures, not invalid observations. The intent-to-treat (ITT) cell contains every valid run assigned to that perturbation, whether or not the perturbation changed the request.

Exposure is measured from the actual outgoing request or returned tool result. For a cell with \(N\) valid, exposure-tracked runs and \(E\) runs where a perturbation was applied, its exposure rate is \(E/N\). The exposed-run pass rate is reported separately and compared with the valid baseline; it is descriptive and does not replace the ITT comparison. A SHA-256 digest records the before/after changed payload pair without duplicating the payload in the exposure event.

## Single-proportion interval

For \(k\) passes in \(n>0\) valid runs, \(\hat p=k/n\). The two-sided Wilson score interval at normal quantile \(z\) is

\[
\frac{\hat p + z^2/(2n) \ \pm\ z\sqrt{\hat p(1-\hat p)/n+z^2/(4n^2)}}{1+z^2/n}.
\]

The default is \(z=1.9599639845\), corresponding to a 95% interval. Bounds are clipped to \([0,1]\). With no observations, the interval is \([0,1]\), not an estimate of a zero pass rate.

## Pooled comparison

For treatment \((k_T,n_T)\) and baseline \((k_C,n_C)\), the observed risk difference is

\[
\hat\Delta = k_T/n_T - k_C/n_C.
\]

Shakedown uses the two-sided Fisher exact test when either arm has fewer than 20 observations or any expected 2-by-2 cell count is below 5. Otherwise it uses the pooled two-proportion z test. Under the null, \(\hat p=(k_T+k_C)/(n_T+n_C)\) and

\[
z = \frac{\hat\Delta}{\sqrt{\hat p(1-\hat p)(1/n_T+1/n_C)}},\qquad
p=2\{1-\Phi(|z|)\}.
\]

Fisher's exact test conditions on the observed margins. Its two-sided p-value is the sum of hypergeometric probabilities no larger than the probability of the observed table.

The Newcombe hybrid-score interval for the difference combines the two arms' Wilson intervals. If their Wilson bounds are \(L_T,U_T,L_C,U_C\), then

\[
L_\Delta=\hat\Delta-\sqrt{(\hat p_T-L_T)^2+(U_C-\hat p_C)^2},\qquad
U_\Delta=\hat\Delta+\sqrt{(U_T-\hat p_T)^2+(\hat p_C-L_C)^2}.
\]

## Task-stratified comparison

For each task stratum \(i\) with treatment size \(n_{Ti}\), baseline size \(n_{Ci}\), treatment successes \(a_i\), and baseline successes \(c_i\), the Cochran–Mantel–Haenszel test sums the conditional departures from the null:

\[
O-E=\sum_i\left(a_i-\frac{n_{Ti}(a_i+c_i)}{n_{Ti}+n_{Ci}}\right),
\]

\[
V=\sum_i\frac{n_{Ti}n_{Ci}(a_i+c_i)(b_i+d_i)}
{(n_{Ti}+n_{Ci})^2(n_{Ti}+n_{Ci}-1)},
\]

where \(b_i=n_{Ti}-a_i\) and \(d_i=n_{Ci}-c_i\). Shakedown uses the continuity-corrected statistic \(\max(0,|O-E|-0.5)^2/V\), with a one-degree-of-freedom chi-square survival probability evaluated as \(\operatorname{erfc}(\sqrt{\chi^2/2})\).

The reported stratified effect is a precision-weighted risk difference. Each task gets weight \(w_i=n_{Ti}n_{Ci}/(n_{Ti}+n_{Ci})\); \(\hat\Delta_{strat}=\sum_i w_i(\hat p_{Ti}-\hat p_{Ci})/\sum_iw_i\). Its interval is the same weighted average of task-specific Newcombe lower and upper bounds. The report shows both pooled and stratified comparisons. Holm/BH adjustment and the verdict label use the CMH p-value and stratified effect when at least one matched task stratum exists; otherwise the report explicitly identifies the pooled comparison as the fallback. Scheduler replicate numbers map to the same task across baseline and perturbation cells.

Per-task rates and Wilson intervals are also reported, using valid observations only. Assigned-run counts include invalid attempts so allocation gaps are visible.

## Multiple comparisons

Shakedown forms separate ablation, chaos, and noise families. Corrections are applied within each family, not across the combined grid. For a family with \(m\) raw p-values sorted as \(p_{(1)}\le\cdots\le p_{(m)}\):

* **Holm step-down:** adjusted values are the cumulative maxima of \(\min(1,(m-i+1)p_{(i)})\); reject sequentially while \(p_{(i)}\le\alpha/(m-i+1)\).
* **Benjamini–Hochberg:** adjusted values are the reverse cumulative minima of \(\min(1,mp_{(i)}/i)\); reject when adjusted \(p\le\alpha\).

Raw pooled and stratified p-values remain separately available in `analysis.json`; each cell shows its raw p-value and its within-family adjusted p-value. A raw p-value is never described as an adjusted-significant result.

## Familywise confidence intervals and detectable effects

For a family of \(m\) tested cells and family alpha \(\alpha\), fixed-N simultaneous Newcombe intervals and the reported cell MDE use per-cell alpha \(\alpha/m\). Under sequential sampling with \(L\) planned looks, each look instead uses \(\alpha/(mL)\). The baseline-only MDE is unadjusted because it is a single-arm design summary. The cell MDE is the smallest absolute difference achieving the target power for the two independent-binomial normal approximation. For candidate treatment rate \(p_T\), baseline rate \(p_C\), and arm sizes \(n_T,n_C\), the standard error is

\[
SE=\sqrt{p_T(1-p_T)/n_T+p_C(1-p_C)/n_C}.
\]

Two-sided power is \(\Phi(-z_{\alpha/2}-|\Delta|/SE)+1-\Phi(z_{\alpha/2}-|\Delta|/SE)\). Shakedown solves this equation by bisection in each feasible direction and reports the smaller detectable absolute effect. The displayed MDE is a design sensitivity, not an observed effect.

## Sequential sampling

At each planned look, Shakedown compares the cell with the baseline and spends at most \(\alpha/(mL)\), where \(m\) is the number of cells in that perturbation family and \(L\) is the planned number of batch looks. It evaluates only at batch boundaries. It stops for an effect when the raw p-value is below this per-look threshold, or for negligible effect when the simultaneous Newcombe interval rules out the configured meaningful drop. Clear cells stop early so remaining runs continue to accumulate on ambiguous cells. Final results still receive the configured Holm or BH correction within their family. The invalid-run threshold is operational rather than statistical: grid execution stops when invalid runs exceed 20% of all attempted runs.

## Oracle validation

`tests/test_stats_oracle.py` runs deterministic randomized comparisons against SciPy and statsmodels for Wilson intervals, Fisher exact p-values, Newcombe intervals, two-proportion z tests, multiple-testing adjustments, normal CDF values, and MDE power. Those libraries are only in the `dev` extra and are not imported by runtime modules.
