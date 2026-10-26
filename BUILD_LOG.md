# Shakedown Build Log

## Checkpoint 1 — statistical core

**Built:** Wilson intervals, two-proportion comparisons with Fisher exact fallback for small counts, Holm and Benjamini–Hochberg correction, minimum detectable effect, and sequential decisions. Sequential looks use a Bonferroni-adjusted p-value and a simultaneous Newcombe interval, so a repeated look cannot use an ordinary 95% interval to claim a perturbation is negligible. Invalid sample counts, probability values, and significance parameters now fail explicitly.

**Validation:** `python -m pytest tests/test_stats.py -q` — 26 passed. Includes the 3/5 versus 4/5 exact-test guard and a case where a 95% interval would stop early but the repeated-look interval correctly continues.

**Spend:** $0.00 model API spend; 0 model calls. No agent runs were made in this checkpoint. The configured hard cap remains `$15`; end-to-end cap enforcement has not yet been validated.

**Known limits:** The working tree already contained implementation files beyond this checkpoint, but they have not been audited or validated as a complete product. No external model endpoint was called. README, end-to-end demo validation, and measured model-run costs remain outstanding.

## Checkpoint 2 — local-model smoke test

**Built and measured:** Ran bugfix_01 once through the local OpenAI-compatible Ollama endpoint. The agent fixed the task and the executable pytest check passed: 1/1 run, 5 model calls, 4 agent steps, 5,224 prompt tokens, 359 completion tokens, 15.0 seconds, and $0.00 at the configured local-model pricing.

**Full-grid estimate:** 4 tasks, 21 cells, 30 baseline runs, up to 30 runs per perturbation cell: 630 runs and 5,040 estimated model calls, approximately 42 minutes at 3 workers. Estimated spend is $0.00 against the $15 cap because this endpoint is local/free. No auto-shrink was required; the estimated call total is below the 6,000-call cap.

**Validation:** `python -m shakedown selftest` — all 4 tasks fail before applying their solution and pass afterward. The one-run model smoke test generated an HTML report; its 100% observed rate is not a meaningful reliability estimate (n=1).

**Known limits:** The cost estimate uses the configured local endpoint and pricing, not a paid hosted endpoint; no paid API was used. The smoke test did not validate perturbation behavior or statistical conclusions. README and complete end-to-end demo validation remain outstanding.

## Checkpoint 3 — runner, proxy, and perturbation smoke

**Built and fixed:** Hardened the proxy to forward the configured upstream credential, cap completion tokens, reserve each request's estimated worst-case cost before forwarding, and atomically enforce global call/spend limits across workers. Failed upstream calls conservatively consume their reservation. Unpriced remote endpoints now fail closed unless explicitly marked free; loopback model servers remain valid at zero price. A full one-run-per-cell matrix exercised all 20 ablation and chaos cells against bugfix_01.

**Statistical correction:** The smoke exposed that a one-run sample could be labeled `dead-weight` when its interval ruled out harm but still allowed a large benefit. Equivalence now requires the entire confidence interval to fit within the meaningful-effect bounds; sequential “negligible” stops use the same two-sided rule. MDE now works at 0% and 100% baseline rates instead of reporting a misleading zero.

**Validation:** 37 unit tests pass, covering statistics, conservative labels, atomic budget reservations, configured proxy credentials, token limits, and request rejection before an upstream call when the cost ceiling exceeds budget. All 4 benchmark self-tests pass. The perturbation smoke completed 21 runs and 111 model calls in 2.9 minutes; baseline was 0/1, so all 20 cells are correctly labeled inconclusive. The report rebuilt from recorded runs.

**Spend:** $0.00 across 22 local-model runs (116 calls total), using `qwen2.5-coder:7b` on local Ollama; no paid endpoint was used. At this checkpoint, the 30-run baseline and polished README/report were still outstanding.

## Checkpoint 4 — baseline reliability slice

**Measured:** `shakedown run --baseline -n 30` across all four tasks produced 14/30 passes (46.7%; Wilson 95% CI 30.2–63.9%), with a 31.4-point minimum detectable effect. The measured per-task rates were 62.5%, 37.5%, 57.1%, and 28.6%. These are the observed results; they do not match the aspirational demo story and must not be tuned or presented as though they do.

**Run cost:** 148 model calls and 176,236 tokens in 3.7 minutes; $0.00 at local Ollama pricing. No API spend.

**Limitations:** The baseline is intentionally noisy and the 31.4-point MDE is large. First-divergence analysis is recorded, but no perturbation effects are inferred from this baseline-only run. A release tag will identify this executable baseline slice.
