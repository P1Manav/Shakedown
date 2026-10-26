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
