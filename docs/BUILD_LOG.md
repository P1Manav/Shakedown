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

**Limitations:** The baseline is intentionally noisy and the 31.4-point MDE is large. First-divergence analysis is recorded, but no perturbation effects are inferred from this baseline-only run. Tag `v0.1-baseline` identifies this executable baseline slice.

## Checkpoint 5 — budget-aware grid sizing

**Built:** Grid estimates now include the global model-call cap, and `shakedown run` applies the reduced per-cell maximum before execution. The default four-task grid estimated 6,880 calls against the 6,000 cap, so it now shrinks from 40 to 30 runs per perturbation cell (5,280 estimated calls) while keeping all cells and the 60-run baseline.

**Validation:** 38 unit tests pass. `python -m shakedown estimate` reports the original upper bound and the actual planned maximum; the shrink-plan test covers this 6,000-call boundary.

**Spend to date:** $0.00 across 52 local-model runs and 264 calls; no paid endpoint has been used.

## Checkpoint 6 — familywise inference and release polish

**Built:** Perturbation difference intervals now use Bonferroni familywise coverage across all planned cells, and cell MDEs use the same adjusted alpha. The report states the familywise confidence level; a 20-cell test confirms that an ordinary 95% interval cannot incorrectly mark a low-powered cell as dead weight. Fix reruns now carry forward the originating experiment's call and spend usage, including prior fix attempts. Cost-risk candidates require at least eight observations so a one-run smoke cannot create a fix-first finding. Added the responsive self-contained HTML report, package metadata/task data, README, and `make demo` entry point.

**Validation:** The full unit suite passes, and all four benchmark self-tests pass both from the checkout and an installed wheel. The one-task end-to-end smoke completed 21 runs (baseline plus 20 perturbations) and 113 model calls in 3.3 minutes at $0.00. Baseline was 1/1; every perturbation remains inconclusive at this sample size (MDE 94 points), so no effects are claimed. The first invocation exposed a Windows CP1252 output error for the arrow in chaos labels; console output now escapes unsupported characters, and the rerun completed and wrote its HTML report.

**Spend to date:** $0.00 across 94 local-model runs and 489 calls, using local Ollama only. No paid model endpoint was used; the configured hard cap remains $15.

**Known limits:** The full default grid and fix remeasurement were not run because the tiny smoke was for pipeline verification, not statistical inference. Results from the one-run smoke are not suitable for conclusions. Temporary workspaces are not OS-level security sandboxes.

## Checkpoint 7 — release validation and full-grid smoke

**Validation:** The full suite passes (44 tests), all four benchmark self-tests pass, and a built wheel installed into a clean virtual environment and passed the same four self-tests. `git diff --check` is clean. A complete 12-run-per-cell demo exercised all 20 perturbation cells across four tasks and generated a self-contained 102 KB HTML report.

**Observed smoke results:** The 12-run baseline was 5/12 (41.7%; Wilson 95% CI 19.3–68.0%). All 20 perturbation cells were correctly labeled inconclusive after Holm correction; the cell MDE was 56.7 points. The empty-result fault had 0/12 passes and 5/12 false-success claims, while truncation had the same 5/12 pass rate as baseline and 1/12 false-success claims. These are exploratory observations, not significant effects; no fix rerun was warranted. The demo did not reproduce the aspirational truncated-output story.

**Next:** Run the adaptive grid at up to 30 runs per cell to improve power before treating any perturbation as a finding. Optional cross-agent proof and model-size comparison remain out of scope for release validation.

## Checkpoint 8 — Audit and v0.3.0 Release Hardening

**CI and Portability:**
- Fixed all cross-platform failures across the 6-matrix GitHub Actions CI pipeline (Ubuntu, macOS, Windows on Python 3.11 and 3.12).
- Placed lint step first for fast failure, added pip wheel caching, concurrency cancel-in-progress, and dynamic ephemeral port binding in wheel smoke testing.
- Added CI status badge to README.

**Fault Semantics & Scheduler:**
- Hardened `error_once` to fire once per run per tool name while maintaining replay stability across identical tool call IDs.
- Converted `empty_result` and `renamed_path` to transient by default with separate explicit persistent variants (`persist: true`).
- Included `added_latency` and Tier 3 input noise cells (`noise:reorder_tools`, `noise:typos`, `noise:paraphrase`) in the full grid.

**Statistical Rigor & Exposure Tracking:**
- Implemented an exact conditional stratified test (convolution of per-stratum hypergeometric distributions ordered by probability) for small strata, verified against numerical regression baselines.
- Exposed per-cell exposure tracking (exposed runs, exposure rate, and effect on exposed runs alongside intent-to-treat). Cells with <80% exposure are conservatively labeled "insufficient exposure".

**Verification Hardening:**
- Verification workspace strips all `__pycache__` and `*.pyc` files, executes under `PYTHONSAFEPATH=1` and `PYTHONDONTWRITEBYTECODE=1`, performs case-insensitive casefold path comparisons, and detects any newly added top-level files that shadow installed modules.
- Added isolated `hidden_tests/` across all bundled tasks, verified against agents hardcoding visible assertions.
- Replaced secret name blocklists with an explicit environment variable allowlist.

**Decoupled Architecture & BYOA:**
- Decoupled `tool_parser.py` from demo agent; runner does not depend on the demo agent.
- Created `examples/bring_your_own_agent/agent.py` (~45 lines) and validated it with CI test `tests/test_byoa_loop.py` against a mock upstream server.
- Removed obfuscation in proxy authentication, used SPDX `license = "MIT"` with `license-files = ["LICENSE"]`.

**Real Regenerated Benchmark Data:**
- Selected `llama3.1:8b-instruct-q4_K_M` via local Ollama ($0.00 spend) after comparative model piloting.
- Executed the full adaptive grid across 4 tasks and 27 cells: 216 runs, 2,510 model calls, 4.05M tokens, $0.00 spend, 0 invalid runs (0.0%). Baseline reliability: 4/8 = 50.0% (Wilson 95% CI [21.5, 78.5]).
- Evaluated the automated fix loop on top fragility `remove identity` with an equal-N interleaved no-fix control (8 control vs 8 fix runs): observed lift +50.0 pts (25% to 75%, p=0.1319, MDE=57.2 pts), reported honestly as "no statistically significant lift".
- Updated `examples/sample_analysis.json`, `examples/sample_report.html`, `docs/index.html`, and `docs/assets/report.png`. Built `scripts/render_results.py` and regression test `tests/test_readme_results.py`.

