# Shakedown Build Log

## Checkpoint 1 — statistical core

**Built:** Wilson intervals, two-proportion comparisons with Fisher exact fallback for small counts, Holm and Benjamini–Hochberg correction, minimum detectable effect, and sequential decisions. Sequential looks use a Bonferroni-adjusted p-value and a simultaneous Newcombe interval, so a repeated look cannot use an ordinary 95% interval to claim a perturbation is negligible. Invalid sample counts, probability values, and significance parameters now fail explicitly.

**Validation:** `python -m pytest tests/test_stats.py -q` — 26 passed. Includes the 3/5 versus 4/5 exact-test guard and a case where a 95% interval would stop early but the repeated-look interval correctly continues.

**Spend:** $0.00 model API spend; 0 model calls. No agent runs were made in this checkpoint. The configured hard cap remains `$15`; end-to-end cap enforcement has not yet been validated.

**Known limits:** The working tree already contained implementation files beyond this checkpoint, but they have not been audited or validated as a complete product. No external model endpoint was called. README, end-to-end demo validation, and measured model-run costs remain outstanding.
