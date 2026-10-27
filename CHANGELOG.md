# Changelog

## 0.2.0

- Move the Python distribution to a `src/` layout and bundle benchmark and prompt data inside `shakedown`.
- Keep the `shakedown` CLI while using the available `shakedown-agent-profiler` distribution name.
- Add paired, fresh-control fix experiments with trace-derived guidance and uncertainty reporting.
- Add cross-platform CI, clean-wheel installation smoke coverage, and automated regression testing.
- Add `shakedown check` CI gate with regression detection (exit code 2) and badge output.
- Re-architect reverse proxy with streaming SSE passthrough, deterministic cache controls, and strict model/temperature enforcement.
- Provide self-contained interactive HTML report with light/dark modes, sensitivity heatmap, first-divergence analysis, and failure taxonomy.
- Commit 252-run open-source model benchmark on `qwen2.5-coder:7b` with zero external spend.
- Provide Bring-Your-Own-Agent (BYOA) integration for any OpenAI-compatible client.

