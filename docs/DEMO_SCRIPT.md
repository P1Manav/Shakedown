# Shakedown Demo Script (90 Seconds)

A structured walkthrough for a high-signal 90-second video demo on X / Twitter, backed entirely by real, unmanipulated numbers from the committed 252-run benchmark (`examples/sample_report.html`).

---

## Video Walkthrough & Voiceover

### 0:00 – 0:15 | The Core Problem & Philosophy
- **Screen:** Show terminal running `shakedown demo` and inspecting the task workspace.
- **Voiceover:**
  > "Most agent benchmarks test an agent once and declare victory if it passes. But an agent that passes today can break tomorrow from a minor prompt tweak or an unexpected tool response. Shakedown is a sensitivity profiler for AI agents. We run controlled prompt ablations and tool-result chaos faults against real tasks. Crucially: success is never determined by an LLM judge. It is an executable pytest exit code—pass or fail."

### 0:15 – 0:35 | Baseline Reliability Card
- **Screen:** Switch to browser showing the top of the HTML report (`examples/sample_report.html`) in dark mode. Hover over the hero card.
- **Voiceover:**
  > "Here is our baseline on Qwen 2.5 Coder 7B running locally on Ollama at zero API cost. Across 4 benchmark tasks, baseline pass rate was 42%—5 out of 12 runs passed. Notice we don't just give a point estimate: Shakedown computes the Wilson 95% confidence interval—19% to 68%—and reports the minimum detectable effect: 41 points. With 12 runs, noise is real, and Shakedown makes uncertainty visible."

### 0:35 – 0:60 | The Sensitivity Heatmap & Statistical Rigor
- **Screen:** Scroll down to the Sensitivity Heatmap. Zoom in on `empty_result → read_file` and `truncate_output → run_tests`.
- **Voiceover:**
  > "Next, the perturbation heatmap. Each card displays raw delta, raw p-value, family-adjusted Holm p-value, and family MDE. Look at `empty_result` on `read_file`: pass rate plunged to 0%, a 42-point drop with raw p = 0.051. But after Holm correction across the chaos family, adjusted p is 0.306, so Shakedown rigorously marks it inconclusive rather than hallucinating a false discovery. Meanwhile, `truncate_output` on `run_tests` had a 0.0 delta—the agent was completely unfazed by truncation."

### 0:60 – 0:75 | Action Divergence & Failure Taxonomy
- **Screen:** Scroll to the "First Divergence" timeline and "Failure Taxonomy" stacked bars.
- **Voiceover:**
  > "When runs fail, where do they branch off? Shakedown analyzes normalized tool call trajectories. At Step 1, total variation distance spikes to 0.51: passing agents inspect target source and tests, while failing agents wander into unrelated files. In the failure taxonomy, 5 out of 12 failures under empty tool output falsely claimed success to the user—caught completely without any LLM judge."

### 0:75 – 0:90 | CI Gate & Takeaway
- **Screen:** Switch to terminal, run `shakedown check --baseline examples/sample_analysis.json` showing exit code 0 or 2 with badge output.
- **Voiceover:**
  > "Finally, plug Shakedown into CI with `shakedown check` to block agent regressions before deploying to production. Fully open-source, runnable with local models, and zero marketing fluff. Test your agent's sensitivity with Shakedown."

---

## Social Post Draft (X / Twitter)

```text
An agent can pass your test suite and still be brittle.

We built Shakedown: a sensitivity profiler that subjects AI agents to controlled prompt ablations and tool chaos faults.

No LLM judges. Success is an executable pytest exit code.

Results on Qwen 2.5 Coder 7B (252 runs, 100% local, $0 spend):
• Baseline pass rate: 42% (Wilson 95% CI: 19%–68%)
• Empty tool results dropped pass rate to 0% (raw p=0.051), with 42% of failing runs falsely claiming success
• Truncated test output had 0.0 delta—agents handled truncation cleanly
• Familywise Holm adjustment prevents false discoveries on small samples

Interactive sample report: https://p1manav.github.io/Shakedown/
Repo & quick start: https://github.com/P1Manav/Shakedown

Built with Alan AI.
```
