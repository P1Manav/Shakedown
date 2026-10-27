# Shakedown Demo Script (90 Seconds)

A structured walkthrough for a high-signal 90-second video demo on X / Twitter, backed entirely by real, unmanipulated numbers from the committed 216-run benchmark (`examples/sample_report.html`).

---

## Video Walkthrough & Voiceover

### 0:00 – 0:15 | The Core Problem & Philosophy
- **Screen:** Show terminal running `shakedown demo` and inspecting the task workspace.
- **Voiceover:**
  > "Most agent benchmarks test an agent once and declare victory if it passes. But an agent that passes today can break tomorrow from a minor prompt tweak or an unexpected tool response. Shakedown is a sensitivity profiler for AI agents. We run controlled prompt ablations and tool-result chaos faults against real tasks. Crucially: success is never determined by an LLM judge. It is an executable pytest exit code—pass or fail."

### 0:15 – 0:35 | Baseline Reliability Card
- **Screen:** Switch to browser showing the top of the HTML report (`examples/sample_report.html`) in dark mode. Hover over the hero card.
- **Voiceover:**
  > "Here is our baseline on Llama 3.1 8B Instruct running locally on Ollama at zero API cost. Across 4 benchmark tasks, baseline pass rate was 50%—4 out of 8 runs passed. Notice we don't just give a point estimate: Shakedown computes the Wilson 95% confidence interval—21% to 79%—and reports the minimum detectable effect: 50 points. Noise is real, and Shakedown makes uncertainty visible."

### 0:35 – 0:60 | The Sensitivity Heatmap & Statistical Rigor
- **Screen:** Scroll down to the Sensitivity Heatmap. Zoom in on `remove identity` and `renamed_path → read_file`.
- **Voiceover:**
  > "Next, the perturbation heatmap. Each card displays raw delta, raw p-value, family-adjusted Holm p-value, and family MDE. Look at removing `identity`: pass rate plunged to 0%, a 50-point drop with raw p = 0.083. But after Holm correction across the ablation family, adjusted p is 1.000, so Shakedown rigorously marks it inconclusive rather than hallucinating a false discovery. Meanwhile, `malformed_json` on `run_tests` had a 0.0 delta—the agent was completely unfazed."

### 0:60 – 0:75 | Action Divergence & Failure Taxonomy
- **Screen:** Scroll to the "First Divergence" timeline and "Failure Taxonomy" stacked bars.
- **Voiceover:**
  > "When runs fail, where do they branch off? Shakedown analyzes normalized tool call trajectories. At early steps, total variation distance spikes: passing agents inspect target source and tests, while failing agents stall or tamper with test files. In the failure taxonomy, 5 out of 8 failures under identity ablation falsely claimed success to the user—caught completely without any LLM judge."

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

Results on Llama 3.1 8B Instruct (216 runs, 100% local, $0 spend):
• Baseline pass rate: 50% (Wilson 95% CI: 21%–79%)
• Removing identity dropped pass rate to 0% (raw p=0.083), with 62% of failing runs falsely claiming success
• Malformed runner JSON had 0.0 delta—agents handled formatting glitches cleanly
• Familywise Holm adjustment prevents false discoveries on small samples

Interactive sample report: https://p1manav.github.io/Shakedown/
Repo & quick start: https://github.com/P1Manav/Shakedown

Built with Alan AI.
```
