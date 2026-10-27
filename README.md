# Shakedown

[![CI](https://github.com/P1Manav/Shakedown/actions/workflows/ci.yml/badge.svg)](https://github.com/P1Manav/Shakedown/actions/workflows/ci.yml)
[![Interactive Report](https://img.shields.io/badge/Report-Live%20Demo-brightgreen)](https://p1manav.github.io/Shakedown/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/Python-3.11+-blue.svg)](pyproject.toml)

![Shakedown Sensitivity Report](docs/assets/report.png)

> **[Open Interactive Live Report](https://p1manav.github.io/Shakedown/)** generated directly from real runs.

Shakedown is a sensitivity profiler for AI agents. Give it an agent command, a task workspace, and an executable pass/fail check: it measures baseline reliability and subjects the agent to controlled prompt ablations and tool-result chaos faults.

**Key Principle:** Success is always determined by the task's executable test exit code (e.g. `pytest` exit code 0). Shakedown never uses an LLM judge to grade agent outputs.

---

## Built with Alan AI & Open-Source Models Only

Shakedown was built with **Alan AI** and is engineered from the ground up for open-source model evaluation:

- **Model Name:** `qwen2.5-coder:7b` (Qwen 2.5 Coder 7B, 7.6B parameters, Q4_K_M quantization)
- **Model Size:** 4.7 GB
- **Endpoint:** Local Ollama runner at `http://127.0.0.1:11434/v1`
- **Total Paid Spend:** **$0.00** across all 252 recorded experiment runs (zero external API calls)

---

## Quick Start

Python 3.11 or newer is required.

### Windows (PowerShell)
```powershell
# 1. Create and activate virtual environment
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1

# 2. Install Shakedown with development dependencies
python -m pip install -e ".[dev]"

# 3. Pull the open-source model into Ollama
ollama pull qwen2.5-coder:7b

# 4. Copy environment configuration (defaults to local Ollama)
Copy-Item .env.example .env

# 5. Run the full demo benchmark
python -m shakedown demo --runs 60 --max-runs 40 --min-runs 16 --workers 3 --max-minutes 60 --max-usd 15
```

### macOS & Linux (Bash)
```bash
# 1. Create and activate virtual environment
python3.11 -m venv .venv
source .venv/bin/activate

# 2. Install Shakedown with development dependencies
python -m pip install -e ".[dev]"

# 3. Pull the open-source model into Ollama
ollama pull qwen2.5-coder:7b

# 4. Copy environment configuration
cp .env.example .env

# 5. Run the full demo benchmark
make demo
# or directly:
python -m shakedown demo --runs 60 --max-runs 40 --min-runs 16 --workers 3 --max-minutes 60 --max-usd 15
```

For a free baseline-only measurement:
```bash
python -m shakedown run --baseline -n 30
```

To estimate grid cost and runtime before executing:
```bash
python -m shakedown estimate
```

---

## Results from the Committed Sample Run

The following results come directly from the committed benchmark run in [`examples/sample_analysis.json`](examples/sample_analysis.json) (252 runs across 4 tasks on `qwen2.5-coder:7b`, 1,181 model calls, 1.37M tokens, $0.00 spend):

| Condition / Cell | Type | Pass Rate | Δ vs Base | Raw p | Adjusted p (Holm) | Formally Conclusive? |
|---|---|---|---|---|---|---|
| **Baseline (no perturbation)** | Baseline | 5/12 (41.7%) | — | — | — | Wilson 95%: [19.3%, 68.0%] |
| `empty_result → read_file` | Chaos | 0/12 (0.0%) | -41.7 pts | 0.051 | 0.306 | Inconclusive (sample size underpowered) |
| `remove identity` | Ablation | 1/12 (8.3%) | -33.3 pts | 0.153 | 1.000 | Inconclusive |
| `remove workflow` | Ablation | 1/12 (8.3%) | -33.3 pts | 0.153 | 1.000 | Inconclusive |
| `renamed_path → read_file` | Chaos | 1/12 (8.3%) | -33.3 pts | 0.171 | 0.855 | Inconclusive |
| `remove tool_run_tests` | Ablation | 3/12 (25.0%) | -16.7 pts | 0.673 | 1.000 | Inconclusive |
| `shuffle tool_write_file` | Ablation | 3/12 (25.0%) | -16.7 pts | 0.673 | 1.000 | Inconclusive |
| `truncate_output → run_tests` | Chaos | 5/12 (41.7%) | 0.0 pts | 1.000 | 1.000 | Inconclusive (unaffected) |
| `stale_file → read_file` | Chaos | 5/12 (41.7%) | 0.0 pts | 1.000 | 1.000 | Inconclusive |
| `shuffle workflow` | Ablation | 5/12 (41.7%) | 0.0 pts | 1.000 | 1.000 | Inconclusive |

### What this teaches us:
1. **Uncertainty is honest:** While `empty_result` caused a steep drop (0/12 vs 5/12 baseline), Holm multiple-testing correction across the chaos family correctly flags this as inconclusive ($p_{adj} = 0.306$). Shakedown refuses to fabricate statistical significance on small samples.
2. **Silent Failure Detection:** Under `empty_result → read_file`, **5 of the 12 failing runs falsely claimed success to the user** (claiming the bug was fixed despite zero edits and failing test suites).
3. **Robustness to Truncation:** When `run_tests` output was truncated to 200 characters, the agent's pass rate was completely identical to baseline (41.7%), demonstrating resilience to truncated tool returns.

---

## Architecture

```mermaid
flowchart LR
  Spec["Task Spec + Shared Perturbations"] --> Scheduler["Interleaved Grid Scheduler"]
  Scheduler --> Workspace["Isolated Workspace Setup"]
  Workspace --> Agent["Agent Subprocess (OPENAI_BASE_URL)"]
  Agent --> Proxy["Reverse Proxy + Fault Injector"]
  Proxy --> LLM["Open Model Endpoint (Ollama)"]
  Proxy --> Trace["Full SSE/JSON Request Trace"]
  Workspace --> ExecCheck["Executable Test (pytest)"]
  ExecCheck --> Analysis["Statistical Inference & Taxonomy"]
  Trace --> Analysis
  Analysis --> Report["Self-Contained Responsive HTML Report"]
```

---

## Point Any Agent at the Proxy

You do not need to rewrite your agent to use Shakedown. Any agent that speaks the OpenAI Chat Completions API can be profiled by pointing its `OPENAI_BASE_URL` (or equivalent client base URL) at Shakedown's proxy.

See [`examples/bring_your_own_agent/agent.py`](examples/bring_your_own_agent/agent.py) for a minimal ~25-line example:

```python
import os, json
from urllib.request import Request, urlopen

# Shakedown injects OPENAI_BASE_URL into the agent process environment
base_url = os.environ["OPENAI_BASE_URL"].rstrip("/")

req = Request(
    f"{base_url}/chat/completions",
    data=json.dumps({
        "model": os.environ.get("OPENAI_MODEL", "qwen2.5-coder:7b"),
        "messages": [{"role": "user", "content": "Fix the bug in calc.py"}],
        "temperature": 0,
    }).encode(),
    headers={"Content-Type": "application/json"},
    method="POST",
)
with urlopen(req, timeout=60) as response:
    result = json.load(response)
print(result["choices"][0]["message"]["content"])
```

During execution:
- The proxy forces configured model and temperature parameters.
- Perturbations (prompt edits or tool result corruptions) are injected transparently.
- All requests and responses are traced with exact token counts and latency.

---

## CI Gate Usage (`shakedown check`)

Shakedown acts as an automated regression gate in continuous integration pipelines:

```bash
# Compare current agent baseline against a committed reference analysis
python -m shakedown check --baseline examples/sample_analysis.json -n 30
```

- **Exit code 0:** No statistically significant regression detected.
- **Exit code 2:** Statistically significant regression detected ($p < \alpha$ and $\Delta < 0$).
- Outputs a standardized badge string for CI dashboards, e.g.:
  `[check] ok | badge: 70% [56-81]`

---

## Limitations

1. **Workspace Sandboxing:** Temporary workspaces isolate files and git states between runs, but they are not kernel-level OS sandboxes. Untrusted agents should be run inside containerized environments.
2. **Statistical Power & Sample Size:** Small sample runs ($N < 30$) have wide confidence intervals ($\text{MDE} > 30\text{ pts}$). Holm and Benjamini–Hochberg family adjustments prevent false discoveries by requiring sufficient sample sizes to declare an effect.
3. **Scope:** Perturbations measure sensitivity to context structure, tool formatting, and simulated network faults. They do not predict performance on completely unseen problem domains.

---

## Development

```bash
# Run the test suite (136 unit & regression tests)
python -m pytest

# Validate task self-tests (checks buggy exit != 0 and fixed exit == 0)
python -m shakedown selftest

# Test packaging and wheel installation smoke
python scripts/wheel_smoke.py
```
