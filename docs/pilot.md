# Model Pilot Evaluation & Selection

This document records the empirical pilot evaluation conducted across candidate open-source models to select the benchmark model for Shakedown v0.4.

## Candidate Models Evaluated

All candidates were run locally via Ollama ($0.00 compute spend) against the Shakedown benchmark tasks (`bugfix_01_median`, `bugfix_02_slugify`, `bugfix_03_cart`, `bugfix_04_chunker`) under baseline conditions.

| Candidate Model | Parameter Size / Quant | Baseline Pass Rate | Avg Tokens / Run | Avg Sec / Run | Tool Protocol Reliability | Selection Decision |
|---|---|---|---|---|---|---|
| **Llama-3.1-8B-Instruct** | 8B (Q4_K_M) | **50.0%** (4/8) | 12,372 | **13.6s** | **High** (clean JSON tool calls, zero hallucinated tools) | **SELECTED** |
| **Qwen-2.5-Coder-7B** | 7B (Q4_K_M) | 25.0% (8/32) | 5,700 | 13.5s | Medium (intermittent syntax errors on tool arguments) | Rejected |
| **Qwen-2.5-Coder-14B** | 14B (Q4_K_M) | 43.8% (14/32) | 6,882 | 154.4s | High (accurate edits but excessive latency per run) | Rejected (11x slower) |

## Empirical Pilot Run Logs

The underlying run directories committed and recorded locally in `runs/`:
- `runs/pilot-llama2`: Llama 3.1 8B Instruct (50.0% pass rate, 12,372 tokens/run, 13.6s/run).
- `runs/pilot-llama-t02`: Llama 3.1 8B Instruct with temp=0.2 (50.0% pass rate, 21,883 tokens/run, 17.5s/run).
- `runs/pilot-qwen25-coder-7b-v02`: Qwen 2.5 Coder 7B (25.0% pass rate, 5,700 tokens/run, 13.5s/run).
- `runs/pilot-qwen25-coder-14b-v02`: Qwen 2.5 Coder 14B (43.8% pass rate, 6,882 tokens/run, 154.4s/run).

## Selection Rationale

1. **Baseline Signal-to-Noise**: Llama 3.1 8B achieves a 50.0% baseline pass rate on the benchmark tasks, which provides maximum statistical sensitivity (variance $p(1-p) = 0.25$ at $p=0.5$), making drops under ablation and chaos perturbations maximally detectable with the lowest sample size.
2. **Execution Latency**: At 13.6 seconds per run, Llama 3.1 8B can complete powered multi-cell grids within reasonable time frames, unlike 14B models which require >150 seconds per run on local consumer GPUs (RTX 3060 Ti).
3. **Tool Compliance**: Llama 3.1 8B adheres strictly to tool definitions, avoiding malformed JSON envelopes and invalid tool name hallucinations.
