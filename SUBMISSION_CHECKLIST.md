# Submission Checklist

This checklist verifies all submission requirements for **Shakedown v0.4.0**.

> **GitHub Pages Configuration:**
> To host the live interactive report from the repository, configure GitHub Pages via:
> **Settings -> Pages -> Build and deployment -> Source: Deploy from a branch -> Branch: master /docs -> Save**


- [x] **Public repository:** [https://github.com/P1Manav/Shakedown](https://github.com/P1Manav/Shakedown)
- [x] **Working live report link:** [https://p1manav.github.io/Shakedown/](https://p1manav.github.io/Shakedown/) (HTTP 200 verified live at 2026-10-27 12:28:35Z)
- [x] **Open models only:** Exclusively powered by local open-weight models (`llama3.1:8b-instruct-q4_K_M` via Ollama at `http://127.0.0.1:11434/v1`). $0.00 model API spend; zero external network requests during profiling.
- [x] **Executable test validation (no LLM judge):** All pass/fail verdicts are determined by deterministic process exit codes (`pytest` returncode 0), never an LLM evaluator.
- [x] **Built with Alan AI:** Autonomous agent development workflow, end-to-end regression testing, and verification.
- [ ] **X demo video (1–3 min):** Video script documented with exact timestamps and real recorded figures in [docs/DEMO_SCRIPT.md](docs/DEMO_SCRIPT.md); post URL pending recording publication.
- [x] **CI gate & packaging:** `shakedown check` command returns exit code 2 on detected regression, generates badge markdown, and wheel packages cleanly via standard `pyproject.toml` `src/` layout.
