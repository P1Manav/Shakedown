# Submission Checklist

This checklist verifies all submission requirements for **Shakedown v0.3.0**.

- [x] **Public repository:** [https://github.com/P1Manav/Shakedown](https://github.com/P1Manav/Shakedown)
- [ ] **Working live report link:** [https://p1manav.github.io/Shakedown/](https://p1manav.github.io/Shakedown/) (curl status: 404; pending GitHub Pages activation in repository settings pointing to `/docs`)
- [x] **Open models only:** Exclusively powered by local open-weight models (`llama3.1:8b-instruct-q4_K_M` via Ollama at `http://127.0.0.1:11434/v1`). $0.00 model API spend; zero external network requests during profiling.
- [x] **Executable test validation (no LLM judge):** All pass/fail verdicts are determined by deterministic process exit codes (`pytest` returncode 0), never an LLM evaluator.
- [x] **Built with Alan AI:** Autonomous agent development workflow, end-to-end regression testing, and verification.
- [ ] **X demo video (1–3 min):** Video script documented with exact timestamps and real recorded figures in [docs/DEMO_SCRIPT.md](docs/DEMO_SCRIPT.md); post URL pending recording publication.
- [x] **CI gate & packaging:** `shakedown check` command returns exit code 2 on detected regression, generates badge markdown, and wheel packages cleanly via standard `pyproject.toml` `src/` layout.
