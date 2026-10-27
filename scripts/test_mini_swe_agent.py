"""Point mini-swe-agent v2.4.6 at Shakedown reverse proxy and record trace analytics."""

from __future__ import annotations

import os
import shutil
import tempfile
import time
from pathlib import Path

import yaml

import minisweagent
from minisweagent.agents import get_agent
from minisweagent.environments.local import LocalEnvironment
from minisweagent.models.litellm_model import LitellmModel
from shakedown.claims import detect_claim
from shakedown.cost import BudgetTracker, Pricing
from shakedown.perturb.base import Baseline
from shakedown.proxy import ProxyServer, RunContext
from shakedown.store import Store

RUN_ID = f"mini-swe-agent-proof-{int(time.time())}"
EXPERIMENT_ID = "mini-swe-agent-proof"
ROOT = Path(__file__).resolve().parent.parent


def run_mini_swe_agent_proof() -> dict:
    os.environ["MSWEA_COST_TRACKING"] = "ignore_errors"
    os.environ["OPENAI_API_KEY"] = "ollama"

    # Setup isolated test workspace with bugfix_01
    work_dir = Path(tempfile.mkdtemp(prefix="mini-swe-"))
    task_source = ROOT / "src" / "shakedown" / "data" / "tasks" / "bugfix_01" / "repo"
    if task_source.exists():
        shutil.copytree(task_source, work_dir, dirs_exist_ok=True)
    else:
        (work_dir / "calc.py").write_text(
            "def median(items):\n    s = sorted(items)\n    n = len(s)\n    if n % 2 == 1:\n        return s[n // 2]\n    return (s[n // 2 - 1] + s[n // 2]) / 2.0\n",
            encoding="utf-8",
        )

    store = Store(EXPERIMENT_ID)
    budget = BudgetTracker(max_calls=15, max_usd=1.0)
    pricing = Pricing(input_per_1m=0.0, output_per_1m=0.0)
    proxy = ProxyServer("http://127.0.0.1:11434/v1", store, budget, pricing)
    proxy.start()

    ctx = RunContext(
        run_id=RUN_ID,
        task_id="bugfix_01_median",
        cell_id="baseline",
        perturbation=Baseline(),
        max_calls=15,
        store=store,
        pricing=pricing,
    )
    proxy.register(ctx)
    proxy_base_url = proxy.base_url(RUN_ID)

    # Point OPENAI_BASE_URL exclusively at Shakedown proxy
    os.environ["OPENAI_BASE_URL"] = proxy_base_url
    print(f"Registered proxy route: {proxy_base_url}")

    # Load default mini-swe-agent agent config
    cfg_path = Path(minisweagent.package_dir) / "config" / "default.yaml"
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    agent_cfg = cfg.get("agent", {})
    agent_cfg["step_limit"] = 3

    model = LitellmModel(model_name="openai/llama3.1:8b-instruct-q4_K_M", cost_tracking="ignore_errors")
    env = LocalEnvironment(cwd=str(work_dir))
    agent = get_agent(model=model, env=env, config=agent_cfg, default_type="default")

    task_prompt = (
        "Inspect calc.py in the repository. It has a function median(items). "
        "Check how median handles even length lists. If needed, propose a fix."
    )

    t0 = time.perf_counter()
    try:
        agent.run(task=task_prompt)
    except Exception as exc:
        print(f"Agent run finished with: {exc}")
    elapsed = time.perf_counter() - t0

    proxy.unregister(RUN_ID)
    proxy.stop()

    # Parse trace metrics from Shakedown proxy
    trace_events = store.load_trace(RUN_ID)
    model_calls = ctx.state.calls
    prompt_tokens = ctx.state.prompt_tokens
    completion_tokens = ctx.state.completion_tokens

    assistant_messages = []
    actions = []
    for ev in trace_events:
        if ev.get("event") == "chat_completion":
            body = ev.get("response_body", {})
            choices = body.get("choices", [])
            if choices:
                msg = choices[0].get("message", {})
                content = msg.get("content", "")
                if content:
                    assistant_messages.append(content)
                    if "```" in content:
                        actions.append("code_block")
                tool_calls = msg.get("tool_calls", [])
                for tc in tool_calls:
                    fn = tc.get("function", {}).get("name", "tool")
                    actions.append(f"tool:{fn}")

    final_message = assistant_messages[-1] if assistant_messages else ""
    claimed = detect_claim(final_message) if final_message else False

    analytics = {
        "agent": "mini-swe-agent",
        "agent_version": minisweagent.__version__,
        "run_id": RUN_ID,
        "elapsed_seconds": round(elapsed, 2),
        "proxy_route_used": proxy_base_url,
        "model_calls_intercepted": model_calls,
        "prompt_tokens_intercepted": prompt_tokens,
        "completion_tokens_intercepted": completion_tokens,
        "actions_detected": actions,
        "final_message_preview": (final_message[:200] + "...") if len(final_message) > 200 else final_message,
        "claimed_success": claimed,
        "proxy_trace_events_logged": len(trace_events),
    }

    shutil.rmtree(work_dir, ignore_errors=True)
    return analytics


if __name__ == "__main__":
    out = run_mini_swe_agent_proof()
    print("=== MINI-SWE-AGENT REAL PROOF RESULT ===")
    for k, v in out.items():
        print(f"{k}: {v}")
