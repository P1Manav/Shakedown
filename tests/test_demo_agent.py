from shakedown.demo_agent.agent import build_prompt, load_chunks


def test_demo_agent_prompt_uses_native_tool_calls():
    system_prompt, _ = build_prompt(load_chunks()["chunks"])

    assert "Use the available tools through their function-calling interface" in system_prompt
    assert "single JSON object" not in system_prompt
