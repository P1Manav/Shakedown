import json

from shakedown.demo_agent.agent import build_prompt, load_chunks
from shakedown.demo_agent.agent import parse_text_tool_call


def test_demo_agent_prompt_uses_native_tool_calls():
    system_prompt, _ = build_prompt(load_chunks()["chunks"])

    assert "Use the available tools through their function-calling interface" in system_prompt
    assert "single JSON object" not in system_prompt


def test_text_tool_parser_handles_narrative_with_triple_quoted_file_content():
    response = r'''The tests failed; here is the complete replacement:
```json
{"name": "write_file", "arguments": {"path": "module.py", "content": """\"\"\"module doc\"\"\"\\nVALUE = 1\\n""" }}
```
```json
{"name": "run_tests", "arguments": {}}
```'''

    call = parse_text_tool_call(response, 0)

    assert call["function"]["name"] == "write_file"
    arguments = json.loads(call["function"]["arguments"])
    assert arguments["content"] == '"""module doc"""\nVALUE = 1\n'
