from pathlib import Path

from shakedown.demo_agent.agent import DEFAULT_MODEL_NAME as AGENT_MODEL_NAME
from shakedown.model_config import DEFAULT_MODEL_NAME
from shakedown.spec import ModelSpec


def test_default_model_name_is_shared_across_runtime_and_documentation():
    root = Path(__file__).resolve().parents[1]
    example = (root / ".env.example").read_text(encoding="utf-8")
    readme = (root / "README.md").read_text(encoding="utf-8")

    assert ModelSpec.name == DEFAULT_MODEL_NAME
    assert AGENT_MODEL_NAME == DEFAULT_MODEL_NAME
    assert f"MODEL_NAME={DEFAULT_MODEL_NAME}" in example
    assert f"`{DEFAULT_MODEL_NAME}`" in readme
