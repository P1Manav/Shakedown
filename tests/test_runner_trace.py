from types import SimpleNamespace

from shakedown.runner import Runner
from shakedown.store import Store


class TraceProxy:
    def __init__(self):
        self.context = None

    def register(self, context):
        self.context = context

    def base_url(self, run_id):
        return f"http://proxy/run/{run_id}"

    def unregister(self, run_id):
        pass


def test_runner_derives_agent_metrics_from_proxy_trace_not_stdout_markers(
    monkeypatch, tmp_path
):
    source = tmp_path / "source"
    source.mkdir()
    (source / "calc.py").write_text("value = 1\n", encoding="utf-8")
    store = Store("trace-metrics", tmp_path / "runs")
    proxy = TraceProxy()
    task = SimpleNamespace(
        id="task",
        name="Task",
        source=source,
        path=tmp_path / "task.yaml",
        agent_command="agent --task {task_file} --workdir {workdir}",
        agent_timeout_s=10,
        max_model_calls=10,
        success_command="success-check",
        success_timeout_s=10,
        chunks=[],
        model=SimpleNamespace(
            name="test-model",
            temperature=0.0,
            supports_seed=False,
            price_input_per_1m=0.0,
            price_output_per_1m=0.0,
        ),
    )

    def completion(call_id):
        return {
            "event": "response",
            "body": {"choices": [{"message": {"tool_calls": [{
                "id": call_id,
                "type": "function",
                "function": {
                    "name": "read_file",
                    "arguments": '{"path":"calc.py"}',
                },
            }]}}]},
        }

    def run_command(command, cwd, env, timeout_s):
        if command.startswith("agent "):
            store.append_trace(proxy.context.run_id, completion("call-1"))
            store.append_trace(proxy.context.run_id, {
                "event": "response",
                "stream": True,
                "body": {"stream": True, "events": [
                    {"choices": [{"delta": {"tool_calls": [{
                        "index": 0,
                        "id": "call-2",
                        "function": {"name": "read_file", "arguments": '{"path":'},
                    }]}}]},
                    {"choices": [{"delta": {"tool_calls": [{
                        "index": 0,
                        "function": {"arguments": '"calc.py"}'},
                    }]}}]},
                ]},
            })
            store.append_trace(proxy.context.run_id, completion("call-3"))
            store.append_trace(proxy.context.run_id, {
                "event": "response",
                "body": {"choices": [{"message": {"content": "DONE: all tests passed"}}]},
            })
            stdout = (
                'SHAKEDOWN_EVENT: {"event":"done","steps":999,'
                '"actions":["write_file:fake.py"]}\n'
                'SHAKEDOWN_EVENT: {"event":"loop_detected","action":"fake"}\n'
                "SHAKEDOWN_FINAL: I could not fix this task\n"
            )
            return 0, stdout, "", False
        return 0, "all tests pass", "", False

    monkeypatch.setattr("shakedown.runner.run_command", run_command)
    result = Runner(store, proxy).run(
        task, "baseline", "Baseline", "baseline", SimpleNamespace(id="baseline")
    )

    assert result.steps == 3
    assert result.actions == ["read_file:calc.py"] * 3
    assert result.normalized_actions == ['read_file({"path":"calc.py"})'] * 3
    assert result.final_message == "DONE: all tests passed"
    assert result.claimed_success is True
    assert {"event": "loop_detected", "action": "read_file:calc.py"} in result.events
    assert result.agent_markers["actions"] == ["write_file:fake.py"]
    assert result.agent_markers["final"] == "I could not fix this task"
