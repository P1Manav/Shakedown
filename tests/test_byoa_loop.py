import json
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path


class MockOpenAIServer(BaseHTTPRequestHandler):
    step = 0

    def log_message(self, format, *args):
        pass

    def do_POST(self):
        if self.path == "/v1/chat/completions" or self.path == "/chat/completions":
            content_length = int(self.headers.get("Content-Length", 0))
            payload = json.loads(self.rfile.read(content_length))
            assert "messages" in payload

            MockOpenAIServer.step += 1
            if MockOpenAIServer.step == 1:
                # First step: ask agent to write a file
                resp = {
                    "id": "chatcmpl-test1",
                    "object": "chat.completion",
                    "choices": [{
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [{
                                "id": "call_write_1",
                                "type": "function",
                                "function": {
                                    "name": "write_file",
                                    "arguments": json.dumps({"path": "output.txt", "content": "BYOA success!"}),
                                },
                            }],
                        },
                        "finish_reason": "tool_calls",
                    }],
                }
            else:
                # Second step: finish
                resp = {
                    "id": "chatcmpl-test2",
                    "object": "chat.completion",
                    "choices": [{
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": "File created successfully.",
                        },
                        "finish_reason": "stop",
                    }],
                }

            body = json.dumps(resp).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        else:
            self.send_response(404)
            self.end_headers()


def test_bring_your_own_agent_loop_edits_file_against_mock_proxy(tmp_path):
    MockOpenAIServer.step = 0
    server = HTTPServer(("127.0.0.1", 0), MockOpenAIServer)
    port = server.server_port
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    byoa_script = Path("examples/bring_your_own_agent/agent.py").resolve()
    target_file = tmp_path / "output.txt"

    env = {
        "OPENAI_BASE_URL": f"http://127.0.0.1:{port}/v1",
        "OPENAI_API_KEY": "test-key",
        "MODEL_NAME": "test-model",
        "PATH": sys.exec_prefix,
        "PYTHONPATH": ".",
        "SYSTEMROOT": sys.platform,
    }
    import os
    for k in ("PATH", "SystemRoot", "WINDIR", "TEMP", "TMP"):
        if k in os.environ:
            env[k] = os.environ[k]

    res = subprocess.run(
        [sys.executable, str(byoa_script)],
        cwd=str(tmp_path),
        env=env,
        capture_output=True,
        text=True,
        timeout=15,
    )
    server.shutdown()

    assert res.returncode == 0, f"BYOA failed: stdout={res.stdout}, stderr={res.stderr}"
    assert "Agent finished" in res.stdout
    assert target_file.exists()
    assert target_file.read_text(encoding="utf-8") == "BYOA success!"
