"""Build a wheel and exercise its CLI from an isolated virtual environment."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
import venv
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run(command: list[str], cwd: Path, env: dict[str, str]) -> None:
    subprocess.run(command, cwd=cwd, env=env, check=True)


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="shakedown-wheel-") as temporary:
        temp = Path(temporary)
        wheels = temp / "wheels"
        wheels.mkdir()
        for generated in (
            ROOT / "build",
            ROOT / "shakedown.egg-info",
            ROOT / "shakedown_agent_profiler.egg-info",
        ):
            if generated.exists():
                shutil.rmtree(generated)
        run(
            [sys.executable, "-m", "pip", "wheel", "--no-deps", "--wheel-dir",
             str(wheels), str(ROOT)],
            ROOT,
            os.environ.copy(),
        )
        wheel = next(wheels.glob("*.whl"))
        with zipfile.ZipFile(wheel) as package:
            contents = set(package.namelist())
        required_data = {
            "shakedown/data/tasks/perturbations.yaml",
            "shakedown/data/tasks/bugfix_01/task.yaml",
            "shakedown/data/agent/prompt_chunks/manifest.yaml",
            "shakedown/report/template.html.j2",
        }
        if not required_data.issubset(contents) or any(
            name.startswith("tasks/") for name in contents
        ):
            raise RuntimeError("wheel is missing bundled resources or includes a top-level tasks package")
        environment = temp / "venv"
        venv.EnvBuilder(with_pip=True).create(environment)
        python = environment / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
        smoke = temp / "outside-checkout"
        smoke.mkdir()
        clean_env = {
            key: value for key, value in os.environ.items()
            if key not in {"PYTHONPATH", "PYTHONHOME", "SHAKEDOWN_HOME"}
        }
        run([str(python), "-m", "pip", "install", str(wheel)], smoke, clean_env)
        run([str(python), "-m", "shakedown", "run", "--help"], smoke, clean_env)
        run([str(python), "-m", "shakedown", "selftest"], smoke, clean_env)
        server_code = """
import json
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer
class Handler(BaseHTTPRequestHandler):
    def respond(self, value):
        body = json.dumps(value).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)
    def do_GET(self):
        self.respond({"data": [{"id": "smoke-model", "object": "model"}]})
    def do_POST(self):
        self.rfile.read(int(self.headers.get("Content-Length", "0")))
        self.respond({"choices": [{"message": {"role": "assistant",
            "content": "No changes needed."}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1}})
    def log_message(self, *_):
        pass
server = HTTPServer(("127.0.0.1", 0), Handler)
print(server.server_port, flush=True)
server.serve_forever()
"""
        server = subprocess.Popen(
            [str(python), "-u", "-c", server_code],
            cwd=smoke,
            env=clean_env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        port_line = server.stdout.readline().strip()
        if not port_line or not port_line.isdigit():
            stderr_out = server.stderr.read()
            server.terminate()
            raise RuntimeError(f"smoke model server failed to start: {stderr_out}")
        port = int(port_line)
        model_env = {
            **clean_env,
            "MODEL_BASE_URL": f"http://127.0.0.1:{port}/v1",
            "MODEL_NAME": "smoke-model",
            "MODEL_API_KEY": "wheel-smoke",
            "MODEL_API_FREE": "true",
        }
        try:
            run([
                str(python), "-m", "shakedown", "run", "--baseline",
                "--tasks", "bugfix_01", "-n", "1", "--max-runs", "1",
                "--min-runs", "1", "--workers", "1", "--max-minutes", "2",
                "--max-usd", "1", "--experiment", "wheel-run",
            ], smoke, model_env)
            run([str(python), "-m", "shakedown", "report", "runs/wheel-run"],
                smoke, model_env)
            if not (smoke / "runs" / "wheel-run" / "report.html").is_file():
                raise RuntimeError("installed report command did not create report.html")
        finally:
            server.terminate()
            server.wait(timeout=10)


if __name__ == "__main__":
    main()
