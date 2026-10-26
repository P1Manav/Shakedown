import os
import subprocess
import sys


def test_cli_output_does_not_crash_on_unsupported_console_characters():
    env = {**os.environ, "PYTHONIOENCODING": "cp1252:strict"}
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "from shakedown.cli import _configure_cli_output; "
            "_configure_cli_output(); print('\\u2192')",
        ],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )

    assert result.returncode == 0
    assert result.stdout.strip() == r"\u2192"
