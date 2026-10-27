from importlib import resources
import os
from pathlib import Path

from shakedown.spec import discover_tasks, load_dotenv
from shakedown.store import Store


def test_default_task_discovery_resolves_bundled_resources(monkeypatch, tmp_path):
    original_files = resources.files
    packages = []

    def tracked_files(package):
        packages.append(package)
        return original_files(package)

    monkeypatch.setattr(resources, "files", tracked_files)
    monkeypatch.chdir(tmp_path)

    tasks = discover_tasks(names=["bugfix_01"])

    assert "shakedown" in packages
    assert [task.id for task in tasks] == ["bugfix_01"]
    assert tasks[0].source.is_dir()


def test_run_store_uses_current_directory_or_shakedown_home(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("SHAKEDOWN_HOME", raising=False)
    local = Store("local-experiment")

    assert local.dir == tmp_path / "runs" / "local-experiment"

    home = tmp_path / "alternate-home"
    monkeypatch.setenv("SHAKEDOWN_HOME", str(home))
    overridden = Store("home-experiment")

    assert overridden.dir == home / "runs" / "home-experiment"


def test_dotenv_loads_cwd_before_home_without_overriding_existing_values(
    monkeypatch, tmp_path
):
    cwd = tmp_path / "project"
    home = tmp_path / "home"
    cwd.mkdir()
    home.mkdir()
    (cwd / ".env").write_text(
        "SHAKEDOWN_TEST_VALUE=project\nSHAKEDOWN_CWD_ONLY=present\n",
        encoding="utf-8",
    )
    (home / ".env").write_text(
        "SHAKEDOWN_TEST_VALUE=home\nSHAKEDOWN_HOME_ONLY=present\n",
        encoding="utf-8",
    )
    for key in ("SHAKEDOWN_TEST_VALUE", "SHAKEDOWN_CWD_ONLY",
                "SHAKEDOWN_HOME_ONLY", "SHAKEDOWN_PRESET"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("SHAKEDOWN_PRESET", "environment")
    monkeypatch.chdir(cwd)
    monkeypatch.setattr(Path, "home", lambda: home)

    load_dotenv()

    assert os.environ["SHAKEDOWN_TEST_VALUE"] == "project"
    assert os.environ["SHAKEDOWN_CWD_ONLY"] == "present"
    assert os.environ["SHAKEDOWN_HOME_ONLY"] == "present"
    assert os.environ["SHAKEDOWN_PRESET"] == "environment"
