from pathlib import Path
import tomllib


ROOT = Path(__file__).resolve().parents[1]


def test_distribution_uses_src_layout_without_a_top_level_tasks_package():
    assert (ROOT / "src" / "shakedown" / "__init__.py").is_file()
    assert not (ROOT / "shakedown").exists()
    assert not (ROOT / "tasks" / "__init__.py").exists()

    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert project["project"]["name"] != "shakedown"
    assert project["project"]["scripts"]["shakedown"] == "shakedown.cli:main"
    assert project["tool"]["setuptools"]["packages"]["find"]["where"] == ["src"]
    assert all(
        "**" not in pattern
        for patterns in project["tool"]["setuptools"]["package-data"].values()
        for pattern in patterns
    )
    assert (ROOT / "scripts" / "wheel_smoke.py").is_file()
    assert (ROOT / ".github" / "workflows" / "ci.yml").is_file()
