import yaml
import pytest

from shakedown.scheduler import build_cells
from shakedown.spec import discover_tasks, load_task


def _write_task(tmp_path, addition="", prompt_chunks="[]"):
    task_dir = tmp_path / "strict-task"
    source = task_dir / "repo"
    source.mkdir(parents=True, exist_ok=True)
    path = task_dir / "task.yaml"
    path.write_text(
        "\ufeffname: strict task\n"
        "description: Strict configuration fixture.\n"
        "agent:\n"
        "  command: 'python -m agent --task {task_file} --workdir {workdir}'\n"
        "workspace:\n"
        "  source: repo\n"
        "success:\n"
        "  command: python -m pytest\n"
        f"prompt_chunks: {prompt_chunks}\n"
        f"{addition}",
        encoding="utf-8",
    )
    return path


@pytest.mark.parametrize(
    "addition",
    [
        "unexpected: true\n",
        "budget:\n  mystery: 2\n",
        "success:\n  pass_if: exit_code_0\n",
        "model:\n  mystery: value\n",
        "agent:\n  mystery: value\n",
    ],
)
def test_unknown_task_and_budget_keys_are_rejected(tmp_path, addition):
    path = _write_task(tmp_path, addition)

    with pytest.raises(ValueError, match="unknown"):
        load_task(path)


def test_task_yaml_accepts_utf8_bom(tmp_path):
    path = _write_task(tmp_path)

    assert load_task(path).name == "strict task"


def test_unknown_shared_perturbation_keys_are_rejected(tmp_path):
    path = _write_task(tmp_path)
    (tmp_path / "perturbations.yaml").write_text(
        "ablation: []\nchaos: []\nnoise: []\nextra: true\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="unknown"):
        load_task(path)


def test_chunk_ids_must_be_unique_and_valid(tmp_path):
    path = _write_task(
        tmp_path,
        prompt_chunks=(
            "\n  - {id: same, file: chunk.md, kind: system}\n"
            "  - {id: same, file: chunk.md, kind: system}"
        ),
    )
    chunk = path.parent / "chunk.md"
    chunk.write_text("The prompt text.", encoding="utf-8")

    with pytest.raises(ValueError, match="duplicate prompt chunk id"):
        load_task(path)

    path = _write_task(
        tmp_path,
        prompt_chunks="\n  - {id: 'not valid', file: chunk.md, kind: system}",
    )
    with pytest.raises(ValueError, match="invalid prompt chunk id"):
        load_task(path)


def test_perturbation_grid_comes_from_one_shared_spec():
    tasks = discover_tasks()
    raw_tasks = [
        yaml.safe_load(task.path.read_text(encoding="utf-8-sig"))
        for task in tasks
    ]
    assert all("perturbations" not in raw for raw in raw_tasks)
    assert all(task.noise_modes == ["reorder_tools", "typos", "paraphrase"]
               for task in tasks)
    cells = build_cells(tasks, noise=True)
    assert {cell.id for cell in cells if cell.kind == "noise"} == {
        "noise:reorder_tools", "noise:typos", "noise:paraphrase",
    }
