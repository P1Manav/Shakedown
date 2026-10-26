"""Task and budget specification loading."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
TASKS_DIR = ROOT / "tasks"


def load_dotenv(path: Path | None = None) -> None:
    """Minimal .env loader; never overrides variables already set."""
    path = path or ROOT / ".env"
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key, value = key.strip(), value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


_ENV_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


def expand_env(value: Any) -> Any:
    if isinstance(value, str):
        return _ENV_RE.sub(lambda m: os.environ.get(m.group(1), ""), value)
    if isinstance(value, list):
        return [expand_env(v) for v in value]
    if isinstance(value, dict):
        return {k: expand_env(v) for k, v in value.items()}
    return value


@dataclass(frozen=True)
class ChunkSpec:
    id: str
    file: str
    kind: str          # system | tool_description
    tool: str | None
    text: str


@dataclass(frozen=True)
class AblationConfig:
    chunks: list[str]
    modes: list[str]


@dataclass(frozen=True)
class ChaosConfig:
    fault: str
    tool: str
    params: dict = field(default_factory=dict)

    @property
    def id(self) -> str:
        extra = ",".join(f"{k}={v}" for k, v in sorted(self.params.items()))
        return f"chaos:{self.fault}:{self.tool}" + (f":{extra}" if extra else "")

    @property
    def label(self) -> str:
        return f"{self.fault} → {self.tool}"


@dataclass(frozen=True)
class BudgetSpec:
    baseline_runs: int = 40
    min_runs_per_cell: int = 10
    max_runs_per_cell: int = 30
    alpha: float = 0.05
    correction: str = "holm"
    stop_rule: str = "sequential"
    meaningful_drop: float = 0.25
    max_total_calls: int = 6000
    max_usd: float = 15.0
    max_workers: int = 3
    max_minutes: float = 180.0
    batch_size: int = 4


@dataclass(frozen=True)
class ModelSpec:
    base_url: str = "http://127.0.0.1:11434/v1"
    name: str = "qwen2.5-coder:7b"
    temperature: float = 0.7
    price_input_per_1m: float = 0.0
    price_output_per_1m: float = 0.0


@dataclass(frozen=True)
class TaskSpec:
    name: str
    description: str
    path: Path
    dir: Path
    agent_command: str
    agent_timeout_s: float
    max_model_calls: int
    source: Path
    success_command: str
    success_timeout_s: float
    chunks: list[ChunkSpec]
    ablation: list[AblationConfig]
    chaos: list[ChaosConfig]
    budget: BudgetSpec
    model: ModelSpec

    @property
    def id(self) -> str:
        return self.dir.name

    def chunk(self, chunk_id: str) -> ChunkSpec:
        for c in self.chunks:
            if c.id == chunk_id:
                return c
        raise KeyError(chunk_id)


def _project_root_for(task_dir: Path) -> Path:
    for parent in [task_dir, *task_dir.parents]:
        if (parent / "shakedown").is_dir() and (parent / "tasks").is_dir():
            return parent
    return ROOT


def _resolve_chunk_file(file: str, task_dir: Path) -> Path:
    candidates = [task_dir / file, _project_root_for(task_dir) / file, ROOT / file]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"prompt chunk not found: {file} (task {task_dir.name})")


def load_task(path: str | Path, budget_override: dict | None = None) -> TaskSpec:
    path = Path(path).resolve()
    task_dir = path.parent
    raw = expand_env(yaml.safe_load(path.read_text(encoding="utf-8")) or {})

    for key in ("name", "description", "agent", "workspace", "success"):
        if key not in raw:
            raise ValueError(f"{path}: missing required key '{key}'")

    agent = raw["agent"]
    command = agent.get("command", "")
    if "{task_file}" not in command or "{workdir}" not in command:
        raise ValueError(f"{path}: agent.command must contain {{task_file}} and {{workdir}}")

    budget_raw = {**raw.get("budget", {})}
    if budget_override:
        budget_raw.update({k: v for k, v in budget_override.items() if v is not None})
    budget = BudgetSpec(**{k: v for k, v in budget_raw.items() if k in BudgetSpec.__dataclass_fields__})

    model_raw = raw.get("model", {}) or {}
    model = ModelSpec(
        base_url=model_raw.get("base_url") or os.environ.get("MODEL_BASE_URL", ModelSpec.base_url),
        name=model_raw.get("name") or os.environ.get("MODEL_NAME", ModelSpec.name),
        temperature=float(model_raw.get("temperature", 0.7)),
        price_input_per_1m=float(os.environ.get("PRICE_INPUT_PER_1M",
                                                model_raw.get("price_input_per_1m", 0))),
        price_output_per_1m=float(os.environ.get("PRICE_OUTPUT_PER_1M",
                                                 model_raw.get("price_output_per_1m", 0))),
    )

    chunks = []
    for item in raw.get("prompt_chunks", []):
        file_path = _resolve_chunk_file(item["file"], task_dir)
        chunks.append(ChunkSpec(
            id=item["id"], file=item["file"], kind=item.get("kind", "system"),
            tool=item.get("tool"), text=file_path.read_text(encoding="utf-8").strip()))

    def _as_list(value, default):
        if value is None:
            return list(default)
        return value if isinstance(value, list) else [value]

    ablation = [AblationConfig(
        chunks=_as_list(a.get("chunks"), ["all"]),
        modes=_as_list(a.get("modes"), ["remove"]))
        for a in (raw.get("perturbations", {}) or {}).get("ablation", [])]
    chaos = []
    for item in (raw.get("perturbations", {}) or {}).get("chaos", []):
        params = {k: v for k, v in item.items() if k not in {"fault", "tool"}}
        chaos.append(ChaosConfig(fault=item["fault"], tool=item.get("tool", ""), params=params))

    source = (task_dir / raw["workspace"]["source"]).resolve()
    if not source.is_dir():
        raise FileNotFoundError(f"{path}: workspace.source not found: {source}")

    return TaskSpec(
        name=raw["name"],
        description=str(raw["description"]).strip(),
        path=path,
        dir=task_dir,
        agent_command=command,
        agent_timeout_s=float(agent.get("timeout_s", 240)),
        max_model_calls=int(agent.get("max_model_calls", 18)),
        source=source,
        success_command=raw["success"].get("command", "python -m pytest -q tests/"),
        success_timeout_s=float(raw["success"].get("timeout_s", 120)),
        chunks=chunks,
        ablation=ablation,
        chaos=chaos,
        budget=budget,
        model=model,
    )


def discover_tasks(root: Path | None = None, names: list[str] | None = None,
                   budget_override: dict | None = None) -> list[TaskSpec]:
    root = root or TASKS_DIR
    paths = sorted(p / "task.yaml" for p in root.iterdir() if (p / "task.yaml").exists())
    if names:
        wanted = set(names)
        paths = [p for p in paths if p.parent.name in wanted or p.parent.name.split("_")[0] in wanted]
    tasks = [load_task(p, budget_override) for p in paths]
    if not tasks:
        raise SystemExit(f"no tasks found under {root}")
    return tasks