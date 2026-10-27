"""Task and budget specification loading."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path
from typing import Any, Mapping

import yaml

from shakedown.model_config import DEFAULT_MODEL_NAME


def load_dotenv(path: Path | None = None) -> None:
    """Minimal .env loader; never overrides variables already set."""
    paths = [path] if path is not None else [
        Path.cwd() / ".env",
        Path.home() / ".env",
    ]
    for dotenv_path in paths:
        if not dotenv_path.exists():
            continue
        for line in dotenv_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key, value = key.strip(), value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value


_ENV_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")
_CHUNK_ID_RE = re.compile(r"^[a-z][a-z0-9_]*$")
_TASK_KEYS = {
    "name", "description", "agent", "workspace", "success", "prompt_chunks",
    "budget", "model",
}


def _mapping(value: Any, context: str) -> dict:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError(f"{context}: expected a mapping")
    return dict(value)


def _validate_keys(value: Mapping, allowed: set[str], context: str) -> None:
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise ValueError(f"{context}: unknown key(s): {', '.join(unknown)}")


def _load_shared_perturbations(task_dir: Path) -> dict:
    path = task_dir.parent / "perturbations.yaml"
    if not path.is_file():
        return {}
    raw = yaml.safe_load(path.read_text(encoding="utf-8-sig")) or {}
    shared = _mapping(raw, "shared perturbations")
    _validate_keys(shared, {"ablation", "chaos", "noise"}, "shared perturbations")
    return shared


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
    baseline_runs: int = 60
    min_runs_per_cell: int = 16
    max_runs_per_cell: int = 40
    alpha: float = 0.05
    correction: str = "holm"
    stop_rule: str = "sequential"
    meaningful_drop: float = 0.25
    max_total_calls: int = 8000
    max_usd: float = 15.0
    max_workers: int = 3
    max_minutes: float = 60.0
    batch_size: int = 4


_BUDGET_KEYS = set(BudgetSpec.__dataclass_fields__)


@dataclass(frozen=True)
class ModelSpec:
    base_url: str = "http://127.0.0.1:11434/v1"
    name: str = DEFAULT_MODEL_NAME
    temperature: float = 0.7
    supports_seed: bool = False
    price_input_per_1m: float = 0.0
    price_output_per_1m: float = 0.0


_MODEL_KEYS = set(ModelSpec.__dataclass_fields__)


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
    noise_modes: list[str]
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


def _resolve_chunk_file(file: str, task_dir: Path) -> Path:
    candidates = [
        task_dir / file,
        Path.cwd() / file,
        Path(str(resources.files("shakedown"))) / file,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"prompt chunk not found: {file} (task {task_dir.name})")


def load_task(path: str | Path, budget_override: dict | None = None) -> TaskSpec:
    path = Path(path).resolve()
    task_dir = path.parent
    raw = _mapping(
        expand_env(yaml.safe_load(path.read_text(encoding="utf-8-sig")) or {}),
        str(path),
    )
    _validate_keys(raw, _TASK_KEYS, str(path))

    for key in ("name", "description", "agent", "workspace", "success"):
        if key not in raw:
            raise ValueError(f"{path}: missing required key '{key}'")

    agent = _mapping(raw["agent"], f"{path}: agent")
    _validate_keys(agent, {"command", "timeout_s", "max_model_calls"},
                   f"{path}: agent")
    command = agent.get("command", "")
    if not isinstance(command, str) or "{task_file}" not in command or "{workdir}" not in command:
        raise ValueError(f"{path}: agent.command must contain {{task_file}} and {{workdir}}")

    workspace = _mapping(raw["workspace"], f"{path}: workspace")
    _validate_keys(workspace, {"source", "isolation"}, f"{path}: workspace")
    success = _mapping(raw["success"], f"{path}: success")
    _validate_keys(success, {"command", "timeout_s"}, f"{path}: success")

    budget_raw = _mapping(raw.get("budget"), f"{path}: budget")
    _validate_keys(budget_raw, _BUDGET_KEYS, f"{path}: budget")
    if budget_override:
        _validate_keys(budget_override, _BUDGET_KEYS, f"{path}: budget override")
        budget_raw.update({k: v for k, v in budget_override.items() if v is not None})
    budget = BudgetSpec(**budget_raw)

    model_raw = _mapping(raw.get("model"), f"{path}: model")
    _validate_keys(model_raw, _MODEL_KEYS, f"{path}: model")
    model = ModelSpec(
        base_url=model_raw.get("base_url") or os.environ.get("MODEL_BASE_URL", ModelSpec.base_url),
        name=model_raw.get("name") or os.environ.get("MODEL_NAME", ModelSpec.name),
        temperature=float(model_raw.get("temperature", 0.7)),
        supports_seed=bool(model_raw.get("supports_seed", False)),
        price_input_per_1m=float(os.environ.get("PRICE_INPUT_PER_1M",
                                                model_raw.get("price_input_per_1m", 0))),
        price_output_per_1m=float(os.environ.get("PRICE_OUTPUT_PER_1M",
                                                 model_raw.get("price_output_per_1m", 0))),
    )

    chunk_items = raw.get("prompt_chunks", []) or []
    if not isinstance(chunk_items, list):
        raise ValueError(f"{path}: prompt_chunks must be a list")
    chunks = []
    seen_chunk_ids = set()
    for index, item in enumerate(chunk_items):
        context = f"{path}: prompt_chunks[{index}]"
        chunk = _mapping(item, context)
        _validate_keys(chunk, {"id", "file", "kind", "tool"}, context)
        chunk_id = chunk.get("id")
        if not isinstance(chunk_id, str) or not _CHUNK_ID_RE.fullmatch(chunk_id):
            raise ValueError(f"{context}: invalid prompt chunk id {chunk_id!r}")
        if chunk_id in seen_chunk_ids:
            raise ValueError(f"{context}: duplicate prompt chunk id {chunk_id!r}")
        seen_chunk_ids.add(chunk_id)
        kind = chunk.get("kind", "system")
        if kind not in {"system", "tool_description"}:
            raise ValueError(f"{context}: unsupported prompt chunk kind {kind!r}")
        tool = chunk.get("tool")
        if kind == "tool_description" and not isinstance(tool, str):
            raise ValueError(f"{context}: tool_description chunks require a tool name")
        file = chunk.get("file")
        if not isinstance(file, str) or not file:
            raise ValueError(f"{context}: file must be a non-empty path")
        file_path = _resolve_chunk_file(file, task_dir)
        text = file_path.read_text(encoding="utf-8").strip()
        if not text:
            raise ValueError(f"{context}: prompt chunk text is empty")
        chunks.append(ChunkSpec(
            id=chunk_id, file=file, kind=kind, tool=tool, text=text))

    def _as_list(value, default):
        if value is None:
            return list(default)
        return value if isinstance(value, list) else [value]

    shared = _load_shared_perturbations(task_dir)
    ablation_raw = shared.get("ablation", []) or []
    if not isinstance(ablation_raw, list):
        raise ValueError(f"{task_dir.parent / 'perturbations.yaml'}: ablation must be a list")
    ablation = []
    for index, item in enumerate(ablation_raw):
        context = f"shared perturbations.ablation[{index}]"
        config = _mapping(item, context)
        _validate_keys(config, {"chunks", "modes"}, context)
        chunks_config = _as_list(config.get("chunks"), ["all"])
        modes = _as_list(config.get("modes"), ["remove"])
        if not all(isinstance(value, str) for value in chunks_config):
            raise ValueError(f"{context}: chunks must contain strings")
        if not all(mode in {"remove", "shuffle"} for mode in modes):
            raise ValueError(f"{context}: modes must be remove or shuffle")
        ablation.append(AblationConfig(chunks=chunks_config, modes=modes))

    chaos = []
    chaos_raw = shared.get("chaos", []) or []
    if not isinstance(chaos_raw, list):
        raise ValueError("shared perturbations.chaos must be a list")
    supported_faults = {
        "truncate_output", "empty_result", "malformed_json", "error_once",
        "stale_file", "renamed_path", "added_latency",
    }
    for index, item in enumerate(chaos_raw):
        context = f"shared perturbations.chaos[{index}]"
        config = _mapping(item, context)
        _validate_keys(config, {"fault", "tool", "keep_chars", "delay_ms"}, context)
        fault = config.get("fault")
        tool = config.get("tool")
        if fault not in supported_faults or not isinstance(tool, str) or not tool:
            raise ValueError(f"{context}: supported fault and non-empty tool are required")
        params = {k: v for k, v in config.items() if k not in {"fault", "tool"}}
        allowed_params = {
            "truncate_output": {"keep_chars"},
            "added_latency": {"delay_ms"},
        }.get(fault, set())
        _validate_keys(params, allowed_params, context)
        chaos.append(ChaosConfig(fault=fault, tool=tool, params=params))

    noise_modes = shared.get("noise", []) or []
    if not isinstance(noise_modes, list) or not all(
        mode in {"reorder_tools", "typos", "paraphrase"} for mode in noise_modes
    ):
        raise ValueError("shared perturbations.noise must list supported noise modes")
    if len(noise_modes) != len(set(noise_modes)):
        raise ValueError("shared perturbations.noise contains duplicate modes")

    source = (task_dir / workspace["source"]).resolve()
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
        noise_modes=noise_modes,
        budget=budget,
        model=model,
    )


def discover_tasks(root: Path | None = None, names: list[str] | None = None,
                   budget_override: dict | None = None) -> list[TaskSpec]:
    root = root or Path(str(resources.files("shakedown").joinpath("data", "tasks")))
    paths = sorted(p / "task.yaml" for p in root.iterdir() if (p / "task.yaml").exists())
    if names:
        wanted = set(names)
        paths = [p for p in paths if p.parent.name in wanted or p.parent.name.split("_")[0] in wanted]
    tasks = [load_task(p, budget_override) for p in paths]
    if not tasks:
        raise SystemExit(f"no tasks found under {root}")
    return tasks