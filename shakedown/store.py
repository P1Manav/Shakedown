"""On-disk experiment store: metadata, run records, and per-run traces."""

from __future__ import annotations

import json
import threading
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
RUNS_DIR = ROOT / "runs"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass
class RunResult:
    run_id: str
    experiment_id: str
    task_id: str
    task_name: str
    cell_id: str
    cell_label: str
    cell_kind: str                 # baseline | ablation | chaos | noise
    replicate: int = -1
    params: dict = field(default_factory=dict)
    started_at: str = ""
    duration_s: float = 0.0
    success: bool = False
    invalid: bool = False
    tamper_detected: bool = False
    perturbation_applied: bool = False
    perturbation_records: list[dict] = field(default_factory=list)
    perturbation_diff_hashes: list[str] = field(default_factory=list)
    exposure_reason: str = ""
    exit_code: int | None = None
    agent_exit: int | None = None
    timed_out: bool = False
    model_calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0
    steps: int = 0
    actions: list[str] = field(default_factory=list)
    normalized_actions: list[str] = field(default_factory=list)
    final_message: str = ""
    claimed_success: bool = False
    failure_class: str = ""
    events: list[dict] = field(default_factory=list)
    agent_markers: dict = field(default_factory=dict)
    test_output: str = ""
    error: str = ""
    trace_path: str = ""
    cached_calls: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


class Store:
    def __init__(self, experiment_id: str, root: Path | None = None):
        self.experiment_id = experiment_id
        self.dir = (root or RUNS_DIR) / experiment_id
        self.traces = self.dir / "traces"
        self.traces.mkdir(parents=True, exist_ok=True)
        self.runs_path = self.dir / "runs.jsonl"
        self._lock = threading.Lock()

    # -- runs ---------------------------------------------------------------
    def append_run(self, run: RunResult) -> None:
        with self._lock:
            with self.runs_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(run.to_dict(), ensure_ascii=False) + "\n")

    def load_runs(self) -> list[dict]:
        if not self.runs_path.exists():
            return []
        rows = []
        with self.runs_path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    try:
                        rows.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        return rows

    # -- traces -------------------------------------------------------------
    def trace_path(self, run_id: str) -> Path:
        return self.traces / f"{run_id}.jsonl"

    def append_trace(self, run_id: str, event: dict) -> None:
        with self._lock:
            with self.trace_path(run_id).open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(event, ensure_ascii=False) + "\n")

    def load_trace(self, run_id: str) -> list[dict]:
        path = self.trace_path(run_id)
        if not path.exists():
            return []
        events = []
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    try:
                        events.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        return events

    # -- metadata and analysis ---------------------------------------------
    def write_json(self, name: str, payload: Any) -> Path:
        path = self.dir / name
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        return path

    def read_json(self, name: str, default: Any = None) -> Any:
        path = self.dir / name
        if not path.exists():
            return default
        return json.loads(path.read_text(encoding="utf-8"))

    def save_meta(self, meta: dict) -> None:
        self.write_json("meta.json", meta)

    def load_meta(self) -> dict:
        return self.read_json("meta.json", {}) or {}


def new_experiment_id(prefix: str = "run") -> str:
    return f"{prefix}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"

def open_experiment(path: str | Path) -> Store:
    return Store(Path(path).name, Path(path).parent)