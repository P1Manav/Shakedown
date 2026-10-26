"""Render the self-contained HTML report."""

from __future__ import annotations

from pathlib import Path

from jinja2 import Environment, FileSystemLoader, select_autoescape

from shakedown.analysis import build_analysis
from shakedown.store import Store

HERE = Path(__file__).resolve().parent


def build_report(store: Store, analysis: dict | None = None,
                 out: Path | None = None) -> Path:
    analysis = analysis or build_analysis(store, store.load_meta())
    env = Environment(loader=FileSystemLoader(str(HERE)),
                      autoescape=select_autoescape(["html", "xml", "j2"]))
    template = env.get_template("template.html.j2")
    html = template.render(a=analysis)
    out = out or (store.dir / "report.html")
    out.write_text(html, encoding="utf-8")
    return out