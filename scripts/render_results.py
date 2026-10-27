#!/usr/bin/env python3
"""Render README and DEMO_SCRIPT results tables and numbers from sample_analysis.json."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def format_table(data: dict) -> str:
    lines = [
        "| Condition / Cell | Type | Pass Rate | Δ vs Base | Raw p | Adjusted p (Holm) | Formally Conclusive? |",
        "|---|---|---|---|---|---|---|",
    ]
    base = data["baseline"]
    base_k = base["k"]
    base_n = base["n"]
    base_rate = base["rate"] * 100
    w_low, w_high = base["wilson"][0] * 100, base["wilson"][1] * 100
    lines.append(
        f"| **Baseline (no perturbation)** | Baseline | {base_k}/{base_n} ({base_rate:.1f}%) | — | — | — | Wilson 95%: [{w_low:.1f}%, {w_high:.1f}%] |"
    )
    sorted_cells = sorted(data["cells"], key=lambda c: (c.get("delta_points", 0), c.get("p_raw", 1.0)))
    for c in sorted_cells:
        label = c["label"]
        kind = c["kind"].capitalize()
        k, n = c["k"], c["n"]
        rate = c["rate"] * 100
        delta = c.get("delta_points", 0.0)
        delta_str = f"{delta:+.1f} pts" if abs(delta) > 1e-6 else "0.0 pts"
        p_raw = c.get("p_raw", 1.0)
        p_adj = c.get("p_adjusted", 1.0)
        raw_str = f"{p_raw:.3f}" if p_raw < 0.9995 else "1.000"
        adj_str = f"{p_adj:.3f}" if p_adj < 0.9995 else "1.000"
        reason = (c.get("reason") or "").lower()
        if c.get("significant"):
            conclusive = f"**Significant** ({c.get('reason', '')})"
        elif "insufficient exposure" in reason:
            conclusive = "Inconclusive (insufficient exposure)"
        elif "unaffected" in reason:
            conclusive = "Inconclusive (unaffected)"
        elif "underpowered" in reason:
            conclusive = "Inconclusive (sample size underpowered)"
        else:
            conclusive = "Inconclusive"
        lines.append(
            f"| `{label}` | {kind} | {k}/{n} ({rate:.1f}%) | {delta_str} | {raw_str} | {adj_str} | {conclusive} |"
        )
    return "\n".join(lines)


def update_readme(readme_path: Path, data: dict, check_only: bool = False) -> bool:
    content = readme_path.read_text(encoding="utf-8")
    table = format_table(data)

    # Replace results table in README
    pattern = re.compile(
        r"(\| Condition / Cell \| Type \| Pass Rate \|.*?)(?=\n\n###|\n\n---|\Z)",
        re.DOTALL
    )
    if not pattern.search(content):
        print(f"Error: Could not find results table in {readme_path}", file=sys.stderr)
        return False

    new_content = pattern.sub(table, content)

    # Update summary counts line if present
    summary = data.get("summary", {})
    total_runs = summary.get("runs", summary.get("total_runs", 216))
    model_name = data.get("meta", {}).get("model", {}).get("name", "llama3.1:8b-instruct-q4_K_M")
    total_calls = summary.get("calls", summary.get("total_calls", 2510))
    tokens = summary.get("tokens", summary.get("total_tokens", 0))

    count_pattern = re.compile(
        r"\((\d+ runs across \d+ tasks on `[^`]+`, [^)]+)\)"
    )
    if count_pattern.search(new_content):
        tokens_str = f"{tokens / 1_000_000:.2f}M tokens" if tokens else "0.00M tokens"
        spend_str = f"${summary.get('spend_usd', 0.0):.2f} spend"
        new_summary_line = f"({total_runs} runs across 4 tasks on `{model_name}`, {total_calls:,} model calls, {tokens_str}, {spend_str})"
        new_content = count_pattern.sub(new_summary_line, new_content)

    if new_content == content:
        return True

    if check_only:
        return False

    readme_path.write_text(new_content, encoding="utf-8")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description="Render README table from analysis.json")
    parser.add_argument("--analysis", default=str(ROOT / "examples" / "sample_analysis.json"))
    parser.add_argument("--readme", default=str(ROOT / "README.md"))
    parser.add_argument("--check", action="store_true", help="Fail if README is not up to date")
    args = parser.parse_args()

    analysis_path = Path(args.analysis)
    readme_path = Path(args.readme)

    if not analysis_path.exists():
        print(f"File not found: {analysis_path}", file=sys.stderr)
        return 1

    data = json.loads(analysis_path.read_text(encoding="utf-8"))
    ok = update_readme(readme_path, data, check_only=args.check)

    if args.check:
        if not ok:
            print("README.md table does not match sample_analysis.json. Run scripts/render_results.py to update.", file=sys.stderr)
            return 1
        print("README.md table matches sample_analysis.json.")
        return 0

    print("Updated README.md successfully.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
