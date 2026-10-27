"""Recompute metrics for manually labeled runs from a recorded runs.jsonl."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from shakedown.taxonomy import classify_run


def calculate_metrics(rows: list[dict]) -> dict:
    labels = sorted({
        label
        for row in rows
        for label in (row["hand_label"], row["classifier_label"])
    })
    classes = {}
    correct = 0
    for row in rows:
        correct += row["hand_label"] == row["classifier_label"]
    for label in labels:
        support = sum(row["hand_label"] == label for row in rows)
        predicted = sum(row["classifier_label"] == label for row in rows)
        true_positive = sum(
            row["hand_label"] == label and row["classifier_label"] == label
            for row in rows
        )
        classes[label] = {
            "support": support,
            "predicted": predicted,
            "true_positive": true_positive,
            "precision": true_positive / predicted if predicted else None,
            "recall": true_positive / support if support else None,
        }
    return {
        "n": len(rows),
        "correct": correct,
        "accuracy": correct / len(rows) if rows else None,
        "micro_precision": correct / len(rows) if rows else None,
        "micro_recall": correct / len(rows) if rows else None,
        "classes": classes,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", type=Path, help="recorded runs.jsonl")
    parser.add_argument(
        "--labels",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "docs" / "taxonomy-labels.csv",
    )
    args = parser.parse_args()
    recorded = {
        run["run_id"]: run
        for run in (json.loads(line) for line in args.runs.read_text(
            encoding="utf-8"
        ).splitlines() if line.strip())
    }
    with args.labels.open(encoding="utf-8", newline="") as stream:
        labels = list(csv.DictReader(stream))
    for row in labels:
        run = recorded.get(row["run_id"])
        if run is None:
            raise ValueError(f"labeled run missing from recording: {row['run_id']}")
        predicted = classify_run(run)
        if predicted != row["classifier_label"]:
            raise ValueError(
                f"classifier mismatch for {row['run_id']}: "
                f"recorded label={row['classifier_label']}, current={predicted}"
            )
    metrics = calculate_metrics(labels)
    print(f"n={metrics['n']} accuracy={metrics['accuracy']:.3f} "
          f"micro-P={metrics['micro_precision']:.3f} "
          f"micro-R={metrics['micro_recall']:.3f}")
    for label, values in metrics["classes"].items():
        precision = "n/a" if values["precision"] is None else f"{values['precision']:.3f}"
        recall = "n/a" if values["recall"] is None else f"{values['recall']:.3f}"
        print(f"{label}: support={values['support']} predicted={values['predicted']} "
              f"precision={precision} recall={recall}")


if __name__ == "__main__":
    main()
