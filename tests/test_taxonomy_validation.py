from scripts.validate_taxonomy import calculate_metrics


def test_taxonomy_metrics_report_class_precision_recall_and_micro_scores():
    rows = [
        {"hand_label": "other", "classifier_label": "other"},
        {"hand_label": "other", "classifier_label": "false-success"},
        {"hand_label": "false-success", "classifier_label": "false-success"},
    ]

    metrics = calculate_metrics(rows)

    assert metrics["accuracy"] == 2 / 3
    assert metrics["classes"]["other"] == {
        "support": 2, "predicted": 1, "true_positive": 1,
        "precision": 1.0, "recall": 0.5,
    }
