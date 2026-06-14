"""Padronized evaluation metrics for v2 experiments.

All experiments must report accuracy, macro precision/recall/F1 and the
confusion matrix over the fixed label set ``(negativo, neutro, positivo)``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)


LABELS: tuple[str, ...] = ("negativo", "neutro", "positivo")
_METRIC_DECIMALS = 4


@dataclass(frozen=True)
class EvaluationResult:
    accuracy: float
    precision_macro: float
    recall_macro: float
    f1_macro: float
    confusion_matrix: list[list[int]]
    labels: tuple[str, ...] = LABELS
    per_class: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "accuracy": self.accuracy,
            "precision_macro": self.precision_macro,
            "recall_macro": self.recall_macro,
            "f1_macro": self.f1_macro,
            "labels": list(self.labels),
            "confusion_matrix": [list(row) for row in self.confusion_matrix],
            "per_class": {
                label: dict(metrics) for label, metrics in self.per_class.items()
            },
        }


def evaluate(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    labels: Sequence[str] = LABELS,
) -> EvaluationResult:
    """Compute the standard metrics over a fixed label order.

    Unknown labels in ``y_true`` or ``y_pred`` are accepted by scikit-learn but
    do not contribute to per-class scores; this is intentional because the v2
    contract requires the canonical three classes.
    """
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length")
    if len(y_true) == 0:
        raise ValueError("y_true must contain at least one element")

    label_list = list(labels)
    acc = float(accuracy_score(y_true, y_pred))
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=label_list,
        average="macro",
        zero_division=0,
    )
    matrix = confusion_matrix(y_true, y_pred, labels=label_list).tolist()
    report = classification_report(
        y_true,
        y_pred,
        labels=label_list,
        output_dict=True,
        zero_division=0,
    )
    per_class = {
        label: {
            "precision": round(float(report[label]["precision"]), _METRIC_DECIMALS),
            "recall": round(float(report[label]["recall"]), _METRIC_DECIMALS),
            "f1": round(float(report[label]["f1-score"]), _METRIC_DECIMALS),
            "support": int(report[label]["support"]),
        }
        for label in label_list
        if label in report
    }
    return EvaluationResult(
        accuracy=round(acc, _METRIC_DECIMALS),
        precision_macro=round(float(precision), _METRIC_DECIMALS),
        recall_macro=round(float(recall), _METRIC_DECIMALS),
        f1_macro=round(float(f1), _METRIC_DECIMALS),
        confusion_matrix=[[int(value) for value in row] for row in matrix],
        labels=tuple(label_list),
        per_class=per_class,
    )
