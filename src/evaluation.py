"""Padronized evaluation metrics for v2 experiments.

All experiments must report accuracy, macro precision/recall/F1 and the
confusion matrix over the fixed label set ``(negativo, neutro, positivo)``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)


LABELS: tuple[str, ...] = ("negativo", "neutro", "positivo")


@dataclass(frozen=True)
class EvaluationResult:
    accuracy: float
    precision_macro: float
    recall_macro: float
    f1_macro: float
    confusion_matrix: list[list[int]]
    labels: tuple[str, ...] = LABELS

    def as_dict(self) -> dict:
        return {
            "accuracy": self.accuracy,
            "precision_macro": self.precision_macro,
            "recall_macro": self.recall_macro,
            "f1_macro": self.f1_macro,
            "labels": list(self.labels),
            "confusion_matrix": [list(row) for row in self.confusion_matrix],
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
    return EvaluationResult(
        accuracy=round(acc, 4),
        precision_macro=round(float(precision), 4),
        recall_macro=round(float(recall), 4),
        f1_macro=round(float(f1), 4),
        confusion_matrix=[[int(value) for value in row] for row in matrix],
        labels=tuple(label_list),
    )
