"""Tests for v2/src/evaluation.py."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import evaluation  # noqa: E402


def test_labels_are_fixed_in_canonical_order():
    assert evaluation.LABELS == ("negativo", "neutro", "positivo")


def test_evaluate_perfect_predictions():
    y_true = ["negativo", "neutro", "positivo", "positivo", "negativo"]
    y_pred = list(y_true)

    result = evaluation.evaluate(y_true, y_pred)

    assert result.accuracy == 1.0
    assert result.precision_macro == 1.0
    assert result.recall_macro == 1.0
    assert result.f1_macro == 1.0
    assert result.labels == evaluation.LABELS
    assert result.confusion_matrix == [
        [2, 0, 0],
        [0, 1, 0],
        [0, 0, 2],
    ]


def test_evaluate_handles_mistakes_and_uses_macro_average():
    y_true = ["negativo", "negativo", "neutro", "positivo"]
    y_pred = ["negativo", "neutro", "neutro", "positivo"]

    result = evaluation.evaluate(y_true, y_pred)

    assert result.accuracy == 0.75
    assert 0.0 < result.f1_macro < 1.0
    assert result.confusion_matrix == [
        [1, 1, 0],
        [0, 1, 0],
        [0, 0, 1],
    ]


def test_evaluate_returns_zero_when_class_absent_from_predictions():
    y_true = ["negativo", "neutro", "positivo"]
    y_pred = ["negativo", "negativo", "negativo"]

    result = evaluation.evaluate(y_true, y_pred)

    assert result.accuracy == pytest.approx(1 / 3, abs=1e-4)
    assert result.f1_macro < result.accuracy
    assert result.confusion_matrix == [
        [1, 0, 0],
        [1, 0, 0],
        [1, 0, 0],
    ]


def test_evaluate_rejects_length_mismatch():
    with pytest.raises(ValueError):
        evaluation.evaluate(["positivo"], ["positivo", "negativo"])


def test_evaluate_rejects_empty_input():
    with pytest.raises(ValueError):
        evaluation.evaluate([], [])


def test_as_dict_preserves_metrics_and_matrix():
    result = evaluation.evaluate(
        ["positivo", "negativo"], ["positivo", "negativo"]
    )

    payload = result.as_dict()

    assert payload["accuracy"] == 1.0
    assert payload["labels"] == list(evaluation.LABELS)
    assert payload["confusion_matrix"] == [
        [1, 0, 0],
        [0, 0, 0],
        [0, 0, 1],
    ]


def test_evaluate_retorna_per_class():
    """evaluate() deve incluir metricas por classe no resultado."""
    y_true = ["negativo", "neutro", "positivo", "positivo", "negativo", "neutro"]
    y_pred = ["negativo", "neutro", "positivo", "negativo", "negativo", "positivo"]

    result = evaluation.evaluate(y_true, y_pred)

    assert hasattr(result, "per_class")
    pc = result.per_class
    for label in ("negativo", "neutro", "positivo"):
        assert label in pc
        for campo in ("precision", "recall", "f1", "support"):
            assert campo in pc[label]


def test_per_class_metricas_arredondadas():
    """Metricas de per_class devem ter no maximo 4 casas decimais."""
    y_true = ["negativo", "neutro", "positivo", "positivo", "negativo", "neutro"]
    y_pred = ["negativo", "neutro", "positivo", "negativo", "negativo", "positivo"]

    result = evaluation.evaluate(y_true, y_pred, labels=evaluation.LABELS)
    pc = result.per_class
    for label, metrics in pc.items():
        for campo in ("precision", "recall", "f1"):
            val = metrics[campo]
            assert val == round(val, 4)
        assert isinstance(metrics["support"], int)


def test_as_dict_inclui_per_class():
    """as_dict() deve conter chave per_class."""
    y_true = ["negativo", "neutro", "positivo", "positivo"]
    y_pred = ["negativo", "neutro", "positivo", "negativo"]

    result = evaluation.evaluate(y_true, y_pred, labels=evaluation.LABELS)
    d = result.as_dict()

    assert "per_class" in d
    assert isinstance(d["per_class"], dict)
    for label in evaluation.LABELS:
        assert label in d["per_class"]
