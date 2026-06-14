"""Experimento E1 da v2: TF-IDF + modelo linear.

Esta funcao recebe dados ja carregados e um split ja definido. Nao baixa
recursos, nao escreve em disco e nao depende de modulos externos a `v2/`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.pipeline import Pipeline

from ..models.linear import LinearModelConfig, build_linear_model
from ..representations.tfidf import TfidfConfig, build_tfidf

DEFAULT_LABEL_ORDER: tuple[str, ...] = ("negativo", "neutro", "positivo")


@dataclass(frozen=True)
class TfidfLinearResult:
    """Resultado do experimento TF-IDF + linear."""

    accuracy: float
    precision_macro: float
    recall_macro: float
    f1_macro: float
    confusion_matrix: list[list[int]]
    label_order: tuple[str, ...]
    model_name: str
    pipeline: Pipeline


def build_tfidf_linear_pipeline(
    tfidf_config: Optional[TfidfConfig] = None,
    model_config: Optional[LinearModelConfig] = None,
) -> Pipeline:
    """Monta a pipeline `tfidf -> modelo linear`."""

    return Pipeline(
        [
            ("tfidf", build_tfidf(tfidf_config)),
            ("model", build_linear_model(model_config)),
        ]
    )


def run_tfidf_linear(
    X_train: Sequence[str],
    y_train: Sequence[str],
    X_test: Sequence[str],
    y_test: Sequence[str],
    *,
    tfidf_config: Optional[TfidfConfig] = None,
    model_config: Optional[LinearModelConfig] = None,
    label_order: Sequence[str] = DEFAULT_LABEL_ORDER,
) -> TfidfLinearResult:
    """Treina TF-IDF + modelo linear e devolve metricas + pipeline treinada.

    Os argumentos `X_*` e `y_*` representam um split ja realizado por outro
    componente (por exemplo, `v2/src/splitting.py` quando estiver disponivel).
    """

    pipeline = build_tfidf_linear_pipeline(tfidf_config, model_config)

    X_train_list = list(X_train)
    y_train_list = list(y_train)
    X_test_list = list(X_test)
    y_test_list = list(y_test)
    labels = list(label_order)

    pipeline.fit(X_train_list, y_train_list)
    predictions = pipeline.predict(X_test_list)

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test_list,
        predictions,
        average="macro",
        zero_division=0,
        labels=labels,
    )
    accuracy = accuracy_score(y_test_list, predictions)
    matrix = confusion_matrix(y_test_list, predictions, labels=labels)

    resolved_model = (model_config or LinearModelConfig()).name

    return TfidfLinearResult(
        accuracy=float(accuracy),
        precision_macro=float(precision),
        recall_macro=float(recall),
        f1_macro=float(f1),
        confusion_matrix=matrix.tolist(),
        label_order=tuple(labels),
        model_name=resolved_model,
        pipeline=pipeline,
    )


__all__ = [
    "DEFAULT_LABEL_ORDER",
    "TfidfLinearResult",
    "build_tfidf_linear_pipeline",
    "run_tfidf_linear",
]
