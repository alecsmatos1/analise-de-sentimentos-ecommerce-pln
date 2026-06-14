"""Baseline: classificador majoritário (Most Frequent).

Serve como ponto de comparação mínimo para os experimentos da v2: prediz
sempre a classe mais frequente do conjunto de treino. Expõe `run()` no mesmo
contrato dos demais experimentos (``v2/src/experiments/run_tfidf_linear.py``)
para que ``v2/run_experiment.py --experiment baseline`` produza um JSON com a
mesma estrutura de ``tfidf-linear.json``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


@dataclass
class BaselineResult:
    """Resultado do baseline majoritário, no contrato de reporting."""

    accuracy: float
    precision_macro: float
    recall_macro: float
    f1_macro: float
    labels: list[str]
    per_class: dict[str, Any] = field(default_factory=dict)
    confusion_matrix: list[list[int]] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "accuracy": self.accuracy,
            "precision_macro": self.precision_macro,
            "recall_macro": self.recall_macro,
            "f1_macro": self.f1_macro,
            "labels": self.labels,
            "per_class": self.per_class,
            "confusion_matrix": self.confusion_matrix,
        }


def run(corpus_path: str | Path | None = None) -> BaselineResult:
    """Treina ``DummyClassifier(most_frequent)`` e avalia no conjunto de teste.

    Args:
        corpus_path: Caminho para o corpus CSV. Se ``None``, usa
            ``v2.src.config.PROCESSED_CORPUS_PATH``.

    Returns:
        ``BaselineResult`` com métricas macro, per-class e matriz de confusão.

    Raises:
        FileNotFoundError: se ``corpus_path`` não for fornecido nem definido em
            ``config``, ou se o arquivo apontado não existir.
    """
    from v2.src import config
    from v2.src.splitting import stratified_split

    if corpus_path is None:
        corpus_path = getattr(config, "PROCESSED_CORPUS_PATH", None)
    if corpus_path is None:
        raise FileNotFoundError(
            "corpus_path nao fornecido e config.PROCESSED_CORPUS_PATH nao definido."
        )

    corpus_path = Path(corpus_path)
    if not corpus_path.exists():
        raise FileNotFoundError(f"Corpus nao encontrado: {corpus_path}")

    df = pd.read_csv(corpus_path)

    split = stratified_split(df)
    train_df = split.train
    test_df = split.test

    label_list = sorted(set(train_df["label"]).union(test_df["label"]))

    X_train = train_df["clean_text"].tolist()
    y_train = train_df["label"].tolist()
    X_test = test_df["clean_text"].tolist()
    y_test = test_df["label"].tolist()

    clf = DummyClassifier(strategy="most_frequent", random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    report = classification_report(
        y_test, y_pred, labels=label_list, output_dict=True, zero_division=0
    )
    per_class = {
        lbl: {
            "precision": round(report[lbl]["precision"], 4),
            "recall": round(report[lbl]["recall"], 4),
            "f1": round(report[lbl]["f1-score"], 4),
            "support": int(report[lbl]["support"]),
        }
        for lbl in label_list
        if lbl in report
    }

    cm = confusion_matrix(y_test, y_pred, labels=label_list).tolist()

    return BaselineResult(
        accuracy=round(accuracy_score(y_test, y_pred), 4),
        precision_macro=round(
            precision_score(
                y_test, y_pred, average="macro", labels=label_list, zero_division=0
            ),
            4,
        ),
        recall_macro=round(
            recall_score(
                y_test, y_pred, average="macro", labels=label_list, zero_division=0
            ),
            4,
        ),
        f1_macro=round(
            f1_score(
                y_test, y_pred, average="macro", labels=label_list, zero_division=0
            ),
            4,
        ),
        labels=label_list,
        per_class=per_class,
        confusion_matrix=cm,
    )


__all__ = ["BaselineResult", "run"]
