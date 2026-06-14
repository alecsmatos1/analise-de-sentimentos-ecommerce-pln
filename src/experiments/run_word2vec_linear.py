"""Experimento E2 da v2: Word2Vec NILC + modelo linear.

Treina um classificador linear (LogisticRegression por padrao) sobre os
vetores produzidos por ``Word2VecVectorizer`` (media dos embeddings NILC
pre-treinados). Expoe ``run()`` no mesmo contrato dos demais experimentos
para que ``v2/run_experiment.py --experiment word2vec-linear`` produza um
JSON com a mesma estrutura de ``tfidf-linear.json``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Union

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)

from ..models.linear import LinearModelConfig, build_linear_model
from ..representations.word2vec_nilc import Word2VecConfig, Word2VecVectorizer

_METRIC_DECIMALS = 4


@dataclass(frozen=True)
class Word2VecLinearResult:
    """Resultado do experimento Word2Vec NILC + modelo linear."""

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


def run(
    corpus_path: Union[Path, str, None] = None,
    *,
    word2vec_config: Word2VecConfig | None = None,
    model_config: LinearModelConfig | None = None,
) -> Word2VecLinearResult:
    """Treina Word2Vec NILC + modelo linear e avalia no conjunto de teste.

    Args:
        corpus_path: Caminho para o corpus CSV. Se ``None``, usa
            ``v2.src.config.PROCESSED_CORPUS_PATH``.
        word2vec_config: Configuracao opcional do vectorizer NILC.
        model_config: Configuracao opcional do modelo linear.

    Returns:
        ``Word2VecLinearResult`` com metricas macro, per-class e matriz de
        confusao.

    Raises:
        FileNotFoundError: se o corpus nao for fornecido nem definido em
            ``config``, ou se o arquivo apontado nao existir.
    """
    import pandas as pd

    from .. import config as _config
    from ..data import coerce_corpus
    from ..splitting import stratified_split

    if corpus_path is None:
        corpus_path = getattr(_config, "PROCESSED_CORPUS_PATH", None)
    if corpus_path is None:
        raise FileNotFoundError(
            "corpus_path nao fornecido e config.PROCESSED_CORPUS_PATH nao definido."
        )

    corpus_path = Path(corpus_path)
    if not corpus_path.exists():
        raise FileNotFoundError(f"Corpus nao encontrado: {corpus_path}")

    df = coerce_corpus(pd.read_csv(corpus_path))
    split = stratified_split(df)

    X_train = split.train["clean_text"].tolist()
    y_train = split.train["label"].tolist()
    X_test = split.test["clean_text"].tolist()
    y_test = split.test["label"].tolist()

    vectorizer = Word2VecVectorizer(word2vec_config)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    clf = build_linear_model(model_config)
    clf.fit(X_train_vec, y_train)
    y_pred = clf.predict(X_test_vec)

    label_list = sorted(set(y_train).union(y_test))
    report = classification_report(
        y_test, y_pred, labels=label_list, output_dict=True, zero_division=0
    )
    per_class = {
        lbl: {
            "precision": round(float(report[lbl]["precision"]), _METRIC_DECIMALS),
            "recall": round(float(report[lbl]["recall"]), _METRIC_DECIMALS),
            "f1": round(float(report[lbl]["f1-score"]), _METRIC_DECIMALS),
            "support": int(report[lbl]["support"]),
        }
        for lbl in label_list
        if lbl in report
    }
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average="macro", labels=label_list, zero_division=0
    )
    cm = confusion_matrix(y_test, y_pred, labels=label_list).tolist()

    return Word2VecLinearResult(
        accuracy=round(float(accuracy_score(y_test, y_pred)), _METRIC_DECIMALS),
        precision_macro=round(float(precision), _METRIC_DECIMALS),
        recall_macro=round(float(recall), _METRIC_DECIMALS),
        f1_macro=round(float(f1), _METRIC_DECIMALS),
        labels=label_list,
        per_class=per_class,
        confusion_matrix=cm,
    )


__all__ = ["Word2VecLinearResult", "run"]
