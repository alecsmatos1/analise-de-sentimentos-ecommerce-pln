"""Experimento E3 da v2: BERTimbau (frozen) + modelo linear.

Extrai embeddings do BERTimbau (`neuralmind/bert-base-portuguese-cased`) usando
mean pooling e treina um classificador linear (LogisticRegression por padrao)
sobre essas representacoes. O modelo BERT permanece congelado: apenas o
classificador e treinado.

Segue o mesmo contrato de ``run_tfidf_linear.py``: recebe um split ja definido
ou e invocado via ``run(corpus_path=...)`` para carregar o corpus processado.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence, Union

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.pipeline import Pipeline

from ..models.linear import LinearModelConfig, build_linear_model
from ..representations.bertimbau_embeddings import (
    BertimbauConfig,
    BertimbauVectorizer,
)

DEFAULT_LABEL_ORDER: tuple[str, ...] = ("negativo", "neutro", "positivo")
_METRIC_DECIMALS = 4


@dataclass(frozen=True)
class BertEmbeddingsResult:
    """Resultado do experimento BERTimbau (frozen) + modelo linear.

    Mesmo contrato serializavel adotado pelos demais experimentos. O campo
    ``pipeline`` fica disponivel para inspecao/reuso mas nao entra em
    ``as_dict()`` (nao serializavel).
    """

    accuracy: float
    precision_macro: float
    recall_macro: float
    f1_macro: float
    labels: list
    per_class: dict = field(default_factory=dict)
    confusion_matrix: list = field(default_factory=list)
    pipeline: Optional[Pipeline] = None

    def as_dict(self) -> dict:
        return {
            "accuracy": self.accuracy,
            "precision_macro": self.precision_macro,
            "recall_macro": self.recall_macro,
            "f1_macro": self.f1_macro,
            "confusion_matrix": self.confusion_matrix,
            "labels": list(self.labels),
            "per_class": {
                label: dict(metrics) for label, metrics in self.per_class.items()
            },
        }


def build_bert_linear_pipeline(
    bert_config: Optional[BertimbauConfig] = None,
    model_config: Optional[LinearModelConfig] = None,
) -> Pipeline:
    """Monta a pipeline ``bertimbau -> modelo linear``."""

    return Pipeline(
        [
            ("bertimbau", BertimbauVectorizer(bert_config)),
            ("model", build_linear_model(model_config)),
        ]
    )


def run_bert_embeddings_linear(
    X_train: Sequence[str],
    y_train: Sequence[str],
    X_test: Sequence[str],
    y_test: Sequence[str],
    *,
    bert_config: Optional[BertimbauConfig] = None,
    model_config: Optional[LinearModelConfig] = None,
    label_order: Sequence[str] = DEFAULT_LABEL_ORDER,
) -> BertEmbeddingsResult:
    """Treina BERTimbau (frozen) + modelo linear e devolve metricas.

    Contrato de entrada textual: ``X_train`` e ``X_test`` devem ser sequencias
    de textos ja normalizados (``clean_text`` do corpus canonico da v2).
    """

    pipeline = build_bert_linear_pipeline(bert_config, model_config)

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
    report = classification_report(
        y_test_list,
        predictions,
        labels=labels,
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
        for label in labels
        if label in report
    }

    return BertEmbeddingsResult(
        accuracy=round(float(accuracy), _METRIC_DECIMALS),
        precision_macro=round(float(precision), _METRIC_DECIMALS),
        recall_macro=round(float(recall), _METRIC_DECIMALS),
        f1_macro=round(float(f1), _METRIC_DECIMALS),
        labels=labels,
        per_class=per_class,
        confusion_matrix=matrix.tolist(),
        pipeline=pipeline,
    )


def run(corpus_path: Union[Path, str, None] = None) -> dict:
    """Entry point para ``run_experiment.py`` no modo nao-fixture.

    Args:
        corpus_path: caminho para o CSV/parquet do corpus processado. Se
            ``None``, tenta usar ``v2.src.config.PROCESSED_CORPUS_PATH``.

    Returns:
        dict serializavel compativel com ``reporting.save_all`` (mesmo
        contrato de ``BertEmbeddingsResult.as_dict()``).

    Raises:
        FileNotFoundError: se o corpus nao for fornecido nem definido em
            ``config``, ou se o arquivo apontado nao existir.
    """

    import pandas as pd

    from .. import config as _config
    from ..splitting import stratified_split

    if corpus_path is None:
        corpus_path = getattr(_config, "PROCESSED_CORPUS_PATH", None)
    if corpus_path is None:
        raise FileNotFoundError(
            "corpus_path nao fornecido e config.PROCESSED_CORPUS_PATH nao "
            "definido. Use --corpus-path ou defina PROCESSED_CORPUS_PATH em "
            "config.py."
        )
    corpus_path = Path(corpus_path)
    if not corpus_path.exists():
        raise FileNotFoundError(
            f"Corpus nao encontrado: {corpus_path}. "
            "Prepare o corpus processado antes de executar o experimento."
        )

    ext = corpus_path.suffix.lower()
    if ext == ".csv":
        df = pd.read_csv(corpus_path)
    elif ext == ".parquet":
        df = pd.read_parquet(corpus_path)
    else:
        raise ValueError(
            f"Formato nao suportado: '{ext}' em '{corpus_path.name}'. "
            "Formatos suportados: .csv, .parquet."
        )

    split = stratified_split(df)
    result = run_bert_embeddings_linear(
        split.train["clean_text"].tolist(),
        split.train["label"].tolist(),
        split.test["clean_text"].tolist(),
        split.test["label"].tolist(),
    )
    return result.as_dict()


__all__ = [
    "DEFAULT_LABEL_ORDER",
    "BertEmbeddingsResult",
    "build_bert_linear_pipeline",
    "run",
    "run_bert_embeddings_linear",
]
