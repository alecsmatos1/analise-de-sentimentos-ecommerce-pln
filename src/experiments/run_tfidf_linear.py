"""Experimento E1 da v2: TF-IDF + modelo linear.

Esta funcao recebe dados ja carregados e um split ja definido. Nao baixa
recursos, nao escreve em disco e nao depende de modulos externos a `v2/`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, Union

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.pipeline import Pipeline

from ..models.linear import LinearModelConfig, build_linear_model
from ..representations.tfidf import TfidfConfig, build_tfidf

DEFAULT_LABEL_ORDER: tuple[str, ...] = ("negativo", "neutro", "positivo")
_METRIC_DECIMALS = 4


@dataclass(frozen=True)
class TfidfLinearResult:
    """Resultado do experimento TF-IDF + linear.

    O campo `pipeline` e dado interno do experimento: fica disponivel para
    reuso (re-score, debug, inspecao de features) mas nao deve aparecer em
    serializacoes (CSV/JSON) — por isso esta fora de `as_dict()`.
    """

    accuracy: float
    precision_macro: float
    recall_macro: float
    f1_macro: float
    confusion_matrix: list[list[int]]
    label_order: tuple[str, ...]
    model_name: str
    pipeline: Pipeline

    def as_dict(self) -> dict:
        """Projecao serializavel compativel com o contrato de `reporting.py`."""

        return {
            "accuracy": self.accuracy,
            "precision_macro": self.precision_macro,
            "recall_macro": self.recall_macro,
            "f1_macro": self.f1_macro,
            "confusion_matrix": self.confusion_matrix,
            "labels": list(self.label_order),
        }


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

    Contrato de entrada textual:
        `X_train` e `X_test` devem ser sequencias de textos ja normalizados
        — tipicamente a coluna `clean_text` produzida por `coerce_corpus()`
        / `normalize_text` da sprint core-data. Nao passe a coluna `text`
        crua: o TF-IDF foi calibrado para o texto limpo do contrato canonico
        da v2.
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
        accuracy=round(float(accuracy), _METRIC_DECIMALS),
        precision_macro=round(float(precision), _METRIC_DECIMALS),
        recall_macro=round(float(recall), _METRIC_DECIMALS),
        f1_macro=round(float(f1), _METRIC_DECIMALS),
        confusion_matrix=matrix.tolist(),
        label_order=tuple(labels),
        model_name=resolved_model,
        pipeline=pipeline,
    )


def run(corpus_path: Union[Path, str, None] = None) -> dict:
    """Entry point para `run_experiment.py` no modo nao-fixture.

    Carrega o corpus processado, faz split estratificado, treina e avalia o
    pipeline TF-IDF + modelo linear.

    Args:
        corpus_path: caminho para o CSV/parquet do corpus processado. Se
            ``None``, tenta usar ``v2.src.config.PROCESSED_CORPUS_PATH``.

    Returns:
        dict serializavel compativel com ``reporting.save_all`` (mesmo
        contrato de ``TfidfLinearResult.as_dict()``).

    Raises:
        FileNotFoundError: se o corpus nao for fornecido nem estiver definido
            em ``config``, ou se o arquivo apontado nao existir.
    """

    import pandas as pd

    from .. import config as _config
    from ..data import coerce_corpus
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

    if corpus_path.suffix.lower() == ".csv":
        df = pd.read_csv(corpus_path)
    else:
        df = pd.read_parquet(corpus_path)

    corpus = coerce_corpus(df)
    split = stratified_split(corpus)

    result = run_tfidf_linear(
        split.train["clean_text"].tolist(),
        split.train["label"].tolist(),
        split.test["clean_text"].tolist(),
        split.test["label"].tolist(),
    )
    return result.as_dict()


__all__ = [
    "DEFAULT_LABEL_ORDER",
    "TfidfLinearResult",
    "build_tfidf_linear_pipeline",
    "run",
    "run_tfidf_linear",
]
