"""Experimento E4 da v2: BERTimbau fine-tuned para classificacao.

Faz fine-tuning completo do BERTimbau (``neuralmind/bert-base-portuguese-cased``)
na tarefa de classificacao de sentimentos em 3 classes
(``negativo``/``neutro``/``positivo``). Em contraste com o experimento E3
(``run_bert_embeddings_linear.py``), aqui o encoder e treinado junto com o
classification head.

O contrato segue o padrao dos outros ``run_*.py``: a funcao ``run()`` carrega
o corpus processado, aplica o split estratificado padrao e devolve um dict
serializavel; ``run_bert_finetune()`` aceita ``X_train/X_test`` ja prontos e
devolve um ``BertFinetuneResult``.
"""

from __future__ import annotations

import platform
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence, Union

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)

from ..models.bertimbau_finetune import (
    DEFAULT_LABEL_ORDER,
    FinetuneConfig,
    train_and_evaluate,
)

_METRIC_DECIMALS = 4


@dataclass(frozen=True)
class BertFinetuneResult:
    """Resultado do experimento BERTimbau fine-tuned.

    Mesmo contrato serializavel adotado pelos demais experimentos. O campo
    ``meta`` carrega informacoes especificas (epochs, modelo, hardware, tempo)
    requeridas pela sprint S07.
    """

    accuracy: float
    precision_macro: float
    recall_macro: float
    f1_macro: float
    labels: list
    per_class: dict = field(default_factory=dict)
    confusion_matrix: list = field(default_factory=list)
    meta: dict = field(default_factory=dict)

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
            "meta": dict(self.meta),
        }


def run_bert_finetune(
    X_train: Sequence[str],
    y_train: Sequence[str],
    X_test: Sequence[str],
    y_test: Sequence[str],
    *,
    config: Optional[FinetuneConfig] = None,
    label_order: Sequence[str] = DEFAULT_LABEL_ORDER,
) -> BertFinetuneResult:
    """Faz fine-tuning do BERTimbau e devolve metricas no conjunto de teste."""

    cfg = config or FinetuneConfig()
    labels = list(label_order)

    X_train_list = list(X_train)
    y_train_list = list(y_train)
    X_test_list = list(X_test)
    y_test_list = list(y_test)

    output = train_and_evaluate(
        X_train_list,
        y_train_list,
        X_test_list,
        y_test_list,
        config=cfg,
        label_order=labels,
    )
    predictions = output.predictions

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

    meta = {
        "epochs": cfg.num_epochs,
        "model": cfg.model_name,
        "hardware": output.device_used,
        "training_time_seconds": output.training_time_seconds,
        "batch_size": cfg.batch_size,
        "learning_rate": cfg.learning_rate,
        "max_length": cfg.max_length,
        "seed": cfg.random_state,
        "python_version": platform.python_version(),
    }

    return BertFinetuneResult(
        accuracy=round(float(accuracy), _METRIC_DECIMALS),
        precision_macro=round(float(precision), _METRIC_DECIMALS),
        recall_macro=round(float(recall), _METRIC_DECIMALS),
        f1_macro=round(float(f1), _METRIC_DECIMALS),
        labels=labels,
        per_class=per_class,
        confusion_matrix=matrix.tolist(),
        meta=meta,
    )


def run(corpus_path: Union[Path, str, None] = None) -> dict:
    """Entry point para ``run_experiment.py`` no modo nao-fixture.

    Args:
        corpus_path: caminho para o CSV/parquet do corpus processado. Se
            ``None``, tenta usar ``v2.src.config.PROCESSED_CORPUS_PATH``.

    Returns:
        dict serializavel compativel com ``reporting.save_all``.

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
    result = run_bert_finetune(
        split.train["clean_text"].tolist(),
        split.train["label"].tolist(),
        split.test["clean_text"].tolist(),
        split.test["label"].tolist(),
    )
    return result.as_dict()


__all__ = [
    "BertFinetuneResult",
    "run",
    "run_bert_finetune",
]
