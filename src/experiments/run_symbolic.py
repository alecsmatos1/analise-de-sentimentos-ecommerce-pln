"""Experimento E+ : analisador simbolico da v1 avaliado no corpus v2.

Reaproveita ``analyze_symbolic_sentiment`` definido em ``sentiment_analyzer.py``
(raiz do repositorio, v1) e avalia sobre o mesmo conjunto de teste produzido
pelo split estratificado da v2 (``v2/src/splitting.py``), para permitir
comparacao direta com TF-IDF, Word2Vec e BERTimbau.

A v1 nao expoe uma classe ``SentimentAnalyzer``; apenas a funcao
``analyze_symbolic_sentiment(text) -> dict``. Para preservar o contrato
sugerido na sprint (``analyzer.analyze(text) -> str``) e permitir
mockagem nos testes, ``_load_symbolic_analyzer`` retorna um adaptador
fino com metodo ``analyze``.
"""
from __future__ import annotations

import importlib.util
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Union

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)


_METRIC_DECIMALS = 4
LABEL_ORDER: list[str] = ["negativo", "neutro", "positivo"]

# Mapeamento defensivo: o analisador da v1 ja retorna labels em portugues
# ("negativo", "neutro", "positivo"), mas mantemos o map para tolerar mocks
# em testes que retornem versoes em ingles ou em caixa-alta.
_LABEL_MAP: dict[str, str] = {
    "positive": "positivo",
    "negative": "negativo",
    "neutral": "neutro",
    "positivo": "positivo",
    "negativo": "negativo",
    "neutro": "neutro",
}


@dataclass(frozen=True)
class SymbolicResult:
    """Resultado do analisador simbolico no contrato dos demais experimentos."""

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


class _SymbolicAdapter:
    """Adapta ``analyze_symbolic_sentiment`` para o contrato ``analyze(text) -> str``."""

    def __init__(self, fn) -> None:
        self._fn = fn

    def analyze(self, text: str) -> str:
        result = self._fn(text)
        if isinstance(result, dict):
            return str(result.get("label", "neutro"))
        return str(result)


def _load_symbolic_analyzer():
    """Carrega o analisador simbolico da v1 sem depender de ``__init__.py``.

    Importa ``sentiment_analyzer.py`` da raiz do repositorio via
    ``importlib.util`` para evitar acoplamento com a estrutura de pacotes da
    v1 (que nao expoe ``__init__.py``). Retorna um adaptador com metodo
    ``analyze(text) -> str``.
    """
    repo_root = Path(__file__).resolve().parents[3]
    analyzer_path = repo_root / "sentiment_analyzer.py"
    if not analyzer_path.exists():
        raise FileNotFoundError(
            f"sentiment_analyzer.py nao encontrado em {repo_root}. "
            "Verifique se a v1 esta presente no repositorio."
        )
    spec = importlib.util.spec_from_file_location(
        "sentiment_analyzer_v1", analyzer_path
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Falha ao criar spec para {analyzer_path}.")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    fn = getattr(mod, "analyze_symbolic_sentiment", None)
    if fn is None:
        raise AttributeError(
            "sentiment_analyzer.py nao expoe analyze_symbolic_sentiment(). "
            "Assinatura esperada pela sprint v2-s08-symbolic-v1 nao encontrada."
        )
    return _SymbolicAdapter(fn)


def run(corpus_path: Union[Path, str, None] = None) -> SymbolicResult:
    """Avalia o analisador simbolico no conjunto de teste da v2.

    Args:
        corpus_path: Caminho para o corpus processado. Se ``None``, usa
            ``v2.src.config.PROCESSED_CORPUS_PATH``.

    Returns:
        ``SymbolicResult`` com metricas macro, per-class e matriz de confusao.

    Raises:
        FileNotFoundError: se nao houver corpus disponivel para avaliar.
    """
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
    test_df = split.test

    analyzer = _load_symbolic_analyzer()
    y_true = test_df["label"].tolist()
    y_pred_raw = [analyzer.analyze(text) for text in test_df["raw_text"].tolist()]
    y_pred = [_LABEL_MAP.get(str(p).lower(), "neutro") for p in y_pred_raw]

    label_list = LABEL_ORDER
    report = classification_report(
        y_true, y_pred, labels=label_list, output_dict=True, zero_division=0
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
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", labels=label_list, zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=label_list).tolist()

    return SymbolicResult(
        accuracy=round(float(accuracy_score(y_true, y_pred)), _METRIC_DECIMALS),
        precision_macro=round(float(prec), _METRIC_DECIMALS),
        recall_macro=round(float(rec), _METRIC_DECIMALS),
        f1_macro=round(float(f1), _METRIC_DECIMALS),
        labels=label_list,
        per_class=per_class,
        confusion_matrix=cm,
    )


__all__ = ["SymbolicResult", "run"]
