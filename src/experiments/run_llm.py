"""Experimento E5: classificacao zero-shot via LLM.

Avalia o ``LLMClassifier`` (Google Gemini, ver
``v2/src/models/llm_classifier.py``) em uma amostra estratificada do conjunto
de teste produzido por ``v2/src/splitting.py``. O modelo padrao usado e o
default da ``LLMConfig`` (``gemini-2.5-flash-lite``), reflexo da migracao
OpenAI -> Gemini consolidada em commits anteriores (ver historico).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    precision_recall_fscore_support,
)

from ..models.llm_classifier import LLMClassifier, LLMConfig

_METRIC_DECIMALS = 4
_SAMPLE_SIZE = 500
_RANDOM_STATE = 42

LABEL_ORDER = ["negativo", "neutro", "positivo"]


@dataclass(frozen=True)
class LLMResult:
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
            "labels": self.labels,
            "per_class": self.per_class,
            "confusion_matrix": self.confusion_matrix,
            "meta": self.meta,
        }


def run(corpus_path: Union[Path, str, None] = None,
        sample_size: int = _SAMPLE_SIZE) -> LLMResult:
    """Classifica amostra do conjunto de teste via LLM."""
    from .. import config as _config
    from ..data import coerce_corpus
    from ..splitting import stratified_split

    if corpus_path is None:
        corpus_path = getattr(_config, "PROCESSED_CORPUS_PATH", None)
    if corpus_path is None or not Path(corpus_path).exists():
        raise FileNotFoundError("Corpus nao encontrado.")

    df = coerce_corpus(pd.read_csv(corpus_path))
    split = stratified_split(df)
    test_df = split.test

    rng = np.random.default_rng(_RANDOM_STATE)
    sample_per_class = sample_size // len(LABEL_ORDER)
    frames = []
    for lbl in LABEL_ORDER:
        chunk = test_df[test_df["label"] == lbl]
        n = min(sample_per_class, len(chunk))
        idx = rng.choice(len(chunk), size=n, replace=False)
        frames.append(chunk.iloc[idx])
    sample_df = pd.concat(frames, ignore_index=True).sample(
        frac=1, random_state=_RANDOM_STATE
    ).reset_index(drop=True)

    texts = sample_df["raw_text"].tolist()
    y_true = sample_df["label"].tolist()

    cfg = LLMConfig(temperature=0.0)
    clf = LLMClassifier(cfg)
    y_pred = clf.predict(texts)

    label_list = LABEL_ORDER
    report = classification_report(y_true, y_pred, labels=label_list,
                                   output_dict=True, zero_division=0)
    per_class = {
        lbl: {
            "precision": round(float(report[lbl]["precision"]), _METRIC_DECIMALS),
            "recall":    round(float(report[lbl]["recall"]), _METRIC_DECIMALS),
            "f1":        round(float(report[lbl]["f1-score"]), _METRIC_DECIMALS),
            "support":   int(report[lbl]["support"]),
        }
        for lbl in label_list if lbl in report
    }
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", labels=label_list, zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=label_list).tolist()

    return LLMResult(
        accuracy=round(float(accuracy_score(y_true, y_pred)), _METRIC_DECIMALS),
        precision_macro=round(float(prec), _METRIC_DECIMALS),
        recall_macro=round(float(rec), _METRIC_DECIMALS),
        f1_macro=round(float(f1), _METRIC_DECIMALS),
        labels=label_list,
        per_class=per_class,
        confusion_matrix=cm,
        meta={
            "model": cfg.model,
            "temperature": cfg.temperature,
            "sample_size": len(sample_df),
            "full_test_size": len(test_df),
        },
    )


__all__ = ["LLMResult", "run"]
