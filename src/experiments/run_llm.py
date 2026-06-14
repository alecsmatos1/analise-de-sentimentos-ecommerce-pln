"""Experimento E5: classificacao zero-shot via LLM (OpenAI gpt-4.1-nano).

Avalia o ``LLMClassifier`` em uma amostra estratificada do conjunto de teste.
Suporta execucao incremental: predicoes sao salvas individualmente em
``v2/outputs/llm_predictions.csv`` e reutilizadas em execucoes posteriores
sem repetir chamadas pagas a API.
"""
from __future__ import annotations

import datetime
import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)

from ..models.llm_classifier import LLMClassifier, LLMConfig

_METRIC_DECIMALS = 4
_SAMPLE_SIZE = 500
_RANDOM_STATE = 42

LABEL_ORDER = ["negativo", "neutro", "positivo"]

_DEFAULT_PREDICTIONS_PATH = (
    Path(__file__).resolve().parents[2] / "outputs" / "llm_predictions.csv"
)

PREDICTION_COLUMNS = [
    "sample_id",
    "original_index",
    "source",
    "raw_text",
    "true_label",
    "predicted_label",
    "model",
    "temperature",
    "prompt_hash",
    "created_at",
]


def _compute_prompt_hash(template: str) -> str:
    """MD5 curto do template de prompt — identifica versao do prompt no cache."""
    return hashlib.md5(template.encode("utf-8")).hexdigest()[:8]


def _load_cache(predictions_path: Path) -> pd.DataFrame:
    """Carrega predicoes salvas; retorna DataFrame vazio se o arquivo nao existe."""
    if predictions_path.exists():
        try:
            return pd.read_csv(predictions_path, dtype=str)
        except Exception:
            pass
    return pd.DataFrame(columns=PREDICTION_COLUMNS)


def _cache_lookup(
    cache_df: pd.DataFrame,
    sample_id: str,
    model: str,
    temperature: str,
    prompt_hash: str,
) -> str | None:
    """Retorna o label em cache para (sample_id, model, temperature, prompt_hash) ou None."""
    if cache_df.empty:
        return None
    mask = (
        (cache_df["sample_id"] == sample_id)
        & (cache_df["model"] == model)
        & (cache_df["temperature"] == temperature)
        & (cache_df["prompt_hash"] == prompt_hash)
    )
    hits = cache_df[mask]
    return str(hits.iloc[0]["predicted_label"]) if len(hits) > 0 else None


def _append_prediction(predictions_path: Path, row: dict) -> None:
    """Escreve uma linha no CSV de predicoes (append imediato; cria header se necessario)."""
    write_header = not predictions_path.exists()
    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([row]).to_csv(
        predictions_path, mode="a", header=write_header, index=False
    )


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


def run(
    corpus_path: Union[Path, str, None] = None,
    sample_size: int = _SAMPLE_SIZE,
    predictions_path: Union[Path, str, None] = None,
    write_predictions: bool = True,
) -> LLMResult:
    """Classifica amostra estratificada do teste via LLM com cache incremental.

    Args:
        corpus_path: CSV processado. Usa ``config.PROCESSED_CORPUS_PATH`` se None.
        sample_size: Numero de amostras (distribuidas igualmente por classe).
        predictions_path: CSV de cache. Padrao: ``v2/outputs/llm_predictions.csv``.
        write_predictions: Se False, nao persiste predicoes (util com --no-write).
    """
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

    # Amostragem estratificada deterministica
    rng = np.random.default_rng(_RANDOM_STATE)
    sample_per_class = sample_size // len(LABEL_ORDER)
    frames = []
    actual_per_class: dict[str, int] = {}
    for lbl in LABEL_ORDER:
        chunk = test_df[test_df["label"] == lbl]
        n = min(sample_per_class, len(chunk))
        actual_per_class[lbl] = n
        idx = rng.choice(len(chunk), size=n, replace=False)
        frames.append(chunk.iloc[idx])

    # Mantém índice original do corpus para uso como sample_id estável
    sample_df = pd.concat(frames).sample(frac=1, random_state=_RANDOM_STATE)

    # Cache
    pred_path = (
        Path(predictions_path) if predictions_path else _DEFAULT_PREDICTIONS_PATH
    )
    cache_df = _load_cache(pred_path)

    cfg = LLMConfig(temperature=0.0)
    clf = LLMClassifier(cfg)
    prompt_hash = _compute_prompt_hash(cfg.prompt_template)
    temperature_str = str(cfg.temperature)

    started_at = datetime.datetime.utcnow().isoformat()
    cache_hits = 0
    api_calls = 0
    y_true: list[str] = []
    y_pred: list[str] = []

    for orig_idx, row in sample_df.iterrows():
        sample_id = str(orig_idx)
        true_label = str(row["label"])

        cached = _cache_lookup(
            cache_df, sample_id, cfg.model, temperature_str, prompt_hash
        )
        if cached is not None:
            predicted = cached
            cache_hits += 1
        else:
            predicted = clf.predict_one(str(row["raw_text"]))
            api_calls += 1
            if write_predictions:
                new_row = {
                    "sample_id": sample_id,
                    "original_index": orig_idx,
                    "source": row.get("source", ""),
                    "raw_text": row["raw_text"],
                    "true_label": true_label,
                    "predicted_label": predicted,
                    "model": cfg.model,
                    "temperature": temperature_str,
                    "prompt_hash": prompt_hash,
                    "created_at": datetime.datetime.utcnow().isoformat(),
                }
                _append_prediction(pred_path, new_row)
                cache_df = pd.concat(
                    [cache_df, pd.DataFrame([{k: str(v) for k, v in new_row.items()}])],
                    ignore_index=True,
                )

        y_true.append(true_label)
        y_pred.append(predicted)

    finished_at = datetime.datetime.utcnow().isoformat()

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
            "sample_size_requested": sample_size,
            "sample_size_evaluated": len(sample_df),
            "full_test_size": len(test_df),
            "prompt_hash": prompt_hash,
            "predictions_path": str(pred_path),
            "cache_hits": cache_hits,
            "api_calls": api_calls,
            "started_at": started_at,
            "finished_at": finished_at,
        },
    )


__all__ = [
    "LLMResult",
    "PREDICTION_COLUMNS",
    "_append_prediction",
    "_cache_lookup",
    "_compute_prompt_hash",
    "_load_cache",
    "run",
]
