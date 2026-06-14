"""Contrato interno de corpus para a v2.

A v2 reusa o mapeamento por nota da v1 para padronizar os rotulos em
`negativo`, `neutro` e `positivo` (ver `data_processing.rating_to_label`), mas
mantem um contrato minimo proprio para evitar acoplamento com a estrutura
historica de `OUTPUT_COLUMNS`. Cada experimento da v2 deve consumir DataFrames
que respeitem `REQUIRED_COLUMNS`.
"""

from __future__ import annotations

from typing import Iterable

import pandas as pd


LABEL_NEGATIVE = "negativo"
LABEL_NEUTRAL = "neutro"
LABEL_POSITIVE = "positivo"

SENTIMENT_LABELS: tuple[str, ...] = (LABEL_NEGATIVE, LABEL_NEUTRAL, LABEL_POSITIVE)

REQUIRED_COLUMNS: tuple[str, ...] = ("text", "label", "source")


def rating_to_label(rating: int | float) -> str:
    """Mapeia nota numerica para rotulo de sentimento.

    Reusa a convencao da v1: 1-2 -> negativo, 3 -> neutro, 4-5 -> positivo.
    Notas fora desse intervalo retornam string vazia para sinalizar descarte
    explicito a quem chama (a v1 usa "indefinido", a v2 prefere filtrar antes
    de chegar ao contrato).
    """

    if rating is None:
        return ""
    try:
        value = int(rating)
    except (TypeError, ValueError):
        return ""
    if value in (1, 2):
        return LABEL_NEGATIVE
    if value == 3:
        return LABEL_NEUTRAL
    if value in (4, 5):
        return LABEL_POSITIVE
    return ""


def coerce_corpus(
    df: pd.DataFrame,
    *,
    allowed_labels: Iterable[str] = SENTIMENT_LABELS,
) -> pd.DataFrame:
    """Valida e normaliza um DataFrame para o contrato da v2.

    Garante que existam as colunas `text`, `label` e `source`, remove linhas com
    texto vazio ou rotulo fora de `allowed_labels`, e retorna uma copia ordenada
    com apenas essas colunas. Nao faz limpeza linguistica; isso e
    responsabilidade dos modulos de representacao (TF-IDF, etc.).
    """

    missing = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(
            f"corpus invalido: colunas obrigatorias ausentes {missing}"
        )

    allowed = set(allowed_labels)
    if not allowed:
        raise ValueError("allowed_labels nao pode ser vazio")

    cleaned = df.loc[:, list(REQUIRED_COLUMNS)].copy()
    cleaned["text"] = cleaned["text"].fillna("").astype(str).str.strip()
    cleaned["label"] = cleaned["label"].fillna("").astype(str).str.strip()
    cleaned["source"] = cleaned["source"].fillna("").astype(str).str.strip()

    mask = cleaned["text"].str.len().gt(0) & cleaned["label"].isin(allowed)
    cleaned = cleaned.loc[mask].reset_index(drop=True)
    return cleaned
