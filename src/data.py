"""Contrato interno de corpus para a v2.

A v2 reusa o mapeamento por nota da v1 para padronizar os rotulos em
`negativo`, `neutro` e `positivo` (ver `data_processing.rating_to_label`), mas
mantem um contrato proprio para evitar acoplamento com a estrutura historica de
`OUTPUT_COLUMNS`. Cada experimento da v2 deve consumir DataFrames que respeitem
`REQUIRED_COLUMNS`: o corpus expoe `raw_text` (texto preservado, usado por
BERTimbau, LLM e analisadores que precisam de pontuacao/acentos) e
`clean_text` (texto normalizado para representacoes esparsas como TF-IDF e
agregacoes Word2Vec). Veja `v2/arquitetura.md` para a justificativa.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Iterable

import pandas as pd


LABEL_NEGATIVE = "negativo"
LABEL_NEUTRAL = "neutro"
LABEL_POSITIVE = "positivo"

SENTIMENT_LABELS: tuple[str, ...] = (LABEL_NEGATIVE, LABEL_NEUTRAL, LABEL_POSITIVE)

REQUIRED_COLUMNS: tuple[str, ...] = ("raw_text", "clean_text", "label", "source")

_REQUIRED_INPUT_COLUMNS: tuple[str, ...] = ("raw_text", "label", "source")

_NON_ALPHANUMERIC = re.compile(r"[^a-z0-9\s]")
_WHITESPACE = re.compile(r"\s+")


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


def normalize_text(text: str | None) -> str:
    """Normaliza texto para uso em representacoes esparsas (TF-IDF).

    Aplica lowercase, remove acentos via decomposicao NFD, descarta qualquer
    caractere que nao seja letra ASCII, digito ou espaco, e colapsa espacos.
    Usa somente a stdlib para manter o contrato sem dependencias extras.
    Aceita `None` para alinhar com chamadores que ainda nao passaram pela
    `coerce_corpus` (que protege via `.fillna("")`).
    """

    if text is None:
        return ""
    text = str(text).lower()
    text = unicodedata.normalize("NFD", text)
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn")
    text = _NON_ALPHANUMERIC.sub(" ", text)
    text = _WHITESPACE.sub(" ", text).strip()
    return text


def coerce_corpus(
    df: pd.DataFrame,
    *,
    allowed_labels: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Valida e normaliza um DataFrame para o contrato da v2.

    Exige `raw_text`, `label` e `source` no DataFrame de entrada. Se
    `clean_text` ja estiver presente, e mantido sem reprocessar (para permitir
    pre-calculo por cache ou pipelines externos); caso contrario, e gerado a
    partir de `raw_text` via `normalize_text`. Remove linhas com `raw_text`
    vazio ou rotulo fora de `allowed_labels`, e retorna uma copia com exatamente
    as colunas de `REQUIRED_COLUMNS` na ordem canonica.
    """

    # Materializa allowed_labels imediatamente: aceitar Iterable significa que
    # geradores tambem sao validos e nao podem ser iterados duas vezes.
    _allowed: set[str] = (
        set(allowed_labels) if allowed_labels is not None else set(SENTIMENT_LABELS)
    )

    missing = [column for column in _REQUIRED_INPUT_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(
            f"corpus invalido: colunas obrigatorias ausentes {missing}"
        )

    if not _allowed:
        raise ValueError("allowed_labels nao pode ser vazio")

    cleaned = df.copy()
    cleaned["raw_text"] = cleaned["raw_text"].fillna("").astype(str).str.strip()
    cleaned["label"] = cleaned["label"].fillna("").astype(str).str.strip()
    cleaned["source"] = cleaned["source"].fillna("").astype(str).str.strip()

    if "clean_text" in cleaned.columns:
        cleaned["clean_text"] = cleaned["clean_text"].fillna("").astype(str)
    else:
        cleaned["clean_text"] = cleaned["raw_text"].apply(normalize_text)

    mask = cleaned["raw_text"].str.len().gt(0) & cleaned["label"].isin(_allowed)
    cleaned = cleaned.loc[mask, list(REQUIRED_COLUMNS)].reset_index(drop=True)
    return cleaned
