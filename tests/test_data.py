"""Testes do contrato de corpus da v2."""

from __future__ import annotations

import pandas as pd
import pytest

from v2.src.data import (
    LABEL_NEGATIVE,
    LABEL_NEUTRAL,
    LABEL_POSITIVE,
    REQUIRED_COLUMNS,
    SENTIMENT_LABELS,
    coerce_corpus,
    rating_to_label,
)


def _fixture_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "text": [
                "produto excelente, amei",
                "veio quebrado, pessimo",
                "ok, nem bom nem ruim",
                "  ",
                "entrega rapida",
            ],
            "label": [
                LABEL_POSITIVE,
                LABEL_NEGATIVE,
                LABEL_NEUTRAL,
                LABEL_POSITIVE,
                "indefinido",
            ],
            "source": ["b2w", "olist", "meli", "b2w", "olist"],
        }
    )


def test_rating_to_label_mapeia_intervalos():
    assert rating_to_label(1) == LABEL_NEGATIVE
    assert rating_to_label(2) == LABEL_NEGATIVE
    assert rating_to_label(3) == LABEL_NEUTRAL
    assert rating_to_label(4) == LABEL_POSITIVE
    assert rating_to_label(5) == LABEL_POSITIVE


def test_rating_to_label_aceita_float_e_descarta_invalidos():
    assert rating_to_label(4.0) == LABEL_POSITIVE
    assert rating_to_label(0) == ""
    assert rating_to_label(6) == ""
    assert rating_to_label(None) == ""
    assert rating_to_label("a") == ""


def test_sentiment_labels_tem_tres_classes_padrao():
    assert SENTIMENT_LABELS == (LABEL_NEGATIVE, LABEL_NEUTRAL, LABEL_POSITIVE)


def test_coerce_corpus_remove_texto_vazio_e_label_invalido():
    df = _fixture_df()
    result = coerce_corpus(df)

    assert list(result.columns) == list(REQUIRED_COLUMNS)
    assert len(result) == 3
    assert set(result["label"]) <= set(SENTIMENT_LABELS)
    assert (result["text"].str.len() > 0).all()


def test_coerce_corpus_falha_se_coluna_obrigatoria_ausente():
    df = pd.DataFrame({"text": ["a"], "source": ["b2w"]})
    with pytest.raises(ValueError, match="colunas obrigatorias ausentes"):
        coerce_corpus(df)


def test_coerce_corpus_falha_com_allowed_labels_vazio():
    df = _fixture_df()
    with pytest.raises(ValueError, match="allowed_labels"):
        coerce_corpus(df, allowed_labels=[])


def test_coerce_corpus_respeita_allowed_labels_customizado():
    df = _fixture_df()
    result = coerce_corpus(df, allowed_labels=[LABEL_POSITIVE])
    assert set(result["label"]) == {LABEL_POSITIVE}
    assert (result["text"].str.len() > 0).all()
