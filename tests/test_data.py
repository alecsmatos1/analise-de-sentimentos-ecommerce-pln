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
    normalize_text,
    rating_to_label,
)


def _make_minimal_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "raw_text": [
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


def test_normalize_text_aplica_lowercase_e_remove_acento_e_pontuacao():
    assert normalize_text("Produto ÓTIMO, amei!") == "produto otimo amei"
    assert normalize_text("Não gostei... veio quebrado.") == "nao gostei veio quebrado"
    assert normalize_text("  espaços  duplos  ") == "espacos duplos"
    assert normalize_text("") == ""
    # Caracteres nao ASCII sao filtrados, mas digitos sao preservados.
    assert normalize_text("Comprei 2 unidades :)") == "comprei 2 unidades"


def test_coerce_corpus_remove_texto_vazio_e_label_invalido():
    df = _make_minimal_df()
    result = coerce_corpus(df)

    assert list(result.columns) == list(REQUIRED_COLUMNS)
    assert len(result) == 3
    assert set(result["label"]) <= set(SENTIMENT_LABELS)
    assert (result["raw_text"].str.len() > 0).all()


def test_coerce_corpus_gera_clean_text_quando_ausente():
    df = _make_minimal_df()
    result = coerce_corpus(df)

    assert "clean_text" in result.columns
    # clean_text deve refletir normalize_text do raw_text correspondente.
    esperado = [normalize_text(raw) for raw in result["raw_text"]]
    assert list(result["clean_text"]) == esperado


def test_coerce_corpus_preserva_clean_text_quando_presente():
    df = _make_minimal_df()
    # Fornece clean_text pre-computado distinto da normalizacao default para
    # garantir que o valor existente seja preservado sem reprocessar.
    df["clean_text"] = [
        "pre computado positivo",
        "pre computado negativo",
        "pre computado neutro",
        "ignorado vazio",
        "ignorado invalido",
    ]
    result = coerce_corpus(df)

    assert list(result.columns) == list(REQUIRED_COLUMNS)
    assert list(result["clean_text"]) == [
        "pre computado positivo",
        "pre computado negativo",
        "pre computado neutro",
    ]


def test_coerce_corpus_falha_se_coluna_obrigatoria_ausente():
    df = pd.DataFrame({"raw_text": ["a"], "source": ["b2w"]})
    with pytest.raises(ValueError, match="colunas obrigatorias ausentes"):
        coerce_corpus(df)


def test_coerce_corpus_falha_com_allowed_labels_vazio():
    df = _make_minimal_df()
    with pytest.raises(ValueError, match="allowed_labels"):
        coerce_corpus(df, allowed_labels=[])


def test_coerce_corpus_respeita_allowed_labels_customizado():
    df = _make_minimal_df()
    result = coerce_corpus(df, allowed_labels=[LABEL_POSITIVE])
    assert set(result["label"]) == {LABEL_POSITIVE}
    assert (result["raw_text"].str.len() > 0).all()


def test_normalize_text_com_none_retorna_string_vazia():
    assert normalize_text(None) == ""


def test_normalize_text_com_string_normal_funciona():
    assert normalize_text("Texto COM Acento ção") == "texto com acento cao"


def test_coerce_corpus_aceita_allowed_labels_como_generator():
    df = pd.DataFrame(
        {
            "raw_text": ["bom", "ruim", "ok"],
            "label": [LABEL_POSITIVE, LABEL_NEGATIVE, LABEL_NEUTRAL],
            "source": ["s"] * 3,
        }
    )
    gen = (x for x in [LABEL_POSITIVE, LABEL_NEGATIVE, LABEL_NEUTRAL])
    result = coerce_corpus(df, allowed_labels=gen)
    assert len(result) == 3
