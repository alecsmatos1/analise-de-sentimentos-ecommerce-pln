"""Testes para v2.src.corpus_loader.

Le os CSVs reais em data/ (presentes na maquina, nao versionados). Caso o
diretorio de dados nao exista no ambiente, os testes sao pulados com `skip`,
permitindo que a sprint rode validacoes leves em CI sem dados.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from v2.src.corpus_loader import (
    _DATA_DIR,
    _load_b2w,
    load_corpus,
    save_corpus,
)


pytestmark = pytest.mark.skipif(
    not _DATA_DIR.exists(),
    reason="diretorio data/ ausente; CSVs brutos nao versionados",
)


def test_load_b2w_retorna_colunas_canonicas():
    df = _load_b2w()
    assert {"raw_text", "label", "source"}.issubset(df.columns)
    assert df["source"].unique()[0] == "b2w"


def test_load_corpus_todas_as_fontes():
    df = load_corpus()
    assert set(df.columns) == {"raw_text", "clean_text", "label", "source"}
    assert set(df["source"].unique()) >= {"b2w", "olist", "mercadolivre"}
    assert set(df["label"].unique()) == {"positivo", "neutro", "negativo"}


def test_load_corpus_fonte_unica():
    df = load_corpus(sources=["mercadolivre"])
    assert df["source"].unique()[0] == "mercadolivre"
    assert len(df) > 0


def test_load_corpus_max_per_class_limita():
    df = load_corpus(max_per_class=100)
    for label in ["positivo", "neutro", "negativo"]:
        assert (df["label"] == label).sum() <= 100


def test_load_corpus_fonte_invalida_levanta_valueerror():
    with pytest.raises(ValueError, match="Nenhuma fonte valida"):
        load_corpus(sources=["nao_existe"])


def test_save_corpus_cria_csv(tmp_path: Path):
    df = load_corpus(sources=["mercadolivre"], max_per_class=10)
    path = save_corpus(df, output_path=tmp_path / "corpus.csv")
    assert path.exists()
    reloaded = pd.read_csv(path)
    assert list(reloaded.columns) == ["raw_text", "clean_text", "label", "source"]
