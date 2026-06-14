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


# Aplicado apenas aos testes que leem CSVs reais em data/. Os testes que
# monkeypatcham `_load_all` nao precisam dos arquivos brutos e portanto
# rodam em qualquer ambiente.
requires_real_data = pytest.mark.skipif(
    not _DATA_DIR.exists(),
    reason="diretorio data/ ausente; CSVs brutos nao versionados",
)


@requires_real_data
def test_load_b2w_retorna_colunas_canonicas():
    df = _load_b2w()
    assert {"raw_text", "label", "source"}.issubset(df.columns)
    assert df["source"].unique()[0] == "b2w"


@requires_real_data
def test_load_corpus_todas_as_fontes():
    df = load_corpus()
    assert set(df.columns) == {"raw_text", "clean_text", "label", "source"}
    assert set(df["source"].unique()) >= {"b2w", "olist", "mercadolivre"}
    assert set(df["label"].unique()) == {"positivo", "neutro", "negativo"}


@requires_real_data
def test_load_corpus_fonte_unica():
    df = load_corpus(sources=["mercadolivre"])
    assert df["source"].unique()[0] == "mercadolivre"
    assert len(df) > 0


@requires_real_data
def test_load_corpus_max_per_class_limita():
    df = load_corpus(max_per_class=100)
    for label in ["positivo", "neutro", "negativo"]:
        assert (df["label"] == label).sum() <= 100


def test_load_corpus_fonte_invalida_levanta_valueerror():
    with pytest.raises(ValueError, match="Nenhuma fonte valida"):
        load_corpus(sources=["nao_existe"])


@requires_real_data
def test_save_corpus_cria_csv(tmp_path: Path):
    df = load_corpus(sources=["mercadolivre"], max_per_class=10)
    path = save_corpus(df, output_path=tmp_path / "corpus.csv")
    assert path.exists()
    reloaded = pd.read_csv(path)
    assert list(reloaded.columns) == ["raw_text", "clean_text", "label", "source"]


def test_load_corpus_balanced_flag(monkeypatch):
    """balanced=True deve truncar todas as classes para o tamanho da menor."""
    import v2.src.corpus_loader as cl

    # Corpus desbalanceado: 100 pos, 20 neg, 30 neutro.
    fake = pd.DataFrame(
        {
            "raw_text": ["t"] * 150,
            "clean_text": ["t"] * 150,
            "label": (
                ["positivo"] * 100 + ["negativo"] * 20 + ["neutro"] * 30
            ),
            "source": ["test"] * 150,
        }
    )

    monkeypatch.setattr(cl, "_load_all", lambda *a, **k: fake)

    result = load_corpus(balanced=True)
    counts = result["label"].value_counts()
    # Todas as classes devem ter o mesmo tamanho (tamanho da menor = 20).
    assert counts.max() == counts.min(), f"Contagens: {counts.to_dict()}"
    assert counts.max() <= 20


def test_load_corpus_balanced_max_per_class_override(monkeypatch):
    """balanced=True + max_per_class explicito usa o valor fornecido."""
    import v2.src.corpus_loader as cl

    fake = pd.DataFrame(
        {
            "raw_text": ["t"] * 90,
            "clean_text": ["t"] * 90,
            "label": (
                ["positivo"] * 60 + ["negativo"] * 20 + ["neutro"] * 10
            ),
            "source": ["test"] * 90,
        }
    )

    monkeypatch.setattr(cl, "_load_all", lambda *a, **k: fake)

    result = load_corpus(balanced=True, max_per_class=15)
    counts = result["label"].value_counts()
    # neutro tem 10 amostras; com max_per_class=15 fica com 10 (nao 15).
    assert counts.max() <= 15


def test_load_corpus_sem_balanced_mantem_distribuicao(monkeypatch):
    """Sem balanced=True (e sem max_per_class) o corpus mantem a distribuicao."""
    import v2.src.corpus_loader as cl

    fake = pd.DataFrame(
        {
            "raw_text": ["t"] * 150,
            "clean_text": ["t"] * 150,
            "label": (
                ["positivo"] * 100 + ["negativo"] * 30 + ["neutro"] * 20
            ),
            "source": ["test"] * 150,
        }
    )

    monkeypatch.setattr(cl, "_load_all", lambda *a, **k: fake)

    result = load_corpus(balanced=False)
    counts = result["label"].value_counts()
    assert counts["positivo"] == 100
