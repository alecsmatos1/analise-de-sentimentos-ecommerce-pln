"""Testes do split estratificado comum da v2."""

from __future__ import annotations

import pandas as pd
import pytest

from v2.src.data import LABEL_NEGATIVE, LABEL_NEUTRAL, LABEL_POSITIVE, REQUIRED_COLUMNS
from v2.src.splitting import StratifiedSplit, stratified_split


def _balanced_fixture(per_class: int = 10) -> pd.DataFrame:
    rows = []
    for i in range(per_class):
        rows.append((f"texto positivo {i}", LABEL_POSITIVE, "b2w"))
        rows.append((f"texto negativo {i}", LABEL_NEGATIVE, "olist"))
        rows.append((f"texto neutro {i}", LABEL_NEUTRAL, "meli"))
    return pd.DataFrame(rows, columns=["text", "label", "source"])


def test_stratified_split_preserva_proporcoes_e_tamanho():
    df = _balanced_fixture(per_class=10)
    split = stratified_split(df, seed=42, test_size=0.2)

    assert isinstance(split, StratifiedSplit)
    assert split.seed == 42
    assert split.test_size == 0.2

    total = len(split.train) + len(split.test)
    assert total == len(df)
    assert len(split.test) == 6
    assert len(split.train) == 24

    for label in (LABEL_POSITIVE, LABEL_NEGATIVE, LABEL_NEUTRAL):
        assert (split.train["label"] == label).sum() == 8
        assert (split.test["label"] == label).sum() == 2


def test_stratified_split_e_deterministico_com_mesma_seed():
    df = _balanced_fixture(per_class=10)
    a = stratified_split(df, seed=7)
    b = stratified_split(df, seed=7)
    pd.testing.assert_frame_equal(a.train, b.train)
    pd.testing.assert_frame_equal(a.test, b.test)


def test_stratified_split_difere_com_seed_diferente():
    df = _balanced_fixture(per_class=10)
    a = stratified_split(df, seed=1)
    b = stratified_split(df, seed=2)
    # Mesmas linhas no total, mas ordens distintas em pelo menos um lado.
    different = not a.train.equals(b.train) or not a.test.equals(b.test)
    assert different


def test_stratified_split_mantem_contrato_minimo_de_colunas():
    df = _balanced_fixture()
    split = stratified_split(df)
    assert list(split.train.columns) == list(REQUIRED_COLUMNS)
    assert list(split.test.columns) == list(REQUIRED_COLUMNS)


def test_stratified_split_falha_com_test_size_invalido():
    df = _balanced_fixture()
    with pytest.raises(ValueError, match="test_size"):
        stratified_split(df, test_size=0.0)
    with pytest.raises(ValueError, match="test_size"):
        stratified_split(df, test_size=1.0)


def test_stratified_split_falha_se_classe_tem_um_exemplo():
    df = pd.DataFrame(
        {
            "text": ["a", "b", "c", "d", "e"],
            "label": [
                LABEL_POSITIVE,
                LABEL_POSITIVE,
                LABEL_NEGATIVE,
                LABEL_NEGATIVE,
                LABEL_NEUTRAL,
            ],
            "source": ["b2w"] * 5,
        }
    )
    with pytest.raises(ValueError, match="estratificacao"):
        stratified_split(df)


def test_stratified_split_falha_com_corpus_vazio_apos_coercao():
    df = pd.DataFrame({"text": ["", ""], "label": ["x", "y"], "source": ["b2w", "olist"]})
    with pytest.raises(ValueError, match="corpus vazio"):
        stratified_split(df)
