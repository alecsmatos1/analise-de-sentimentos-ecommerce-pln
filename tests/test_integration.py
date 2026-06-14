"""Testes de integracao end-to-end do pipeline v2 (dados sinteticos).

Cobre: coerce_corpus -> stratified_split -> run_tfidf_linear -> reporting.save_all
Sem dependencia de dados externos.
"""
from __future__ import annotations

import pandas as pd

from v2.src.data import coerce_corpus
from v2.src.splitting import stratified_split
from v2.src.experiments.run_tfidf_linear import run_tfidf_linear
import v2.src.reporting as reporting


def _synthetic_corpus(n_per_class: int = 12) -> pd.DataFrame:
    """Corpus sintetico balanceado com as 4 colunas canonicas."""
    labels = ["positivo", "neutro", "negativo"]
    rows = []
    for label in labels:
        for i in range(n_per_class):
            rows.append({
                "raw_text": f"exemplo {label} numero {i}",
                "label": label,
                "source": "synthetic",
            })
    return pd.DataFrame(rows)


class TestPipelineEndToEnd:
    """Pipeline completo: coerce -> split -> experimento -> reporting."""

    def test_coerce_gera_clean_text(self):
        """coerce_corpus() deve gerar coluna clean_text a partir de raw_text."""
        df = _synthetic_corpus()
        result = coerce_corpus(df)
        assert "clean_text" in result.columns
        assert result["clean_text"].notna().all()

    def test_split_preserva_todas_classes(self):
        """stratified_split() deve conter as 3 classes em train e test."""
        df = coerce_corpus(_synthetic_corpus())
        split = stratified_split(df)
        for part in (split.train, split.test):
            assert set(part["label"].unique()) == {"positivo", "neutro", "negativo"}

    def test_run_tfidf_linear_aceita_saida_do_split(self):
        """run_tfidf_linear() deve aceitar as colunas produzidas pelo split."""
        df = coerce_corpus(_synthetic_corpus())
        split = stratified_split(df)
        result = run_tfidf_linear(
            split.train["clean_text"].tolist(),
            split.train["label"].tolist(),
            split.test["clean_text"].tolist(),
            split.test["label"].tolist(),
        )
        assert 0.0 <= result.accuracy <= 1.0
        assert result.f1_macro >= 0.0

    def test_as_dict_compativel_com_reporting(self, tmp_path):
        """as_dict() de TfidfLinearResult deve ser aceito por reporting.save_all()."""
        df = coerce_corpus(_synthetic_corpus())
        split = stratified_split(df)
        result = run_tfidf_linear(
            split.train["clean_text"].tolist(),
            split.train["label"].tolist(),
            split.test["clean_text"].tolist(),
            split.test["label"].tolist(),
        )
        # nao deve lancar excecao
        written = reporting.save_all("tfidf-linear", result.as_dict(), output_dir=tmp_path)
        # verifica artefatos gerados (usa o Path retornado por save_all,
        # evita acoplar o teste a convencao interna de nomes do reporting)
        assert written["result_json"].exists()

    def test_pipeline_completo_gera_tres_artefatos(self, tmp_path):
        """Pipeline completo deve gerar JSON + CSV de metricas + CSV de confusao."""
        df = coerce_corpus(_synthetic_corpus())
        split = stratified_split(df)
        result = run_tfidf_linear(
            split.train["clean_text"].tolist(),
            split.train["label"].tolist(),
            split.test["clean_text"].tolist(),
            split.test["label"].tolist(),
        )
        reporting.save_all("tfidf-linear", result.as_dict(), output_dir=tmp_path)
        files = list(tmp_path.iterdir())
        assert len(files) >= 3, f"Esperado >= 3 artefatos, encontrado: {[f.name for f in files]}"

    def test_pipeline_deterministico_com_mesma_seed(self):
        """Duas execucoes com mesma seed devem produzir f1_macro identico."""
        df = coerce_corpus(_synthetic_corpus())
        split_a = stratified_split(df, seed=42)
        split_b = stratified_split(df, seed=42)
        r_a = run_tfidf_linear(
            split_a.train["clean_text"].tolist(), split_a.train["label"].tolist(),
            split_a.test["clean_text"].tolist(), split_a.test["label"].tolist(),
        )
        r_b = run_tfidf_linear(
            split_b.train["clean_text"].tolist(), split_b.train["label"].tolist(),
            split_b.test["clean_text"].tolist(), split_b.test["label"].tolist(),
        )
        assert r_a.f1_macro == r_b.f1_macro
