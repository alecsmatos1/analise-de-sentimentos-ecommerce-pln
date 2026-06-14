"""Testes para v2/src/reporting_compare.py."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import reporting_compare  # noqa: E402


def _tfidf_result_payload() -> dict:
    return {
        "accuracy": 0.8004,
        "f1_macro": 0.7096,
        "precision_macro": 0.699,
        "recall_macro": 0.7347,
        "per_class": {
            "negativo": {"f1": 0.8518},
            "neutro":   {"f1": 0.4021},
            "positivo": {"f1": 0.8748},
        },
    }


def test_build_comparison_table_com_jsons_parciais(tmp_path: Path):
    """Experimentos ausentes ficam com placeholder; presentes com valores reais."""
    (tmp_path / "tfidf-linear.json").write_text(
        json.dumps(_tfidf_result_payload()), encoding="utf-8"
    )

    df = reporting_compare.build_comparison_table(outputs_dir=tmp_path)

    tfidf_row = df[df["Abordagem"] == "TF-IDF + LR"].iloc[0]
    assert tfidf_row["F1 Macro"] == 0.7096

    baseline_row = df[df["Abordagem"].str.contains("Baseline")].iloc[0]
    assert baseline_row["F1 Macro"] == reporting_compare.MISSING


def test_save_comparison_csv(tmp_path: Path):
    """save_comparison_csv cria arquivo CSV legivel."""
    df = reporting_compare.build_comparison_table(outputs_dir=tmp_path)

    path = reporting_compare.save_comparison_csv(df, outputs_dir=tmp_path)

    assert path.exists()
    loaded = pd.read_csv(path)
    assert "Abordagem" in loaded.columns
    assert "F1 Macro" in loaded.columns


def test_save_comparison_report_contem_markdown(tmp_path: Path):
    """save_comparison_report gera arquivo com tabela Markdown."""
    (tmp_path / "tfidf-linear.json").write_text(
        json.dumps(_tfidf_result_payload()), encoding="utf-8"
    )
    df = reporting_compare.build_comparison_table(outputs_dir=tmp_path)

    path = reporting_compare.save_comparison_report(df, outputs_dir=tmp_path)

    content = path.read_text(encoding="utf-8")
    assert "|" in content
    assert "TF-IDF" in content
    assert "Melhor F1 Macro" in content
