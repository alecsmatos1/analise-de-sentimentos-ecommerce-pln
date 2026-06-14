"""Testes para run_baseline.py."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def _make_corpus(tmp_path: Path, n_per_class: int = 30) -> Path:
    rows = []
    for lbl in ["positivo", "negativo", "neutro"]:
        for i in range(n_per_class):
            rows.append({
                "raw_text":   f"{lbl} review {i}",
                "clean_text": f"{lbl} review {i}",
                "label":      lbl,
                "source":     "test",
            })
    df = pd.DataFrame(rows)
    p = tmp_path / "corpus.csv"
    df.to_csv(p, index=False)
    return p


def test_baseline_run_retorna_result(tmp_path):
    """run() deve retornar BaselineResult com metricas preenchidas."""
    from v2.src.experiments.run_baseline import run
    result = run(corpus_path=_make_corpus(tmp_path))
    assert 0 <= result.accuracy <= 1
    assert 0 <= result.f1_macro <= 1


def test_baseline_as_dict_tem_campos_esperados(tmp_path):
    """as_dict() deve conter todos os campos do contrato."""
    from v2.src.experiments.run_baseline import run
    d = run(corpus_path=_make_corpus(tmp_path)).as_dict()
    for campo in ("accuracy", "precision_macro", "recall_macro",
                  "f1_macro", "labels", "per_class", "confusion_matrix"):
        assert campo in d, f"Campo ausente: {campo}"


def test_baseline_per_class_preenchido(tmp_path):
    """per_class deve ter entrada para cada label."""
    from v2.src.experiments.run_baseline import run
    d = run(corpus_path=_make_corpus(tmp_path)).as_dict()
    assert len(d["per_class"]) >= 2
    for lbl, metrics in d["per_class"].items():
        assert "precision" in metrics and "recall" in metrics and "f1" in metrics


def test_baseline_corpus_inexistente_lanca_erro(tmp_path):
    """run() deve levantar FileNotFoundError se o corpus nao existir."""
    from v2.src.experiments.run_baseline import run
    with pytest.raises(FileNotFoundError):
        run(corpus_path=tmp_path / "inexistente.csv")


def test_baseline_confusion_matrix_dimensoes(tmp_path):
    """confusion_matrix deve ser N x N onde N = numero de labels."""
    from v2.src.experiments.run_baseline import run
    result = run(corpus_path=_make_corpus(tmp_path))
    n = len(result.labels)
    cm = result.confusion_matrix
    assert len(cm) == n
    assert all(len(row) == n for row in cm)
