"""Testes para v2/src/experiments/run_symbolic.py.

Os testes nao executam o analisador real da v1 (depende de spaCy, lexicons
externos e do corpus completo): em vez disso, mockam ``_load_symbolic_analyzer``
para isolar a logica de orquestracao do experimento (split, metricas, contrato
do resultado).
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def _make_corpus(tmp_path: Path, n_per_class: int = 30) -> Path:
    rows = []
    for lbl in ["positivo", "negativo", "neutro"]:
        for i in range(n_per_class):
            rows.append(
                {
                    "raw_text": f"{lbl} review {i}",
                    "clean_text": f"{lbl} review {i}",
                    "label": lbl,
                    "source": "test",
                }
            )
    df = pd.DataFrame(rows)
    p = tmp_path / "corpus.csv"
    df.to_csv(p, index=False)
    return p


def test_run_symbolic_com_mock(tmp_path):
    """run() com mock do analisador retorna campos no contrato esperado."""
    from v2.src.experiments.run_symbolic import run

    mock_analyzer = MagicMock()
    mock_analyzer.analyze.side_effect = lambda t: "positivo"

    with patch(
        "v2.src.experiments.run_symbolic._load_symbolic_analyzer",
        return_value=mock_analyzer,
    ):
        result = run(corpus_path=_make_corpus(tmp_path))

    d = result.as_dict()
    for campo in (
        "accuracy",
        "precision_macro",
        "recall_macro",
        "f1_macro",
        "labels",
        "per_class",
        "confusion_matrix",
    ):
        assert campo in d, f"Campo ausente: {campo}"


def test_run_symbolic_metricas_no_intervalo(tmp_path):
    """Metricas macro devem ficar entre 0 e 1."""
    from v2.src.experiments.run_symbolic import run

    mock_analyzer = MagicMock()
    mock_analyzer.analyze.side_effect = lambda t: "positivo"

    with patch(
        "v2.src.experiments.run_symbolic._load_symbolic_analyzer",
        return_value=mock_analyzer,
    ):
        result = run(corpus_path=_make_corpus(tmp_path))

    assert 0 <= result.accuracy <= 1
    assert 0 <= result.precision_macro <= 1
    assert 0 <= result.recall_macro <= 1
    assert 0 <= result.f1_macro <= 1


def test_run_symbolic_matriz_confusao_dimensoes(tmp_path):
    """confusion_matrix deve ser N x N onde N = numero de labels."""
    from v2.src.experiments.run_symbolic import run

    mock_analyzer = MagicMock()
    mock_analyzer.analyze.side_effect = lambda t: "neutro"

    with patch(
        "v2.src.experiments.run_symbolic._load_symbolic_analyzer",
        return_value=mock_analyzer,
    ):
        result = run(corpus_path=_make_corpus(tmp_path))

    n = len(result.labels)
    cm = result.confusion_matrix
    assert len(cm) == n
    assert all(len(row) == n for row in cm)


def test_run_symbolic_per_class_preenchido(tmp_path):
    """per_class deve ter entrada para cada label presente."""
    from v2.src.experiments.run_symbolic import run

    mock_analyzer = MagicMock()
    # alterna predicoes para garantir matches em mais de uma classe
    seq = iter(["positivo", "negativo", "neutro"] * 1000)
    mock_analyzer.analyze.side_effect = lambda t: next(seq)

    with patch(
        "v2.src.experiments.run_symbolic._load_symbolic_analyzer",
        return_value=mock_analyzer,
    ):
        result = run(corpus_path=_make_corpus(tmp_path))

    assert len(result.per_class) >= 2
    for lbl, metrics in result.per_class.items():
        assert "precision" in metrics
        assert "recall" in metrics
        assert "f1" in metrics
        assert "support" in metrics


def test_run_symbolic_label_map_normaliza_ingles(tmp_path):
    """Labels em ingles devem ser mapeadas para o vocabulario da v2."""
    from v2.src.experiments.run_symbolic import run

    mock_analyzer = MagicMock()
    # Retorna labels em ingles para simular variacoes possiveis do analisador.
    seq = iter(["positive", "negative", "neutral"] * 1000)
    mock_analyzer.analyze.side_effect = lambda t: next(seq)

    with patch(
        "v2.src.experiments.run_symbolic._load_symbolic_analyzer",
        return_value=mock_analyzer,
    ):
        result = run(corpus_path=_make_corpus(tmp_path))

    # Se o map nao funcionasse, todos cairiam em "neutro" pelo default e
    # accuracy ficaria proxima de 1/3. Com o map correto, a distribuicao de
    # predicoes cobre as 3 classes, permitindo acertos nas 3.
    assert set(result.per_class.keys()) == {"negativo", "neutro", "positivo"}


def test_run_symbolic_corpus_inexistente_lanca_erro(tmp_path):
    """run() deve levantar FileNotFoundError se o corpus nao existir."""
    from v2.src.experiments.run_symbolic import run

    with pytest.raises(FileNotFoundError):
        run(corpus_path=tmp_path / "inexistente.csv")
