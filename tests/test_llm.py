"""Testes do experimento E5 (run_llm) com mock do LLMClassifier."""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from v2.src.experiments.run_llm import LABEL_ORDER, LLMResult, run


def _synthetic_corpus(tmp_path: Path) -> Path:
    rows = []
    for lbl in LABEL_ORDER:
        for i in range(40):
            rows.append(
                {
                    "raw_text": f"{lbl} {i}",
                    "clean_text": f"{lbl} {i}",
                    "label": lbl,
                    "source": "test",
                }
            )
    path = tmp_path / "corpus.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def _predict_fixed(label: str):
    """Mock side_effect que devolve `label` para cada texto recebido."""

    def _side(texts):
        return [label] * len(texts)

    return _side


def test_run_llm_com_mock(tmp_path):
    """run() com mock do LLMClassifier retorna LLMResult com campos corretos."""
    corpus = _synthetic_corpus(tmp_path)

    with patch("v2.src.experiments.run_llm.LLMClassifier") as MockClf:
        instance = MagicMock()
        instance.predict.side_effect = _predict_fixed("positivo")
        MockClf.return_value = instance
        result = run(corpus_path=str(corpus), sample_size=30)

    assert isinstance(result, LLMResult)
    d = result.as_dict()
    assert "accuracy" in d and "f1_macro" in d
    assert "meta" in d and isinstance(d["meta"]["model"], str)
    assert d["meta"]["temperature"] == 0.0
    assert d["meta"]["sample_size"] > 0
    assert d["labels"] == LABEL_ORDER
    assert "per_class" in d and set(d["per_class"]).issubset(set(LABEL_ORDER))
    assert len(d["confusion_matrix"]) == len(LABEL_ORDER)


def test_run_llm_meta_full_test_size(tmp_path):
    """meta.full_test_size reflete o tamanho real do conjunto de teste."""
    corpus = _synthetic_corpus(tmp_path)

    with patch("v2.src.experiments.run_llm.LLMClassifier") as MockClf:
        instance = MagicMock()
        instance.predict.side_effect = _predict_fixed("neutro")
        MockClf.return_value = instance
        result = run(corpus_path=str(corpus), sample_size=30)

    meta = result.as_dict()["meta"]
    assert meta["full_test_size"] > 0
    assert meta["sample_size"] <= meta["full_test_size"]


def test_run_llm_corpus_inexistente(tmp_path):
    """FileNotFoundError quando o corpus_path nao existe."""
    with pytest.raises(FileNotFoundError):
        run(corpus_path=str(tmp_path / "nao_existe.csv"), sample_size=30)
