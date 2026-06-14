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

from v2.src.experiments.run_llm import (
    LABEL_ORDER,
    LLMResult,
    _append_prediction,
    _compute_prompt_hash,
    _load_cache,
    run,
)
from v2.src.models.llm_classifier import LLMConfig


# ── helpers ──────────────────────────────────────────────────────────────────

def _synthetic_corpus(tmp_path: Path) -> Path:
    rows = []
    for lbl in LABEL_ORDER:
        for i in range(40):
            rows.append(
                {
                    "raw_text": f"{lbl} review {i}",
                    "clean_text": f"{lbl} review {i}",
                    "label": lbl,
                    "source": "test",
                }
            )
    path = tmp_path / "corpus.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def _mock_clf(return_label: str) -> MagicMock:
    instance = MagicMock()
    instance.predict_one.return_value = return_label
    return instance


# ── testes existentes (atualizados para predict_one) ─────────────────────────

def test_run_llm_com_mock(tmp_path):
    """run() com mock retorna LLMResult com campos corretos."""
    corpus = _synthetic_corpus(tmp_path)
    pred_path = tmp_path / "preds.csv"

    with patch("v2.src.experiments.run_llm.LLMClassifier") as MockClf:
        MockClf.return_value = _mock_clf("positivo")
        result = run(corpus_path=str(corpus), sample_size=9, predictions_path=pred_path)

    assert isinstance(result, LLMResult)
    d = result.as_dict()
    assert "accuracy" in d and "f1_macro" in d
    assert "meta" in d and isinstance(d["meta"]["model"], str)
    assert d["meta"]["temperature"] == 0.0
    assert d["meta"]["sample_size_evaluated"] > 0
    assert d["labels"] == LABEL_ORDER
    assert set(d["per_class"]).issubset(set(LABEL_ORDER))
    assert len(d["confusion_matrix"]) == len(LABEL_ORDER)


def test_run_llm_meta_full_test_size(tmp_path):
    """meta.full_test_size reflete o tamanho real do conjunto de teste."""
    corpus = _synthetic_corpus(tmp_path)
    pred_path = tmp_path / "preds.csv"

    with patch("v2.src.experiments.run_llm.LLMClassifier") as MockClf:
        MockClf.return_value = _mock_clf("neutro")
        result = run(corpus_path=str(corpus), sample_size=9, predictions_path=pred_path)

    meta = result.as_dict()["meta"]
    assert meta["full_test_size"] > 0
    assert meta["sample_size_evaluated"] <= meta["full_test_size"]


def test_run_llm_corpus_inexistente(tmp_path):
    """FileNotFoundError quando o corpus_path nao existe."""
    with pytest.raises(FileNotFoundError):
        run(corpus_path=str(tmp_path / "nao_existe.csv"), sample_size=9)


# ── testes de cache ───────────────────────────────────────────────────────────

def test_cache_miss_chama_api_e_salva(tmp_path):
    """Sem cache: API e chamada e predicoes sao salvas no CSV."""
    corpus = _synthetic_corpus(tmp_path)
    pred_path = tmp_path / "preds.csv"

    with patch("v2.src.experiments.run_llm.LLMClassifier") as MockClf:
        MockClf.return_value = _mock_clf("neutro")
        result = run(corpus_path=str(corpus), sample_size=9, predictions_path=pred_path)

    assert result.meta["api_calls"] > 0
    assert result.meta["cache_hits"] == 0
    assert pred_path.exists()
    df = pd.read_csv(pred_path)
    assert len(df) == result.meta["sample_size_evaluated"]
    assert set(PREDICTION_COLUMNS).issubset(set(df.columns))


def test_cache_hit_nao_chama_api(tmp_path):
    """Segunda execucao identica nao faz nenhuma chamada de API."""
    corpus = _synthetic_corpus(tmp_path)
    pred_path = tmp_path / "preds.csv"

    # Primeira execucao — popula cache
    with patch("v2.src.experiments.run_llm.LLMClassifier") as MockClf:
        MockClf.return_value = _mock_clf("positivo")
        result1 = run(corpus_path=str(corpus), sample_size=9, predictions_path=pred_path)
    first_calls = result1.meta["api_calls"]
    assert first_calls > 0

    # Segunda execucao — deve usar somente cache
    with patch("v2.src.experiments.run_llm.LLMClassifier") as MockClf2:
        instance2 = _mock_clf("negativo")  # resposta diferente — nao deve ser usada
        MockClf2.return_value = instance2
        result2 = run(corpus_path=str(corpus), sample_size=9, predictions_path=pred_path)

    assert instance2.predict_one.call_count == 0
    assert result2.meta["cache_hits"] == first_calls
    assert result2.meta["api_calls"] == 0
    # Predicoes devem ser identicas (vindas do cache)
    assert result1.f1_macro == result2.f1_macro
    assert result1.accuracy == result2.accuracy


def test_segunda_execucao_usa_cache(tmp_path):
    """Alias semantico: garante que a segunda execucao nunca chama predict_one."""
    corpus = _synthetic_corpus(tmp_path)
    pred_path = tmp_path / "preds.csv"

    with patch("v2.src.experiments.run_llm.LLMClassifier") as MockClf:
        MockClf.return_value = _mock_clf("positivo")
        run(corpus_path=str(corpus), sample_size=9, predictions_path=pred_path)

    with patch("v2.src.experiments.run_llm.LLMClassifier") as MockClf2:
        instance2 = MagicMock()
        MockClf2.return_value = instance2
        result2 = run(corpus_path=str(corpus), sample_size=9, predictions_path=pred_path)

    assert instance2.predict_one.call_count == 0
    assert result2.meta["api_calls"] == 0


def test_sample_size_chega_no_run(tmp_path):
    """sample_size controla quantas amostras sao processadas."""
    corpus = _synthetic_corpus(tmp_path)
    pred_path = tmp_path / "preds.csv"

    with patch("v2.src.experiments.run_llm.LLMClassifier") as MockClf:
        MockClf.return_value = _mock_clf("neutro")
        result = run(corpus_path=str(corpus), sample_size=6, predictions_path=pred_path)

    assert result.meta["sample_size_requested"] == 6
    assert result.meta["sample_size_evaluated"] <= 6


def test_metricas_calculadas_do_cache(tmp_path):
    """Metricas da segunda execucao (cache total) sao identicas a primeira."""
    corpus = _synthetic_corpus(tmp_path)
    pred_path = tmp_path / "preds.csv"

    with patch("v2.src.experiments.run_llm.LLMClassifier") as MockClf:
        MockClf.return_value = _mock_clf("positivo")
        result1 = run(corpus_path=str(corpus), sample_size=9, predictions_path=pred_path)

    with patch("v2.src.experiments.run_llm.LLMClassifier") as MockClf2:
        MockClf2.return_value = MagicMock()
        result2 = run(corpus_path=str(corpus), sample_size=9, predictions_path=pred_path)

    assert result1.accuracy == result2.accuracy
    assert result1.f1_macro == result2.f1_macro
    assert result1.confusion_matrix == result2.confusion_matrix


def test_predicoes_salvas_com_interrupcao(tmp_path):
    """CSV parcial (execucao interrompida) nao quebra na proxima execucao."""
    corpus = _synthetic_corpus(tmp_path)
    pred_path = tmp_path / "preds.csv"

    call_count = [0]

    def _fail_after_3(text: str) -> str:
        call_count[0] += 1
        if call_count[0] > 3:
            raise RuntimeError("Interrupcao simulada")
        return "negativo"

    # Primeira execucao interrompida apos 3 chamadas
    with patch("v2.src.experiments.run_llm.LLMClassifier") as MockClf:
        instance = MagicMock()
        instance.predict_one.side_effect = _fail_after_3
        MockClf.return_value = instance
        with pytest.raises(RuntimeError):
            run(corpus_path=str(corpus), sample_size=9, predictions_path=pred_path)

    # CSV deve ter exatamente 3 linhas salvas
    assert pred_path.exists()
    partial = pd.read_csv(pred_path)
    assert len(partial) == 3

    # Segunda execucao completa com sucesso usando as 3 linhas em cache
    with patch("v2.src.experiments.run_llm.LLMClassifier") as MockClf2:
        instance2 = _mock_clf("neutro")
        MockClf2.return_value = instance2
        result = run(corpus_path=str(corpus), sample_size=9, predictions_path=pred_path)

    assert result.meta["cache_hits"] == 3
    assert result.meta["api_calls"] == result.meta["sample_size_evaluated"] - 3
    assert isinstance(result, LLMResult)


def test_no_write_nao_cria_csv(tmp_path):
    """write_predictions=False nao cria o arquivo de predicoes."""
    corpus = _synthetic_corpus(tmp_path)
    pred_path = tmp_path / "preds.csv"

    with patch("v2.src.experiments.run_llm.LLMClassifier") as MockClf:
        MockClf.return_value = _mock_clf("positivo")
        run(
            corpus_path=str(corpus),
            sample_size=9,
            predictions_path=pred_path,
            write_predictions=False,
        )

    assert not pred_path.exists()


# ── helpers exportados ────────────────────────────────────────────────────────

from v2.src.experiments.run_llm import PREDICTION_COLUMNS  # noqa: E402
