"""Testes do contrato LLMClassifier (mock da API Gemini)."""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from v2.src.models.llm_classifier import (
    ALLOWED_LABELS,
    LLMClassifier,
    LLMConfig,
    build_llm_classifier,
)


def _make_mock_model(response_text: str) -> MagicMock:
    """Cria mock do GenerativeModel Gemini que retorna response_text fixo."""
    model = MagicMock()
    response = MagicMock()
    response.text = response_text
    model.generate_content.return_value = response
    return model


@pytest.fixture
def clf_mock_positivo():
    clf = LLMClassifier()
    clf._model = _make_mock_model('{"label": "positivo"}')
    with patch.dict("os.environ", {"GOOGLE_API_KEY": "mock-key"}):
        yield clf


def test_predict_retorna_label_valido(clf_mock_positivo):
    """predict() deve retornar um label da lista permitida."""
    results = clf_mock_positivo.predict(["otimo produto"])
    assert len(results) == 1
    assert results[0] in ALLOWED_LABELS


def test_predict_normaliza_resposta_com_pontuacao():
    """Resposta JSON com capitalização deve ser normalizada."""
    clf = LLMClassifier()
    clf._model = _make_mock_model('{"label": "Positivo"}')
    with patch.dict("os.environ", {"GOOGLE_API_KEY": "mock-key"}):
        assert clf.predict_one("texto") == "positivo"


def test_predict_fallback_para_resposta_invalida():
    """Resposta invalida deve retornar o fallback_label."""
    clf = LLMClassifier(LLMConfig(fallback_label="neutro"))
    clf._model = _make_mock_model('{"label": "desconhecido"}')
    with patch.dict("os.environ", {"GOOGLE_API_KEY": "mock-key"}):
        result = clf.predict_one("texto ambiguo")
    assert result == "neutro"


def test_sem_api_key_lanca_erro(monkeypatch):
    """EnvironmentError quando GOOGLE_API_KEY nao esta definida."""
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.setitem(sys.modules, "google", MagicMock())
    monkeypatch.setitem(sys.modules, "google.generativeai", MagicMock())
    clf = LLMClassifier()
    with pytest.raises(EnvironmentError, match="GOOGLE_API_KEY"):
        clf._classify_one("texto")


def test_build_llm_classifier_retorna_instancia():
    """build_llm_classifier retorna LLMClassifier."""
    clf = build_llm_classifier()
    assert isinstance(clf, LLMClassifier)


def test_predict_multiplos_textos(clf_mock_positivo):
    """predict() retorna lista com um label por texto."""
    texts = ["bom", "ruim", "ok"]
    results = clf_mock_positivo.predict(texts)
    assert len(results) == 3
    assert all(r in ALLOWED_LABELS for r in results)
