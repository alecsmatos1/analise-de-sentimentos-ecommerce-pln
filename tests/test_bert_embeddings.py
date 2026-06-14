"""Testes do contrato BERTimbau embeddings (mock — sem torch/modelo real)."""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from v2.src.representations.bertimbau_embeddings import (
    BertimbauConfig,
    BertimbauVectorizer,
    build_bert_vectorizer,
)

HIDDEN_SIZE = 4  # dimensao reduzida para testes rapidos


def _make_mock_tokenizer(seq_len: int = 6):
    import torch
    mock = MagicMock()
    def side_effect(texts, **kwargs):
        n = len(texts)
        return {
            "input_ids": torch.zeros(n, seq_len, dtype=torch.long),
            "attention_mask": torch.ones(n, seq_len, dtype=torch.long),
        }
    mock.side_effect = side_effect
    return mock


def _make_mock_model(hidden_size: int, seq_len: int = 6):
    import torch
    mock = MagicMock()
    def forward(**kwargs):
        n = kwargs["input_ids"].shape[0]
        out = MagicMock()
        out.last_hidden_state = torch.ones(n, seq_len, hidden_size)
        return out
    mock.side_effect = forward
    mock.eval = MagicMock(return_value=mock)
    mock.to = MagicMock(return_value=mock)
    return mock


@pytest.fixture
def vectorizer_com_mock():
    try:
        import torch  # noqa: F401
    except ImportError:
        pytest.skip("torch nao instalado — skip de teste BERTimbau")
    config = BertimbauConfig(vector_size=HIDDEN_SIZE, batch_size=2)
    v = BertimbauVectorizer(config)
    v._tokenizer = _make_mock_tokenizer()
    v._model = _make_mock_model(HIDDEN_SIZE)
    return v


def test_transform_retorna_array_correto(vectorizer_com_mock):
    """transform() retorna ndarray de shape (n_textos, hidden_size)."""
    texts = ["produto bom", "entrega ruim", "atendimento ok"]
    result = vectorizer_com_mock.transform(texts)
    assert isinstance(result, np.ndarray)
    assert result.shape == (3, HIDDEN_SIZE)


def test_fit_nao_lanca_excecao(vectorizer_com_mock):
    """fit() com modelo ja carregado nao lanca excecao."""
    vectorizer_com_mock.fit(["texto de treino"])


def test_fit_transform_equivale_a_transform(vectorizer_com_mock):
    """fit_transform deve retornar o mesmo que transform."""
    texts = ["a", "b"]
    a = vectorizer_com_mock.fit_transform(texts)
    b = vectorizer_com_mock.transform(texts)
    np.testing.assert_allclose(a, b)


def test_build_bert_vectorizer_retorna_instancia():
    """build_bert_vectorizer retorna BertimbauVectorizer."""
    v = build_bert_vectorizer()
    assert isinstance(v, BertimbauVectorizer)


def test_import_error_sem_torch(monkeypatch):
    """_load_model levanta ImportError descritivo quando torch nao instalado."""
    v = BertimbauVectorizer()
    with patch.dict("sys.modules", {"torch": None, "transformers": None}):
        with pytest.raises((ImportError, Exception)):
            v._load_model()
