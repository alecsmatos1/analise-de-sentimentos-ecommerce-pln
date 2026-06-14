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


class _FakeBertVectorizer:
    """Vectorizer falso compativel com sklearn Pipeline.

    Mapeia heuristicamente termos de cada classe para vetores distintos, para
    que o classificador linear consiga separar as classes mesmo com features
    sinteticas. Evita carregar torch/BERT real nos testes de ``run()``.

    A assinatura aceita um argumento posicional opcional para imitar
    ``BertimbauVectorizer(config)``.
    """

    def __init__(self, config=None):
        self.config = config
        self.dim = 8
        self._keywords = {
            0: ("bom", "otimo", "amei", "excelente", "recomendo"),
            1: ("ruim", "pessimo", "horrivel", "decepcionante", "quebrado"),
            2: ("comum", "normal", "razoavel", "regular", "padrao"),
        }

    def fit(self, texts, y=None):
        return self

    def transform(self, texts):
        vectors = []
        for text in texts:
            v = np.zeros(self.dim, dtype=np.float32)
            lower = text.lower()
            for idx, words in self._keywords.items():
                for w in words:
                    if w in lower:
                        v[idx] += 1.0
                        v[idx + 3] += 0.5
            vectors.append(v)
        return np.vstack(vectors) if vectors else np.zeros((0, self.dim))

    def fit_transform(self, texts, y=None):
        return self.fit(texts, y).transform(texts)


def _synthetic_corpus():
    train_texts = [
        "produto otimo entrega rapida amei recomendo loja excelente",
        "muito bom recomendo demais loja confiavel adorei produto otimo",
        "adorei o produto chegou rapido qualidade excelente amei",
        "super recomendo produto otimo entrega rapida amei perfeito",
        "excelente atendimento nota dez otima loja recomendo",
        "comum nada demais entrega normal regular padrao",
        "razoavel produto medio nada surpreendente padrao normal",
        "ok funcionou esperava mais nada demais padrao comum",
        "regular dentro do esperado nada de especial comum normal",
        "normal aceitavel produto entregue padrao razoavel",
        "pessimo produto chegou quebrado nao recomendo horrivel",
        "horrivel atendimento demorou nao gostei pessimo quebrado",
        "ruim produto com defeito devolvi nao recomendo pessimo",
        "nao recomendo pessimo qualidade ruim horrivel defeito",
        "decepcionante produto quebrado nao funciona pessimo ruim",
    ]
    train_labels = (
        ["positivo"] * 5 + ["neutro"] * 5 + ["negativo"] * 5
    )
    test_texts = [
        "produto otimo recomendo entrega rapida amei",
        "razoavel nada demais comum padrao",
        "pessimo nao recomendo produto ruim horrivel",
    ]
    test_labels = ["positivo", "neutro", "negativo"]
    return train_texts, train_labels, test_texts, test_labels


def test_run_bert_embeddings_linear_com_mock():
    """``run_bert_embeddings_linear`` retorna BertEmbeddingsResult com per_class."""
    from v2.src.experiments.run_bert_embeddings_linear import (
        BertEmbeddingsResult,
        run_bert_embeddings_linear,
    )

    X_train, y_train, X_test, y_test = _synthetic_corpus()

    with patch(
        "v2.src.experiments.run_bert_embeddings_linear.BertimbauVectorizer",
        _FakeBertVectorizer,
    ):
        result = run_bert_embeddings_linear(X_train, y_train, X_test, y_test)

    assert isinstance(result, BertEmbeddingsResult)
    assert 0.0 <= result.accuracy <= 1.0
    assert 0.0 <= result.f1_macro <= 1.0
    assert set(result.labels) == {"negativo", "neutro", "positivo"}

    payload = result.as_dict()
    assert "per_class" in payload
    assert set(payload["per_class"].keys()) == {"negativo", "neutro", "positivo"}
    for cls, metrics in payload["per_class"].items():
        assert set(metrics.keys()) >= {"precision", "recall", "f1", "support"}
    assert "confusion_matrix" in payload
    assert len(payload["confusion_matrix"]) == 3
    assert payload["labels"] == ["negativo", "neutro", "positivo"]


def test_run_bert_embeddings_linear_as_dict_serializable():
    """``as_dict()`` produz somente tipos JSON-serializaveis."""
    import json

    from v2.src.experiments.run_bert_embeddings_linear import (
        run_bert_embeddings_linear,
    )

    X_train, y_train, X_test, y_test = _synthetic_corpus()
    with patch(
        "v2.src.experiments.run_bert_embeddings_linear.BertimbauVectorizer",
        _FakeBertVectorizer,
    ):
        result = run_bert_embeddings_linear(X_train, y_train, X_test, y_test)
    payload = result.as_dict()
    json.dumps(payload)  # nao deve lancar


def test_run_module_run_lanca_quando_corpus_inexistente(tmp_path):
    """``run(corpus_path=missing)`` levanta FileNotFoundError descritivo."""
    from v2.src.experiments.run_bert_embeddings_linear import run

    missing = tmp_path / "no_such_corpus.csv"
    with pytest.raises(FileNotFoundError):
        run(corpus_path=missing)
