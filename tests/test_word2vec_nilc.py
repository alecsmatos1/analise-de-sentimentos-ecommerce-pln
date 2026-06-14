"""Testes do contrato Word2Vec NILC (mock - sem arquivo binario)."""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from v2.src.representations.word2vec_nilc import (  # noqa: E402
    Word2VecConfig,
    Word2VecVectorizer,
    build_word2vec_vectorizer,
)


def _make_mock_model(vocab: dict[str, np.ndarray]) -> MagicMock:
    """Cria mock de KeyedVectors com vocabulario fixo."""
    mock = MagicMock()
    mock.__contains__ = lambda self, key: key in vocab
    mock.__getitem__ = lambda self, key: vocab[key]
    return mock


VOCAB = {
    "produto": np.array([1.0, 0.0, 0.0], dtype=np.float32),
    "bom": np.array([0.0, 1.0, 0.0], dtype=np.float32),
    "ruim": np.array([0.0, 0.0, 1.0], dtype=np.float32),
}


@pytest.fixture
def vectorizer_com_mock():
    """Word2VecVectorizer com modelo NILC mockado."""
    config = Word2VecConfig(vector_size=3)
    v = Word2VecVectorizer(config)
    v._model = _make_mock_model(VOCAB)
    return v


def test_vectorizer_retorna_media_dos_tokens(vectorizer_com_mock):
    """Vetorizador retorna media dos vetores das palavras conhecidas."""
    vec = vectorizer_com_mock.transform(["produto bom"])
    expected = np.mean([VOCAB["produto"], VOCAB["bom"]], axis=0)
    np.testing.assert_allclose(vec[0], expected)


def test_vectorizer_zeros_para_oov(vectorizer_com_mock):
    """Texto sem tokens conhecidos retorna vetor zero."""
    vec = vectorizer_com_mock.transform(["xyzabc"])
    assert vec.shape == (1, 3)
    np.testing.assert_allclose(vec[0], np.zeros(3))


def test_vectorizer_ignora_oov_na_media(vectorizer_com_mock):
    """Token OOV nao afeta a media dos tokens conhecidos."""
    vec_com_oov = vectorizer_com_mock.transform(["produto bom xyzabc"])
    vec_sem_oov = vectorizer_com_mock.transform(["produto bom"])
    np.testing.assert_allclose(vec_com_oov[0], vec_sem_oov[0])


def test_fit_transform_retorna_array(vectorizer_com_mock):
    """fit_transform retorna ndarray com shape (n_samples, vector_size)."""
    texts = ["produto bom", "ruim entrega"]
    result = vectorizer_com_mock.fit_transform(texts)
    assert isinstance(result, np.ndarray)
    assert result.shape == (2, 3)


def test_build_word2vec_vectorizer_retorna_instancia():
    """build_word2vec_vectorizer retorna Word2VecVectorizer."""
    v = build_word2vec_vectorizer()
    assert isinstance(v, Word2VecVectorizer)


def test_arquivo_ausente_lanca_erro(tmp_path, monkeypatch):
    """FileNotFoundError quando o binario NILC nao existe."""
    # Injeta modulos gensim mockados em sys.modules para que o import
    # interno em _load_model nao falhe (e o erro testado seja o de
    # arquivo ausente, nao o de dependencia ausente).
    fake_gensim = MagicMock()
    fake_gensim_models = MagicMock()
    fake_gensim_models.KeyedVectors = MagicMock()
    monkeypatch.setitem(sys.modules, "gensim", fake_gensim)
    monkeypatch.setitem(sys.modules, "gensim.models", fake_gensim_models)

    config = Word2VecConfig(model_path=tmp_path / "inexistente.bin", vector_size=3)
    v = Word2VecVectorizer(config)
    with pytest.raises(FileNotFoundError, match="nilc"):
        v._load_model()
