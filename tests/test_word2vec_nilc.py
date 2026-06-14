"""Testes do contrato Word2Vec NILC (mock - sem arquivo binario)."""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from v2.src.experiments.run_word2vec_linear import (  # noqa: E402
    Word2VecLinearResult,
    run,
)
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


# ---------------------------------------------------------------------------
# Testes de integracao: run() com corpus sintetico + mock do vectorizer
# ---------------------------------------------------------------------------


_SYNTHETIC_CORPUS_ROWS = [
    ("produto otimo recomendo muito", "positivo"),
    ("excelente entrega rapida amei", "positivo"),
    ("perfeito adorei loja confiavel", "positivo"),
    ("super recomendo nota dez", "positivo"),
    ("amei produto top demais", "positivo"),
    ("comum nada demais regular", "neutro"),
    ("razoavel mediano ok padrao", "neutro"),
    ("normal aceitavel dentro do esperado", "neutro"),
    ("regular comum padrao mediano", "neutro"),
    ("ok funcionou esperava mais", "neutro"),
    ("pessimo produto quebrado horrivel", "negativo"),
    ("horrivel ruim nao recomendo", "negativo"),
    ("ruim defeito devolvi pessimo", "negativo"),
    ("nao recomendo decepcionante ruim", "negativo"),
    ("pessimo nao funciona quebrado", "negativo"),
]


def _write_synthetic_corpus(path: Path) -> None:
    rows = [
        {
            "raw_text": text,
            "clean_text": text,
            "label": label,
            "source": "synthetic",
        }
        for text, label in _SYNTHETIC_CORPUS_ROWS
    ]
    pd.DataFrame(rows).to_csv(path, index=False)


class _FakeWord2VecVectorizer:
    """Stub do vectorizer NILC: gera vetores determinsticos por hash do texto.

    Evita carregar o arquivo binario NILC (~1,8 GB). Mantem assinatura
    (fit/transform/fit_transform) compativel com o real para que
    run_word2vec_linear.run() funcione sem mudancas.
    """

    def __init__(self, *args, **kwargs) -> None:
        self.vector_size = 8

    def fit(self, texts, y=None):
        return self

    def transform(self, texts):
        rng = np.random.default_rng(42)
        vectors = []
        for text in texts:
            seed = abs(hash(text)) % (2**32)
            local = np.random.default_rng(seed)
            vectors.append(local.standard_normal(self.vector_size).astype(np.float32))
        # mantem rng para nao ser otimizado pelo linter; nao usado
        _ = rng.standard_normal(1)
        return np.array(vectors)

    def fit_transform(self, texts, y=None):
        return self.fit(texts, y).transform(texts)


def test_run_com_corpus_sintetico_e_mock_do_vectorizer(tmp_path):
    """run() executa fim-a-fim com vectorizer mockado e corpus sintetico."""
    corpus_path = tmp_path / "corpus.csv"
    _write_synthetic_corpus(corpus_path)

    with patch(
        "v2.src.experiments.run_word2vec_linear.Word2VecVectorizer",
        _FakeWord2VecVectorizer,
    ):
        result = run(corpus_path=corpus_path)

    assert isinstance(result, Word2VecLinearResult)
    assert 0.0 <= result.accuracy <= 1.0
    assert 0.0 <= result.f1_macro <= 1.0
    assert set(result.labels).issubset({"negativo", "neutro", "positivo"})
    assert len(result.confusion_matrix) == len(result.labels)
    assert all(len(row) == len(result.labels) for row in result.confusion_matrix)


def test_run_lanca_erro_quando_corpus_nao_existe(tmp_path):
    """run() falha com FileNotFoundError quando o corpus nao existe."""
    with pytest.raises(FileNotFoundError):
        run(corpus_path=tmp_path / "inexistente.csv")


def test_as_dict_contem_todos_os_campos_esperados():
    """as_dict() expoe o contrato compativel com reporting.save_all."""
    result = Word2VecLinearResult(
        accuracy=0.8,
        precision_macro=0.79,
        recall_macro=0.78,
        f1_macro=0.785,
        labels=["negativo", "neutro", "positivo"],
        per_class={
            "negativo": {"precision": 0.8, "recall": 0.8, "f1": 0.8, "support": 5},
            "neutro": {"precision": 0.7, "recall": 0.7, "f1": 0.7, "support": 5},
            "positivo": {"precision": 0.9, "recall": 0.9, "f1": 0.9, "support": 5},
        },
        confusion_matrix=[[4, 1, 0], [1, 3, 1], [0, 1, 4]],
    )
    payload = result.as_dict()
    expected_keys = {
        "accuracy",
        "precision_macro",
        "recall_macro",
        "f1_macro",
        "labels",
        "per_class",
        "confusion_matrix",
    }
    assert expected_keys.issubset(payload.keys())
    assert payload["accuracy"] == 0.8
    assert payload["labels"] == ["negativo", "neutro", "positivo"]
    assert payload["confusion_matrix"] == [[4, 1, 0], [1, 3, 1], [0, 1, 4]]
