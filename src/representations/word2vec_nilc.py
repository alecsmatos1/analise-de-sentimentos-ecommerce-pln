"""Representacao Word2Vec NILC pre-treinado."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class Word2VecConfig:
    """Configuracao do vectorizer Word2Vec NILC."""

    model_path: Path = Path("data/embeddings/nilc/cbow_s300.bin")
    vector_size: int = 300
    unknown_strategy: str = "zeros"  # "zeros" | "random"
    random_state: int = 42


class Word2VecVectorizer:
    """Vetoriza textos usando Word2Vec NILC pre-treinado (media dos tokens).

    O modelo nao e carregado no __init__ - somente na primeira chamada a
    fit() ou transform(), para evitar erro quando o arquivo nao existe.
    """

    def __init__(self, config: Word2VecConfig | None = None) -> None:
        self.config = config or Word2VecConfig()
        self._model = None

    def _load_model(self) -> None:
        """Carrega o modelo binario NILC via gensim."""
        try:
            from gensim.models import KeyedVectors
        except ImportError as exc:
            raise ImportError(
                "gensim nao esta instalado. Execute: pip install gensim"
            ) from exc
        path = Path(self.config.model_path)
        if not path.exists():
            raise FileNotFoundError(
                f"Vetores NILC nao encontrados em '{path}'. "
                "Baixe em: http://nilc.icmc.usp.br/embeddings (cbow_s300.zip) "
                "e descompacte em data/embeddings/nilc/cbow_s300.bin"
            )
        self._model = KeyedVectors.load_word2vec_format(str(path), binary=True)

    def _vectorize_text(self, text: str) -> np.ndarray:
        """Retorna vetor medio das palavras presentes no vocabulario."""
        tokens = text.lower().split()
        vectors = []
        for token in tokens:
            if token in self._model:
                vectors.append(self._model[token])
        if vectors:
            return np.mean(vectors, axis=0)
        if self.config.unknown_strategy == "random":
            rng = np.random.default_rng(self.config.random_state)
            return rng.standard_normal(self.config.vector_size).astype(np.float32)
        return np.zeros(self.config.vector_size, dtype=np.float32)

    def fit(self, texts: Sequence[str], y=None) -> "Word2VecVectorizer":
        if self._model is None:
            self._load_model()
        return self

    def transform(self, texts: Sequence[str]) -> np.ndarray:
        if self._model is None:
            self._load_model()
        return np.array([self._vectorize_text(t) for t in texts])

    def fit_transform(self, texts: Sequence[str], y=None) -> np.ndarray:
        return self.fit(texts, y).transform(texts)


def build_word2vec_vectorizer(
    config: Word2VecConfig | None = None,
) -> Word2VecVectorizer:
    """Instancia o vectorizer Word2Vec com a configuracao fornecida."""
    return Word2VecVectorizer(config)


__all__ = ["Word2VecConfig", "Word2VecVectorizer", "build_word2vec_vectorizer"]
