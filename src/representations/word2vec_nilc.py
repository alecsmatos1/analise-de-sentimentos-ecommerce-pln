"""Representacao Word2Vec NILC pre-treinado."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

# Caminho absoluto para o diretório de embeddings NILC, independente do CWD
_NILC_DIR = Path(__file__).resolve().parents[3] / "v2" / "data" / "embeddings" / "nilc"
_DEFAULT_MODEL_PATH = _NILC_DIR / "cbow_s300.bin"


@dataclass(frozen=True)
class Word2VecConfig:
    """Configuracao do vectorizer Word2Vec NILC."""

    model_path: Path = _DEFAULT_MODEL_PATH
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
        """Carrega o modelo NILC — suporta safetensors (HuggingFace) ou .bin (gensim)."""
        model_dir = Path(self.config.model_path).parent
        safetensors_path = model_dir / "embeddings.safetensors"
        vocab_path = model_dir / "vocab.txt"
        binary_path = Path(self.config.model_path)

        if safetensors_path.exists() and vocab_path.exists():
            self._load_from_safetensors(safetensors_path, vocab_path)
        elif binary_path.exists():
            self._load_from_binary(binary_path)
        else:
            raise FileNotFoundError(
                f"Vetores NILC nao encontrados. Esperado:\n"
                f"  (a) {safetensors_path} + {vocab_path}  "
                "[HuggingFace: nilc-nlp/word2vec-cbow-300d]\n"
                f"  (b) {binary_path}  [formato gensim binario]"
            )

    def _load_from_safetensors(self, emb_path: Path, vocab_path: Path) -> None:
        try:
            from safetensors.numpy import load_file
        except ImportError as exc:
            raise ImportError(
                "safetensors nao esta instalado. Execute: pip install safetensors"
            ) from exc
        tensors = load_file(str(emb_path))
        matrix = tensors[next(iter(tensors))].astype(np.float32)
        with open(vocab_path, encoding="utf-8") as fh:
            words = [line.strip() for line in fh if line.strip()]
        self._model = dict(zip(words, matrix))

    def _load_from_binary(self, path: Path) -> None:
        try:
            from gensim.models import KeyedVectors
        except ImportError as exc:
            raise ImportError(
                "gensim nao esta instalado. Execute: pip install gensim"
            ) from exc
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
