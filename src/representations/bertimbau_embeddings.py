"""Representacao BERTimbau como extrator de embeddings congelados."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class BertimbauConfig:
    """Configuracao do extrator de embeddings BERTimbau."""
    model_name: str = "neuralmind/bert-base-portuguese-cased"
    max_length: int = 128
    batch_size: int = 32
    device: str = "cpu"
    cache_dir: str | None = None
    vector_size: int = 768


class BertimbauVectorizer:
    """Extrai embeddings contextuais do BERTimbau (frozen).

    Usa mean pooling dos tokens de saida (excluindo tokens especiais
    [CLS] e [SEP]) para produzir um vetor de tamanho fixo por review.

    O modelo nao e carregado no __init__ — somente no primeiro fit() ou
    transform(), para evitar erro de import quando torch nao esta instalado.
    """

    def __init__(self, config: BertimbauConfig | None = None) -> None:
        self.config = config or BertimbauConfig()
        self._tokenizer = None
        self._model = None

    def _load_model(self) -> None:
        """Carrega tokenizer e modelo BERTimbau via transformers."""
        try:
            import torch  # noqa: F401
            from transformers import AutoModel, AutoTokenizer
        except ImportError as exc:
            raise ImportError(
                "torch e/ou transformers nao estao instalados. Execute:\n"
                "pip install torch transformers"
            ) from exc
        cfg = self.config
        self._tokenizer = AutoTokenizer.from_pretrained(
            cfg.model_name, cache_dir=cfg.cache_dir
        )
        self._model = AutoModel.from_pretrained(
            cfg.model_name, cache_dir=cfg.cache_dir
        )
        self._model.eval()
        try:
            import torch
            self._model = self._model.to(torch.device(cfg.device))
        except Exception:
            pass

    def _embed_batch(self, texts: list[str]) -> np.ndarray:
        """Processa um batch de textos e retorna embeddings (N, 768)."""
        import torch
        cfg = self.config
        enc = self._tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=cfg.max_length,
            return_tensors="pt",
        )
        enc = {k: v.to(torch.device(cfg.device)) for k, v in enc.items()}
        with torch.no_grad():
            out = self._model(**enc)
        # last_hidden_state: (batch, seq_len, hidden)
        hidden = out.last_hidden_state
        # attention_mask: 1 para tokens reais, 0 para padding
        mask = enc["attention_mask"].unsqueeze(-1).float()
        # excluir [CLS] (pos 0) e usar mask para mean pooling
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
        return pooled.cpu().numpy()

    def fit(self, texts: Sequence[str], y=None) -> "BertimbauVectorizer":
        if self._model is None:
            self._load_model()
        return self

    def transform(self, texts: Sequence[str]) -> np.ndarray:
        if self._model is None:
            self._load_model()
        texts_list = list(texts)
        cfg = self.config
        batches = [
            texts_list[i: i + cfg.batch_size]
            for i in range(0, len(texts_list), cfg.batch_size)
        ]
        return np.vstack([self._embed_batch(b) for b in batches])

    def fit_transform(self, texts: Sequence[str], y=None) -> np.ndarray:
        return self.fit(texts, y).transform(texts)


def build_bert_vectorizer(
    config: BertimbauConfig | None = None,
) -> BertimbauVectorizer:
    """Instancia o vetorizador BERTimbau com a configuracao fornecida."""
    return BertimbauVectorizer(config)
