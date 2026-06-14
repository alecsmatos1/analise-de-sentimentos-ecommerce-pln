"""Transformador TF-IDF configuravel para os experimentos da v2."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Union

from sklearn.feature_extraction.text import TfidfVectorizer


@dataclass(frozen=True)
class TfidfConfig:
    """Parametros usados para instanciar o TfidfVectorizer.

    Os defaults seguem o baseline da v1 (n-gramas 1-2, sublinear TF), com
    `min_df` levemente mais permissivo para suportar corpora menores.
    """

    ngram_range: tuple[int, int] = (1, 2)
    min_df: Union[int, float] = 2
    max_df: Union[int, float] = 0.95
    sublinear_tf: bool = True
    lowercase: bool = True
    max_features: Optional[int] = None
    strip_accents: Optional[str] = None
    norm: Optional[str] = "l2"


def build_tfidf(config: Optional[TfidfConfig] = None) -> TfidfVectorizer:
    """Cria um TfidfVectorizer a partir de uma configuracao opcional."""

    cfg = config or TfidfConfig()
    return TfidfVectorizer(
        ngram_range=cfg.ngram_range,
        min_df=cfg.min_df,
        max_df=cfg.max_df,
        sublinear_tf=cfg.sublinear_tf,
        lowercase=cfg.lowercase,
        max_features=cfg.max_features,
        strip_accents=cfg.strip_accents,
        norm=cfg.norm,
    )


__all__ = ["TfidfConfig", "build_tfidf"]
