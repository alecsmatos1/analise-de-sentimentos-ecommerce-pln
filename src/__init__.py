"""Pacote interno da v2.

Reune contrato de corpus, configuracao e split estratificado comuns a todos os
experimentos da v2. Modulos externos como representacoes, modelos e CLI sao
criados em sprints posteriores.
"""

from .config import V2Config, default_config
from .data import (
    LABEL_NEGATIVE,
    LABEL_NEUTRAL,
    LABEL_POSITIVE,
    REQUIRED_COLUMNS,
    SENTIMENT_LABELS,
    coerce_corpus,
    rating_to_label,
)
from .splitting import StratifiedSplit, stratified_split

__all__ = [
    "LABEL_NEGATIVE",
    "LABEL_NEUTRAL",
    "LABEL_POSITIVE",
    "REQUIRED_COLUMNS",
    "SENTIMENT_LABELS",
    "StratifiedSplit",
    "V2Config",
    "coerce_corpus",
    "default_config",
    "rating_to_label",
    "stratified_split",
]
