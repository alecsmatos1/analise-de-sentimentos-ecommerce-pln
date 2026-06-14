"""Configuracao compartilhada do pytest para a suite interna da v2.

Sem dependencia de NILC, BERTimbau, LLM ou dataset bruto. As fixtures aqui
expostas alimentam testes de contrato que asseguram que futuras ondas da v2
respeitem as colunas e os rotulos comuns acordados em ``v2/avaliacao.md`` e
``v2/arquitetura.md``.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Mapping

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from v2.tests.fixtures import SAMPLE_REVIEWS

REQUIRED_COLUMNS = frozenset({"raw_text", "clean_text", "label", "source"})
OPTIONAL_COLUMNS = frozenset({"score"})
ALLOWED_LABELS = frozenset({"positivo", "neutro", "negativo"})
SCORE_TO_LABEL = {
    1: "negativo",
    2: "negativo",
    3: "neutro",
    4: "positivo",
    5: "positivo",
}


@pytest.fixture(scope="session")
def sample_reviews() -> List[Mapping[str, object]]:
    return [dict(row) for row in SAMPLE_REVIEWS]


@pytest.fixture(scope="session")
def required_columns() -> frozenset:
    return REQUIRED_COLUMNS


@pytest.fixture(scope="session")
def allowed_labels() -> frozenset:
    return ALLOWED_LABELS


@pytest.fixture(scope="session")
def score_to_label() -> Mapping[int, str]:
    return dict(SCORE_TO_LABEL)
