"""Configuracao compartilhada do pytest para a suite interna da v2.

Sem dependencia de NILC, BERTimbau, LLM ou dataset bruto. As fixtures aqui
expostas alimentam testes de contrato que asseguram que futuras ondas da v2
respeitem as colunas e os rotulos comuns acordados em ``v2/avaliacao.md`` e
``v2/arquitetura.md``.

As constantes ``REQUIRED_COLUMNS`` e ``ALLOWED_LABELS`` sao importadas de
``v2.src.data`` quando disponivel (cenario integrado, apos merge das sprints).
Em worktrees isolados, onde ``v2/src/data.py`` ainda nao existe, os mesmos
valores sao definidos localmente como fallback. O teste
``test_required_columns_consistente_com_src_data`` em ``test_contracts.py``
verifica automaticamente que ambas as definicoes nao divergem.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Mapping

import pytest

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

REPO_ROOT = _PROJECT_ROOT  # alias used by sprint tests from core-data

try:
    from v2.src.data import (  # type: ignore[import-not-found]
        REQUIRED_COLUMNS as _SRC_COLS,
        SENTIMENT_LABELS as _SRC_LABELS,
    )

    REQUIRED_COLUMNS = frozenset(_SRC_COLS)
    ALLOWED_LABELS = frozenset(_SRC_LABELS)
except ImportError:
    REQUIRED_COLUMNS = frozenset({"raw_text", "clean_text", "label", "source"})
    ALLOWED_LABELS = frozenset({"positivo", "neutro", "negativo"})

OPTIONAL_COLUMNS = frozenset({"score"})
SCORE_TO_LABEL: Mapping[int, str] = {
    1: "negativo",
    2: "negativo",
    3: "neutro",
    4: "positivo",
    5: "positivo",
}

from v2.tests.fixtures import SAMPLE_REVIEWS  # noqa: E402


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
