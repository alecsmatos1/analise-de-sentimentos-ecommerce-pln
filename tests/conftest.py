"""Configuracao de testes: garante que `v2.src` seja importavel.

Sem arquivo de configuracao de projeto na raiz, pytest nao adiciona o repo
ao `sys.path`. Este `conftest.py` insere o diretorio raiz para permitir
`from v2.src.data import ...` nos testes desta sprint.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
