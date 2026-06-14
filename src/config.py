"""Configuracao interna da v2.

Define caminhos locais para dados, embeddings, caches e saidas, alem da semente
padrao usada por split, modelos e quaisquer rotinas estocasticas. Esta sprint
cobre apenas o nucleo (Core data); valores especificos de cada experimento
(TF-IDF, NILC, BERTimbau, LLM) serao configurados em sprints posteriores que
adicionarao novos campos sem quebrar o contrato basico.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SEED = 42
DEFAULT_TEST_SIZE = 0.2


@dataclass(frozen=True)
class V2Config:
    """Configuracao reutilizavel pela v2.

    Campos sao caminhos absolutos para evitar ambiguidade entre execucoes em
    diretorios diferentes. `seed` e `test_size` sao usados pelo split comum.
    """

    repo_root: Path = REPO_ROOT
    data_dir: Path = REPO_ROOT / "data"
    embeddings_dir: Path = REPO_ROOT / "data" / "embeddings"
    cache_dir: Path = REPO_ROOT / "data" / "cache"
    outputs_dir: Path = REPO_ROOT / "v2" / "outputs"
    seed: int = DEFAULT_SEED
    test_size: float = DEFAULT_TEST_SIZE
    label_order: tuple[str, ...] = field(
        default=("negativo", "neutro", "positivo")
    )

    def with_overrides(self, **overrides: object) -> "V2Config":
        """Retorna uma copia com os campos sobrescritos.

        Caminhos em string sao convertidos para `Path` para manter a invariante
        do tipo dos campos.
        """

        cleaned: dict[str, object] = {}
        for key, value in overrides.items():
            if key in {"repo_root", "data_dir", "embeddings_dir", "cache_dir", "outputs_dir"}:
                cleaned[key] = Path(value) if not isinstance(value, Path) else value
            else:
                cleaned[key] = value
        return replace(self, **cleaned)


def default_config() -> V2Config:
    """Atalho para construir a configuracao padrao."""

    return V2Config()
