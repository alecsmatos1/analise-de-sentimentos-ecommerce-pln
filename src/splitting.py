"""Split estratificado comum para a v2.

Garante que todos os experimentos da v2 usem o mesmo particionamento (80/20 por
padrao, estratificado por rotulo, semente fixa), conforme `v2/avaliacao.md`.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.model_selection import train_test_split

from .config import DEFAULT_SEED, DEFAULT_TEST_SIZE
from .data import REQUIRED_COLUMNS, coerce_corpus


@dataclass(frozen=True)
class StratifiedSplit:
    """Resultado de um split estratificado.

    Os DataFrames preservam o contrato da v2 (`raw_text`, `clean_text`,
    `label`, `source`) e possuem indices reiniciados para evitar surpresas em
    iteracoes.
    """

    train: pd.DataFrame
    test: pd.DataFrame
    seed: int
    test_size: float


def stratified_split(
    df: pd.DataFrame,
    *,
    seed: int = DEFAULT_SEED,
    test_size: float = DEFAULT_TEST_SIZE,
) -> StratifiedSplit:
    """Divide o corpus em treino e teste com estratificacao por `label`.

    O DataFrame de entrada e passado por `coerce_corpus` para garantir o
    contrato da v2 antes do split. `test_size` deve estar em (0, 1).
    """

    if not 0.0 < test_size < 1.0:
        raise ValueError("test_size deve estar entre 0 e 1, exclusive")

    corpus = coerce_corpus(df)
    if corpus.empty:
        raise ValueError("corpus vazio apos coercao; nada para dividir")

    counts = corpus["label"].value_counts()
    if (counts < 2).any():
        rare = counts[counts < 2].index.tolist()
        raise ValueError(
            "estratificacao exige pelo menos 2 exemplos por classe; "
            f"classes com poucos exemplos: {rare}"
        )

    train_df, test_df = train_test_split(
        corpus,
        test_size=test_size,
        random_state=seed,
        stratify=corpus["label"],
    )

    train_df = train_df.loc[:, list(REQUIRED_COLUMNS)].reset_index(drop=True)
    test_df = test_df.loc[:, list(REQUIRED_COLUMNS)].reset_index(drop=True)

    return StratifiedSplit(
        train=train_df,
        test=test_df,
        seed=seed,
        test_size=test_size,
    )
