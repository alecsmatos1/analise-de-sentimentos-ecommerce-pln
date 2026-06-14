"""Carregamento e unificacao dos corpora reais para experimentos v2.

Nenhuma dependencia de rede, NILC, BERTimbau ou LLM. Apenas leitura
de CSVs locais em data/ e coercao via v2.src.data.coerce_corpus().
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from v2.src.data import coerce_corpus

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DATA_DIR = _REPO_ROOT / "data"

SCORE_TO_LABEL: dict[int, str] = {
    1: "negativo",
    2: "negativo",
    3: "neutro",
    4: "positivo",
    5: "positivo",
}


def _load_b2w() -> pd.DataFrame:
    path = _DATA_DIR / "B2W-Reviews01.csv"
    df = pd.read_csv(path, usecols=["review_text", "overall_rating"], low_memory=False)
    df = df.rename(columns={"review_text": "raw_text", "overall_rating": "score"})
    df["label"] = df["score"].map(SCORE_TO_LABEL)
    df["source"] = "b2w"
    return df[["raw_text", "label", "source"]].dropna(subset=["raw_text", "label"])


def _load_olist() -> pd.DataFrame:
    path = _DATA_DIR / "olist_order_reviews_dataset.csv"
    df = pd.read_csv(path, usecols=["review_comment_message", "review_score"])
    df = df.rename(
        columns={"review_comment_message": "raw_text", "review_score": "score"}
    )
    df["label"] = df["score"].map(SCORE_TO_LABEL)
    df["source"] = "olist"
    return df[["raw_text", "label", "source"]].dropna(subset=["raw_text", "label"])


def _load_mercadolivre() -> pd.DataFrame:
    path = _DATA_DIR / "mercadolivre_reviews.csv"
    df = pd.read_csv(path, usecols=["review_text", "rating"])
    df = df.rename(columns={"review_text": "raw_text", "rating": "score"})
    df["label"] = df["score"].map(SCORE_TO_LABEL)
    df["source"] = "mercadolivre"
    return df[["raw_text", "label", "source"]].dropna(subset=["raw_text", "label"])


def load_corpus(
    sources: list[str] | None = None,
    min_per_class: int | None = None,
    max_per_class: int | None = None,
) -> pd.DataFrame:
    """Carrega e unifica os corpora reais, retornando DataFrame no formato canonico.

    Args:
        sources: lista de fontes a incluir ('b2w', 'olist', 'mercadolivre').
            None = todas.
        min_per_class: descarta classes com menos de N exemplos apos coercao.
        max_per_class: limita a N exemplos por classe (amostragem aleatoria,
            seed=42 para reprodutibilidade).

    Returns:
        DataFrame com colunas (raw_text, clean_text, label, source) validado
        por coerce_corpus().
    """
    loaders = {
        "b2w": _load_b2w,
        "olist": _load_olist,
        "mercadolivre": _load_mercadolivre,
    }
    if sources is None:
        sources = list(loaders.keys())

    parts = [loaders[s]() for s in sources if s in loaders]
    if not parts:
        raise ValueError(f"Nenhuma fonte valida em: {sources}")

    raw = pd.concat(parts, ignore_index=True)
    df = coerce_corpus(raw)

    if min_per_class is not None:
        counts = df["label"].value_counts()
        valid = counts[counts >= min_per_class].index
        df = df[df["label"].isin(valid)]

    if max_per_class is not None:
        # pandas 3.x exclui a coluna de agrupamento do resultado de groupby+apply;
        # iterar explicitamente preserva o contrato (raw_text, clean_text, label,
        # source) sem depender de comportamento que mudou entre versoes.
        parts_sampled = [
            group.sample(min(len(group), max_per_class), random_state=42)
            for _, group in df.groupby("label", sort=False)
        ]
        df = pd.concat(parts_sampled, ignore_index=True)

    return df


def save_corpus(df: pd.DataFrame, output_path: Path | str | None = None) -> Path:
    """Persiste o corpus processado em CSV.

    Args:
        df: DataFrame retornado por load_corpus().
        output_path: caminho destino. Default: v2/data/processed/corpus.csv.

    Returns:
        Path do arquivo salvo.
    """
    if output_path is None:
        output_path = _REPO_ROOT / "v2" / "data" / "processed" / "corpus.csv"
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    return output_path
