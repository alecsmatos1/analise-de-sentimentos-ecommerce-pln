"""Gera tabela comparativa unificada de todos os experimentos.

Le os JSONs de resultado em ``v2/outputs/`` (um por experimento) e monta uma
tabela linha-por-experimento com as metricas principais. Quando um JSON nao
existe, a linha aparece com placeholders ``"-"`` para manter a ordem canonica
das abordagens no artigo.

Saidas geradas pelas funcoes ``save_*``:

- ``v2/outputs/comparison_table.csv``: tabela em CSV (utf-8-sig).
- ``v2/outputs/comparison_report.md``: relatorio Markdown com a tabela e
  destaque automatico do melhor F1 Macro.

A renderizacao do Markdown e feita por ``_df_to_markdown`` para evitar a
dependencia opcional ``tabulate`` exigida por ``pandas.DataFrame.to_markdown``.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


EXPERIMENT_ORDER = [
    ("baseline",               "Baseline (majoritario)"),
    ("symbolic",               "Simbolico (v1)"),
    ("tfidf-linear",           "TF-IDF + LR"),
    ("word2vec-linear",        "Word2Vec NILC + LR"),
    ("bert-embeddings-linear", "BERTimbau frozen + LR"),
    ("bert-finetune",          "BERTimbau fine-tuned"),
    ("llm",                    "LLM GPT-4.1-nano (n=500)"),
]

METRIC_COLS = [
    ("accuracy",        "Acc"),
    ("f1_macro",        "F1 Macro"),
    ("precision_macro", "P Macro"),
    ("recall_macro",    "R Macro"),
    ("f1_negativo",     "F1 neg"),
    ("f1_neutro",       "F1 neu"),
    ("f1_positivo",     "F1 pos"),
]

MISSING = "-"


def _load_result(outputs_dir: Path, experiment_key: str) -> dict | None:
    """Carrega o JSON de resultado de um experimento, ou ``None`` se ausente."""
    path = outputs_dir / f"{experiment_key}.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def build_comparison_table(outputs_dir: Path | str = "v2/outputs") -> pd.DataFrame:
    """Le todos os JSONs disponiveis e monta DataFrame comparativo."""
    outputs_dir = Path(outputs_dir)
    rows: list[dict] = []
    for key, label in EXPERIMENT_ORDER:
        result = _load_result(outputs_dir, key)
        row: dict = {"Abordagem": label}
        if result is None:
            for _, col in METRIC_COLS:
                row[col] = MISSING
        else:
            row["Acc"]      = result.get("accuracy", MISSING)
            row["F1 Macro"] = result.get("f1_macro", MISSING)
            row["P Macro"]  = result.get("precision_macro", MISSING)
            row["R Macro"]  = result.get("recall_macro", MISSING)
            pc = result.get("per_class", {}) or {}
            row["F1 neg"]   = (pc.get("negativo") or {}).get("f1", MISSING)
            row["F1 neu"]   = (pc.get("neutro") or {}).get("f1", MISSING)
            row["F1 pos"]   = (pc.get("positivo") or {}).get("f1", MISSING)
        rows.append(row)
    df = pd.DataFrame(rows)
    display_cols = ["Abordagem"] + [col for _, col in METRIC_COLS]
    return df[display_cols]


def save_comparison_csv(
    df: pd.DataFrame, outputs_dir: Path | str = "v2/outputs"
) -> Path:
    """Salva a tabela comparativa em CSV (utf-8-sig)."""
    target_dir = Path(outputs_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    out = target_dir / "comparison_table.csv"
    df.to_csv(out, index=False, encoding="utf-8-sig")
    return out


def _format_cell(value: object) -> str:
    """Formata uma celula para Markdown.

    Numeros viram strings com 4 casas; placeholders ficam intactos.
    """
    if isinstance(value, float):
        return f"{value:.4f}"
    if isinstance(value, int) and not isinstance(value, bool):
        return f"{value:.4f}"
    return str(value)


def _df_to_markdown(df: pd.DataFrame) -> str:
    """Renderiza um DataFrame como tabela Markdown sem depender de ``tabulate``."""
    headers = [str(col) for col in df.columns]
    rows = [[_format_cell(v) for v in row] for row in df.itertuples(index=False, name=None)]
    header_line = "| " + " | ".join(headers) + " |"
    sep_line = "| " + " | ".join(["---"] * len(headers)) + " |"
    body_lines = ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join([header_line, sep_line, *body_lines])


def save_comparison_report(
    df: pd.DataFrame, outputs_dir: Path | str = "v2/outputs"
) -> Path:
    """Salva relatorio Markdown com a tabela e analise automatica."""
    target_dir = Path(outputs_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    out = target_dir / "comparison_report.md"

    lines = [
        "# Tabela Comparativa - Experimentos v2",
        "",
        "Corpus: B2W + Olist + Mercado Livre (171.745 reviews, desbalanceado).",
        "Split: 80/20 estratificado, seed=42.",
        "LLM avaliado em amostra estratificada de 500 reviews do conjunto de teste.",
        "",
        _df_to_markdown(df),
        "",
    ]

    numeric_rows = df[df["F1 Macro"] != MISSING].copy()
    if not numeric_rows.empty:
        numeric_rows["F1 Macro"] = pd.to_numeric(numeric_rows["F1 Macro"])
        best = numeric_rows.loc[numeric_rows["F1 Macro"].idxmax()]
        lines.append(
            f"**Melhor F1 Macro:** {best['Abordagem']} ({best['F1 Macro']:.4f})"
        )
        lines.append("")

    out.write_text("\n".join(lines), encoding="utf-8")
    return out
