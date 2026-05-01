from __future__ import annotations

from pathlib import Path

import matplotlib
import pandas as pd

from data_processing import DOCS_DIR


matplotlib.use("Agg")
import matplotlib.pyplot as plt


def save_confusion_matrix(matrix, labels: list[str], output_path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    image = ax.imshow(matrix, cmap="Blues")
    fig.colorbar(image, ax=ax)
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicao")
    ax.set_ylabel("Classe real")
    ax.set_title(title)
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, str(matrix[i, j]), ha="center", va="center")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def summarize_dataset(name: str, df: pd.DataFrame) -> dict[str, object]:
    return {
        "experimento": name,
        "linhas": int(len(df)),
        "negativo": int((df["label"] == "negativo").sum()),
        "neutro": int((df["label"] == "neutro").sum()),
        "positivo": int((df["label"] == "positivo").sum()),
    }


def metrics_markdown_table(df: pd.DataFrame) -> str:
    header = " | ".join(df.columns)
    separator = " | ".join(["---"] * len(df.columns))
    rows = [" | ".join(map(str, row)) for row in df.values.tolist()]
    return "\n".join([header, separator, *rows])


def write_report(
    availability: dict[str, str],
    dataset_summaries: list[dict[str, object]],
    metrics_df: pd.DataFrame,
    recommendations_df: pd.DataFrame | None = None,
) -> None:
    best_rows = (
        metrics_df.sort_values(["experimento", "f1_macro", "accuracy"], ascending=[True, False, False])
        .groupby("experimento", as_index=False)
        .first()
    )
    dataset_df = pd.DataFrame(dataset_summaries)
    recommendations_df = pd.DataFrame() if recommendations_df is None else recommendations_df
    if recommendations_df.empty:
        recommendation_text = "Nenhuma recomendacao foi gerada nesta execucao."
    else:
        recommendation_sample = recommendations_df.head(10)[
            [
                "tipo",
                "user_id",
                "rank",
                "source",
                "product_name",
                "category",
                "recommendation_score",
                "motivo",
            ]
        ].copy()
        recommendation_text = metrics_markdown_table(recommendation_sample)
    result_lines = []
    for _, row in best_rows.iterrows():
        result_lines.append(
            f"- {row['experimento']}: melhor F1 macro = {row['f1_macro']:.4f} com {row['modelo']}."
        )
    result_text = "\n".join(result_lines)

    report = f"""# Relatorio Final

## Introducao

Este projeto apresenta um pipeline simples de analise de sentimentos em avaliacoes de e-commerce brasileiro. A base principal e o corpus B2W-Reviews01. As bases complementares sao o Olist Brazilian E-Commerce Public Dataset e a coleta simples de reviews do Mercado Livre.

## Base teorica breve

A analise de sentimentos busca classificar a polaridade predominante de um texto. Em reviews de e-commerce, essa tarefa ajuda a resumir a percepcao do consumidor. Neste trabalho, a polaridade foi inferida a partir da nota da avaliacao, com tres classes: negativo, neutro e positivo.

## Bases de dados

- B2W-Reviews01: base principal, em portugues brasileiro, com reviews de produtos do comercio eletronico.
- Olist Brazilian E-Commerce Public Dataset: base complementar, usando a tabela de reviews.
- Mercado Livre simples: base complementar real obtida por coleta via API oficial, integrada ao pipeline no mesmo formato das demais bases.

Disponibilidade na execucao:

- B2W: {availability['b2w']}
- Olist: {availability['olist']}
- Mercado Livre simples: {availability['meli_simples']}

Resumo dos conjuntos usados:

{metrics_markdown_table(dataset_df)}

## Metodologia

O pipeline seguiu as etapas de carregamento dos dados, uniao dos campos textuais, limpeza do texto, remocao de stopwords, radicalizacao simples por sufixos, rotulagem por nota (1-2 negativo, 3 neutro, 4-5 positivo), vetorizacao com TF-IDF, treinamento de Regressao Logistica e Linear SVC e avaliacao com accuracy, precision, recall, F1-score macro e matriz de confusao. Alem dos modelos supervisionados, foi avaliado um analisador simbolico paralelo baseado em lexico de sentimento e regras discursivas simples. Foram considerados o experimento principal com B2W e experimentos combinados com Olist e Mercado Livre simples.

## Recomendacao de itens

Para atender ao uso academico de recomendacao, foi implementado um baseline explicavel baseado nas informacoes de sentimento. O campo `sentiment_label` representa a aproximacao de sentimento derivada da nota da avaliacao, usando a mesma regra das classes supervisionadas. Na B2W, o historico positivo de cada usuario define categorias de interesse, e o sistema recomenda produtos bem avaliados da mesma categoria, excluindo itens ja avaliados pelo usuario. Quando nao ha usuario disponivel, como na coleta simples do Mercado Livre, o sistema gera um ranking global de produtos por categoria usando media de nota, proporcao de avaliacoes positivas, sentimento medio e volume de reviews.

Exemplo de recomendacoes geradas:

{recommendation_text}

## Resultados

{metrics_markdown_table(metrics_df)}

{result_text}

## Limitacoes

O estudo possui limitacoes importantes: ruido textual, erros ortograficos, abreviacoes, ambiguidades semanticas, desbalanceamento entre classes e a propria limitacao de usar a nota numerica como aproximacao de sentimento textual. A recomendacao ainda e um baseline simples: Olist nao entra na recomendacao personalizada por falta de metadados de produto e usuario nesta versao, e Mercado Livre entra como recomendacao global porque a coleta local nao possui identificador de usuario.

## Conclusao

O projeto fornece uma linha de base simples e reproduzivel para classificacao de sentimentos em reviews de e-commerce brasileiro e para recomendacao inicial de itens baseada nesses sentimentos. A estrutura foi mantida propositalmente enxuta para facilitar apresentacao academica e execucao.

## Referencias

OLIST. Brazilian E-Commerce Public Dataset by Olist. Kaggle, 2018. Disponivel em: <https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce>. Acesso em: 19 abr. 2026.

REAL, L.; OSHIRO, M.; MAFRA, A. B2W-Reviews01: an open product reviews corpus. 2019. Disponivel em: <https://github.com/b2wdigital/b2w-reviews01>. Acesso em: 19 abr. 2026.

MERCADO LIVRE. Documentacao da API de opinioes sobre um produto. Disponivel em: <https://developers.mercadolivre.com.br/pt_br/opinioes-sobre-um-produto>. Acesso em: 19 abr. 2026.
"""
    (DOCS_DIR / "relatorio_final.md").write_text(report, encoding="utf-8")
