from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from data_processing import DOCS_DIR, ROOT


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


def evaluate_on_gold(
    trained_pipelines: dict,
    df_gold: pd.DataFrame,
    train_experiment_name: str = "b2w_principal",
) -> None:
    """
    Avalia modelos supervisionados e simbólico sobre um corpus gold externo.

    Corpus gold: Olist (avaliação cruzada entre plataformas — modelo treinado em
    B2W, avaliado em Olist, que é uma plataforma independente). Ver decisão D2-gold.

    df_gold deve ter colunas: raw_text (simbólico), clean_text (supervisionados), label.
    Salva tabela em artifacts/gold_evaluation.md.
    """
    from sentiment_analyzer import analyze_symbolic_sentiment, LABEL_ORDER

    artifacts_dir = ROOT / "artifacts"
    artifacts_dir.mkdir(exist_ok=True)

    y_true = df_gold["label"].tolist()
    rows = []

    # Modelos supervisionados
    for model_name, pipeline in trained_pipelines.items():
        preds = pipeline.predict(df_gold["clean_text"]).tolist()
        acc = accuracy_score(y_true, preds)
        prec, rec, f1, _ = precision_recall_fscore_support(
            y_true, preds, average="macro", zero_division=0, labels=LABEL_ORDER
        )
        rows.append({
            "modelo": model_name,
            "corpus_treino": train_experiment_name,
            "corpus_gold": "olist",
            "linhas_gold": len(df_gold),
            "accuracy": round(float(acc), 4),
            "precision_macro": round(float(prec), 4),
            "recall_macro": round(float(rec), 4),
            "f1_macro": round(float(f1), 4),
        })

    # Analisador simbólico
    sym_preds = [analyze_symbolic_sentiment(t)["label"] for t in df_gold["raw_text"]]
    acc = accuracy_score(y_true, sym_preds)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, sym_preds, average="macro", zero_division=0, labels=LABEL_ORDER
    )
    rows.append({
        "modelo": "symbolic_rules",
        "corpus_treino": "n/a",
        "corpus_gold": "olist",
        "linhas_gold": len(df_gold),
        "accuracy": round(float(acc), 4),
        "precision_macro": round(float(prec), 4),
        "recall_macro": round(float(rec), 4),
        "f1_macro": round(float(f1), 4),
    })

    df_results = pd.DataFrame(rows)
    table = metrics_markdown_table(df_results)

    report = f"""# Avaliacao Gold — Avaliacao Cruzada entre Plataformas

## Configuracao

- **Corpus gold:** Olist Brazilian E-Commerce Public Dataset (plataforma independente de B2W)
- **Modelo supervisionado treinado em:** {train_experiment_name}
- **Justificativa:** Os modelos supervisionados nunca viram dados Olist durante o treino.
  Isso testa generalizacao entre plataformas de e-commerce brasileiras distintas.
- **Limitacao:** Rotulos derivados de notas numericas (mesma metodologia do treino).
  Substituir por corpus com anotacao humana quando disponivel (SentiBR estava inacessivel em 2026-06-04).

## Resultados

{table}

## Interpretacao

Comparar estes resultados com os resultados em-dominio (metricas.csv) revela o gap de
generalizacao entre plataformas. Uma queda menor indica maior robustez do modelo.
"""
    (artifacts_dir / "gold_evaluation.md").write_text(report, encoding="utf-8")
    print(f"\nAvaliacao gold salva em artifacts/gold_evaluation.md")
    print(df_results.to_string(index=False))


def evaluate_on_repro(
    trained_pipelines: dict,
    path: str = "data/gold/repro.csv",
) -> None:
    """
    Avalia os três sistemas (LR, LinearSVC, simbólico) no corpus gold RePro.

    RePro é usado APENAS para avaliação — nunca para treino. Os modelos supervisionados
    recebem o texto pré-processado via clean_text(); o analisador simbólico recebe o texto bruto.

    Referência: dos Santos Silva et al. (2024), PROPOR,
    "RePro: a benchmark for Opinion Mining in Brazilian Portuguese".

    Parâmetros
    ----------
    trained_pipelines : dict {"logistic_regression": Pipeline, "linear_svc": Pipeline}
        Modelos treinados no experimento b2w_principal.
    path : str
        Caminho para o arquivo CSV do RePro (padrão: data/gold/repro.csv).
        Se o arquivo não existir, a função imprime aviso e retorna sem erro.
    """
    from pathlib import Path as _Path
    from sentiment_analyzer import analyze_symbolic_sentiment, LABEL_ORDER
    from data_processing import clean_text

    p = _Path(path)
    if not p.exists():
        print(f"\nRePro não encontrado em {path} — avaliação gold pulada.")
        return

    df = pd.read_csv(p)

    # Formato real do RePro (inspecionado em 2026-06-04):
    #   Texto: coluna "review_text" (review_title concatenado quando disponível)
    #   Rótulo: coluna "polarity" com strings de listas, ex: "['POSITIVO']", "['NEGATIVO']",
    #           "['NEUTRO']", "['NEGATIVO', 'POSITIVO']"
    #
    # Mapeamento de rótulos:
    #   ['POSITIVO']            → positivo
    #   ['NEGATIVO']            → negativo
    #   ['NEUTRO']              → neutro
    #   ['NEGATIVO','POSITIVO'] → neutro (sentimentos conflitantes = polaridade mista ≈ neutro)
    #
    # Distribuição no corpus (10.003 exemplos):
    #   positivo: 4.127 | negativo: 3.449 | neutro: 409 | misto→neutro: 2.018

    title = df["review_title"].fillna("").astype(str).str.strip()
    body  = df["review_text"].fillna("").astype(str).str.strip()
    df["_text"] = (title + " " + body).str.strip()

    def _map_polarity(val: str) -> str:
        val = str(val).upper()
        has_pos = "POSITIVO" in val
        has_neg = "NEGATIVO" in val
        if has_pos and has_neg:
            return "neutro"
        if has_pos:
            return "positivo"
        if has_neg:
            return "negativo"
        return "neutro"

    df["_label"] = df["polarity"].apply(_map_polarity)
    df["_clean"] = df["_text"].apply(clean_text)

    y_true = df["_label"].tolist()
    rows = []

    for model_name, pipeline in trained_pipelines.items():
        preds = pipeline.predict(df["_clean"]).tolist()
        acc = accuracy_score(y_true, preds)
        prec, rec, f1, _ = precision_recall_fscore_support(
            y_true, preds, average="macro", zero_division=0, labels=LABEL_ORDER
        )
        rows.append({
            "modelo": model_name, "corpus_treino": "b2w_principal", "corpus_gold": "repro",
            "linhas_gold": len(df),
            "accuracy": round(float(acc), 4), "precision_macro": round(float(prec), 4),
            "recall_macro": round(float(rec), 4), "f1_macro": round(float(f1), 4),
        })

    sym_preds = [analyze_symbolic_sentiment(t)["label"] for t in df["_text"]]
    acc = accuracy_score(y_true, sym_preds)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, sym_preds, average="macro", zero_division=0, labels=LABEL_ORDER
    )
    rows.append({
        "modelo": "symbolic_rules", "corpus_treino": "n/a", "corpus_gold": "repro",
        "linhas_gold": len(df),
        "accuracy": round(float(acc), 4), "precision_macro": round(float(prec), 4),
        "recall_macro": round(float(rec), 4), "f1_macro": round(float(f1), 4),
    })

    df_res = pd.DataFrame(rows)
    artifacts_dir = ROOT / "artifacts"
    artifacts_dir.mkdir(exist_ok=True)

    report = f"""# Avaliacao Gold — RePro

## Configuracao

- **Corpus gold:** RePro (dos Santos Silva et al., PROPOR 2024) — anotacao manual em PT-BR
- **Modelos avaliados:** treinados em b2w_principal; jamais viram dados do RePro
- **Uso:** somente avaliacao

## Resultados

{metrics_markdown_table(df_res)}
"""
    (artifacts_dir / "repro_evaluation.md").write_text(report, encoding="utf-8")
    print("\n=== Avaliacao no corpus gold RePro ===")
    print(df_res.to_string(index=False))
    print("\nResultados salvos em artifacts/repro_evaluation.md")
