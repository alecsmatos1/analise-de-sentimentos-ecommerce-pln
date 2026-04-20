from __future__ import annotations

import re
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlretrieve

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

# Define o backend nao interativo para permitir a geracao de figuras
# em ambiente local sem dependencia de interface grafica.
matplotlib.use("Agg")

# Estabelece os caminhos centrais do projeto e mantem a estrutura minima
# necessaria para dados brutos e documentacao final.
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
DOCS_DIR = ROOT / "docs"

# Registra as fontes publicas utilizadas para garantir reprodutibilidade
# e permitir o reaproveitamento automatizado dos datasets.
B2W_URL = "https://raw.githubusercontent.com/b2wdigital/b2w-reviews01/master/B2W-Reviews01.csv"
OLIST_URL = (
    "https://raw.githubusercontent.com/Athospd/work-at-olist-data/master/"
    "datasets/olist_order_reviews_dataset.csv"
)

B2W_PATH = DATA_DIR / "B2W-Reviews01.csv"
OLIST_PATH = DATA_DIR / "olist_order_reviews_dataset.csv"
MELI_SIMPLE_PATH = DATA_DIR / "mercadolivre_reviews_simple.csv"
LABEL_ORDER = ["negativo", "neutro", "positivo"]


def ensure_directories() -> None:
    # Garante a existencia das pastas minimas do projeto para evitar
    # falhas operacionais durante o download e a escrita do relatorio.
    DATA_DIR.mkdir(exist_ok=True)
    DOCS_DIR.mkdir(exist_ok=True)


def download_if_needed(url: str, path: Path) -> tuple[bool, str]:
    # Prioriza o reaproveitamento local dos arquivos para reduzir custo
    # computacional e manter a execucao simples e reproduzivel.
    if path.exists():
        return True, "arquivo local"
    try:
        urlretrieve(url, path)
        return True, "download concluido"
    except URLError as exc:
        return False, f"falha no download: {exc}"


def clean_text(text: str) -> str:
    # Realiza uma limpeza textual padrao para reduzir ruido superficial,
    # preservando o conteudo lexical principal para a vetorizacao TF-IDF.
    text = "" if text is None else str(text).strip().lower()
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"[\r\n\t]+", " ", text)
    text = re.sub(r"[^0-9a-zA-ZÀ-ÿ\s]", " ", text)
    text = re.sub(r"\d+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def rating_to_label(value: int) -> str:
    # Converte a nota numerica em rotulos discretos de sentimento segundo
    # a regra metodologica definida para o trabalho academico.
    if value in (1, 2):
        return "negativo"
    if value == 3:
        return "neutro"
    if value in (4, 5):
        return "positivo"
    return "indefinido"


def load_b2w() -> pd.DataFrame:
    # Carrega a base principal e combina titulo e corpo da avaliacao,
    # formando a unidade textual usada no experimento principal.
    df = pd.read_csv(B2W_PATH, low_memory=False)
    df["raw_text"] = (
        df["review_title"].fillna("").astype(str).str.strip() + " " +
        df["review_text"].fillna("").astype(str).str.strip()
    ).str.strip()
    df["rating"] = pd.to_numeric(df["overall_rating"], errors="coerce")
    df["label"] = df["rating"].apply(lambda x: rating_to_label(int(x)) if pd.notna(x) else "indefinido")
    df["clean_text"] = df["raw_text"].apply(clean_text)
    df = df[(df["label"] != "indefinido") & (df["clean_text"] != "")].copy()
    df["source"] = "b2w"
    return df[["source", "rating", "label", "clean_text"]]


def load_olist() -> pd.DataFrame:
    # Carrega a base complementar e aproveita os campos textuais de review
    # para ampliar a cobertura do dominio de e-commerce brasileiro.
    df = pd.read_csv(OLIST_PATH, low_memory=False)
    df["raw_text"] = (
        df["review_comment_title"].fillna("").astype(str).str.strip() + " " +
        df["review_comment_message"].fillna("").astype(str).str.strip()
    ).str.strip()
    df["rating"] = pd.to_numeric(df["review_score"], errors="coerce")
    df["label"] = df["rating"].apply(lambda x: rating_to_label(int(x)) if pd.notna(x) else "indefinido")
    df["clean_text"] = df["raw_text"].apply(clean_text)
    df = df[(df["label"] != "indefinido") & (df["clean_text"] != "")].copy()
    df["source"] = "olist"
    return df[["source", "rating", "label", "clean_text"]]


def load_meli_simple() -> pd.DataFrame:
    # Carrega a coleta complementar simples do Mercado Livre no mesmo
    # formato das demais bases textuais do projeto.
    df = pd.read_csv(MELI_SIMPLE_PATH, low_memory=False)
    df["raw_text"] = (
        df["review_title"].fillna("").astype(str).str.strip() + " " +
        df["review_text"].fillna("").astype(str).str.strip()
    ).str.strip()
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
    df["label"] = df["rating"].apply(lambda x: rating_to_label(int(x)) if pd.notna(x) else "indefinido")
    df["clean_text"] = df["raw_text"].apply(clean_text)
    df = df[(df["label"] != "indefinido") & (df["clean_text"] != "")].copy()
    df["source"] = "mercadolivre_simples"
    return df[["source", "rating", "label", "clean_text"]]


def build_pipeline(model_name: str) -> Pipeline:
    # Define classificadores tradicionais adequados a representacoes
    # esparsas, mantendo a proposta de baseline simples e interpretavel.
    if model_name == "logistic_regression":
        model = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
    else:
        model = LinearSVC(class_weight="balanced", random_state=42)
    # Encadeia a vetorizacao TF-IDF com o modelo supervisionado para
    # assegurar um fluxo unico e reprodutivel de treinamento e previsao.
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=False,
                    ngram_range=(1, 2),
                    min_df=3,
                    max_df=0.95,
                    sublinear_tf=True,
                ),
            ),
            ("model", model),
        ]
    )


def save_confusion_matrix(matrix, labels: list[str], output_path: Path, title: str) -> None:
    # Gera a matriz de confusao para apoiar uma leitura qualitativa dos
    # erros de classificacao alem das metricas agregadas.
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
    # Resume o volume e a distribuicao das classes para documentar a base
    # efetivamente utilizada em cada experimento.
    return {
        "experimento": name,
        "linhas": int(len(df)),
        "negativo": int((df["label"] == "negativo").sum()),
        "neutro": int((df["label"] == "neutro").sum()),
        "positivo": int((df["label"] == "positivo").sum()),
    }


def run_experiment(name: str, df: pd.DataFrame) -> list[dict[str, object]]:
    # Separa treino e teste com estratificacao para preservar a proporcao
    # das classes e produzir uma avaliacao mais consistente.
    X_train, X_test, y_train, y_test = train_test_split(
        df["clean_text"],
        df["label"],
        test_size=0.2,
        stratify=df["label"],
        random_state=42,
    )

    rows: list[dict[str, object]] = []
    for model_name in ["logistic_regression", "linear_svc"]:
        # Treina dois modelos tradicionais para permitir comparacao entre
        # baselines lineares frequentemente usados em classificacao textual.
        pipeline = build_pipeline(model_name)
        pipeline.fit(X_train, y_train)
        predictions = pipeline.predict(X_test)

        # Calcula metricas macro para reduzir a influencia exclusiva da
        # classe majoritaria na leitura do desempenho geral.
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_test,
            predictions,
            average="macro",
            zero_division=0,
        )
        accuracy = accuracy_score(y_test, predictions)
        matrix = confusion_matrix(y_test, predictions, labels=LABEL_ORDER)

        image_name = f"cm_{name}_{model_name}.png"
        save_confusion_matrix(matrix, LABEL_ORDER, ROOT / image_name, f"{name} - {model_name}")

        rows.append(
            {
                "experimento": name,
                "modelo": model_name,
                "linhas": int(len(df)),
                "accuracy": round(float(accuracy), 4),
                "precision_macro": round(float(precision), 4),
                "recall_macro": round(float(recall), 4),
                "f1_macro": round(float(f1), 4),
                "matriz_confusao": image_name,
            }
        )
    return rows


def metrics_markdown_table(df: pd.DataFrame) -> str:
    # Converte tabelas em Markdown simples para inserir os resultados no
    # relatorio final sem depender de bibliotecas adicionais.
    header = " | ".join(df.columns)
    separator = " | ".join(["---"] * len(df.columns))
    rows = [" | ".join(map(str, row)) for row in df.values.tolist()]
    return "\n".join([header, separator, *rows])


def write_report(
    availability: dict[str, str],
    dataset_summaries: list[dict[str, object]],
    metrics_df: pd.DataFrame,
) -> None:
    # Seleciona o melhor resultado por experimento para facilitar a
    # interpretacao final dos achados no relatorio academico.
    best_rows = (
        metrics_df.sort_values(["experimento", "f1_macro", "accuracy"], ascending=[True, False, False])
        .groupby("experimento", as_index=False)
        .first()
    )
    dataset_df = pd.DataFrame(dataset_summaries)
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

O pipeline seguiu as etapas de carregamento dos dados, uniao dos campos textuais, limpeza simples do texto, rotulagem por nota (1-2 negativo, 3 neutro, 4-5 positivo), vetorizacao com TF-IDF, treinamento de Regressao Logistica e Linear SVC e avaliacao com accuracy, precision, recall, F1-score macro e matriz de confusao. Foram considerados o experimento principal com B2W e experimentos combinados com Olist e Mercado Livre simples.

## Resultados

{metrics_markdown_table(metrics_df)}

{result_text}

## Limitacoes

O estudo possui limitacoes importantes: ruido textual, erros ortograficos, abreviacoes, ambiguidades semanticas, desbalanceamento entre classes e a propria limitacao de usar a nota numerica como aproximacao de sentimento textual.

## Conclusao

O projeto fornece uma linha de base simples e reproduzivel para classificacao de sentimentos em reviews de e-commerce brasileiro. A estrutura foi mantida propositalmente enxuta para facilitar apresentacao academica e execucao.

## Referencias

OLIST. Brazilian E-Commerce Public Dataset by Olist. Kaggle, 2018. Disponivel em: <https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce>. Acesso em: 19 abr. 2026.

REAL, L.; OSHIRO, M.; MAFRA, A. B2W-Reviews01: an open product reviews corpus. 2019. Disponivel em: <https://github.com/b2wdigital/b2w-reviews01>. Acesso em: 19 abr. 2026.

MERCADO LIVRE. Documentacao da API de opinioes sobre um produto. Disponivel em: <https://developers.mercadolivre.com.br/pt_br/opinioes-sobre-um-produto>. Acesso em: 19 abr. 2026.
"""
    (DOCS_DIR / "relatorio_final.md").write_text(report, encoding="utf-8")


def main() -> None:
    # Inicia a execucao garantindo que o projeto tenha os diretorios
    # minimos necessarios para dados e documentacao.
    ensure_directories()

    # Verifica a disponibilidade da base principal, cuja ausencia impede
    # a execucao do estudo proposto.
    availability: dict[str, str] = {}
    ok_b2w, msg_b2w = download_if_needed(B2W_URL, B2W_PATH)
    availability["b2w"] = msg_b2w
    if not ok_b2w:
        raise SystemExit(f"B2W indisponivel: {msg_b2w}")

    # Tenta incorporar a base complementar sem tornar sua disponibilidade
    # obrigatoria para a realizacao do experimento principal.
    ok_olist, msg_olist = download_if_needed(OLIST_URL, OLIST_PATH)
    availability["olist"] = msg_olist
    availability["meli_simples"] = "arquivo local" if MELI_SIMPLE_PATH.exists() else "nao disponivel"

    # Monta o experimento principal e, quando possivel, o experimento
    # combinado para comparar o efeito da base complementar.
    b2w_df = load_b2w()
    experiments = {"b2w_principal": b2w_df}
    olist_df = None
    meli_df = None

    if ok_olist and OLIST_PATH.exists():
        olist_df = load_olist()
        if not olist_df.empty:
            experiments["b2w_mais_olist"] = pd.concat([b2w_df, olist_df], ignore_index=True)

    if MELI_SIMPLE_PATH.exists():
        meli_df = load_meli_simple()
        if not meli_df.empty:
            experiments["b2w_mais_meli_simples"] = pd.concat([b2w_df, meli_df], ignore_index=True)

    if olist_df is not None and meli_df is not None and not olist_df.empty and not meli_df.empty:
        experiments["b2w_mais_olist_mais_meli_simples"] = pd.concat(
            [b2w_df, olist_df, meli_df],
            ignore_index=True,
        )

    # Consolida os resumos dos conjuntos e executa os modelos definidos
    # para cada configuracao experimental.
    dataset_summaries = [summarize_dataset(name, df) for name, df in experiments.items()]
    metric_rows: list[dict[str, object]] = []
    for name, df in experiments.items():
        metric_rows.extend(run_experiment(name, df))

    # Salva as metricas tabulares e atualiza automaticamente o relatorio
    # final, mantendo a estrutura documental minima do projeto.
    metrics_df = pd.DataFrame(metric_rows)
    metrics_df.to_csv(ROOT / "metricas.csv", index=False)
    write_report(availability, dataset_summaries, metrics_df)

    # Exibe no terminal um resumo objetivo da execucao para verificacao
    # imediata do usuario apos o processamento.
    print("Execucao concluida.")
    print(metrics_df.to_string(index=False))


if __name__ == "__main__":
    main()
