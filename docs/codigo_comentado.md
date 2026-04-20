# Codigo Comentado do Pipeline

Este documento apresenta o codigo do projeto em formato Markdown, acompanhado de explicacoes em linguagem mais formal, adequada ao contexto de um trabalho de conclusao de curso ou relatorio academico. O objetivo e justificar cada etapa do pipeline de analise de sentimentos implementado em `main.py`.

## 1. Importacao das bibliotecas

Nesta etapa, sao importadas as bibliotecas necessarias para manipulacao dos dados, pre-processamento textual, treinamento dos modelos, avaliacao de desempenho e geracao das visualizacoes. A selecao dessas dependencias foi orientada pela necessidade de manter o projeto simples, reprodutivel e compatível com uma abordagem classica de classificacao de textos.

```python
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
```

## 2. Configuracao inicial do ambiente e dos caminhos

O backend grafico nao interativo do `matplotlib` e definido para possibilitar a geracao das figuras em ambiente local sem dependencia de interface grafica. Em seguida, sao estabelecidos os caminhos centrais do projeto, o que permite manter a organizacao minima da solucao e facilitar sua reproducao em outros ambientes.

```python
matplotlib.use("Agg")

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
DOCS_DIR = ROOT / "docs"

B2W_URL = "https://raw.githubusercontent.com/b2wdigital/b2w-reviews01/master/B2W-Reviews01.csv"
OLIST_URL = (
    "https://raw.githubusercontent.com/Athospd/work-at-olist-data/master/"
    "datasets/olist_order_reviews_dataset.csv"
)

B2W_PATH = DATA_DIR / "B2W-Reviews01.csv"
OLIST_PATH = DATA_DIR / "olist_order_reviews_dataset.csv"
LABEL_ORDER = ["negativo", "neutro", "positivo"]
```

## 3. Preparacao da estrutura minima do projeto

Esta funcao garante a existencia das pastas estritamente necessarias para o funcionamento do projeto. O objetivo e evitar falhas de execucao relacionadas a escrita de arquivos, sem introduzir uma organizacao excessivamente complexa.

```python
def ensure_directories() -> None:
    DATA_DIR.mkdir(exist_ok=True)
    DOCS_DIR.mkdir(exist_ok=True)
```

## 4. Download ou reaproveitamento dos datasets

O pipeline foi planejado para reutilizar arquivos ja presentes localmente, reduzindo custo computacional e tempo de execucao. Quando o dataset nao esta disponivel no diretorio `data/`, o codigo tenta realizar o download a partir de uma fonte publica.

```python
def download_if_needed(url: str, path: Path) -> tuple[bool, str]:
    if path.exists():
        return True, "arquivo local"
    try:
        urlretrieve(url, path)
        return True, "download concluido"
    except URLError as exc:
        return False, f"falha no download: {exc}"
```

## 5. Limpeza textual

O pre-processamento textual adota uma limpeza simples, com a finalidade de reduzir ruídos superficiais sem descaracterizar o conteudo lexical das avaliacoes. Sao removidos links, marcas HTML, digitos, excesso de espacos e caracteres especiais. Essa escolha e coerente com a proposta de utilizar representacoes tradicionais como TF-IDF.

```python
def clean_text(text: str) -> str:
    text = "" if text is None else str(text).strip().lower()
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"[\r\n\t]+", " ", text)
    text = re.sub(r"[^0-9a-zA-ZÀ-ÿ\s]", " ", text)
    text = re.sub(r"\d+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text
```

## 6. Conversao da nota em rotulo de sentimento

Como o projeto utiliza aprendizagem supervisionada, foi necessario transformar a nota numerica em classes de sentimento. A regra adotada segue uma estrategia comum na literatura aplicada: notas 1 e 2 foram interpretadas como negativas, nota 3 como neutra e notas 4 e 5 como positivas.

```python
def rating_to_label(value: int) -> str:
    if value in (1, 2):
        return "negativo"
    if value == 3:
        return "neutro"
    if value in (4, 5):
        return "positivo"
    return "indefinido"
```

## 7. Carregamento da base principal B2W

Nesta funcao, a base principal e carregada, os campos de titulo e texto da avaliacao sao concatenados e a nota e convertida em rotulo de sentimento. Em seguida, os textos sao limpos e os registros inviaveis para modelagem sao removidos.

```python
def load_b2w() -> pd.DataFrame:
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
```

## 8. Carregamento da base complementar Olist

A base complementar segue o mesmo principio da base principal, aproveitando os campos textuais das reviews para compor uma amostra adicional. Essa etapa permite comparar o desempenho do experimento principal com um experimento combinado.

```python
def load_olist() -> pd.DataFrame:
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
```

## 9. Definicao do pipeline de modelagem

O pipeline combina duas etapas: vetorizacao TF-IDF e classificacao supervisionada. Foram escolhidos dois modelos tradicionais bastante utilizados em classificacao de texto, a Regressao Logistica e o Linear SVC, por apresentarem bom desempenho com representacoes esparsas e baixa complexidade de implementacao.

```python
def build_pipeline(model_name: str) -> Pipeline:
    if model_name == "logistic_regression":
        model = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
    else:
        model = LinearSVC(class_weight="balanced", random_state=42)
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
```

## 10. Geracao da matriz de confusao

Para complementar as metricas numericas, a matriz de confusao e utilizada como recurso de interpretacao visual. Ela permite identificar em quais classes o modelo apresenta maior dificuldade, especialmente em cenarios com desbalanceamento entre rotulos.

```python
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
```

## 11. Resumo dos conjuntos de dados

Antes da modelagem, o codigo produz um resumo simples com o volume total de registros e a distribuicao das classes. Isso e importante para documentar a composicao efetiva dos experimentos realizados.

```python
def summarize_dataset(name: str, df: pd.DataFrame) -> dict[str, object]:
    return {
        "experimento": name,
        "linhas": int(len(df)),
        "negativo": int((df["label"] == "negativo").sum()),
        "neutro": int((df["label"] == "neutro").sum()),
        "positivo": int((df["label"] == "positivo").sum()),
    }
```

## 12. Execucao dos experimentos

Nesta etapa, os dados sao divididos em conjuntos de treino e teste com estratificacao, preservando a distribuicao das classes. Em seguida, os dois modelos sao treinados e avaliados por meio das metricas `accuracy`, `precision`, `recall` e `F1-score` macro.

```python
def run_experiment(name: str, df: pd.DataFrame) -> list[dict[str, object]]:
    X_train, X_test, y_train, y_test = train_test_split(
        df["clean_text"],
        df["label"],
        test_size=0.2,
        stratify=df["label"],
        random_state=42,
    )

    rows: list[dict[str, object]] = []
    for model_name in ["logistic_regression", "linear_svc"]:
        pipeline = build_pipeline(model_name)
        pipeline.fit(X_train, y_train)
        predictions = pipeline.predict(X_test)

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
```

## 13. Conversao das tabelas para Markdown

Como o projeto busca manter simplicidade estrutural, optou-se por gerar tabelas em Markdown manualmente, sem depender de ferramentas adicionais para a producao do relatorio final.

```python
def metrics_markdown_table(df: pd.DataFrame) -> str:
    header = " | ".join(df.columns)
    separator = " | ".join(["---"] * len(df.columns))
    rows = [" | ".join(map(str, row)) for row in df.values.tolist()]
    return "\n".join([header, separator, *rows])
```

## 14. Escrita automatica do relatorio final

Esta funcao sintetiza os resultados da execucao em um unico arquivo Markdown. O texto segue uma estrutura academica simples, contendo introducao, fundamentacao breve, bases de dados, metodologia, resultados, limitacoes, conclusao e referencias.

```python
def write_report(
    availability: dict[str, str],
    dataset_summaries: list[dict[str, object]],
    metrics_df: pd.DataFrame,
) -> None:
    best_rows = (
        metrics_df.sort_values(["experimento", "f1_macro", "accuracy"], ascending=[True, False, False])
        .groupby("experimento", as_index=False)
        .first()
    )
    dataset_df = pd.DataFrame(dataset_summaries)

    if "b2w_mais_olist" in set(metrics_df["experimento"]):
        result_text = (
            "No experimento principal com a base B2W, o melhor resultado em F1 macro foi "
            f"{best_rows.loc[best_rows['experimento'] == 'b2w_principal', 'f1_macro'].iloc[0]:.4f}. "
            "No experimento combinado B2W + Olist, o melhor F1 macro foi "
            f"{best_rows.loc[best_rows['experimento'] == 'b2w_mais_olist', 'f1_macro'].iloc[0]:.4f}. "
            "A combinacao das bases ampliou o volume de dados, mas nao melhorou o F1 macro em relacao ao experimento principal."
        )
    else:
        result_text = (
            "A execucao final foi realizada apenas com a base B2W, pois a base complementar Olist nao ficou disponivel."
        )

    report = f"""# Relatorio Final

## Introducao

Este projeto apresenta um pipeline simples de analise de sentimentos em avaliacoes de e-commerce brasileiro. A base principal e o corpus B2W-Reviews01, enquanto a base Olist e usada de forma complementar quando disponivel.

## Base teorica breve

A analise de sentimentos busca classificar a polaridade predominante de um texto. Em reviews de e-commerce, essa tarefa ajuda a resumir a percepcao do consumidor. Neste trabalho, a polaridade foi inferida a partir da nota da avaliacao, com tres classes: negativo, neutro e positivo.

## Bases de dados

- B2W-Reviews01: base principal, em portugues brasileiro, com reviews de produtos do comercio eletronico.
- Olist Brazilian E-Commerce Public Dataset: base complementar, usando a tabela de reviews.

Disponibilidade na execucao:

- B2W: {availability['b2w']}
- Olist: {availability['olist']}

Resumo dos conjuntos usados:

{metrics_markdown_table(dataset_df)}

## Metodologia

O pipeline seguiu as etapas de carregamento dos dados, uniao dos campos textuais, limpeza simples do texto, rotulagem por nota (1-2 negativo, 3 neutro, 4-5 positivo), vetorizacao com TF-IDF, treinamento de Regressao Logistica e Linear SVC e avaliacao com accuracy, precision, recall, F1-score macro e matriz de confusao.

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
"""
    (DOCS_DIR / "relatorio_final.md").write_text(report, encoding="utf-8")
```

## 15. Funcao principal

Por fim, a funcao principal organiza toda a execucao do pipeline. Ela prepara as pastas, verifica a disponibilidade das bases, monta os experimentos, executa os modelos, salva as metricas e atualiza automaticamente o relatorio final.

```python
def main() -> None:
    ensure_directories()

    availability: dict[str, str] = {}
    ok_b2w, msg_b2w = download_if_needed(B2W_URL, B2W_PATH)
    availability["b2w"] = msg_b2w
    if not ok_b2w:
        raise SystemExit(f"B2W indisponivel: {msg_b2w}")

    ok_olist, msg_olist = download_if_needed(OLIST_URL, OLIST_PATH)
    availability["olist"] = msg_olist

    b2w_df = load_b2w()
    experiments = {"b2w_principal": b2w_df}

    if ok_olist and OLIST_PATH.exists():
        olist_df = load_olist()
        if not olist_df.empty:
            experiments["b2w_mais_olist"] = pd.concat([b2w_df, olist_df], ignore_index=True)

    dataset_summaries = [summarize_dataset(name, df) for name, df in experiments.items()]
    metric_rows: list[dict[str, object]] = []
    for name, df in experiments.items():
        metric_rows.extend(run_experiment(name, df))

    metrics_df = pd.DataFrame(metric_rows)
    metrics_df.to_csv(ROOT / "metricas.csv", index=False)
    write_report(availability, dataset_summaries, metrics_df)

    print("Execucao concluida.")
    print(metrics_df.to_string(index=False))


if __name__ == "__main__":
    main()
```

## Consideracao final

O codigo foi mantido em um unico arquivo para atender ao requisito de simplicidade estrutural. Ainda assim, o pipeline contempla as etapas centrais esperadas em um trabalho academico introdutorio de PLN aplicado a analise de sentimentos.
