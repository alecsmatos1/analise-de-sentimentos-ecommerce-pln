# Codigo Comentado do Projeto

Este documento descreve a organizacao atual do codigo do projeto. A implementacao deixou de estar concentrada em um unico arquivo e foi separada em modulos pequenos, mantendo a estrutura simples para fins academicos.

## 1. Visao geral

O projeto executa um pipeline de analise de sentimentos em reviews de e-commerce brasileiro. A classe de sentimento e inferida a partir da nota da review:

- notas 1 e 2: `negativo`;
- nota 3: `neutro`;
- notas 4 e 5: `positivo`.

A avaliacao compara tres abordagens:

- `logistic_regression`: Regressao Logistica com vetorizacao TF-IDF;
- `linear_svc`: Linear SVC com vetorizacao TF-IDF;
- `symbolic_rules`: analisador simbolico baseado em regras lexicais e discursivas.

## 2. Arquivos principais

### `main.py`

O arquivo `main.py` e o ponto de entrada do projeto. Ele nao concentra mais a logica interna de processamento, modelagem e relatorio. Sua funcao e orquestrar a execucao:

1. garantir a existencia dos diretorios;
2. verificar ou baixar as bases B2W e Olist;
3. registrar a disponibilidade da base simples do Mercado Livre;
4. carregar as bases disponiveis;
5. montar os experimentos individuais e combinados;
6. executar os modelos supervisionados e o analisador simbolico;
7. salvar `metricas.csv`;
8. atualizar `docs/relatorio_final.md`.

Essa separacao deixa o arquivo principal curto e torna mais facil apresentar o fluxo geral do trabalho.

### `data_processing.py`

Este modulo concentra as funcoes de dados:

- definicao dos caminhos do projeto (`ROOT`, `DATA_DIR`, `DOCS_DIR`);
- URLs e caminhos das bases (`B2W_PATH`, `OLIST_PATH`, `MELI_SIMPLE_PATH`);
- criacao de diretorios;
- download quando o arquivo nao existe localmente;
- limpeza textual;
- conversao da nota numerica em rotulo;
- carregamento padronizado das bases B2W, Olist e Mercado Livre simples.

Os loaders retornam um `DataFrame` com as colunas:

- `source`;
- `rating`;
- `label`;
- `raw_text`;
- `clean_text`.

A coluna `raw_text` preserva o texto original para o analisador simbolico. A coluna `clean_text` e usada pelos modelos supervisionados com TF-IDF.

### `sentiment_analyzer.py`

Este modulo concentra a parte de analise de sentimentos.

Para os modelos supervisionados, ele define:

- `build_pipeline(model_name)`;
- `run_experiment(name, df)`.

O pipeline supervisionado usa:

- `TfidfVectorizer`;
- `LogisticRegression`;
- `LinearSVC`;
- divisao treino/teste estratificada com `random_state=42`;
- metricas `accuracy`, `precision_macro`, `recall_macro` e `f1_macro`.

O mesmo modulo tambem implementa o analisador simbolico:

- lexicos positivos e negativos de dominio;
- padroes fortes, como nao recebimento, defeito, devolucao, expectativa violada, entrega e recomendacao;
- emojis;
- intensificadores, atenuadores, hedges e condicionais;
- regras de contraste, concessao, caixa-alta, alongamento e pontuacao expressiva.

A funcao principal dessa parte e:

```python
analyze_symbolic_sentiment(raw_text: str) -> dict[str, object]
```

Ela retorna:

- `label`;
- `score`;
- `confidence`;
- `positive_score`;
- `negative_score`;
- `neutral_evidence`;
- `rule_hits`.

O analisador simbolico e avaliado por `run_symbolic_experiment(name, df)`, como baseline paralelo. Ele nao corrige nem substitui as predicoes dos modelos supervisionados.

### `report_generator.py`

Este modulo concentra a producao de artefatos e relatorio:

- `save_confusion_matrix`;
- `summarize_dataset`;
- `metrics_markdown_table`;
- `write_report`.

As matrizes de confusao sao salvas no diretorio raiz com o padrao:

```text
cm_<experimento>_<modelo>.png
```

O relatorio final e escrito em:

```text
docs/relatorio_final.md
```

## 3. Fluxo de execucao

O fluxo geral e:

```text
main.py
  -> data_processing.py
       carrega e prepara as bases
  -> sentiment_analyzer.py
       executa modelos supervisionados e regras simbolicas
  -> report_generator.py
       salva matrizes, metricas e relatorio
```

## 4. Experimentos realizados

Quando as bases estao disponiveis, o projeto monta os seguintes conjuntos:

- `b2w_principal`;
- `b2w_mais_olist`;
- `b2w_mais_meli_simples`;
- `b2w_mais_olist_mais_meli_simples`.

Para cada conjunto, sao avaliados:

- Regressao Logistica;
- Linear SVC;
- analisador simbolico `symbolic_rules`.

## 5. Consideracoes metodologicas

Os modelos supervisionados representam uma linha de base estatistica simples e reprodutivel. O analisador simbolico foi adicionado como comparacao academica, com foco em interpretabilidade e cobertura de fenomenos linguisticos recorrentes em reviews de e-commerce.

As regras simbolicas nao usam dependencias externas como WordNet-PT, spaCy ou LIWC. Elas sao baseadas em expressoes regulares, lexicos pequenos e padroes discursivos conservadores.

## 6. Consideracao final

A nova organizacao preserva a simplicidade do projeto, mas separa responsabilidades que antes estavam misturadas. Isso facilita manutencao, apresentacao academica e futuras extensoes, como aprimorar as regras simbolicas ou adicionar novos modelos.
