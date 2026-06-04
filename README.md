# Analise de Sentimentos em E-commerce com PLN

Trabalho universitario de Processamento de Linguagem Natural voltado para classificacao de sentimentos em avaliacoes de e-commerce brasileiro.

**Disciplina:** SCC0633 / SCC5908 â€” Processamento de Linguagem Natural
**Professores:** Renato M. Silva e Graca Nunes
**Equipe:** PLN Rocks â€” Alecsander Goncalves de Matos, Victor Anthony Pereira Alves, Vitor Rodrigues Tonon

---

## Objetivo

Comparar abordagens de classificacao de sentimentos em reviews de produtos, usando textos em portugues e combinando diferentes fontes de dados para observar o impacto no desempenho.

### Etapa 1 (concluida)

- Modelos supervisionados classicos com TF-IDF, Regressao Logistica e Linear SVC.
- Analisador simbolico baseado em regras lexicais e discursivas, com lexicos internos construidos pela equipe.
- Recomendacao baseline de itens usando sentimento positivo, usuario, produto e categoria.
- Radicalizacao por sufixos manual (`simple_stem`).

### Etapa 2 (em andamento)

Melhorias solicitadas pela professora para fundamentar o trabalho na literatura:

1. **Lexicos externos validados** â€” integrar OpLexicon v3.0 (PUC-RS) e SentiLex-PT 02 (INESC-ID) ao analisador simbolico.
2. **Corpus gold externo** â€” avaliar todos os modelos sobre o SentiBR (tweets PT-BR anotados manualmente), separando avaliacao de dados de treino.
3. **Lematizacao publicada** â€” substituir `simple_stem()` manual pelo RSLPStemmer do NLTK (Orengo & Huyck, 2001).

Plano detalhado em [`docs/etapa2_plano.md`](docs/etapa2_plano.md).

---

## Bases utilizadas

- `B2W-Reviews01` como base principal.
- `Olist Brazilian E-Commerce Public Dataset` como base complementar.
- Base complementar simples do Mercado Livre, utilizada localmente quando `data/mercadolivre_reviews_simple.csv` esta disponivel.
- `SentiBR` como corpus gold de avaliacao (Etapa 2) â€” somente para avaliacao, nunca para treino.

---

## Estrutura do repositorio

- `main.py`: ponto de entrada e orquestrador dos experimentos.
- `data_processing.py`: caminhos, download, limpeza textual, rotulagem por nota e carregamento das bases.
- `sentiment_analyzer.py`: modelos supervisionados, analisador simbolico e execucao dos experimentos.
- `report_generator.py`: matrizes de confusao, tabelas Markdown, resumo das bases e relatorio final.
- `test_symbolic_analyzer.py`: teste rapido do analisador simbolico com `assert` nativo do Python.
- `docs/etapa2_plano.md`: plano de melhorias da Etapa 2 com recursos da literatura.
- `docs/status.md`: estado atual e proximos passos.
- `docs/decisoes.md`: decisoes de modelagem, dados e engenharia.
- `docs/referencias.md`: referencias academicas e fontes dos recursos utilizados.
- `docs/como_executar.md`: guia de execucao do zero, incluindo ambiente virtual e pasta `data/`.
- `docs/relatorio_final.md`: relatorio consolidado da ultima execucao registrada.
- `docs/regras_simbolicas_viabilidade.md`: analise de viabilidade das regras simbolicas.
- `docs/codigo_comentado.md`: explicacao da organizacao atual do codigo.
- `docs/coleta_base_real_simples.md`: documentacao da coleta complementar simples do Mercado Livre.
- `docs/resposta_comentarios_professor.md`: respostas aos comentarios da segunda entrega.

---

## Como executar

Veja o passo a passo completo em [`docs/como_executar.md`](docs/como_executar.md).

Execucao rapida, depois de preparar ambiente e dados:

```powershell
python test_symbolic_analyzer.py
python main.py
```

---

## Observacao sobre dados e arquivos locais

Os datasets brutos, metricas geradas, recomendacoes geradas, scripts auxiliares de coleta local e arquivos da pasta `data/` nao fazem parte da arvore principal publicada no GitHub. Quando os CSVs locais nao existem, o pipeline tenta baixar B2W e Olist a partir das URLs configuradas. A base simples do Mercado Livre depende do arquivo local.

---

## Resultados (Etapa 1)

A execucao do pipeline gera metricas comparando:

- `logistic_regression` â€” melhor F1-macro: 0,7452 (B2W + Mercado Livre simples);
- `linear_svc` â€” melhor acuracia: 0,8418 (B2W + Mercado Livre simples);
- `symbolic_rules` â€” melhor F1-macro: 0,6036 (B2W + Mercado Livre simples).

As metricas sao gravadas em `metricas.csv`, as recomendacoes em `recomendacoes.csv` e ambos sao descritos no relatorio final.
