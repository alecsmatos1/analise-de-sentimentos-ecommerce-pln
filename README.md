# Analise de Sentimentos em E-commerce com PLN

Trabalho universitario de Processamento de Linguagem Natural voltado para classificacao de sentimentos em avaliacoes de e-commerce brasileiro.

## Objetivo

Comparar abordagens de classificacao de sentimentos em reviews de produtos, usando textos em portugues e combinando diferentes fontes de dados para observar o impacto no desempenho.

O projeto agora contempla tres frentes:

- modelos supervisionados classicos com TF-IDF, Regressao Logistica e Linear SVC;
- analisador simbolico baseado em regras lexicais e discursivas, avaliado como baseline paralelo;
- recomendacao baseline de itens usando sentimento positivo, usuario, produto e categoria quando esses metadados existem.

## Bases utilizadas

- `B2W-Reviews01` como base principal.
- `Olist Brazilian E-Commerce Public Dataset` como base complementar.
- Base complementar simples do Mercado Livre, utilizada localmente quando `data/mercadolivre_reviews_simple.csv` esta disponivel.

## Estrutura do repositorio

- `main.py`: ponto de entrada e orquestrador dos experimentos.
- `data_processing.py`: caminhos, download, limpeza textual, rotulagem por nota e carregamento das bases.
- `sentiment_analyzer.py`: modelos supervisionados, analisador simbolico e execucao dos experimentos.
- `report_generator.py`: matrizes de confusao, tabelas Markdown, resumo das bases e relatorio final.
- `test_symbolic_analyzer.py`: teste rapido do analisador simbolico com `assert` nativo do Python.
- `docs/como_executar.md`: guia de execucao do zero, incluindo ambiente virtual e pasta `data/`.
- `docs/relatorio_final.md`: relatorio consolidado da ultima execucao registrada.
- `docs/regras_simbolicas_viabilidade.md`: analise de viabilidade das regras simbolicas.
- `docs/codigo_comentado.md`: explicacao da organizacao atual do codigo.
- `docs/coleta_base_real.md`: documentacao da coleta complementar mais rigorosa do Mercado Livre.
- `docs/coleta_base_real_simples.md`: documentacao da coleta complementar simples do Mercado Livre.

## Como executar

Veja o passo a passo completo em [`docs/como_executar.md`](docs/como_executar.md).

Execucao rapida, depois de preparar ambiente e dados:

```powershell
python test_symbolic_analyzer.py
python main.py
```

## Observacao sobre dados e arquivos locais

Os datasets brutos, metricas geradas, recomendacoes geradas, scripts auxiliares de coleta local e arquivos da pasta `data/` nao fazem parte da arvore principal publicada no GitHub. Quando os CSVs locais nao existem, o pipeline tenta baixar B2W e Olist a partir das URLs configuradas. A base simples do Mercado Livre depende do arquivo local.

## Resultado esperado

A execucao do pipeline gera metricas comparando:

- `logistic_regression`;
- `linear_svc`;
- `symbolic_rules`.

As metricas sao gravadas em `metricas.csv`, as recomendacoes em `recomendacoes.csv` e ambos sao descritos no relatorio final.
