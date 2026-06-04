# SCC0633-SCC5908 - PLN

## O que e

Projeto de Processamento de Linguagem Natural para analise de sentimentos em avaliacoes de e-commerce brasileiro.

## Objetivo

Comparar classificadores supervisionados e uma abordagem simbolica em reviews de produtos, usando bases B2W, Olist e uma coleta simples do Mercado Livre. O projeto tambem inclui um baseline de recomendacao baseado em sentimento positivo.

## Documentos principais

- `como_executar.md`: preparacao do ambiente, dados e execucao.
- `relatorio_final.md`: relatorio consolidado dos experimentos.
- `relatorio_solucao_simbolica.md`: descricao detalhada da solucao simbolica.
- `regras_simbolicas_viabilidade.md`: analise de viabilidade das regras.
- `codigo_comentado.md`: explicacao dos arquivos principais.
- `coleta_base_real.md`: coleta complementar mais rigorosa.
- `coleta_base_real_simples.md`: coleta complementar simples do Mercado Livre.

## Codigo relacionado

- `main.py`: orquestracao dos experimentos.
- `data_processing.py`: carga, limpeza e rotulagem.
- `sentiment_analyzer.py`: modelos supervisionados e regras simbolicas.
- `report_generator.py`: metricas, tabelas e relatorios.
- `test_symbolic_analyzer.py`: validacao rapida do analisador simbolico.

## Como usar a documentacao

Comece por `como_executar.md` para reproduzir, leia `relatorio_final.md` para entender resultados e use `codigo_comentado.md` quando precisar modificar o pipeline.
