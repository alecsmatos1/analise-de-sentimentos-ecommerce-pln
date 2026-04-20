# Relatorio Final

## Introducao

Este projeto apresenta um pipeline simples de analise de sentimentos em avaliacoes de e-commerce brasileiro. A base principal e o corpus B2W-Reviews01. As bases complementares sao o Olist Brazilian E-Commerce Public Dataset e a coleta simples de reviews do Mercado Livre.

## Base teorica breve

A analise de sentimentos busca classificar a polaridade predominante de um texto. Em reviews de e-commerce, essa tarefa ajuda a resumir a percepcao do consumidor. Neste trabalho, a polaridade foi inferida a partir da nota da avaliacao, com tres classes: negativo, neutro e positivo.

## Bases de dados

- B2W-Reviews01: base principal, em portugues brasileiro, com reviews de produtos do comercio eletronico.
- Olist Brazilian E-Commerce Public Dataset: base complementar, usando a tabela de reviews.
- Mercado Livre simples: base complementar real obtida por coleta via API oficial, integrada ao pipeline no mesmo formato das demais bases.

Disponibilidade na execucao:

- B2W: arquivo local
- Olist: arquivo local
- Mercado Livre simples: arquivo local

Resumo dos conjuntos usados:

experimento | linhas | negativo | neutro | positivo
--- | --- | --- | --- | ---
b2w_principal | 132230 | 35712 | 16295 | 80223
b2w_mais_olist | 175460 | 47207 | 20032 | 108221
b2w_mais_meli_simples | 133612 | 35860 | 16341 | 81411
b2w_mais_olist_mais_meli_simples | 176842 | 47355 | 20078 | 109409

## Metodologia

O pipeline seguiu as etapas de carregamento dos dados, uniao dos campos textuais, limpeza simples do texto, rotulagem por nota (1-2 negativo, 3 neutro, 4-5 positivo), vetorizacao com TF-IDF, treinamento de Regressao Logistica e Linear SVC e avaliacao com accuracy, precision, recall, F1-score macro e matriz de confusao. Foram considerados o experimento principal com B2W e experimentos combinados com Olist e Mercado Livre simples.

## Resultados

experimento | modelo | linhas | accuracy | precision_macro | recall_macro | f1_macro | matriz_confusao
--- | --- | --- | --- | --- | --- | --- | ---
b2w_principal | logistic_regression | 132230 | 0.8366 | 0.7543 | 0.798 | 0.7688 | cm_b2w_principal_logistic_regression.png
b2w_principal | linear_svc | 132230 | 0.8516 | 0.7522 | 0.7525 | 0.7519 | cm_b2w_principal_linear_svc.png
b2w_mais_olist | logistic_regression | 175460 | 0.8183 | 0.7222 | 0.7646 | 0.7349 | cm_b2w_mais_olist_logistic_regression.png
b2w_mais_olist | linear_svc | 175460 | 0.8412 | 0.7229 | 0.7231 | 0.7223 | cm_b2w_mais_olist_linear_svc.png
b2w_mais_meli_simples | logistic_regression | 133612 | 0.8367 | 0.7539 | 0.7982 | 0.7684 | cm_b2w_mais_meli_simples_logistic_regression.png
b2w_mais_meli_simples | linear_svc | 133612 | 0.8518 | 0.7517 | 0.7525 | 0.7517 | cm_b2w_mais_meli_simples_linear_svc.png
b2w_mais_olist_mais_meli_simples | logistic_regression | 176842 | 0.8204 | 0.7243 | 0.7678 | 0.7373 | cm_b2w_mais_olist_mais_meli_simples_logistic_regression.png
b2w_mais_olist_mais_meli_simples | linear_svc | 176842 | 0.8446 | 0.7278 | 0.7278 | 0.7272 | cm_b2w_mais_olist_mais_meli_simples_linear_svc.png

- b2w_mais_meli_simples: melhor F1 macro = 0.7684 com logistic_regression.
- b2w_mais_olist: melhor F1 macro = 0.7349 com logistic_regression.
- b2w_mais_olist_mais_meli_simples: melhor F1 macro = 0.7373 com logistic_regression.
- b2w_principal: melhor F1 macro = 0.7688 com logistic_regression.

## Limitacoes

O estudo possui limitacoes importantes: ruido textual, erros ortograficos, abreviacoes, ambiguidades semanticas, desbalanceamento entre classes e a propria limitacao de usar a nota numerica como aproximacao de sentimento textual.

## Conclusao

O projeto fornece uma linha de base simples e reproduzivel para classificacao de sentimentos em reviews de e-commerce brasileiro. A estrutura foi mantida propositalmente enxuta para facilitar apresentacao academica e execucao.

## Referencias

OLIST. Brazilian E-Commerce Public Dataset by Olist. Kaggle, 2018. Disponivel em: <https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce>. Acesso em: 19 abr. 2026.

REAL, L.; OSHIRO, M.; MAFRA, A. B2W-Reviews01: an open product reviews corpus. 2019. Disponivel em: <https://github.com/b2wdigital/b2w-reviews01>. Acesso em: 19 abr. 2026.

MERCADO LIVRE. Documentacao da API de opinioes sobre um produto. Disponivel em: <https://developers.mercadolivre.com.br/pt_br/opinioes-sobre-um-produto>. Acesso em: 19 abr. 2026.
