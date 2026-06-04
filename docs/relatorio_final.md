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
b2w_principal | 132220 | 35708 | 16295 | 80217
b2w_mais_olist | 175423 | 47199 | 20028 | 108196
b2w_mais_meli_simples | 133602 | 35856 | 16341 | 81405
b2w_mais_olist_mais_meli_simples | 176805 | 47347 | 20074 | 109384

## Metodologia

O pipeline seguiu as etapas de carregamento dos dados, uniao dos campos textuais, limpeza do texto, remocao de stopwords, radicalizacao simples por sufixos, rotulagem por nota (1-2 negativo, 3 neutro, 4-5 positivo), vetorizacao com TF-IDF, treinamento de Regressao Logistica e Linear SVC e avaliacao com accuracy, precision, recall, F1-score macro e matriz de confusao. Alem dos modelos supervisionados, foi avaliado um analisador simbolico paralelo baseado em lexico de sentimento e regras discursivas simples. Foram considerados o experimento principal com B2W e experimentos combinados com Olist e Mercado Livre simples.

## Recomendacao de itens

Para atender ao uso academico de recomendacao, foi implementado um baseline explicavel baseado nas informacoes de sentimento. O campo `sentiment_label` representa a aproximacao de sentimento derivada da nota da avaliacao, usando a mesma regra das classes supervisionadas. Na B2W, o historico positivo de cada usuario define categorias de interesse, e o sistema recomenda produtos bem avaliados da mesma categoria, excluindo itens ja avaliados pelo usuario. Quando nao ha usuario disponivel, como na coleta simples do Mercado Livre, o sistema gera um ranking global de produtos por categoria usando media de nota, proporcao de avaliacoes positivas, sentimento medio e volume de reviews.

Exemplo de recomendacoes geradas:

tipo | user_id | rank | source | product_name | category | recommendation_score | motivo
--- | --- | --- | --- | --- | --- | --- | ---
personalizada | d0fb1ca69422530334178f5c8624aa7a99da47907c44de0243719b15d50623ce | 1 | b2w | Notebook 2 em 1 Dell Inspiron I13-5378-A40C Intel Core i7 7ª Geração 8GB 256GB SSD Tela Full HD 13.3" Touch Windows 10 - Cinza | Informática | 0.9634 | categoria com historico positivo do usuario
personalizada | d0fb1ca69422530334178f5c8624aa7a99da47907c44de0243719b15d50623ce | 2 | b2w | MacBook Air MQD32BZ/A com Intel Core i5 Dual Core 8GB 128GB SSD Tela 13" Prata - Apple | Informática | 0.9622 | categoria com historico positivo do usuario
personalizada | d0fb1ca69422530334178f5c8624aa7a99da47907c44de0243719b15d50623ce | 3 | b2w | PC G-Fire Amd A8 7600 8gb 1tb Radeon R7 2gb Integrada Computador Gamer Icarus Ev HTG-23 | Informática | 0.9621 | categoria com historico positivo do usuario
personalizada | d0fb1ca69422530334178f5c8624aa7a99da47907c44de0243719b15d50623ce | 4 | b2w | Teclado Sem Fio Apple Magic Mla22lz/a | Informática | 0.9604 | categoria com historico positivo do usuario
personalizada | d0fb1ca69422530334178f5c8624aa7a99da47907c44de0243719b15d50623ce | 5 | b2w | Notebook Dell Inspiron i15-3567-A30P Intel Core 7ª i5 4GB 1TB Tela LED 15.6" Windows 10 - Preto | Informática | 0.9582 | categoria com historico positivo do usuario
personalizada | 014d6dc5a10aed1ff1e6f349fb2b059a2d3de511c7538a9008da562ead5f5ecd | 1 | b2w | Copo De Vidro Americano 450ml - Nadir | Utilidades Domésticas | 0.9675 | categoria com historico positivo do usuario
personalizada | 014d6dc5a10aed1ff1e6f349fb2b059a2d3de511c7538a9008da562ead5f5ecd | 2 | b2w | Conjunto de Panelas Tramontina Mônaco 5 Peças | Utilidades Domésticas | 0.9634 | categoria com historico positivo do usuario
personalizada | 014d6dc5a10aed1ff1e6f349fb2b059a2d3de511c7538a9008da562ead5f5ecd | 3 | b2w | Formilix Para Formiga Cupim Berne Carrapato Bomba 500 Ml | Utilidades Domésticas | 0.9621 | categoria com historico positivo do usuario
personalizada | 014d6dc5a10aed1ff1e6f349fb2b059a2d3de511c7538a9008da562ead5f5ecd | 4 | b2w | Lixeira Aço Inox para Granito 8 Litros - Tramontina | Utilidades Domésticas | 0.9604 | categoria com historico positivo do usuario
personalizada | 014d6dc5a10aed1ff1e6f349fb2b059a2d3de511c7538a9008da562ead5f5ecd | 5 | b2w | Aparelho de Jantar e Chá Floreal São Luis 30 Peças - Oxford Daily | Utilidades Domésticas | 0.9604 | categoria com historico positivo do usuario

## Resultados

experimento | modelo | linhas | accuracy | precision_macro | recall_macro | f1_macro | matriz_confusao
--- | --- | --- | --- | --- | --- | --- | ---
b2w_principal | logistic_regression | 132220 | 0.8104 | 0.7304 | 0.7734 | 0.7416 | cm_b2w_principal_logistic_regression.png
b2w_principal | linear_svc | 132220 | 0.8389 | 0.7329 | 0.7347 | 0.7334 | cm_b2w_principal_linear_svc.png
b2w_principal | symbolic_rules | 132220 | 0.7534 | 0.6251 | 0.5772 | 0.5934 | cm_b2w_principal_symbolic_rules.png
b2w_mais_olist | logistic_regression | 175423 | 0.807 | 0.7114 | 0.7507 | 0.7219 | cm_b2w_mais_olist_logistic_regression.png
b2w_mais_olist | linear_svc | 175423 | 0.8366 | 0.7129 | 0.7137 | 0.7125 | cm_b2w_mais_olist_linear_svc.png
b2w_mais_olist | symbolic_rules | 175423 | 0.7157 | 0.6033 | 0.5538 | 0.5687 | cm_b2w_mais_olist_symbolic_rules.png
b2w_mais_meli_simples | logistic_regression | 133602 | 0.8101 | 0.7298 | 0.7745 | 0.7414 | cm_b2w_mais_meli_simples_logistic_regression.png
b2w_mais_meli_simples | linear_svc | 133602 | 0.8398 | 0.7335 | 0.735 | 0.7337 | cm_b2w_mais_meli_simples_linear_svc.png
b2w_mais_meli_simples | symbolic_rules | 133602 | 0.754 | 0.6257 | 0.5767 | 0.5932 | cm_b2w_mais_meli_simples_symbolic_rules.png
b2w_mais_olist_mais_meli_simples | logistic_regression | 176805 | 0.8085 | 0.7125 | 0.7523 | 0.7231 | cm_b2w_mais_olist_mais_meli_simples_logistic_regression.png
b2w_mais_olist_mais_meli_simples | linear_svc | 176805 | 0.8376 | 0.7135 | 0.715 | 0.7136 | cm_b2w_mais_olist_mais_meli_simples_linear_svc.png
b2w_mais_olist_mais_meli_simples | symbolic_rules | 176805 | 0.719 | 0.6055 | 0.5567 | 0.5714 | cm_b2w_mais_olist_mais_meli_simples_symbolic_rules.png

- b2w_mais_meli_simples: melhor F1 macro = 0.7414 com logistic_regression.
- b2w_mais_olist: melhor F1 macro = 0.7219 com logistic_regression.
- b2w_mais_olist_mais_meli_simples: melhor F1 macro = 0.7231 com logistic_regression.
- b2w_principal: melhor F1 macro = 0.7416 com logistic_regression.

## Limitacoes

O estudo possui limitacoes importantes: ruido textual, erros ortograficos, abreviacoes, ambiguidades semanticas, desbalanceamento entre classes e a propria limitacao de usar a nota numerica como aproximacao de sentimento textual. A recomendacao ainda e um baseline simples: Olist nao entra na recomendacao personalizada por falta de metadados de produto e usuario nesta versao, e Mercado Livre entra como recomendacao global porque a coleta local nao possui identificador de usuario.

## Conclusao

O projeto fornece uma linha de base simples e reproduzivel para classificacao de sentimentos em reviews de e-commerce brasileiro e para recomendacao inicial de itens baseada nesses sentimentos. A estrutura foi mantida propositalmente enxuta para facilitar apresentacao academica e execucao.

## Referencias

OLIST. Brazilian E-Commerce Public Dataset by Olist. Kaggle, 2018. Disponivel em: <https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce>. Acesso em: 19 abr. 2026.

REAL, L.; OSHIRO, M.; MAFRA, A. B2W-Reviews01: an open product reviews corpus. 2019. Disponivel em: <https://github.com/b2wdigital/b2w-reviews01>. Acesso em: 19 abr. 2026.

MERCADO LIVRE. Documentacao da API de opinioes sobre um produto. Disponivel em: <https://developers.mercadolivre.com.br/pt_br/opinioes-sobre-um-produto>. Acesso em: 19 abr. 2026.
