# Protocolo De Avaliacao

## Objetivo

Garantir que todos os experimentos da v2 sejam comparados de forma justa, usando o mesmo split, os mesmos rotulos e as mesmas metricas.

## Split

Todos os experimentos devem usar o mesmo particionamento:

- treino: 80%;
- teste: 20%;
- estratificacao por classe;
- seed fixa.

Se houver validacao para ajuste de hiperparametros, ela deve ser retirada apenas do conjunto de treino.

## Rotulos

Os rotulos seguem a convencao da v1:

| Nota | Rotulo |
|---|---|
| 1 ou 2 | `negativo` |
| 3 | `neutro` |
| 4 ou 5 | `positivo` |

Em corpora gold com anotacao manual, deve-se mapear os rotulos originais para o mesmo conjunto: `positivo`, `neutro`, `negativo`.

## Metricas obrigatorias

Todos os experimentos devem reportar:

- accuracy;
- precision macro;
- recall macro;
- F1 macro;
- matriz de confusao.

F1 macro deve ser considerada a metrica principal, pois o corpus e desbalanceado.

## Comparacao minima

A tabela final deve conter:

| Experimento | Representacao | Modelo/processo | Accuracy | F1 macro | Custo | Observacoes |
|---|---|---|---:|---:|---|---|
| E1 | TF-IDF | modelo linear | a preencher | a preencher | baixo | baseline |
| E2 | Word2Vec NILC | modelo linear | a preencher | a preencher | medio | embeddings estaticos |
| E3 | BERTimbau embeddings | modelo linear | a preencher | a preencher | medio-alto | BERT congelado |
| E4 | BERTimbau completo | fine-tuning | a preencher | a preencher | alto | contextual treinado |
| E5 | LLM | prompt | a preencher | a preencher | variavel | custo e reproducibilidade |

## Avaliacao de erro

Alem das metricas, cada experimento deve registrar exemplos de erro, preferencialmente cobrindo:

- negacao;
- ironia;
- texto curto;
- review longa;
- polaridade mista;
- classe neutra;
- erro por vocabulario fora do dominio.

## Reprodutibilidade

Todo resultado deve informar:

- versao dos dados;
- seed;
- modelo usado;
- hiperparametros principais;
- data da execucao;
- se houve GPU;
- tempo aproximado de execucao.
