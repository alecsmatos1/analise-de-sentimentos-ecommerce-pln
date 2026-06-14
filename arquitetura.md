# Arquitetura Proposta

## Principio geral

A v2 deve comparar representacoes diferentes sob um protocolo comum. Para isso, todos os experimentos devem usar o mesmo corpus normalizado, os mesmos rotulos e o mesmo conjunto de teste.

## Fluxo comum

```text
Corpus bruto
  -> carregamento usando a logica da v1
  -> normalizacao para colunas padrao
  -> split estratificado unico
  -> representacao textual especifica do experimento
  -> treinamento ou inferencia
  -> avaliacao comum
  -> relatorio comparativo
```

## Entrada padrao

A v2 deve reaproveitar a estrutura conceitual da v1:

| Campo | Uso |
|---|---|
| `raw_text` | entrada para BERTimbau, LLM e analisadores que precisam de texto preservado |
| `clean_text` | entrada para TF-IDF e possivelmente Word2Vec |
| `label` | rotulo esperado: `positivo`, `neutro`, `negativo` |
| `source` | origem do corpus |

## Representacoes

### Bag-of-Words com TF-IDF

Representacao esparsa baseada em frequencia ponderada de termos. Deve continuar sendo o baseline principal.

### Word2Vec NILC

Representacao densa estatica. Cada palavra conhecida recebe um vetor fixo, e a review vira um vetor agregado.

### BERTimbau embeddings

Representacao contextual densa. O BERTimbau gera embeddings dependentes do contexto, mas permanece congelado neste experimento.

### BERTimbau fine-tuned

Processo completo em que o BERTimbau e treinado para a tarefa de classificacao de sentimentos.

### LLM

Processo de classificacao por prompt. Nao segue exatamente o mesmo padrao de treinamento, mas deve seguir o mesmo protocolo de avaliacao.

## Saidas esperadas

Cada experimento deve produzir:

- arquivo de metricas comparaveis;
- matriz de confusao;
- predicoes por exemplo, quando viavel;
- resumo de custo e tempo;
- observacoes de erro.

## Separacao entre v1 e v2

A v1 nao deve ser removida. A v2 deve ser criada como uma camada experimental nova, permitindo comparar:

- baseline simbolico e TF-IDF da v1;
- novas representacoes densas;
- modelos contextuais;
- LLMs.
