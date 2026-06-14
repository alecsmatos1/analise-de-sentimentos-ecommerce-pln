# Recursos Externos E Dependencias

## Principio de versionamento

Recursos grandes ou sensiveis nao devem ser versionados. A documentacao pode indicar onde eles devem ficar localmente, mas os arquivos em si devem permanecer fora do Git.

## Word2Vec NILC

O experimento E2 deve usar vetores pre-treinados do NILC para portugues.

Uso planejado:

- baixar os vetores localmente;
- carregar com biblioteca adequada, como `gensim`;
- mapear tokens da review para vetores;
- agregar os vetores por review;
- treinar modelo linear sobre a representacao agregada.

Caminho local sugerido, nao versionado:

```text
data/embeddings/nilc/
```

## BERTimbau

Os experimentos E3 e E4 devem usar BERTimbau.

Uso planejado:

- E3: BERTimbau congelado como extrator de embeddings;
- E4: BERTimbau fine-tuned para classificacao.

Dependencias provaveis:

```text
transformers
torch
datasets
accelerate
```

Caches locais nao devem ser versionados:

```text
.cache/
data/models/
models/
```

## LLM

O experimento E5 deve ser provider-agnostic nesta etapa documental. A implementacao futura podera usar API externa ou modelo local.

Regras de seguranca:

- nunca versionar `.env`;
- nunca colocar chave de API em codigo ou Markdown;
- registrar apenas o nome do modelo, data do experimento e configuracao geral;
- salvar respostas brutas apenas se nao contiverem dados sensiveis.

Caminho local sugerido para saidas, nao versionado:

```text
outputs/llm/
```

## Datasets

A v2 deve reaproveitar os datasets ja previstos na v1:

- B2W-Reviews01;
- Olist;
- Mercado Livre simples, quando disponivel;
- corpus gold externo, quando existir localmente.

Datasets brutos continuam fora do Git.

## Dependencias documentais

Nesta etapa nao sera alterado `requirements.txt`. As dependencias acima devem ser tratadas como plano para implementacao futura.
