# Riscos E Decisoes Da V2

## Decisoes

### D1 - V2 documental antes de codigo

A v2 inicia apenas com documentacao. Isso evita misturar implementacao nova com a entrega atual e permite alinhar o desenho experimental antes de alterar o pipeline.

### D2 - Manter v1 como baseline

A implementacao atual nao sera removida. Ela continua como baseline historico, especialmente para TF-IDF, modelos lineares e regras simbolicas.

### D3 - Interpretar "Regressao Linear" como modelo linear de classificacao

Para classificacao multiclasse, `LinearRegression` nao e o estimador adequado. A v2 documenta a intencao como modelo linear para classificacao, usando alternativas apropriadas como `LogisticRegression`, `LinearSVC` ou `SGDClassifier`.

### D4 - Word2Vec com NILC

O Word2Vec da v2 deve usar vetores pre-treinados do NILC. Esses arquivos sao grandes e devem ficar somente no ambiente local.

### D5 - BERT com BERTimbau

Os experimentos BERT devem usar BERTimbau:

- como extrator congelado de embeddings no E3;
- como modelo fine-tuned completo no E4.

### D6 - LLM como experimento separado

LLM deve ser avaliado separadamente porque envolve custo, variabilidade de resposta e dependencia de API ou modelo local.

## Riscos

### R1 - Custo computacional

Word2Vec e TF-IDF sao baratos. BERTimbau e LLM podem exigir GPU, tempo maior ou custo financeiro.

### R2 - Comparacao injusta

Se cada experimento usar splits diferentes, a comparacao perde validade. Por isso, o split deve ser comum.

### R3 - Vazamento de dados

Embeddings pre-treinados sao permitidos, mas nenhum dado de teste pode ser usado em ajuste de hiperparametros ou treinamento.

### R4 - Arquivos grandes no Git

Vetores NILC, caches Hugging Face, modelos baixados e saidas brutas de LLM nao devem ser versionados.

### R5 - Reprodutibilidade do LLM

LLMs podem mudar comportamento entre datas, provedores e configuracoes. O experimento deve registrar modelo, data, temperatura, prompt e formato de saida.

### R6 - Classe neutra

A classe neutra e minoritaria e dificil. A v2 deve evitar otimizar apenas acuracia e deve priorizar F1 macro.

## Proximos passos documentais

1. Revisar se os cinco experimentos estao alinhados com o enunciado.
2. Decidir quais dependencias entrarao na implementacao futura.
3. Definir onde ficarao outputs locais nao versionados.
4. Criar, em etapa futura, um plano de implementacao incremental.
