# Experimentos Planejados

## Visao geral

Todos os experimentos devem resolver a mesma tarefa: classificar uma avaliacao textual como `positivo`, `neutro` ou `negativo`. A diferenca entre eles esta na representacao textual e no tipo de modelo usado.

| ID | Entrada textual | Representacao | Classificador/processo | Dependencias principais | Custo computacional | Saida esperada | Metricas |
|---|---|---|---|---|---|---|---|
| E1 | texto limpo da review | Bag-of-Words com TF-IDF | modelo linear para classificacao | scikit-learn | baixo | classe em 3 rotulos | accuracy, precision macro, recall macro, F1 macro, matriz de confusao |
| E2 | tokens da review | media de vetores Word2Vec NILC | modelo linear para classificacao | gensim, vetores NILC | medio | classe em 3 rotulos | mesmas metricas |
| E3 | texto bruto/normalizado | embedding BERTimbau congelado | modelo linear para classificacao | transformers, torch | medio-alto | classe em 3 rotulos | mesmas metricas |
| E4 | texto bruto/normalizado | representacao interna do BERTimbau fine-tuned | cabeca de classificacao do BERT | transformers, torch | alto | classe em 3 rotulos | mesmas metricas |
| E5 | texto bruto | prompt para LLM | classificacao direta por instrucao | API ou modelo local | variavel | classe em 3 rotulos, opcionalmente justificativa | mesmas metricas, custo e consistencia |

## E1 - TF-IDF + modelo linear

Este e o baseline classico. O texto e convertido em uma matriz esparsa de termos por documento usando TF-IDF. O modelo de classificacao deve ser linear, preferencialmente `LogisticRegression` ou `LinearSVC`.

Papel no projeto:

- servir como baseline forte e barato;
- permitir comparacao direta com a v1;
- indicar se representacoes mais complexas realmente trazem ganho.

## E2 - Word2Vec NILC + modelo linear

Este experimento usa vetores pre-treinados do NILC para portugues. Cada review deve ser representada por um unico vetor, calculado pela media dos vetores das palavras presentes no vocabulario do Word2Vec.

Pontos a documentar na implementacao futura:

- tratamento de palavras fora do vocabulario;
- estrategia para reviews sem nenhuma palavra conhecida;
- escolha entre media simples e media ponderada por TF-IDF;
- dimensao do vetor NILC utilizado.

## E3 - BERTimbau embeddings + modelo linear

Neste experimento, o BERTimbau e usado apenas como extrator de embeddings. O modelo nao sera fine-tuned. A review sera transformada em um vetor fixo, por exemplo usando `[CLS]` ou mean pooling. Em seguida, um modelo linear para classificacao sera treinado sobre esses vetores.

Papel no projeto:

- medir a qualidade de embeddings contextuais sem custo de fine-tuning;
- separar o ganho da representacao BERT do ganho do treinamento completo.

## E4 - BERTimbau completo

Este experimento faz fine-tuning do BERTimbau para classificacao em tres classes. Diferente do E3, o proprio BERT e treinado com uma cabeca de classificacao.

Pontos importantes:

- usar o mesmo split dos demais experimentos;
- controlar `random_state`/seed;
- registrar hiperparametros;
- monitorar custo computacional;
- evitar comparar apenas acuracia, pois o corpus e desbalanceado.

## E5 - LLM

Este experimento usa um LLM para classificar diretamente cada review por prompt. Deve ser documentado separadamente, porque envolve custo, variabilidade de resposta e dependencia de provedor ou modelo local.

Formato esperado de saida:

```json
{
  "label": "positivo|neutro|negativo"
}
```

Opcionalmente, a resposta pode incluir justificativa, mas a metrica deve usar apenas o campo `label`.

## Comparacao esperada

A ordem esperada de complexidade e:

```text
TF-IDF < Word2Vec NILC < BERTimbau embeddings < BERTimbau fine-tuned < LLM
```

Essa ordem nao implica necessariamente melhor desempenho. A avaliacao deve verificar ganho real em F1 macro, custo e reprodutibilidade.
