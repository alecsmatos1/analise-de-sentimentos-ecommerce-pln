# V2 - Experimentos de Representacao Textual

## Objetivo

A v2 do projeto reorganiza a investigacao de analise de sentimentos como uma comparacao entre diferentes formas de representar textos em portugues brasileiro. A implementacao atual do projeto permanece como baseline historico, enquanto esta pasta documenta uma linha experimental nova, focada em representacoes vetoriais e modelos modernos.

A primeira etapa foi documental. A proxima etapa de implementacao deve criar primeiro o codigo interno da v2, sem download de modelo, chave de API ou artefato grande. Integracoes com NILC, Hugging Face/BERTimbau e LLM ficam para ondas posteriores.

## Escopo experimental

A v2 sera guiada por cinco experimentos:

1. Bag-of-Words com TF-IDF e modelo linear para classificacao.
2. Word2Vec com vetores pre-treinados do NILC e modelo linear para classificacao.
3. Embeddings pre-treinados do BERTimbau e modelo linear para classificacao.
4. Processo completo usando BERTimbau fine-tuned.
5. Processo completo usando LLM.

## Nota terminologica

O alinhamento original menciona "Regressao Linear para classificacao". Na documentacao da v2, essa ideia sera descrita como **modelo linear para classificacao**, porque `LinearRegression` pura nao e adequada para classificacao multiclasse. Na pratica, os candidatos corretos sao `LogisticRegression`, `LinearSVC` ou `SGDClassifier`.

## Relacao com a v1

A v1 ja possui carregamento de dados, rotulagem por nota, modelos TF-IDF, avaliacao quantitativa, matriz de confusao e relatorios. A v2 deve reaproveitar esse conhecimento, mas organizar os novos experimentos em uma arquitetura comparativa mais clara.

## Documentos desta pasta

- `experimentos.md`: detalha os cinco experimentos planejados.
- `arquitetura.md`: descreve o fluxo comum de dados, representacao, modelo e avaliacao.
- `recursos.md`: lista recursos externos e dependencias esperadas.
- `avaliacao.md`: define protocolo de comparacao e metricas.
- `riscos-e-decisoes.md`: registra decisoes, riscos e limitacoes.

## Execucao em ondas

O plano operacional da implementacao esta em `docs/planejamento/v2-execution-plan.md`. Ele define quais partes da v1 devem ser reaproveitadas, quais recursos externos precisam ser obtidos localmente e como dividir a implementacao em sprints paralelas com agentes.

Ordem esperada:

1. Codigo interno sem integracao externa: dados, split, TF-IDF, modelo linear, avaliacao, CLI e testes.
2. Contratos locais para recursos externos, ainda com mocks/stubs.
3. Integracoes reais com NILC, BERTimbau e LLM.

## Testes internos

A v2 possui uma suite de testes em `v2/tests/` que valida os contratos comuns das ondas: colunas minimas, conjunto de rotulos e mapeamento de nota para classe. A suite e leve, nao baixa modelos e nao depende de NILC, BERTimbau, LLM ou datasets brutos.

Para rodar a suite localmente, a partir da raiz do repositorio:

```powershell
python -m pytest v2/tests
```

Notas:

- A fixture demonstrativa em `v2/tests/fixtures/` contem textos sinteticos, nao reviews comerciais reais.
- O contrato verificado pelos testes considera `raw_text`, `clean_text`, `label` e `source` como colunas obrigatorias, e os rotulos `positivo`, `neutro` e `negativo` como conjunto fechado.
- Novas ondas que precisem ampliar o contrato devem atualizar primeiro os testes em `v2/tests/test_contracts.py` e a fixture correspondente.

> **Nota sobre `conftest.py`**: As constantes `REQUIRED_COLUMNS` e `ALLOWED_LABELS`
> sao importadas de `v2.src.data` quando disponiveis (apos merge das sprints). Em
> worktrees isolados, valores locais sao usados como fallback. O teste
> `test_required_columns_consistente_com_src_data` verifica automaticamente
> a consistencia apos integracao.
