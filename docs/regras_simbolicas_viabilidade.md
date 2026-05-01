# Viabilidade de Regras Simbolicas para Analise de Sentimento

## 1. Contexto do pipeline atual

O projeto atual implementa uma linha de base supervisionada para analise de sentimentos em reviews de e-commerce brasileiro. O fluxo principal esta concentrado em `main.py` e segue as etapas de carregamento dos dados, limpeza textual simples, conversao da nota numerica em rotulo de sentimento, vetorizacao TF-IDF e treinamento de modelos lineares.

Os modelos avaliados atualmente sao:

- Regressao Logistica
- Linear SVC

As bases usadas no experimento atual sao:

- B2W-Reviews01
- Olist Brazilian E-Commerce Public Dataset
- coleta simples de reviews do Mercado Livre, quando o arquivo local esta disponivel

Na versao inicial deste documento, o projeto ainda nao possuia um analisador simbolico implementado. A versao atual ja inclui uma primeira implementacao em `sentiment_analyzer.py`, avaliada como baseline paralelo com o nome `symbolic_rules`.

Essa implementacao atual cobre uma parte das regras de baixo custo, usando expressoes regulares, lexicos pequenos e padroes discursivos conservadores. Ainda nao fazem parte do projeto:

- lexico afetivo amplo ou validado externamente;
- LIWC integrado;
- parser sintatico;
- analise baseada em aspectos (ABSA);
- WordNet-PT ou NLTK como dependencia do pipeline principal;
- spaCy ou modelo `pt_core_news_lg`.

Portanto, qualquer mencao a LIWC neste documento deve ser lida como hipotese de uma futura camada lexical. Ela nao descreve uma dependencia ja existente no projeto.

## 2. Objetivo das regras simbolicas

As regras simbolicas propostas podem funcionar como uma camada complementar ao baseline supervisionado. A ideia nao e substituir os modelos lineares, mas avaliar se alguns fenomenos linguisticos frequentes em reviews de e-commerce podem ser tratados antes ou depois da classificacao estatistica.

Existem tres formas plausiveis de integracao futura:

- pre-processamento: marcar ou normalizar fenomenos antes do TF-IDF;
- features adicionais: transformar sinais simbolicos em colunas numericas combinadas ao modelo;
- pos-processamento: ajustar a decisao final quando uma regra de alta precisao for acionada.

Para o projeto atual, a opcao mais segura para uma primeira implementacao seria registrar sinais simbolicos como features ou relatorios auxiliares, sem alterar imediatamente o rotulo previsto pelo classificador. Isso reduz o risco de piorar metricas globais por regras muito agressivas.

## 3. Matriz Relevancia x Complexidade

| Relevancia | Complexidade baixa | Complexidade media/alta |
| --- | --- | --- |
| Alta | #1 Detector de Pros/Contras; #2 Lexico de dominio; #3 Polarity shifters | #4 Lexico discursivo tipado; #12 Ontologia ABSA; #13 Parsing de dependencias |
| Media | #5 Lexico de emojis; #6 Caixa-alta e alongamento; #7 Hedges; #8 Frames | #9 Comparativos; #10 Ironia; #11 WordNet-PT |

Ordem recomendada por retorno esperado em relacao ao custo:

1. #1 Detector de Pros/Contras
2. #2 Lexico de dominio
3. #3 Lexico de polarity shifters
4. #5 Lexico de emojis
5. #6 Caixa-alta e alongamento
6. #7 Marcadores modais / hedges
7. #4 Lexico discursivo tipado
8. #8 Classificador de frames
9. #10 Padroes de ironia
10. #9 Parser de comparativos
11. #11 Expansao WordNet-PT
12. #12 Ontologia ABSA
13. #13 Parsing de dependencias

## 4. Analise individual das regras

### #1 - Detector de Pros/Contras

**Fenomeno tratado:** reviews que separam explicitamente pontos positivos e negativos por cabecalhos como "Pontos positivos:" e "Pontos negativos:".

**Exemplo:** "Pontos positivos: bateria dura o dia todo. Pontos negativos: tela arranha facil."

**Viabilidade no projeto atual:** alta. Pode ser implementado com regex e segmentacao simples antes da vetorizacao ou em uma camada de analise auxiliar.

**Dependencias necessarias:** nenhuma dependencia externa. Apenas padroes regex e lista de cabecalhos equivalentes.

**Impacto esperado:** alto para reviews mistas, porque evita que sinais positivos e negativos sejam agregados como se pertencessem a uma unica opiniao uniforme.

**Complexidade realista:** baixa.

**Risco principal:** cabecalhos escritos de formas variadas podem reduzir cobertura. Exemplo: "pro", "contra", "gostei", "nao gostei".

**Recomendacao:** implementar agora em uma primeira fase simbolica.

### #2 - Lexico de dominio

**Fenomeno tratado:** termos de e-commerce que carregam polaridade forte e podem nao aparecer em dicionarios academicos, como "travou", "bugou", "top demais", "zero bala", "falso", "quebrado" e "original".

**Exemplo:** "Liguei o celular e ja travou na tela inicial."

**Viabilidade no projeto atual:** alta. Pode ser criado como um arquivo JSON ou CSV com termo, polaridade, intensidade e categoria opcional.

**Dependencias necessarias:** nenhuma dependencia externa. Exige apenas curadoria manual inicial e criterio de revisao.

**Impacto esperado:** alto, especialmente em reviews de 1 e 5 estrelas com vocabulario informal.

**Complexidade realista:** baixa.

**Risco principal:** ambiguidade contextual. Por exemplo, "barato" pode ser positivo em "preco barato" e negativo em "material barato".

**Recomendacao:** implementar agora, com um lexico pequeno e versionado inicialmente.

### #3 - Lexico de polarity shifters

**Fenomeno tratado:** expressoes que invertem ou deslocam polaridade sem usar apenas a palavra "nao", como "deixou de", "parou de", "perdeu a", "falta de", "sem" e "nunca mais".

**Exemplo:** "A bateria deixou de carregar depois de 30 dias."

**Viabilidade no projeto atual:** alta para casos curtos e padroes frequentes. Pode ser feito por regex de janela local antes de qualquer parser sintatico.

**Dependencias necessarias:** nenhuma dependencia externa. Exige lista de shifters tipados e regra de escopo local.

**Impacto esperado:** alto em reviews negativos, pois muitas reclamacoes descrevem perda de funcionamento.

**Complexidade realista:** baixa.

**Risco principal:** escopo incorreto. Uma janela fixa pode inverter a palavra errada em frases longas.

**Recomendacao:** implementar agora com escopo conservador e testes de falsos positivos.

### #4 - Lexico discursivo tipado

**Fenomeno tratado:** conectores que alteram a importancia relativa das partes da frase, especialmente concessao e contraste.

**Exemplo:** "Embora a entrega tenha demorado, o produto e otimo." tende a encerrar com avaliacao positiva. "O produto e otimo, mas a entrega demorou." tende a dar mais peso ao trecho negativo final.

**Viabilidade no projeto atual:** media. Regex e listas resolvem a deteccao inicial, mas a decisao de peso exige desenho cuidadoso.

**Dependencias necessarias:** nenhuma dependencia externa obrigatoria. Pode ser iniciado com lista de conectores e pesos por tipo discursivo.

**Impacto esperado:** alto em reviews mistos, porque o conector frequentemente indica qual parte o autor quer destacar.

**Complexidade realista:** media.

**Risco principal:** supervalorizar o trecho depois de "mas" em frases onde o contraste nao muda a avaliacao principal.

**Recomendacao:** implementar depois das regras lexicais basicas, com avaliacao manual em amostra de reviews mistas.

### #5 - Lexico de emojis

**Fenomeno tratado:** emojis que expressam polaridade ou aspecto sem depender de palavras.

**Exemplo:** "AMEI, chegou rapidinho, recomendo." acompanhado de emojis positivos.

**Viabilidade no projeto atual:** alta. Pode ser tratado antes da limpeza textual, porque o `clean_text` atual remove caracteres especiais e apagaria esse sinal.

**Dependencias necessarias:** nenhuma dependencia externa. Exige tabela de emojis com polaridade, intensidade e possivel aspecto.

**Impacto esperado:** medio. O impacto depende da frequencia de emojis nas bases locais, especialmente na coleta do Mercado Livre.

**Complexidade realista:** baixa.

**Risco principal:** a limpeza atual remove emojis. Se a regra for adicionada no lugar errado, o sinal sera perdido antes de ser lido.

**Recomendacao:** implementar agora apenas se a analise exploratoria confirmar presenca relevante de emojis na base.

### #6 - Caixa-alta e alongamento

**Fenomeno tratado:** enfase textual por caixa-alta, repeticao de letras e pontuacao expressiva.

**Exemplo:** "PESSIMO atendimento!!! pessimoooo."

**Viabilidade no projeto atual:** alta, mas precisa ser aplicada antes da normalizacao para minusculas feita por `clean_text`.

**Dependencias necessarias:** nenhuma dependencia externa. Exige funcoes simples para detectar uppercase, repeticao de caracteres e repeticao de pontuacao.

**Impacto esperado:** medio. A regra nao cria polaridade sozinha; ela aumenta ou reduz a intensidade de um sinal ja detectado.

**Complexidade realista:** baixa.

**Risco principal:** amplificar ruido. Caixa-alta pode aparecer por habito do usuario, nao por enfase afetiva.

**Recomendacao:** implementar agora como multiplicador limitado, nunca como decisao isolada.

### #7 - Marcadores modais / hedges

**Fenomeno tratado:** expressoes de incerteza ou baixa assertividade, como "talvez", "acho que", "parece", "pode ser" e "ainda preciso testar".

**Exemplo:** "Talvez seja o melhor produto nessa faixa de preco."

**Viabilidade no projeto atual:** alta. Pode ser implementado com lista lexical e janela curta sobre termos afetivos proximos.

**Dependencias necessarias:** nenhuma dependencia externa.

**Impacto esperado:** medio. Ajuda a evitar que frases especulativas tenham o mesmo peso de avaliacoes diretas.

**Complexidade realista:** baixa.

**Risco principal:** atenuar opinioes que continuam fortes apesar do hedge, como "acho que e excelente".

**Recomendacao:** implementar agora, com atenuacao moderada e reversivel.

### #8 - Classificador de frames

**Fenomeno tratado:** atos de fala em reviews, como reclamacao, recomendacao, alerta, pergunta ou pedido de suporte.

**Exemplo:** "Nao comprem aqui, e propaganda enganosa."

**Viabilidade no projeto atual:** media. Pode iniciar com padroes de alta precisao, como imperativos e expressoes fixas.

**Dependencias necessarias:** nenhuma dependencia externa obrigatoria. Pode exigir listas de verbos-gatilho e padroes de negacao.

**Impacto esperado:** medio. Recomendacoes negativas diretas costumam ser informativas para sentimento geral.

**Complexidade realista:** baixa a media.

**Risco principal:** confundir pergunta ou pedido de ajuda com opiniao negativa consolidada.

**Recomendacao:** implementar depois das regras lexicais, com cobertura conservadora.

### #9 - Parser de comparativos

**Fenomeno tratado:** comparacoes relativas, superlativas, equativas e direcionais.

**Exemplo:** "Muito melhor que o modelo anterior em autonomia." ou "Tao bom quanto o iPhone, mas com metade do preco."

**Viabilidade no projeto atual:** media. Padroes regex cobrem casos simples, mas uma cobertura robusta exige tratamento sintatico maior.

**Dependencias necessarias:** nenhuma para uma versao simples baseada em regex. spaCy pode ser considerado em uma versao avancada.

**Impacto esperado:** medio a alto, pois comparativos sao argumentos importantes de compra.

**Complexidade realista:** media.

**Risco principal:** extrair comparacoes incompletas ou atribuir polaridade ao alvo errado.

**Recomendacao:** avaliar com calma; iniciar apenas com padroes de alta precisao se houver exemplos suficientes na base.

### #10 - Padroes de ironia

**Fenomeno tratado:** uso de vocabulario positivo com intencao negativa, geralmente acompanhado de um evento negativo.

**Exemplo:** "Produto excelente! Chegou quebrado e nao funciona. Parabens, loja!"

**Viabilidade no projeto atual:** media. E possivel criar padroes conservadores que so disparam quando elogio e evidencia negativa aparecem juntos.

**Dependencias necessarias:** nenhuma dependencia externa obrigatoria. Exige listas de elogios formulaicos, marcadores negativos e padroes como "parabens" em contexto de reclamacao.

**Impacto esperado:** medio. A precisao pode ser boa, mas a cobertura tende a ser baixa.

**Complexidade realista:** media.

**Risco principal:** inverter elogios genuinos. Esse erro e pior do que deixar alguns casos ironicos sem deteccao.

**Recomendacao:** implementar depois, com regras deliberadamente conservadoras e validacao manual.

### #11 - Expansao WordNet-PT

**Fenomeno tratado:** ampliacao de cobertura lexical por sinonimos e antonimos.

**Exemplo:** "O som do fone encantou a familia toda."

**Viabilidade no projeto atual:** incerta. Pode ajudar em termos afetivos ausentes do lexico manual, mas adiciona dependencia externa e pode trazer ruido sem curadoria.

**Dependencias necessarias:** NLTK e recurso WordNet-PT disponivel no ambiente de execucao.

**Impacto esperado:** medio se houver lexico inicial bem definido. Baixo ou negativo se a propagacao de polaridade for automatica demais.

**Complexidade realista:** media.

**Risco principal:** sinonimia fora de contexto e antonimia incompleta, especialmente em portugues brasileiro informal.

**Recomendacao:** apenas avaliar, depois de existir um lexico de dominio curado e uma amostra de validacao.

### #12 - Ontologia ABSA

**Fenomeno tratado:** separacao da opiniao por aspecto, como bateria, tela, camera, entrega, embalagem e atendimento.

**Exemplo:** "Bateria otima, camera fraca, entrega rapida."

**Viabilidade no projeto atual:** baixa para implementacao imediata e alta como objetivo futuro. A utilidade e grande, mas exige mudanca de arquitetura.

**Dependencias necessarias:** ontologia curada de aspectos, regras de ligacao entre opiniao e aspecto e, idealmente, suporte sintatico da regra #13.

**Impacto esperado:** alto para interpretabilidade. Permite relatorios do tipo "reviews elogiam bateria, mas reclamam de camera".

**Complexidade realista:** alta.

**Risco principal:** atribuir opiniao ao aspecto errado, gerando conclusoes enganosas.

**Recomendacao:** implementar depois, como etapa futura separada do classificador geral de sentimento.

### #13 - Parsing de dependencias

**Fenomeno tratado:** escopo sintatico real de negacao, modificadores e relacoes entre substantivos e adjetivos.

**Exemplo:** "Nao achei, mesmo depois de tres meses de uso diario, que o produto fosse bom."

**Viabilidade no projeto atual:** baixa para curto prazo. A regra exige dependencia pesada e altera o desenho do pipeline.

**Dependencias necessarias:** spaCy e modelo `pt_core_news_lg`, alem de instalacao e carregamento do modelo no ambiente local.

**Impacto esperado:** alto para negacao longa e para ABSA.

**Complexidade realista:** alta.

**Risco principal:** custo computacional, instalacao do modelo e erros de parse em texto informal de review.

**Recomendacao:** etapa futura. Deve ser considerada depois de validar ganhos com regras mais simples.

## 5. Dependencias tecnicas

### Sem novas dependencias externas

As seguintes regras podem ser prototipadas com Python padrao e estruturas simples:

- #1 Detector de Pros/Contras
- #2 Lexico de dominio
- #3 Polarity shifters
- #4 Lexico discursivo tipado
- #5 Lexico de emojis
- #6 Caixa-alta e alongamento
- #7 Hedges
- #8 Frames
- #9 Comparativos simples
- #10 Ironia conservadora

### Com possiveis novas dependencias

As seguintes regras exigem avaliacao de dependencias:

- #11 WordNet-PT: possivel uso de NLTK e recurso lexical adicional.
- #12 ABSA: exige ontologia curada e provavel suporte sintatico.
- #13 Parsing de dependencias: exige spaCy e modelo `pt_core_news_lg`.

## 6. Riscos principais

- Regras simbolicas podem melhorar interpretabilidade, mas piorar metricas se forem usadas para alterar rotulos finais sem validacao.
- A limpeza textual atual remove alguns sinais relevantes, como emojis e pontuacao expressiva. Regras que dependem desses sinais precisam ser aplicadas antes de `clean_text`.
- Reviews curtas e informais podem acionar falsos positivos em regras de ironia, comparacao e discurso.
- Bases diferentes podem ter distribuicoes linguisticas diferentes. Uma regra boa para Mercado Livre pode nao se comportar igual em B2W ou Olist.
- LIWC nao esta integrado ao projeto. Qualquer comparacao com LIWC precisa ser documentada como proposta futura, nao como dependencia atual.

## 7. Criterios de validacao

Antes de expandir as regras ou usa-las para corrigir automaticamente predicoes dos modelos supervisionados, recomenda-se validar em uma amostra manual de reviews.

Criterios minimos:

- medir cobertura: quantos reviews acionam a regra;
- medir precisao manual: quantos acionamentos parecem corretos;
- separar casos por base de origem, quando possivel;
- comparar impacto em accuracy e F1 macro se a regra alterar a decisao final;
- registrar exemplos de falso positivo e falso negativo;
- manter as regras conservadoras enquanto nao houver validacao suficiente.

Para uma primeira rodada, a validacao pode usar uma amostra pequena e controlada:

- 50 reviews negativos;
- 50 reviews neutros;
- 50 reviews positivos;
- 50 reviews mistos ou longos, quando disponiveis.

## 8. Recomendacao final

A primeira implementacao ja comecou por regras de baixo custo e alta interpretabilidade, concentradas em `sentiment_analyzer.py`. Ela cobre sinais de dominio, shifters, eventos fortes de e-commerce, contraste, concessao, intensificacao e algumas regras discursivas.

Depois, vale avancar para #4, #8, #9 e #10, desde que os pesos e escopos sejam conservadores. A regra #11 deve ficar condicionada a uma avaliacao especifica de WordNet-PT. As regras #12 e #13 devem ser tratadas como uma fase futura, pois representam uma mudanca de arquitetura em direcao a analise baseada em aspectos e sintaxe de dependencias.

No estado atual do projeto, a melhor estrategia continua sendo validar as regras como camada complementar ao classificador supervisionado, nao como substituicao imediata do pipeline com TF-IDF e modelos lineares. As proximas evolucoes devem priorizar validacao manual dos disparos, melhoria dos lexicos e comparacao das metricas do `symbolic_rules` apos uma nova execucao completa de `python main.py`.
