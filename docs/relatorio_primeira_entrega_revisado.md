# AnÃ¡lise de Sentimentos para Plataformas de E-Commerce

**Disciplina:** Processamento de Linguagem Natural (SCC0633 / SCC5908)
**Professores:** Renato M. Silva e GraÃ§a Nunes
**Equipe:** PLN Rocks
**Integrantes:** Alecsander GonÃ§alves de Matos, Victor Anthony Pereira Alves, Vitor Rodrigues Tonon

## 1. IntroduÃ§Ã£o

Este projeto trata da anÃ¡lise de sentimentos em avaliaÃ§Ãµes de produtos de e-commerce brasileiro. O problema consiste em receber o texto bruto de uma avaliaÃ§Ã£o escrita por um usuÃ¡rio e produzir como saÃ­da uma classe de polaridade: **positivo**, **negativo** ou **neutro**.

Exemplos de entrada e saÃ­da:

| Entrada | SaÃ­da esperada |
| --- | --- |
| `Produto excelente, chegou rÃ¡pido.` | positivo |
| `NÃ£o recebi o produto.` | negativo |
| `Produto entregue ontem.` | neutro |

A tarefa Ã© relevante porque avaliaÃ§Ãµes de e-commerce registram a percepÃ§Ã£o real dos consumidores sobre produto, entrega, atendimento, embalagem e custo-benefÃ­cio. Um sistema capaz de detectar sentimentos pode resumir grandes volumes de reviews, apoiar a comparaÃ§Ã£o entre produtos e alimentar mecanismos simples de recomendaÃ§Ã£o.

O projeto compara duas famÃ­lias de soluÃ§Ã£o. A primeira usa modelos supervisionados com TF-IDF, RegressÃ£o LogÃ­stica e LinearSVC. A segunda usa um analisador simbÃ³lico interpretÃ¡vel, baseado em lÃ©xicos internos e regras explÃ­citas. O analisador simbÃ³lico Ã© tratado como baseline explicÃ¡vel: ele nÃ£o busca substituir os modelos supervisionados, mas fornecer uma referÃªncia simples, rastreÃ¡vel e adequada para discutir fenÃ´menos linguÃ­sticos.

Os principais desafios observados sÃ£o variaÃ§Ã£o linguÃ­stica em portuguÃªs brasileiro, abreviaÃ§Ãµes, erros ortogrÃ¡ficos, polaridade mista, negaÃ§Ã£o, contraste, ironia, intensificaÃ§Ã£o e a limitaÃ§Ã£o de usar a nota numÃ©rica como aproximaÃ§Ã£o do sentimento textual.

## 2. Metodologia

### 2.1 Bases de dados e rotulagem

A base principal Ã© o corpus **B2W-Reviews01**. TambÃ©m foram usados o **Olist Brazilian E-Commerce Public Dataset** e uma coleta simples de avaliaÃ§Ãµes do **Mercado Livre**, integrada ao pipeline no mesmo formato das demais bases.

A classe de sentimento foi inferida a partir da nota da avaliaÃ§Ã£o:

| Nota | Classe |
| --- | --- |
| 1 ou 2 | negativo |
| 3 | neutro |
| 4 ou 5 | positivo |

Resumo dos conjuntos usados:

| Experimento | Linhas | Negativo | Neutro | Positivo |
| --- | ---: | ---: | ---: | ---: |
| B2W principal | 132.220 | 35.708 | 16.295 | 80.217 |
| B2W + Olist | 175.423 | 47.199 | 20.028 | 108.196 |
| B2W + Mercado Livre simples | 133.602 | 35.856 | 16.341 | 81.405 |
| B2W + Olist + Mercado Livre simples | 176.805 | 47.347 | 20.074 | 109.384 |

O corpus Ã© desbalanceado: a classe positiva representa a maior parte das avaliaÃ§Ãµes, enquanto a classe neutra Ã© minoritÃ¡ria. Por isso, alÃ©m de acurÃ¡cia, a avaliaÃ§Ã£o considera precisÃ£o, recall e F1-macro.

### 2.2 Modelos supervisionados

O pipeline supervisionado executa as seguintes etapas:

1. uniÃ£o dos campos textuais da avaliaÃ§Ã£o;
2. limpeza textual;
3. remoÃ§Ã£o de stopwords;
4. radicalizaÃ§Ã£o simples por sufixos;
5. vetorizaÃ§Ã£o TF-IDF com unigramas e bigramas;
6. treino de RegressÃ£o LogÃ­stica e LinearSVC;
7. avaliaÃ§Ã£o em conjunto de teste com 20% dos dados, usando `random_state=42` e estratificaÃ§Ã£o por classe.

Os dois classificadores usam `class_weight='balanced'` para reduzir o impacto do desbalanceamento entre classes.

### 2.3 Analisador simbÃ³lico

O analisador simbÃ³lico estÃ¡ implementado na funÃ§Ã£o `analyze_symbolic_sentiment`, em `sentiment_analyzer.py`. Ele recebe o texto bruto da avaliaÃ§Ã£o e retorna o rÃ³tulo final, os escores positivo e negativo, a confianÃ§a e a lista de regras acionadas.

Fluxo geral:

```text
Review bruto
  -> normalizaÃ§Ã£o simbÃ³lica
  -> detecÃ§Ã£o de padrÃµes fortes
  -> aplicaÃ§Ã£o de lÃ©xicos positivos e negativos
  -> ajuste por contexto linguÃ­stico
  -> tratamento de contraste, concessÃ£o e Ãªnfase
  -> agregaÃ§Ã£o de scores
  -> rÃ³tulo final e explicaÃ§Ã£o das regras acionadas
```

O texto Ã© normalizado para minÃºsculas, tem acentos removidos por NFKD e espaÃ§os normalizados. A decisÃ£o final usa:

```text
score = positive_score - negative_score
```

Se o score for maior que `0,75`, a classe Ã© positiva. Se for menor que `-0,75`, a classe Ã© negativa. Valores dentro desse intervalo sÃ£o classificados como neutros.

## 3. Recursos simbÃ³licos utilizados

Os lÃ©xicos e padrÃµes usados pelo analisador simbÃ³lico foram construÃ­dos pela equipe e estÃ£o definidos diretamente em `sentiment_analyzer.py`. NÃ£o foram importados lÃ©xicos externos como SentiLex, LIWC, WordNet, spaCy ou dicionÃ¡rios prontos.

Os critÃ©rios de inclusÃ£o foram empÃ­ricos e orientados ao domÃ­nio: termos frequentes e semanticamente claros em avaliaÃ§Ãµes de e-commerce, especialmente relacionados a satisfaÃ§Ã£o, recomendaÃ§Ã£o, defeito, entrega, custo-benefÃ­cio e qualidade do produto. Os pesos foram calibrados manualmente para refletir a forÃ§a semÃ¢ntica esperada de cada termo ou expressÃ£o.

### 3.1 LÃ©xico positivo interno

O lÃ©xico positivo contÃ©m **24 termos**. As entradas abaixo aparecem no formato normalizado usado pelo cÃ³digo, isto Ã©, sem acentos:

| Termo | Peso | Termo | Peso | Termo | Peso |
| --- | ---: | --- | ---: | --- | ---: |
| adorei | 1,4 | funciona | 1,0 | rapido | 1,0 |
| amei | 1,5 | gostei | 1,1 | recomendo | 1,3 |
| aprovado | 1,2 | maravilhoso | 1,6 | resistente | 1,0 |
| barato | 0,8 | melhor | 1,0 | satisfeito | 1,2 |
| bom | 1,0 | original | 1,0 | superou | 1,4 |
| bonito | 0,8 | otimo | 1,4 | top | 1,4 |
| confortavel | 1,0 | perfeito | 1,5 | vale a pena | 1,4 |
| cumpre | 0,9 | qualidade | 0,8 | excelente | 1,6 |

Termos como `excelente`, `maravilhoso`, `perfeito` e `amei` recebem pesos maiores por indicarem satisfaÃ§Ã£o explÃ­cita. Termos como `barato`, `bonito` e `qualidade` recebem pesos menores porque podem depender mais do contexto.

### 3.2 LÃ©xico negativo interno

O lÃ©xico negativo contÃ©m **18 termos**, tambÃ©m no formato normalizado usado pelo cÃ³digo:

| Termo | Peso | Termo | Peso | Termo | Peso |
| --- | ---: | --- | ---: | --- | ---: |
| arrependi | 1,4 | defeito | 1,7 | horrivel | 1,6 |
| atrasado | 1,2 | demorou | 1,1 | lento | 1,0 |
| atrasou | 1,4 | desliga | 1,4 | pessimo | 1,7 |
| bugou | 1,4 | enganosa | 1,6 | quebrado | 1,8 |
| caro | 0,9 | falso | 1,7 | ruim | 1,3 |
| decepcionado | 1,5 | fraco | 1,1 | travou | 1,5 |

Termos associados a defeitos objetivos, como `quebrado`, `defeito`, `falso` e `pÃ©ssimo`, tÃªm maior peso porque normalmente indicam avaliaÃ§Ã£o negativa forte.

### 3.3 PadrÃµes fortes de domÃ­nio

AlÃ©m de palavras isoladas, o sistema usa **16 padrÃµes fortes**. Esses padrÃµes recebem peso maior porque capturam eventos ou atos de fala recorrentes em e-commerce.

| Nome | Polaridade | Peso | Exemplos cobertos |
| --- | --- | ---: | --- |
| nao_recebimento | negativa | 2,6 | `nÃ£o recebi`, `produto nÃ£o chegou` |
| defeito_funcionamento | negativa | 2,4 | `nÃ£o funciona`, `veio com defeito`, `chegou quebrado` |
| recomendacao_negativa | negativa | 2,0 | `nÃ£o recomendo`, `nÃ£o comprem`, `evitem` |
| devolucao_garantia | negativa | 1,8 | `quero devoluÃ§Ã£o`, `pedi reembolso`, `procon` |
| expectativa_negativa | negativa | 1,8 | `esperava mais`, `diferente da foto`, `propaganda enganosa` |
| mudanca_temporal_negativa | negativa | 1,8 | inÃ­cio positivo seguido de `mas/depois` e falha |
| valor_negativo | negativa | 1,7 | `nÃ£o vale a pena`, `dinheiro jogado fora` |
| recomendacao_positiva | positiva | 1,7 | `super recomendo`, `podem comprar` |
| ironia_conservadora | negativa | 1,6 | elogio prÃ³ximo de `quebrado`, `nÃ£o funciona` ou `defeito` |
| entrega_negativa | negativa | 1,5 | `entrega atrasou`, `prazo nÃ£o cumprido` |
| valor_positivo | positiva | 1,5 | `bom custo benefÃ­cio`, `preÃ§o justo` |
| entrega_positiva | positiva | 1,4 | `chegou rÃ¡pido`, `chegou antes do prazo` |
| atribuicao_positiva | positiva | 1,1 | familiar prÃ³ximo de `adorou`, `amou`, `gostou` |
| atribuicao_negativa | negativa | 1,1 | familiar prÃ³ximo de `odiou`, `reclamou`, `nÃ£o gostou` |
| mudanca_temporal_positiva | positiva | 1,0 | inÃ­cio negativo seguido de melhora |
| condicional_contrafactual | negativa | 0,9 | `seria/teria/poderia` prÃ³ximo de `se/caso` |

### 3.4 Outros recursos simbÃ³licos

O sistema tambÃ©m usa:

| Recurso | Como atua |
| --- | --- |
| Emojis | 9 mapeamentos: 6 positivos e 3 negativos |
| Intensificadores | `muito`, `super`, `mega`, `extremamente`, `demais`, `bastante`, `totalmente` aumentam peso |
| Atenuadores | `pouco`, `meio`, `quase`, `um pouco`, `mais ou menos` reduzem peso |
| Hedges | `talvez`, `acho que`, `parece`, `pode ser` reduzem confianÃ§a |
| Condicionais | `seria`, `teria`, `poderia`, `se`, `caso` reduzem forÃ§a da evidÃªncia |
| Inversores de polaridade | `nÃ£o`, `nunca`, `sem`, `parou de`, `deixou de`, `falta de` invertem ou deslocam polaridade |
| Contraste | trecho apÃ³s `mas` ou `porÃ©m` recebe maior peso |
| ConcessÃ£o | trecho principal apÃ³s `embora` ou `apesar de` recebe maior peso |
| ÃŠnfase | caixa-alta, alongamento de letras e pontuaÃ§Ã£o expressiva ajustam peso |

## 4. Flexibilidade dos padrÃµes

Os padrÃµes sÃ£o parcialmente flexÃ­veis. Eles nÃ£o fazem anÃ¡lise morfolÃ³gica completa, mas cobrem algumas variaÃ§Ãµes superficiais importantes.

| Tipo de flexibilidade | Coberto? | Mecanismo | Exemplo |
| --- | --- | --- | --- |
| Acentos | Sim | NormalizaÃ§Ã£o NFKD | `rÃ¡pido` e `rapido` sÃ£o tratados de forma equivalente |
| MaiÃºsculas/minÃºsculas | Sim | ConversÃ£o para minÃºsculas | `EXCELENTE` casa com `excelente` |
| EspaÃ§os mÃºltiplos | Sim | Regex aceita `\s+` entre palavras | `chegou  rÃ¡pido` casa com `chegou rÃ¡pido` |
| Fronteira de palavra | Sim | Uso de fronteiras regex | `caro` nÃ£o casa dentro de `encaro` |
| Formas listadas explicitamente | Sim | Alternativas no padrÃ£o | `nÃ£o funciona` e `nÃ£o funcionou` sÃ£o cobertos |
| VariaÃ§Ã£o morfolÃ³gica automÃ¡tica | NÃ£o | NÃ£o hÃ¡ lematizaÃ§Ã£o | `chegaram rÃ¡pido` nÃ£o casa automaticamente com `chegou rÃ¡pido` |
| InserÃ§Ã£o de modificador no meio do padrÃ£o | NÃ£o | PadrÃ£o exige a expressÃ£o esperada | `chegou muito rÃ¡pido` nÃ£o casa com `chegou rÃ¡pido` |

Portanto, `chegou rÃ¡pido` Ã© detectado pelo padrÃ£o de entrega positiva, mas `chegaram rÃ¡pido` nÃ£o Ã© detectado como o mesmo padrÃ£o, a menos que essa forma seja adicionada explicitamente. Essa limitaÃ§Ã£o Ã© uma das causas de perda de cobertura do mÃ©todo simbÃ³lico em corpus real. Uma soluÃ§Ã£o mais robusta exigiria lematizaÃ§Ã£o ou expansÃ£o dos padrÃµes por lema.

## 5. AvaliaÃ§Ã£o formal

### 5.1 SuÃ­te curada do analisador simbÃ³lico

O arquivo `test_symbolic_analyzer.py` contÃ©m 15 casos de teste curados para validar os principais comportamentos do analisador simbÃ³lico. A suÃ­te cobre avaliaÃ§Ãµes positivas simples, recomendaÃ§Ãµes, entrega, valor, concessÃ£o, nÃ£o recebimento, contraste, defeito, devoluÃ§Ã£o, ironia conservadora, mudanÃ§a temporal, condicional e textos neutros.

| NÂº | Categoria | Entrada | Esperado | Obtido | Resultado |
| ---: | --- | --- | --- | --- | --- |
| 1 | Positivo simples | `Produto excelente, chegou rapido.` | positivo | positivo | Acerto |
| 2 | PrÃ³s positivo | `Gostei: produto otimo.` | positivo | positivo | Acerto |
| 3 | Entrega positiva | `Chegou antes do prazo, super recomendo.` | positivo | positivo | Acerto |
| 4 | Valor positivo | `Bom custo beneficio e preco justo.` | positivo | positivo | Acerto |
| 5 | ConcessÃ£o positiva | `Embora tenha demorado, o produto e otimo.` | positivo | positivo | Acerto |
| 6 | NÃ£o recebimento | `Nao recebi o produto.` | negativo | negativo | Acerto |
| 7 | Contraste negativo | `O produto e bonito, mas nao funciona.` | negativo | negativo | Acerto |
| 8 | Contras negativo | `Nao gostei: produto ruim.` | negativo | negativo | Acerto |
| 9 | Defeito e devoluÃ§Ã£o | `Veio com defeito e quero devolucao.` | negativo | negativo | Acerto |
| 10 | RecomendaÃ§Ã£o negativa | `Nao recomendo, dinheiro jogado fora.` | negativo | negativo | Acerto |
| 11 | Ironia conservadora | `Produto excelente! Chegou quebrado e nao funciona.` | negativo | negativo | Acerto |
| 12 | MudanÃ§a temporal negativa | `No comeco era bom, mas depois parou de funcionar.` | negativo | negativo | Acerto |
| 13 | Condicional | `Seria otimo se a bateria durasse mais.` | neutro | neutro | Acerto |
| 14 | Factual neutro | `Produto entregue ontem.` | neutro | neutro | Acerto |
| 15 | Texto vazio | `` | neutro | neutro | Acerto |

Resultado da suÃ­te:

| Classe | Casos | TP | FP | FN | PrecisÃ£o | Recall | F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Positivo | 5 | 5 | 0 | 0 | 1,00 | 1,00 | 1,00 |
| Negativo | 7 | 7 | 0 | 0 | 1,00 | 1,00 | 1,00 |
| Neutro | 3 | 3 | 0 | 0 | 1,00 | 1,00 | 1,00 |
| Macro | 15 | - | - | - | 1,00 | 1,00 | 1,00 |

A suÃ­te mostra que as regras implementadas funcionam nos casos para os quais foram projetadas. Ela nÃ£o substitui a avaliaÃ§Ã£o em corpus real, porque os exemplos sÃ£o controlados e nÃ£o representam toda a variabilidade das avaliaÃ§Ãµes reais.

### 5.2 AvaliaÃ§Ã£o quantitativa em corpus real

A avaliaÃ§Ã£o em corpus real usa divisÃ£o treino/teste de 80%/20%, estratificada por classe, com `random_state=42`. Para o analisador simbÃ³lico, o mesmo conjunto de teste Ã© usado, mas sem etapa de treinamento.

Como `metricas.csv` armazena acurÃ¡cia com quatro casas decimais, os nÃºmeros de acertos e erros abaixo foram reconstruÃ­dos por arredondamento a partir do tamanho do conjunto de teste e da acurÃ¡cia registrada.

| Experimento | Modelo | Teste | Acertos | Erros | Accuracy | Precision macro | Recall macro | F1 macro |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| B2W principal | RegressÃ£o LogÃ­stica | 26.444 | 21.465 | 4.979 | 0,8117 | 0,7307 | 0,7735 | 0,7423 |
| B2W principal | LinearSVC | 26.444 | 22.197 | 4.247 | 0,8394 | 0,7339 | 0,7363 | 0,7347 |
| B2W principal | SimbÃ³lico | 26.444 | 19.653 | 6.791 | 0,7432 | 0,6366 | 0,5859 | 0,6018 |
| B2W + Olist | RegressÃ£o LogÃ­stica | 35.085 | 28.370 | 6.715 | 0,8086 | 0,7125 | 0,7517 | 0,7234 |
| B2W + Olist | LinearSVC | 35.085 | 29.401 | 5.684 | 0,8380 | 0,7151 | 0,7154 | 0,7145 |
| B2W + Olist | SimbÃ³lico | 35.085 | 24.837 | 10.248 | 0,7079 | 0,6251 | 0,5689 | 0,5803 |
| B2W + Mercado Livre | RegressÃ£o LogÃ­stica | 26.721 | 21.751 | 4.970 | 0,8140 | 0,7330 | 0,7775 | 0,7452 |
| B2W + Mercado Livre | LinearSVC | 26.721 | 22.494 | 4.227 | 0,8418 | 0,7377 | 0,7398 | 0,7385 |
| B2W + Mercado Livre | SimbÃ³lico | 26.721 | 19.926 | 6.795 | 0,7457 | 0,6386 | 0,5879 | 0,6036 |
| B2W + Olist + Mercado Livre | RegressÃ£o LogÃ­stica | 35.361 | 28.653 | 6.708 | 0,8103 | 0,7134 | 0,7525 | 0,7242 |
| B2W + Olist + Mercado Livre | LinearSVC | 35.361 | 29.650 | 5.711 | 0,8385 | 0,7148 | 0,7150 | 0,7141 |
| B2W + Olist + Mercado Livre | SimbÃ³lico | 35.361 | 25.135 | 10.226 | 0,7108 | 0,6248 | 0,5701 | 0,5814 |

Os resultados mostram que os modelos supervisionados superam o baseline simbÃ³lico em todas as configuraÃ§Ãµes. A melhor F1-macro geral Ã© da RegressÃ£o LogÃ­stica com B2W + Mercado Livre simples (`0,7452`). A melhor acurÃ¡cia geral Ã© do LinearSVC no mesmo experimento (`0,8418`). O melhor resultado simbÃ³lico ocorre tambÃ©m em B2W + Mercado Livre simples, com acurÃ¡cia `0,7457` e F1-macro `0,6036`.

## 6. AnÃ¡lise dos erros e limitaÃ§Ãµes

A diferenÃ§a entre a suÃ­te controlada e o corpus real ocorre porque as avaliaÃ§Ãµes reais apresentam variaÃ§Ãµes que nÃ£o estÃ£o totalmente cobertas pelas regras.

**Cobertura lexical limitada:** o lÃ©xico tem 24 termos positivos e 18 negativos, uma fraÃ§Ã£o pequena do vocabulÃ¡rio real. Termos como `fantÃ¡stico`, `decepcionante`, `emperrou`, `sumiu` ou gÃ­rias especÃ­ficas podem nÃ£o ser detectados.

**AusÃªncia de lematizaÃ§Ã£o:** o sistema nÃ£o reduz palavras ao lema. Assim, `chegaram rÃ¡pido`, `funcionam`, `funcionaria` ou `quebraram` nÃ£o sÃ£o automaticamente associados a `chegou rÃ¡pido`, `funciona` ou `quebrado`.

**Classe neutra difÃ­cil:** o recall macro do simbÃ³lico fica em torno de `0,57` a `0,59`. Parte desse problema vem da ambiguidade entre textos neutros factuais e textos curtos com opiniÃ£o fraca.

**RuÃ­do ortogrÃ¡fico e abreviaÃ§Ãµes:** avaliaÃ§Ãµes reais contÃªm erros como `otimooo`, `prfeit`, `mto bom`, abreviaÃ§Ãµes e emojis nÃ£o mapeados. A normalizaÃ§Ã£o de acentos nÃ£o resolve esses casos.

**Ironia sutil:** hÃ¡ apenas uma regra conservadora para ironia, acionada quando elogio explÃ­cito aparece prÃ³ximo de evidÃªncia negativa objetiva. Casos sutis continuam sem cobertura.

**Janela de contexto:** modificadores e inversores sÃ£o analisados em janelas locais. NegaÃ§Ãµes ou contrastes distantes podem nÃ£o ser associados ao termo correto.

As melhorias mais diretas seriam expansÃ£o dos lÃ©xicos, lematizaÃ§Ã£o com uma ferramenta como spaCy, normalizaÃ§Ã£o de erros ortogrÃ¡ficos comuns, calibraÃ§Ã£o dos limiares com conjunto anotado e ampliaÃ§Ã£o controlada dos padrÃµes fortes.

## 7. RecomendaÃ§Ã£o de itens

AlÃ©m da anÃ¡lise de sentimentos, foi implementado um baseline simples de recomendaÃ§Ã£o. Na B2W, o histÃ³rico positivo de cada usuÃ¡rio define categorias de interesse, e o sistema recomenda produtos bem avaliados da mesma categoria, excluindo itens jÃ¡ avaliados pelo usuÃ¡rio.

Quando nÃ£o hÃ¡ identificador de usuÃ¡rio, como na coleta simples do Mercado Livre, o sistema gera um ranking global por categoria, usando mÃ©dia de nota, proporÃ§Ã£o de avaliaÃ§Ãµes positivas, sentimento mÃ©dio e volume de reviews. Esse componente Ã© apenas um baseline explicÃ¡vel e depende diretamente da qualidade dos rÃ³tulos de sentimento.

## 8. ConsideraÃ§Ãµes finais

O projeto fornece uma linha de base reprodutÃ­vel para anÃ¡lise de sentimentos em avaliaÃ§Ãµes de e-commerce brasileiro. A avaliaÃ§Ã£o formal mostra dois resultados complementares.

Na suÃ­te curada de 15 testes, o analisador simbÃ³lico acerta todos os casos, indicando que as regras implementadas cobrem os fenÃ´menos linguÃ­sticos previstos: negaÃ§Ã£o, contraste, concessÃ£o, recomendaÃ§Ã£o, defeito, entrega e condicionalidade.

No corpus real, a avaliaÃ§Ã£o quantitativa mostra que os modelos supervisionados generalizam melhor. A melhor F1-macro Ã© `0,7452`, obtida pela RegressÃ£o LogÃ­stica com B2W + Mercado Livre simples. A melhor acurÃ¡cia Ã© `0,8418`, obtida pelo LinearSVC na mesma configuraÃ§Ã£o. O melhor resultado simbÃ³lico Ã© acurÃ¡cia `0,7457` e F1-macro `0,6036`.

O gap de aproximadamente 14 pontos percentuais em F1-macro entre o melhor modelo supervisionado e o simbÃ³lico indica que o baseline por regras Ã© interpretÃ¡vel, mas limitado pela cobertura dos lÃ©xicos e padrÃµes. Sua principal vantagem Ã© a explicabilidade: cada decisÃ£o pode ser rastreada para termos, expressÃµes ou regras acionadas. Sua principal limitaÃ§Ã£o Ã© a baixa flexibilidade diante de variaÃ§Ã£o morfolÃ³gica, ruÃ­do e vocabulÃ¡rio nÃ£o previsto.

Assim, o analisador simbÃ³lico cumpre o papel de baseline explicÃ¡vel, enquanto os modelos supervisionados apresentam melhor desempenho quantitativo. A evoluÃ§Ã£o natural do projeto seria ampliar os lÃ©xicos, incorporar lematizaÃ§Ã£o e validar novas regras em amostra manual antes de usÃ¡-las para alterar decisÃµes do classificador.

## ReferÃªncias

OLIST. *Brazilian E-Commerce Public Dataset by Olist*. Kaggle, 2018. DisponÃ­vel em: <https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce>. Acesso em: 19 abr. 2026.

REAL, L.; OSHIRO, M.; MAFRA, A. *B2W-Reviews01: an open product reviews corpus*. 2019. DisponÃ­vel em: <https://github.com/b2wdigital/b2w-reviews01>. Acesso em: 19 abr. 2026.

MERCADO LIVRE. *DocumentaÃ§Ã£o da API de opiniÃµes sobre um produto*. DisponÃ­vel em: <https://developers.mercadolivre.com.br/pt_br/opinioes-sobre-um-produto>. Acesso em: 19 abr. 2026.
