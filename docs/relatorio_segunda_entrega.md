# AnÃ¡lise de Sentimentos para E-Commerce â€” Segunda Entrega

**Equipe:** PLN Rocks
**Integrantes:** Alecsander GonÃ§alves de Matos, Victor Anthony Pereira Alves, Vitor Rodrigues Tonon
**Disciplina:** Processamento de Linguagem Natural (SCC0633 / SCC5908)
**Professores:** Renato M. Silva e GraÃ§a Nunes

---

## 1. IntroduÃ§Ã£o

Este projeto implementa um pipeline de anÃ¡lise de sentimentos em avaliaÃ§Ãµes de e-commerce brasileiro, combinando dois paradigmas: modelos supervisionados (RegressÃ£o LogÃ­stica e LinearSVC com TF-IDF) e um analisador simbÃ³lico baseado em lÃ©xico e regras discursivas. A base principal Ã© o B2W-Reviews01; bases complementares incluem o Olist Brazilian E-Commerce Public Dataset e uma coleta prÃ³pria de avaliaÃ§Ãµes do Mercado Livre. Adicionalmente, foi implementado um sistema de recomendaÃ§Ã£o de itens baseado nos sentimentos inferidos.

Nesta segunda entrega, o analisador simbÃ³lico foi formalmente avaliado em um conjunto de testes curados e em corpus completo, atendendo ao pedido de avaliaÃ§Ã£o quantitativa levantado na revisÃ£o anterior. Os lÃ©xicos utilizados sÃ£o agora documentados em sua totalidade, a flexibilidade dos padrÃµes de correspondÃªncia Ã© analisada com exemplos concretos, e as consideraÃ§Ãµes finais apresentam nÃºmeros reais de desempenho de todos os modelos.

---

## 2. Metodologia

### 2.1 Bases de Dados

A classificaÃ§Ã£o de sentimento foi derivada da nota numÃ©rica da avaliaÃ§Ã£o: notas 1â€“2 sÃ£o rotuladas como **negativo**, nota 3 como **neutro** e notas 4â€“5 como **positivo**.

| Experimento | Linhas | Negativo | Neutro | Positivo |
|---|---|---|---|---|
| B2W principal | 132.220 | 35.708 | 16.295 | 80.217 |
| B2W + Olist | 175.423 | 47.199 | 20.028 | 108.196 |
| B2W + Mercado Livre simples | 133.602 | 35.856 | 16.341 | 81.405 |
| B2W + Olist + Mercado Livre simples | 176.805 | 47.347 | 20.074 | 109.384 |

O desbalanceamento entre classes Ã© visÃ­vel: avaliaÃ§Ãµes positivas representam cerca de 60% do corpus B2W, enquanto avaliaÃ§Ãµes neutras ficam em torno de 12%. Esse desbalanceamento penaliza especialmente o recall da classe neutra nos modelos avaliados.

### 2.2 PrÃ©-processamento e Modelos Supervisionados

O prÃ©-processamento compreende: (i) unificaÃ§Ã£o dos campos de tÃ­tulo e corpo da avaliaÃ§Ã£o, (ii) conversÃ£o para letras minÃºsculas e remoÃ§Ã£o de caracteres especiais, (iii) remoÃ§Ã£o de stopwords do portuguÃªs, (iv) radicaÃ§Ã£o simples por sufixo. Os textos resultantes sÃ£o vetorizados com TF-IDF e alimentam dois classificadores: RegressÃ£o LogÃ­stica (`max_iter=1000`, `class_weight='balanced'`, `random_state=42`) e LinearSVC (`class_weight='balanced'`, `random_state=42`). A divisÃ£o treino/teste segue o padrÃ£o `train_test_split` com `random_state=42`.

### 2.3 Analisador SimbÃ³lico

O analisador simbÃ³lico opera sobre texto normalizado sem necessidade de treinamento. A normalizaÃ§Ã£o (`normalize_symbolic_text`) converte o texto para minÃºsculas e remove acentos via decomposiÃ§Ã£o NFKD, tornando o sistema insensÃ­vel a variaÃ§Ãµes de acentuaÃ§Ã£o. A pontuaÃ§Ã£o final Ã© dada por:

```
score = positive_score âˆ’ negative_score
```

O rÃ³tulo atribuÃ­do Ã© **positivo** se `score > 0,75`, **negativo** se `score < âˆ’0,75`, e **neutro** caso contrÃ¡rio.

#### 2.3.1 LÃ©xico Positivo

O lÃ©xico positivo contÃ©m **24 termos** com pesos empÃ­ricos calibrados para o domÃ­nio de avaliaÃ§Ãµes de e-commerce brasileiro. Os termos foram selecionados com base na frequÃªncia de uso em contextos positivos de compras online; os pesos refletem a intensidade tÃ­pica de cada termo no domÃ­nio (termos de recomendaÃ§Ã£o explÃ­cita recebem pesos mais altos).

| Termo | Peso | Termo | Peso | Termo | Peso |
|---|---|---|---|---|---|
| adorei | 1,4 | funciona | 1,0 | rapido | 1,0 |
| amei | 1,5 | gostei | 1,1 | recomendo | 1,3 |
| aprovado | 1,2 | maravilhoso | 1,6 | resistente | 1,0 |
| barato | 0,8 | melhor | 1,0 | satisfeito | 1,2 |
| bom | 1,0 | original | 1,0 | superou | 1,4 |
| bonito | 0,8 | otimo | 1,4 | top | 1,4 |
| confortavel | 1,0 | perfeito | 1,5 | vale a pena | 1,4 |
| cumpre | 0,9 | qualidade | 0,8 | excelente | 1,6 |

**Total: 24 termos.** Pesos no intervalo [0,8 ; 1,6].

#### 2.3.2 LÃ©xico Negativo

O lÃ©xico negativo contÃ©m **18 termos** com pesos empÃ­ricos, priorizando defeitos de produto e problemas de entrega.

| Termo | Peso | Termo | Peso | Termo | Peso |
|---|---|---|---|---|---|
| arrependi | 1,4 | defeito | 1,7 | horrivel | 1,6 |
| atrasado | 1,2 | demorou | 1,1 | lento | 1,0 |
| atrasou | 1,4 | desliga | 1,4 | pessimo | 1,7 |
| bugou | 1,4 | enganosa | 1,6 | quebrado | 1,8 |
| caro | 0,9 | falso | 1,7 | ruim | 1,3 |
| decepcionado | 1,5 | fraco | 1,1 | travou | 1,5 |

**Total: 18 termos.** Pesos no intervalo [0,9 ; 1,8]. Os termos com peso mais alto (`quebrado=1,8`, `defeito=1,7`, `falso=1,7`, `pessimo=1,7`) indicam situaÃ§Ãµes objetivamente problemÃ¡ticas e de difÃ­cil reversÃ£o semÃ¢ntica.

#### 2.3.3 PadrÃµes Fortes

Os padrÃµes fortes sÃ£o expressÃµes regulares que detectam construÃ§Ãµes discursivas de maior peso semÃ¢ntico. Cada padrÃ£o Ã© aplicado sobre o texto normalizado com `re.finditer`, e seu peso pode ser amplificado por modificadores contextuais (ver SeÃ§Ã£o 2.3.4).

| Nome | Polaridade | Peso | PadrÃ£o (simplificado) |
|---|---|---|---|
| nao_recebimento | negativo | 2,6 | nao recebi, nunca recebi, nao chegou, produto nao chegou |
| defeito_funcionamento | negativo | 2,4 | nao funciona, nao funcionou, parou de funcionar, veio com defeito, chegou quebrado |
| recomendacao_negativa | negativo | 2,0 | nao recomendo, nao comprem, evitem, nunca mais compro |
| devolucao_garantia | negativo | 1,8 | quero devolucao, pedi reembolso, estorno, procon |
| expectativa_negativa | negativo | 1,8 | esperava mais, diferente da foto, propaganda enganosa, veio errado |
| mudanca_temporal_negativa | negativo | 1,8 | (no comeco / inicialmente) ... (mas / porem) ... (parou / quebrou / defeito) |
| valor_negativo | negativo | 1,7 | nao vale a pena, dinheiro jogado fora, caro pelo que entrega |
| recomendacao_positiva | positivo | 1,7 | super recomendo, recomendo muito, podem comprar |
| ironia_conservadora | negativo | 1,6 | (excelente / otimo) ... (quebrado / nao funciona / defeito) |
| entrega_negativa | negativo | 1,5 | entrega atrasou, demorou para chegar, prazo nao cumprido |
| valor_positivo | positivo | 1,5 | bom custo beneficio, preco justo, vale muito a pena |
| entrega_positiva | positivo | 1,4 | chegou rapido, chegou antes do prazo, entrega perfeita |
| atribuicao_positiva | positivo | 1,1 | (minha filha / meu filho / familia) ... (adorou / amou / gostou) |
| atribuicao_negativa | negativo | 1,1 | (minha filha / meu filho / familia) ... (odiou / reclamou / nao gostou) |
| mudanca_temporal_positiva | positivo | 1,0 | (no comeco / inicialmente) ... (mas / porem) ... (funcionou / melhorou) |
| condicional_contrafactual | negativo | 0,9 | (seria / teria / poderia) ... (se / caso) |

**Total: 16 padrÃµes.** PadrÃµes de alto peso capturam situaÃ§Ãµes de falha objetiva (nÃ£o recebimento, defeito); padrÃµes de baixo peso (condicional) expressam avaliaÃ§Ãµes hipotÃ©ticas com menor certeza.

#### 2.3.4 Modificadores Contextuais

Os modificadores sÃ£o aplicados localmente, examinando uma janela de atÃ© 70 caracteres antes de cada ocorrÃªncia:

| Modificador | PadrÃ£o | Multiplicador |
|---|---|---|
| Intensificador | muito, super, mega, extremamente, demais, bastante, totalmente | x1,25 |
| Atenuador | pouco, meio, quase, um pouco, mais ou menos | x0,65 |
| Hedge (incerteza) | talvez, acho que, parece, pode ser, acredito que, ainda preciso testar | x0,55 |
| Condicional | seria, teria, poderia, se, caso | x0,65 |

**Inversores de polaridade (shifters):** quando uma negaÃ§Ã£o (`nÃ£o`, `nunca`, `sem`, `perdeu`, `deixou de`, `parou de`, `falta de`) precede um termo positivo, o peso Ã© convertido para negativo com fator 1,15; quando precede um termo negativo, converte para positivo com fator 0,80.

Recursos adicionais: detecÃ§Ã£o de emojis (9 mapeamentos: 6 positivos, 3 negativos), seÃ§Ãµes de pros/contras explÃ­citas (`Pontos positivos:`, `NÃ£o gostei:`), anÃ¡lise de contraste pÃ³s-"mas/porÃ©m", anÃ¡lise de concessÃ£o pÃ³s-"embora/apesar de", Ãªnfase por caixa-alta (PALAVRAS EM MAIÃšSCULO), alongamento de letras (`booom`) e pontuaÃ§Ã£o expressiva (`!!`).

#### 2.3.5 Flexibilidade dos PadrÃµes

Esta seÃ§Ã£o responde diretamente Ã  pergunta do professor: *"SÃ£o flexÃ­veis? Tanto 'chegou rÃ¡pido' como 'chegaram rÃ¡pido' sÃ£o detectados?"*

A resposta Ã©: **parcialmente flexÃ­veis**. A tabela abaixo detalha o que Ã© e o que nÃ£o Ã© coberto:

| Tipo de flexibilidade | Coberto? | Mecanismo | Exemplo |
|---|---|---|---|
| Insensibilidade a acentos | Sim | NFKD: `rÃ¡pido` normalizado para `rapido` | "Chegou rÃ¡pido" detectado pelo padrÃ£o chegou rapido |
| Insensibilidade a maiÃºsculas | Sim | NormalizaÃ§Ã£o para minÃºsculas | "EXCELENTE" detectado no lÃ©xico |
| MÃºltiplos espaÃ§os | Sim | `phrase_pattern` usa `\s+` | "chegou  rapido" (duplo espaÃ§o) detectado |
| Fronteira de palavra | Sim | `\b` nas regex | "caro" nÃ£o casa em "encaro" |
| Formas verbais distintas listadas | Sim (explÃ­cito) | Alternativas no padrÃ£o | nao funciona e nao funcionou ambos listados |
| VariaÃ§Ã£o morfolÃ³gica de pessoa/nÃºmero | NÃ£o | NÃ£o hÃ¡ lematizaÃ§Ã£o | "chegaram rapido" nÃ£o detectado (apenas "chegou rapido") |
| VariaÃ§Ã£o de tempo verbal | Parcial | Apenas quando listado | "chegou" nÃ£o equivale a "chega" (nÃ£o listados como variantes) |
| InserÃ§Ã£o de modificador no padrÃ£o forte | NÃ£o | PadrÃ£o exige frase exata | "chegou muito rapido" nÃ£o casa com chegou rapido |

**Exemplos concretos de variaÃ§Ãµes NÃƒO cobertas:**

```
"Chegaram rapido"       -- NAO detectado pelo padrao entrega_positiva
"Chegaria rapido"       -- NAO detectado (condicional muda o sentido)
"O produto funciona"    -- detectado pelo lexico positivo, mas
"Os produtos funcionam" -- NAO detectado pelo lexico (forma plural)
"Nao funcionaria"       -- NAO detectado pelo padrao defeito_funcionamento
```

**LimitaÃ§Ã£o documentada:** os padrÃµes cobrem formas canÃ´nicas do vocabulÃ¡rio de reviews. A expansÃ£o para cobertura morfolÃ³gica completa exigiria lematizaÃ§Ã£o (e.g., spaCy com modelo `pt_core_news_sm`) ou um lÃ©xico por lema (e.g., SentiLex-PT). Esta limitaÃ§Ã£o contribui para o gap de desempenho observado no corpus real (ver SeÃ§Ã£o 3).

---

## 3. AvaliaÃ§Ã£o e Resultados

### 3.1 Conjunto de Testes do Analisador SimbÃ³lico

O arquivo `test_symbolic_analyzer.py` contÃ©m 15 casos de teste que exercitam os principais padrÃµes de linguagem encontrados em reviews de e-commerce. Cada caso inclui um texto de entrada, o rÃ³tulo esperado e a verificaÃ§Ã£o de que ao menos uma regra foi acionada (exceto para o caso de texto vazio).

| NÂº | Categoria | Texto de Entrada | Esperado | Obtido | Resultado |
|---|---|---|---|---|---|
| 1 | Positivo simples | "Produto excelente, chegou rapido." | positivo | positivo | Acerto |
| 2 | Positivo â€” pros explÃ­cito | "Gostei: produto otimo." | positivo | positivo | Acerto |
| 3 | Entrega positiva | "Chegou antes do prazo, super recomendo." | positivo | positivo | Acerto |
| 4 | Valor positivo | "Bom custo beneficio e preco justo." | positivo | positivo | Acerto |
| 5 | ConcessÃ£o positiva | "Embora tenha demorado, o produto e otimo." | positivo | positivo | Acerto |
| 6 | NÃ£o recebimento | "Nao recebi o produto." | negativo | negativo | Acerto |
| 7 | Contraste negativo | "O produto e bonito, mas nao funciona." | negativo | negativo | Acerto |
| 8 | Negativo â€” cons explÃ­cito | "Nao gostei: produto ruim." | negativo | negativo | Acerto |
| 9 | Defeito e devoluÃ§Ã£o | "Veio com defeito e quero devolucao." | negativo | negativo | Acerto |
| 10 | RecomendaÃ§Ã£o negativa | "Nao recomendo, dinheiro jogado fora." | negativo | negativo | Acerto |
| 11 | Ironia conservadora | "Produto excelente! Chegou quebrado e nao funciona." | negativo | negativo | Acerto |
| 12 | MudanÃ§a temporal negativa | "No comeco era bom, mas depois parou de funcionar." | negativo | negativo | Acerto |
| 13 | Condicional â€” baixa certeza | "Seria otimo se a bateria durasse mais." | neutro | neutro | Acerto |
| 14 | Factual neutro | "Produto entregue ontem." | neutro | neutro | Acerto |
| 15 | Texto vazio | "" | neutro | neutro | Acerto |

**Resultado:** 15/15 casos corretos. AcurÃ¡cia na suÃ­te de testes = **100%**.

**MÃ©tricas por classe na suÃ­te de testes:**

| Classe | Casos | TP | FP | FN | PrecisÃ£o | Recall | F1 |
|---|---|---|---|---|---|---|---|
| Positivo | 5 | 5 | 0 | 0 | 1,00 | 1,00 | 1,00 |
| Negativo | 7 | 7 | 0 | 0 | 1,00 | 1,00 | 1,00 |
| Neutro | 3 | 3 | 0 | 0 | 1,00 | 1,00 | 1,00 |
| **Macro** | **15** | â€” | â€” | â€” | **1,00** | **1,00** | **1,00** |

**Nota importante:** a suÃ­te de testes foi projetada para cobrir padrÃµes representativos dos mecanismos implementados â€” nÃ£o Ã© uma amostra aleatÃ³ria do corpus real. O desempenho perfeito nos testes unitÃ¡rios reflete que o analisador funciona corretamente para os casos que ele foi desenhado para tratar; nÃ£o implica desempenho equivalente no corpus completo (ver SeÃ§Ã£o 3.2).

### 3.2 AvaliaÃ§Ã£o em Corpus

A avaliaÃ§Ã£o no corpus completo revela o desempenho real das trÃªs abordagens sobre dados nÃ£o selecionados. Os resultados abaixo foram obtidos com o pipeline completo (`python main.py`):

| Experimento | Modelo | Linhas | Accuracy | Precision macro | Recall macro | F1 macro |
|---|---|---|---|---|---|---|
| B2W principal | RegressÃ£o LogÃ­stica | 132.220 | 0,8117 | 0,7307 | 0,7735 | 0,7423 |
| B2W principal | LinearSVC | 132.220 | 0,8394 | 0,7339 | 0,7363 | 0,7347 |
| B2W principal | SimbÃ³lico | 132.220 | 0,7432 | 0,6366 | 0,5859 | 0,6018 |
| B2W + Olist | RegressÃ£o LogÃ­stica | 175.423 | 0,8086 | 0,7125 | 0,7517 | 0,7234 |
| B2W + Olist | LinearSVC | 175.423 | 0,8380 | 0,7151 | 0,7154 | 0,7145 |
| B2W + Olist | SimbÃ³lico | 175.423 | 0,7079 | 0,6251 | 0,5689 | 0,5803 |
| B2W + Mercado Livre | RegressÃ£o LogÃ­stica | 133.602 | 0,8140 | 0,7330 | 0,7775 | 0,7452 |
| B2W + Mercado Livre | LinearSVC | 133.602 | 0,8418 | 0,7377 | 0,7398 | 0,7385 |
| B2W + Mercado Livre | SimbÃ³lico | 133.602 | 0,7457 | 0,6386 | 0,5879 | 0,6036 |
| B2W + Olist + Mercado Livre | RegressÃ£o LogÃ­stica | 176.805 | 0,8103 | 0,7134 | 0,7525 | 0,7242 |
| B2W + Olist + Mercado Livre | LinearSVC | 176.805 | 0,8385 | 0,7148 | 0,7150 | 0,7141 |
| B2W + Olist + Mercado Livre | SimbÃ³lico | 176.805 | 0,7108 | 0,6248 | 0,5701 | 0,5814 |

**Destaques:**
- Melhor F1-macro geral: **0,7452** â€” RegressÃ£o LogÃ­stica com B2W + Mercado Livre simples
- Melhor acurÃ¡cia geral: **0,8418** â€” LinearSVC com B2W + Mercado Livre simples
- Melhor desempenho do simbÃ³lico: F1-macro = **0,6036**, acurÃ¡cia = **0,7457** (B2W + Mercado Livre)
- Gap simbÃ³lico vs. melhor supervisionado: **~14 pontos percentuais no F1**

As matrizes de confusÃ£o estÃ£o disponÃ­veis em formato PNG para cada experimento (ex.: `cm_b2w_mais_meli_simples_symbolic_rules.png`).

### 3.3 AnÃ¡lise dos Erros

A discrepÃ¢ncia entre a acurÃ¡cia perfeita nos testes unitÃ¡rios e o desempenho de 0,60 F1 no corpus real tem causas identificÃ¡veis:

**1. Cobertura lexical limitada**
O lÃ©xico conta com 24 termos positivos e 18 negativos â€” uma fraÃ§Ã£o Ã­nfima do vocabulÃ¡rio real. Termos como "fantÃ¡stico", "decepcionante", "emperrou", "sumiu" nÃ£o estÃ£o cobertos e resultam em erros de classificaÃ§Ã£o para "neutro" quando deveriam ser positivos ou negativos.

**2. AusÃªncia de variaÃ§Ã£o morfolÃ³gica automÃ¡tica**
Como documentado na SeÃ§Ã£o 2.3.5, o sistema nÃ£o cobre formas morfolÃ³gicas derivadas automaticamente. Uma review como "os produtos chegaram rÃ¡pido" nÃ£o aciona o padrÃ£o `entrega_positiva`, levando a subcontagem de evidÃªncias positivas.

**3. Baixo recall da classe neutra**
O macro recall de 0,5859 indica que o simbÃ³lico frequentemente polariza textos que deveriam ser classificados como neutros. AvaliaÃ§Ãµes factuais curtas (e.g., "Produto ok, dentro do esperado") nÃ£o acionam padrÃµes suficientes para ultrapassar o limiar de Â±0,75, sendo rotuladas como neutras â€” o que Ã© correto â€” mas avaliaÃ§Ãµes neutras com lÃ©xico ambÃ­guo tendem a ser erroneamente polarizadas.

**4. RuÃ­do ortogrÃ¡fico e abreviaÃ§Ãµes**
O corpus real contÃ©m erros como "Ã³timooo", "prfeit", "mto bom", "kkkk", emojis nÃ£o mapeados e gÃ­rias. A normalizaÃ§Ã£o NFKD nÃ£o Ã© suficiente para cobrir essa diversidade.

**5. Ironia e sarcasmo nÃ£o detectados**
O padrÃ£o `ironia_conservadora` cobre apenas um caso especÃ­fico: elogio explÃ­cito seguido de evidÃªncia negativa objetiva. Ironia sutil (e.g., "Que produto incrÃ­vel â€” chegou em pedaÃ§os") nÃ£o Ã© detectada.

**6. Contextos de longa distÃ¢ncia**
Os modificadores de contexto analisam janelas de atÃ© 70â€“80 caracteres. NegaÃ§Ãµes e modificadores que ocorrem em distÃ¢ncias maiores sÃ£o ignorados.

### 3.4 DiscussÃ£o de Melhorias

Com base na anÃ¡lise dos erros, as melhorias mais impactantes seriam:

| Melhoria | Impacto esperado | Complexidade |
|---|---|---|
| ExpansÃ£o dos lÃ©xicos com SentiLex-PT | Alto â€” cobertura muito maior | Baixa (integraÃ§Ã£o direta) |
| LematizaÃ§Ã£o via spaCy (`pt_core_news_sm`) antes da busca | Alto â€” captura variaÃ§Ãµes morfolÃ³gicas | MÃ©dia (dependÃªncia adicional) |
| NormalizaÃ§Ã£o de erros ortogrÃ¡ficos comuns | MÃ©dio â€” cobre parte do ruÃ­do | MÃ©dia |
| CalibraÃ§Ã£o dos limiares com conjunto anotado | MÃ©dio â€” melhora fronteira neutro/polar | Baixa |
| DetecÃ§Ã£o de negaÃ§Ã£o de longo alcance | MÃ©dio â€” menos falsos positivos | Alta |
| ExpansÃ£o dos emojis mapeados | Baixo-mÃ©dio â€” cobertura de novos emojis | Baixa |

---

## 4. RecomendaÃ§Ã£o de Itens

Para uso acadÃªmico, foi implementado um sistema de recomendaÃ§Ã£o baseline explicÃ¡vel. A coluna `sentiment_label` representa a polaridade derivada da nota, usando a mesma regra dos modelos supervisionados.

**RecomendaÃ§Ã£o personalizada (B2W):** o histÃ³rico positivo de cada usuÃ¡rio determina categorias de interesse; o sistema recomenda produtos bem avaliados da mesma categoria que o usuÃ¡rio ainda nÃ£o avaliou.

**RecomendaÃ§Ã£o global (Mercado Livre simples):** sem identificador de usuÃ¡rio, um ranking por categoria Ã© gerado com base em mÃ©dia de nota, proporÃ§Ã£o de avaliaÃ§Ãµes positivas, sentimento mÃ©dio e volume de reviews.

Exemplos de recomendaÃ§Ãµes personalizadas geradas:

| Tipo | user_id (parcial) | Rank | Produto | Categoria | Score |
|---|---|---|---|---|---|
| Personalizada | d0fb1câ€¦ | 1 | Notebook 2 em 1 Dell Inspiron I13-5378 | InformÃ¡tica | 0,9634 |
| Personalizada | d0fb1câ€¦ | 2 | MacBook Air MQD32BZ/A | InformÃ¡tica | 0,9622 |
| Personalizada | 014d6dâ€¦ | 1 | Copo de Vidro Americano 450ml â€” Nadir | Utilidades DomÃ©sticas | 0,9675 |

---

## 5. LimitaÃ§Ãµes

- **RuÃ­do textual:** erros ortogrÃ¡ficos, abreviaÃ§Ãµes e variaÃ§Ãµes lexicais afetam todos os modelos, mas o simbÃ³lico Ã© mais sensÃ­vel por depender de correspondÃªncia exata.
- **Nota como proxy de sentimento:** a nota numÃ©rica Ã© uma aproximaÃ§Ã£o do sentimento textual. Um usuÃ¡rio pode dar nota 4 com crÃ­ticas no texto â€” o rÃ³tulo seria "positivo", mas o conteÃºdo, ambÃ­guo.
- **Desbalanceamento de classes:** a classe neutro representa ~12% do corpus; isso penaliza seu recall em todos os modelos.
- **SimbÃ³lico sem variaÃ§Ã£o morfolÃ³gica:** documentado na SeÃ§Ã£o 2.3.5.
- **RecomendaÃ§Ã£o limitada:** Olist nÃ£o entra na recomendaÃ§Ã£o personalizada por ausÃªncia de metadados de produto e usuÃ¡rio; Mercado Livre entra apenas como ranking global.

---

## 6. ConsideraÃ§Ãµes Finais

Esta entrega apresenta a avaliaÃ§Ã£o quantitativa completa do pipeline, atendendo ao pedido de formalizaÃ§Ã£o levantado na revisÃ£o anterior.

**Resultados numÃ©ricos principais:**

| Abordagem | Melhor F1-macro | Melhor AcurÃ¡cia | ConfiguraÃ§Ã£o |
|---|---|---|---|
| RegressÃ£o LogÃ­stica | 0,7452 | 0,8140 | B2W + Mercado Livre |
| LinearSVC | 0,7385 | 0,8418 | B2W + Mercado Livre |
| SimbÃ³lico | 0,6036 | 0,7457 | B2W + Mercado Livre |

O mÃ©todo supervisionado supera o simbÃ³lico em todas as configuraÃ§Ãµes. A diferenÃ§a no F1-macro chega a **~14 pontos percentuais** no melhor cenÃ¡rio comparÃ¡vel (B2W + Mercado Livre: 0,7452 vs. 0,6036). Em termos de acurÃ¡cia, o simbÃ³lico alcanÃ§a 0,7457 â€” um resultado razoÃ¡vel para um sistema sem treinamento, mas ainda distante dos 0,8418 do LinearSVC.

O que os nÃºmeros revelam: a abordagem supervisionada generaliza melhor porque aprende representaÃ§Ãµes estatÃ­sticas do vocabulÃ¡rio real do corpus; o analisador simbÃ³lico, por sua vez, Ã© limitado pelo lÃ©xico curado e pelos padrÃµes explÃ­citos. A vantagem do simbÃ³lico reside em sua explicabilidade total â€” cada decisÃ£o pode ser rastreada a uma regra especÃ­fica â€” e na ausÃªncia de necessidade de dados rotulados para treinamento.

A adiÃ§Ã£o do Mercado Livre simples ao conjunto de treinamento melhora todos os modelos supervisionados (F1 cresce de 0,7423 para 0,7452 na RegressÃ£o LogÃ­stica), sugerindo que a diversificaÃ§Ã£o de domÃ­nios beneficia a generalizaÃ§Ã£o.

O pipeline Ã© reprodutÃ­vel: a execuÃ§Ã£o de `python main.py` regenera todas as mÃ©tricas apresentadas neste relatÃ³rio. A prÃ³xima etapa natural seria a expansÃ£o lexical com recursos existentes (SentiLex-PT) e a incorporaÃ§Ã£o de lematizaÃ§Ã£o para reduzir o gap de cobertura do analisador simbÃ³lico.

---

## ReferÃªncias

OLIST. *Brazilian E-Commerce Public Dataset by Olist*. Kaggle, 2018. DisponÃ­vel em: https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce. Acesso em: 19 abr. 2026.

REAL, L.; OSHIRO, M.; MAFRA, A. B2W-Reviews01: an open product reviews corpus. 2019. DisponÃ­vel em: https://github.com/b2wdigital/b2w-reviews01. Acesso em: 19 abr. 2026.

MERCADO LIVRE. *DocumentaÃ§Ã£o da API de opiniÃµes sobre um produto*. DisponÃ­vel em: https://developers.mercadolivre.com.br/pt_br/opinioes-sobre-um-produto. Acesso em: 19 abr. 2026.
