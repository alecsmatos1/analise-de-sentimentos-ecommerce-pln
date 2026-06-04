# Resposta aos ComentÃ¡rios do Professor â€” Segunda Entrega

**Equipe:** PLN Rocks
**Disciplina:** Processamento de Linguagem Natural (SCC0633 / SCC5908)

Este documento detalha como cada comentÃ¡rio do professor (mdgvn) foi endereÃ§ado na segunda entrega do projeto.

---

## ComentÃ¡rio 1 â€” AvaliaÃ§Ã£o formal do mÃ©todo

**O que o professor disse:**
> "Mesmo tendo o objetivo de construir um baseline, vocÃªs deveriam ter avaliado formalmente o mÃ©todo, ou seja, ter um conjunto de testes e calcular nÃºmero de acertos e erros; analisar o que errou; discutir melhorias."

**O que foi feito:**

1. **Conjunto de testes documentado:** o arquivo `test_symbolic_analyzer.py` contÃ©m 15 casos de teste curados, cobrindo os principais padrÃµes linguÃ­sticos de reviews de e-commerce (positivos simples, concessÃµes, contrastes, ironia, condicionais, texto neutro factual e texto vazio). Todos os 15 casos passam â€” acurÃ¡cia = 100% na suÃ­te.

2. **Tabela completa de casos de teste:** a SeÃ§Ã£o 3.1 do relatÃ³rio apresenta todos os 15 casos em tabela com entrada, rÃ³tulo esperado, rÃ³tulo obtido e resultado (Acerto/Erro).

3. **MÃ©tricas por classe na suÃ­te:** precisÃ£o, recall e F1 por classe (positivo, negativo, neutro) foram calculados e apresentados na SeÃ§Ã£o 3.1.

4. **AvaliaÃ§Ã£o em corpus real:** a SeÃ§Ã£o 3.2 apresenta as 12 linhas de mÃ©tricas reais obtidas pelo pipeline completo sobre corpus de 132.220 a 176.805 reviews, comparando o simbÃ³lico com RegressÃ£o LogÃ­stica e LinearSVC.

5. **AnÃ¡lise dos erros:** a SeÃ§Ã£o 3.3 identifica 6 causas concretas para o gap de desempenho entre testes unitÃ¡rios (100%) e corpus real (F1=0,60): cobertura lexical limitada, ausÃªncia de variaÃ§Ã£o morfolÃ³gica, baixo recall do neutro, ruÃ­do ortogrÃ¡fico, ironia sutil e distÃ¢ncia de contexto.

6. **DiscussÃ£o de melhorias:** a SeÃ§Ã£o 3.4 apresenta tabela com 6 melhorias possÃ­veis, estimativa de impacto e complexidade de implementaÃ§Ã£o.



---

## ComentÃ¡rio 2 â€” LÃ©xicos nÃ£o detalhados

**O que o professor disse:**
> "Quais lÃ©xicos??"

**O que foi feito:**

Os lÃ©xicos foram extraÃ­dos diretamente do cÃ³digo-fonte (`sentiment_analyzer.py`, linhas 27â€“91) e documentados integralmente no relatÃ³rio:

1. **LÃ©xico positivo:** 24 termos com pesos explÃ­citos, em tabela organizada (SeÃ§Ã£o 2.3.1).
2. **LÃ©xico negativo:** 18 termos com pesos explÃ­citos, em tabela organizada (SeÃ§Ã£o 2.3.2).
3. **PadrÃµes fortes:** 16 regras com nome, polaridade, peso e padrÃ£o regex simplificado, em tabela (SeÃ§Ã£o 2.3.3).
4. **CritÃ©rio de seleÃ§Ã£o:** os termos foram selecionados empiricamente com base na frequÃªncia de uso em contextos de avaliaÃ§Ãµes de e-commerce brasileiro; os pesos foram calibrados manualmente para refletir a intensidade semÃ¢ntica habitual de cada termo no domÃ­nio.
5. **Recursos adicionais:** 9 emojis mapeados (6 positivos, 3 negativos) e 4 tipos de modificadores contextuais (intensificadores, atenuadores, hedges, condicionais) com multiplicadores documentados.


---

## ComentÃ¡rio 3 â€” Flexibilidade dos padrÃµes

**O que o professor disse:**
> "SÃ£o flexÃ­veis? ou seja, tanto 'chegou rÃ¡pido' como 'chegaram rÃ¡pido' sÃ£o detectados como padrÃ£o?"

**O que foi feito:**

A resposta direta Ã©: **parcialmente flexÃ­veis**. A SeÃ§Ã£o 2.3.5 do relatÃ³rio documenta explicitamente:

1. **O que Ã© coberto (com mecanismo explicado):**
   - Insensibilidade a acentos: a normalizaÃ§Ã£o NFKD converte `rÃ¡pido` â†’ `rapido` antes da busca.
   - Insensibilidade a maiÃºsculas: normalizaÃ§Ã£o converte tudo para minÃºsculas.
   - MÃºltiplos espaÃ§os: `phrase_pattern()` usa `\s+` entre palavras.
   - Fronteira de palavra: `\b` evita casamentos em substrings (e.g., `caro` nÃ£o case em `encaro`).
   - Formas verbais distintas listadas explicitamente: `nao funciona` e `nao funcionou` sÃ£o padrÃµes separados no lÃ©xico.

2. **O que NÃƒO Ã© coberto (documentado como limitaÃ§Ã£o):**
   - VariaÃ§Ã£o morfolÃ³gica automÃ¡tica: `"chegaram rapido"` **nÃ£o** Ã© detectado pelo padrÃ£o `entrega_positiva` (que lista apenas `"chegou rapido"`).
   - InserÃ§Ã£o de modificador no meio do padrÃ£o forte: `"chegou muito rapido"` nÃ£o casa com `chegou rapido`.
   - Formas plurais no lÃ©xico: `"os produtos funcionam"` nÃ£o Ã© detectado pelo termo `funciona`.

3. **Tabela de variaÃ§Ãµes cobertas vs. nÃ£o cobertas** com 7 exemplos concretos.

4. **LimitaÃ§Ã£o formal declarada:** expansÃ£o morfolÃ³gica exigiria lematizaÃ§Ã£o (e.g., spaCy com modelo `pt_core_news_sm`) ou uso de lÃ©xico por lema (e.g., SentiLex-PT).


---

## ComentÃ¡rio 4 â€” AusÃªncia de avaliaÃ§Ã£o quantitativa

**O que o professor disse:**
> "NÃ£o fizeram uma avaliaÃ§Ã£o quantitativa?"

**O que foi feito:**

As ConsideraÃ§Ãµes Finais foram reescritas com nÃºmeros reais explÃ­citos, respondendo Ã s trÃªs perguntas implÃ­citas do comentÃ¡rio:

1. **AcurÃ¡cia de cada modelo nos dados de teste:**
   - RegressÃ£o LogÃ­stica (B2W + Mercado Livre): acurÃ¡cia = 0,8140, F1-macro = 0,7452
   - LinearSVC (B2W + Mercado Livre): acurÃ¡cia = **0,8418**, F1-macro = 0,7385
   - SimbÃ³lico (B2W + Mercado Livre): acurÃ¡cia = 0,7457, F1-macro = 0,6036

2. **Qual modelo foi melhor e por quanto:**
   - Melhor F1-macro: RegressÃ£o LogÃ­stica com 0,7452 (diferenÃ§a de ~14pp sobre o simbÃ³lico)
   - Melhor acurÃ¡cia: LinearSVC com 0,8418 (diferenÃ§a de ~10pp sobre o simbÃ³lico)

3. **O que os nÃºmeros revelam:**
   - A abordagem supervisionada supera o simbÃ³lico em todas as 4 configuraÃ§Ãµes de corpus testadas.
   - O simbÃ³lico, apesar do gap, alcanÃ§a acurÃ¡cia de 0,7457 sem qualquer treinamento â€” resultado razoÃ¡vel para um baseline explicÃ¡vel.
   - A adiÃ§Ã£o do Mercado Livre simples ao corpus melhora todos os modelos supervisionados, sugerindo benefÃ­cio da diversificaÃ§Ã£o de domÃ­nio.
