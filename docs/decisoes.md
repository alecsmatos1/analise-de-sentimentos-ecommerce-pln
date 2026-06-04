# Decisoes - SCC0633-SCC5908 PLN

## Modelagem do problema

- Usar notas de avaliacao como aproximacao de sentimento textual.
- Trabalhar com tres classes: negativo, neutro e positivo.
- Manter abordagem supervisionada classica como base comparativa.
- Avaliar regras simbolicas como baseline paralelo e explicavel.

## Bases

- B2W como base principal.
- Olist como base complementar.
- Mercado Livre como coleta complementar simples quando houver CSV local.

## Engenharia

- Manter datasets brutos fora do Git.
- Versionar documentacao e codigo do pipeline.
- Tratar metricas e recomendacoes como artefatos regeneraveis.

---

## Etapa 2 â€” Melhorias solicitadas pela professora

### Lexicos externos

**Decisao:** Integrar OpLexicon v3.0 (PUC-RS) e SentiLex-PT 02 (INESC-ID) como lexicos externos validados na literatura. Manter os lexicos internos manuais como camada de dominio especifico; os externos estendem a cobertura para vocabulario geral.

**Decisao tecnica â€” mapeamento de score (D1):**
Os lexicos externos usam polaridade discreta (+1, 0, -1). O analisador simbolico usa pesos continuos (0.8â€“1.8). Regra de conversao:

- +1 â†’ valor +1.0 no dicionario (positivo com peso base)
- -1 â†’ valor -1.0 no dicionario (negativo com peso base; sinal negativo no valor indica polaridade negativa)
- 0  â†’ entrada ignorada, nao entra no dicionario

O dicionario retornado pelas funcoes `load_oplexicon()` e `load_sentilex()` usa o sinal do float para codificar polaridade: valores positivos indicam termos positivos, valores negativos indicam termos negativos. O chamador usa `abs(valor)` como peso e `sign(valor)` como polaridade. Exemplo: `{"otimo": +1.0, "pessimo": -1.0}`.

No momento do uso em `_external_lexicon_score`, o peso efetivo aplicado e **0.6** (independente do valor do dicionario). Motivacao: uma unica evidencia externa nao deve ser suficiente para cruzar o limiar de classificacao (Â±0.75). Duas ou mais evidencias concordantes (0.6 Ã— 2 = 1.2) cruzam o limiar. Isso e mais conservador que o peso 1.0 do lexico interno, coerente com o status de "cobertura de base" dos lexicos externos. Validacao empirica: sem esse ajuste, "produto entregue ontem" era classificado como negativo porque "entregue" aparece como -1 no OpLexicon em contextos gerais, mas e factual neutro em e-commerce.

Os pesos mais altos (1.3â€“1.8) permanecem exclusivos dos lexicos internos, curados para o dominio. Exemplos: "excelente" (1.6), "quebrado" (1.8), "pessimo" (1.7).

**Motivo da decisao:** Preservar a logica de pesos existente sem recalibrar. Tratar os lexicos externos como cobertura de base, nao evidencia forte. Usar sinal no float e nao dois dicts separados simplifica a interface e permite iteracao uniforme.

**Nao descartar os lexicos internos:** eles cobrem expressoes de e-commerce inexistentes nos lexicos gerais (ex: "vale a pena", "bugou", "travou").

---

### Corpus gold

**Decisao original:** Usar SentiBR (github.com/edilsonacjr/sentibr) como corpus gold externo com rotulos anotados manualmente.

**Decisao revisada (D2-gold):** SentiBR e ReLi estao inacessiveis em 2026-06-04 (repositorio GitHub inativo, servidores PUC-RS fora). Alternativa adotada: **avaliacao cruzada entre plataformas usando Olist como corpus de avaliacao para o modelo treinado somente em B2W**.

Justificativa academica: Olist e uma plataforma independente de B2W (marketplace multi-vendedor vs. varejo direto; empresas distintas, catÃ¡logos distintos, base de clientes distinta). O modelo `b2w_principal` nunca viu dados Olist durante treino. Isso testa generalizacao entre plataformas, uma questao de pesquisa valida em adaptacao de dominio. Os rotulos sao derivados de notas numericas pela mesma metodologia (1-2 negativo, 3 neutro, 4-5 positivo), o que mantem consistencia metodologica.

**Corpus gold:** split de teste do Olist (20% estratificado, random_state=42 â€” mesmo padrao dos demais experimentos). Modelo avaliado: treinado em B2W_principal, sem retreinamento. Todos os tres modelos (LR, LinearSVC, simbolico) sao avaliados no gold.

**Limitacao documentada:** a avaliacao nao elimina completamente a circularidade dos rotulos derivados de notas, mas testa independencia de plataforma. Substituir por um corpus com anotacao humana (SentiBR ou similar) quando disponivel e o proximo passo recomendado.

---

### Lematizacao e radicalizacao

**Decisao tecnica â€” escopo do RSLP (D3-rslp):**
O RSLPStemmer do NLTK (Orengo & Huyck, 2001) substitui simple_stem() **somente no pipeline supervisionado** (funcao clean_text() em data_processing.py). O analisador simbolico **nao usa RSLP**.

**Motivo:** O analisador simbolico opera sobre o texto completo normalizado com padroes regex multi-palavra (ex: "nao recebi", "chegou quebrado", "veio com defeito"). O RSLP transformaria "recebi" em "receb" e "chegou" em "cheg", quebrando todos esses padroes. O simbÃ³lico ja tem normalizacao propria conservadora (normalize_symbolic_text: lowercase + NFKD + espacos) que e suficiente para sua logica de busca.

O nome simple_stem() e mantido para compatibilidade com o resto do codigo; apenas a implementacao interna muda.

**Decisao tecnica â€” spaCy como fallback isolado no simbÃ³lico (D4-spacy):**
O spaCy com modelo pt_core_news_sm entra como etapa isolada **apenas no momento do lookup nos lexicos externos**, nao substitui a normalizacao geral do analisador simbolico.

Fluxo de execucao do analisador com lexicos externos:
1. normalize_symbolic_text() â€” como hoje
2. Padroes fortes e lexico interno â€” como hoje (produzem rule_hits com excerpts)
3. Tokenizar o texto normalizado em palavras individuais (split por espacos)
4. Para cada token: verificar se ja aparece como substring em algum rule_hit["excerpt"] â€” se sim, pular (ja foi contabilizado)
5. Para tokens que nao apareceram em nenhum excerpt: lematizar com spaCy e consultar SentiLex e OpLexicon pelo lema
6. Para hits externos encontrados: aplicar multiplicador de contexto usando a posicao do token no texto normalizado

**Decisao tecnica â€” carga gracosa do spaCy (D5-graceful):**
O spaCy e carregado no inicio do modulo dentro de um try/except. Se o modelo nao estiver instalado, uma variavel `_nlp = None` e definida e o lookup externo e desabilitado silenciosamente. Isso evita que a ausencia do modelo quebre o import do modulo inteiro.

```python
try:
    _nlp = spacy.load("pt_core_news_sm", disable=["parser", "ner"])
except OSError:
    _nlp = None  # fallback externo desabilitado
```

**Decisao tecnica â€” carga gracosa dos lexicos (D6-graceful-lex):**
`load_oplexicon()` e `load_sentilex()` retornam dict vazio (e emitem aviso) se o arquivo nao existir. Isso evita que a ausencia dos arquivos baixados quebre o import de sentiment_analyzer.py.

**Motivo:** O SentiLex e indexado por lema â€” sem lematizacao, "funcionou", "funcionando", "funcionam" nao encontram a entrada "funcionar". O OpLexicon tem formas flexionadas â€” para ele o lookup direto (apos normalizacao NFKD) seria suficiente, mas lematizar por consistencia nao prejudica. O spaCy e carregado uma vez (nao por chamada). A abordagem de fallback preserva todo o comportamento existente.

**Nota sobre o formato do SentiLex-PT02:**
O SentiLex nao e um CSV simples. O formato real e: `forma.lema.POS.flex=POL:valor;ANOT:tipo`. O parser de `load_sentilex()` deve:
1. Separar a linha pelo caractere `=`
2. Extrair o lema do campo composto `forma.lema.POS.flex` (segundo campo apos split por `.`)
3. Extrair o valor de `POL:` do segundo campo
4. Ignorar entradas com `POL:0`
