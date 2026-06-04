# Status - SCC0633-SCC5908 PLN

## Estado atual (Etapa 2 — planejada)

A Etapa 1 esta concluida. O pipeline esta documentado, o relatorio LaTeX esta compilando e os resultados foram entregues.

A Etapa 2 foi planejada conforme solicitacao da professora. O plano esta em `docs/etapa2_plano.md`.

### O que esta concluido (Etapa 1)

- Pipeline de analise de sentimentos com B2W, Olist e Mercado Livre.
- Tres abordagens comparadas: Regressao Logistica, LinearSVC e regras simbolicas.
- Analisador simbolico com lexicos internos, padroes fortes, modificadores e regras discursivas.
- Recomendacao baseline documentada.
- Relatorio LaTeX no formato SBC compilando sem erros.
- Respostas aos comentarios da professora documentadas em `resposta_comentarios_professor.md`.

### Melhores resultados (Etapa 1)

| Modelo | Corpus | Acuracia | F1-macro |
|---|---|---|---|
| Regressao Logistica | B2W + Mercado Livre | 0,8140 | 0,7452 |
| LinearSVC | B2W + Mercado Livre | 0,8418 | 0,7385 |
| Simbolico | B2W + Mercado Livre | 0,7457 | 0,6036 |

## Etapa 2 — Implementada em 2026-06-04

| Sprint | Objetivo | Status |
|---|---|---|
| 0 | Ambiente, .gitignore, pastas | concluida |
| 1 | RSLP substitui simple_stem() | concluida |
| 2 | Download e parsing OpLexicon + SentiLex | concluida |
| 3 | Lexicos externos + spaCy fallback no simbólico | concluida |
| 4 | Avaliacao gold cross-platform (B2W → Olist) | concluida |
| 5 | LaTeX com resultados reais + como_executar.md | concluida |

### Resultados pós-Etapa 2 (B2W + Mercado Livre simples)

| Modelo | F1 Etapa 1 | F1 pós-RSLP | F1 pós-léxicos |
|---|---|---|---|
| Regressão Logística | 0,7452 | 0,7414 | 0,7414 |
| LinearSVC | 0,7385 | 0,7337 | 0,7337 |
| Simbólico | 0,6036 | 0,6036 | 0,5932 |

### Avaliacao gold cross-platform (B2W → Olist, 43.203 exemplos)

| Modelo | F1 gold | Gap vs. em-domínio |
|---|---|---|
| Regressão Logística | 0,6075 | −0,1341 |
| LinearSVC | 0,6319 | −0,1015 |
| Simbólico | 0,5055 | −0,0879 |

## Bloqueios conhecidos

- A base simples do Mercado Livre depende de arquivo local.
- O SentiLex e lematizado: requer lematizacao ativa para uso correto (passo 1 e pre-requisito do passo 2).
- Os arquivos do OpLexicon e SentiLex precisam ser baixados manualmente dos sites institucionais e colocados em `data/lexicons/`.
- O SentiBR precisa ser baixado do GitHub e colocado em `data/gold_corpus/`.
