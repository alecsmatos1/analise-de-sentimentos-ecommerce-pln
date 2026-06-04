# Plano de Melhorias — Pipeline de Análise de Sentimentos

## Contexto

O projeto atual usa léxicos internos manuais, radicalização simples por sufixos e não tem corpus gold para avaliação. A professora solicitou três melhorias para deixar o trabalho mais fundamentado na literatura.

---

## Melhoria 1 — Léxicos Externos da Literatura

### O que adicionar

Integrar dois léxicos validados academicamente para português:

- **OpLexicon v3.0** — léxico de opinião em português, disponível em http://www.inf.pucrs.br/linatural/wordpress/recursos-e-ferramentas/oplexicon/
- **SentiLex-PT 02** — léxico de sentimentos para português europeu e brasileiro, disponível em http://l2f.inesc-id.pt/wiki/index.php/SentiLex-PT

Ambos têm polaridade (positivo/negativo/neutro) e estão em formatos tabulares (CSV ou similar).

### Como integrar ao código atual

1. Criar uma função `load_oplexicon(path)` que lê o arquivo do OpLexicon e retorna um `dict` `{lemma: score}` onde score é numérico (ex: +1, -1, 0).
2. Criar uma função `load_sentilex(path)` análoga para o SentiLex.
3. No analisador simbólico atual, **substituir ou estender** os dicionários `positive_lexicon` e `negative_lexicon` com os termos carregados.
4. Manter os léxicos internos manuais atuais como fallback ou combiná-los via união dos dicionários.
5. Ajustar os pesos: usar o score numérico do léxico externo quando disponível; cair no peso manual quando o termo só existe no léxico interno.

### Ponto de atenção

O SentiLex é lematizado. Os termos precisam ser lematizados antes da busca (ver Melhoria 3). O OpLexicon tem entradas flexionadas, o que facilita o uso com a radicalização existente.

---

## Melhoria 2 — Corpus Gold para Avaliação

### O que é e por que importa

Corpus gold é um conjunto de avaliações com rótulos manuais confiáveis, usado para medir o desempenho do sistema de forma independente dos dados de treino. Sem ele, os resultados de acurácia ficam circulares (o modelo é testado sobre dados com rótulos derivados automaticamente da nota numérica).

### Corpus escolhido

- **SentiBR** — corpus de tweets em português com anotação manual de sentimento (positivo/negativo/neutro). Disponível em https://github.com/edilsonacjr/sentibr (~1.200 exemplos anotados).

Alternativa: **ReLi** — resenhas literárias em português, disponível em https://www.inf.pucrs.br/linatural/wordpress/recursos-e-ferramentas/reli/

### Como integrar ao código atual

1. Baixar o SentiBR e carregar em um DataFrame com colunas `text` e `label`.
2. Criar função `evaluate_on_gold(model, df_gold)` separada da avaliação atual.
3. Aplicar pipeline completo (pré-processamento + predição) sobre `df_gold` e calcular:
   - Acurácia
   - F1-macro
   - Relatório por classe (precision, recall, F1)
4. Reportar os resultados separadamente dos resultados nos corpora B2W/Olist/Mercado Livre.
5. Usar o corpus gold **somente para avaliação**, nunca para treino.

---

## Melhoria 3 — Lematização e Radicalização nos Textos

### Situação atual

O projeto usa `simple_stem()` manual com remoção de sufixos (linhas 111-165 de `data_processing.py`). Isso é frágil: "chegaram" e "chegou" não são reconhecidos como o mesmo radical.

### O que implementar

#### Opção A — RSLP via NLTK (mínimo esforço, pré-processamento supervisionado)

```python
import nltk
nltk.download('rslp')
from nltk.stem import RSLPStemmer
stemmer = RSLPStemmer()

def simple_stem(token: str) -> str:
    return stemmer.stem(token)
```

Substituir a função `simple_stem()` existente. Aplicar no pré-processamento dos modelos supervisionados.

#### Opção B — Lematização com spaCy (recomendada para o analisador simbólico)

```bash
pip install spacy
python -m spacy download pt_core_news_sm
```

```python
import spacy
nlp = spacy.load("pt_core_news_sm")

def lemmatize_text(text):
    doc = nlp(text)
    return " ".join(token.lemma_ for token in doc if not token.is_space)
```

**Aplicar onde**:
- No pré-processamento dos **modelos supervisionados**: substituir `simple_stem()` por `lemmatize_text()` antes do TF-IDF.
- No **analisador simbólico**: lematizar o texto de entrada antes de buscar nos léxicos externos (especialmente o SentiLex que é lematizado).

---

## Ordem de Implementação

1. **Lematização primeiro** (Melhoria 3, RSLP) — pré-requisito para usar o SentiLex corretamente
2. **Léxicos externos** (Melhoria 1) — integrar OpLexicon e SentiLex ao analisador simbólico
3. **Corpus gold** (Melhoria 2) — criar pipeline de avaliação independente com SentiBR

---

## Dependências Novas

| Recurso | Tipo | Como obter |
|---|---|---|
| OpLexicon v3.0 | Léxico externo | http://www.inf.pucrs.br/linatural/wordpress/recursos-e-ferramentas/oplexicon/ |
| SentiLex-PT 02 | Léxico externo | http://l2f.inesc-id.pt/wiki/index.php/SentiLex-PT |
| SentiBR | Corpus gold | https://github.com/edilsonacjr/sentibr |
| NLTK + RSLP | Stemmer | `pip install nltk` |
| spaCy + pt_core_news_sm | Lematizador | `pip install spacy` + `python -m spacy download pt_core_news_sm` |

---

## O Que Não Mudar

- A lógica de negação, intensificadores, atenuadores e padrões simbólicos existentes
- A derivação de rótulos a partir da nota numérica (1-2 negativo, 3 neutro, 4-5 positivo)
- Os corpora B2W, Olist e Mercado Livre como dados de treino/teste principal
- A arquitetura TF-IDF + Regressão Logística + LinearSVC para os modelos supervisionados

---

## Nova Seção no Relatório LaTeX

Adicionar seção **"Revisão e Extensões"** em `latex/gen-latex/relatorio.tex` cobrindo:
- Resumo da Etapa 1 (o que foi feito e resultados)
- Escolhas empíricas identificadas
- Léxicos externos integrados (OpLexicon, SentiLex) com referências
- Lematização com RSLP — referência: Orengo & Huyck (2001)
- Corpus gold SentiBR — avaliação independente
- Tabela comparativa: resultados antes vs depois das melhorias
