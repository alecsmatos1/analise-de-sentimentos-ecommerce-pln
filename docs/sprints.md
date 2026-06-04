# Sprints — Etapa 2

Baseado em `etapa2_plano.md` e nas decisões técnicas em `decisoes.md`.
Revisado criticamente em 2026-06-04 — problemas de implementação corrigidos antes do início.

Ordem obrigatória: Sprint 1 → Sprint 2 → Sprint 3 → Sprint 4 → Sprint 5.

---

## Sprint 0 — Preparação do ambiente (pré-requisito de todas)

**Objetivo:** Garantir que o ambiente está pronto antes de qualquer sprint de código.

### Tarefas

- [ ] Adicionar ao `.gitignore` as pastas de dados novos:
  ```
  data/lexicons/
  data/gold_corpus/
  ```
- [ ] Instalar dependências novas (antecipando Sprints 1 e 3):
  ```
  pip install nltk spacy
  python -m spacy download pt_core_news_sm
  ```
- [ ] Verificar que `python test_symbolic_analyzer.py` passa com 15/15 antes de qualquer mudança (baseline de regressão)
- [ ] Criar as pastas vazias localmente:
  ```
  mkdir data\lexicons
  mkdir data\gold_corpus
  ```

### Critério de aceite

- `.gitignore` atualizado
- `spacy.load("pt_core_news_sm")` executa sem erro
- `nltk.download("rslp", quiet=True)` executa sem erro
- `test_symbolic_analyzer.py` passa 15/15

### Dependências

Nenhuma. Deve ser feita antes de qualquer outra sprint.

---

## Sprint 1 — Radicalização com RSLP

**Objetivo:** Substituir `simple_stem()` manual pelo RSLPStemmer do NLTK no pipeline supervisionado.

**Decisão aplicada:** D3-rslp — RSLP só no pipeline supervisionado, não no analisador simbólico.

### Tarefas

- [ ] Adicionar `nltk` ao `requirements.txt`
- [ ] Em `data_processing.py`:
  - Remover a constante `STEM_SUFFIXES` (linhas 111–141)
  - Substituir a função `simple_stem()` (linhas 159–165) por:
    ```python
    import nltk
    nltk.download("rslp", quiet=True)
    from nltk.stem import RSLPStemmer
    _stemmer = RSLPStemmer()

    def simple_stem(token: str) -> str:
        return _stemmer.stem(token)
    ```
  - O `nltk.download` com `quiet=True` verifica se já foi baixado antes de tentar rede — seguro chamar no nível de módulo
- [ ] Rodar `python test_symbolic_analyzer.py` — deve passar 15/15 (simbólico não usa `simple_stem`)
- [ ] Rodar `python main.py` — deve gerar `metricas.csv` com os 4 experimentos sem erros
- [ ] Registrar os novos números gerados (F1-macro pós-RSLP) — o arquivo `docs/relatorio_final.md` é regenerado automaticamente pelo pipeline

### Critério de aceite

- `test_symbolic_analyzer.py` passa 15/15 sem alteração
- `metricas.csv` gerado com 12 linhas (4 experimentos × 3 modelos)
- F1-macro dos modelos supervisionados pode mudar em relação à Etapa 1 (esperado — radicais RSLP diferentes dos sufixos manuais)

### Dependências

Sprint 0 concluída.

---

## Sprint 2 — Download e parsing dos léxicos externos

**Objetivo:** Obter os arquivos do OpLexicon v3.0 e SentiLex-PT 02, inspecionar os formatos reais e criar `lexicon_loader.py`.

**Decisão aplicada:** D1 (mapeamento de score), D6-graceful-lex (retorno de dict vazio se arquivo ausente).

### Aviso sobre formatos

Os formatos reais dos arquivos só podem ser confirmados após download. Formatos esperados com base na literatura:

- **OpLexicon v3.0**: provavelmente CSV com colunas `term, class, polarity, frequency` ou similar. Separador pode ser vírgula ou tabulação.
- **SentiLex-PT02**: formato específico, **não é CSV**. Cada linha tem estrutura `forma.lema.POS.flex=POL:valor;ANOT:tipo`. Requer parser próprio (ver D4-spacy em decisoes.md).

### Tarefas

- [ ] Baixar OpLexicon v3.0:
  - URL: http://www.inf.pucrs.br/linatural/wordpress/recursos-e-ferramentas/oplexicon/
  - Salvar em `data/lexicons/` com o nome original do arquivo
  - **Antes de escrever o loader:** abrir o arquivo e anotar: separador, nomes das colunas, encoding, exemplo de 3 linhas. Registrar no comentário inicial de `lexicon_loader.py`
- [ ] Baixar SentiLex-PT 02:
  - URL: http://l2f.inesc-id.pt/wiki/index.php/SentiLex-PT
  - Salvar em `data/lexicons/` com o nome original do arquivo
  - **Antes de escrever o loader:** inspecionar o formato. Confirmar se corresponde a `forma.lema.POS.flex=POL:valor;ANOT:tipo` ou se é diferente. Anotar no comentário de `lexicon_loader.py`
- [ ] Criar `lexicon_loader.py` na raiz do projeto:
  ```python
  def load_oplexicon(path) -> dict[str, float]:
      """
      Retorna {termo_normalizado_NFKD: valor_com_sinal}.
      Valor positivo = termo positivo (+1.0).
      Valor negativo = termo negativo (-1.0).
      Entradas com polaridade 0 são ignoradas.
      Retorna dict vazio com aviso se arquivo não existir (D6-graceful-lex).
      """

  def load_sentilex(path) -> dict[str, float]:
      """
      Retorna {lema_normalizado_NFKD: valor_com_sinal}.
      Formato de entrada: forma.lema.POS.flex=POL:valor;ANOT:tipo
      Parser extrai: lema (campo [1] do split por '.') e POL (após 'POL:').
      Mesmo contrato de retorno que load_oplexicon.
      Retorna dict vazio com aviso se arquivo não existir (D6-graceful-lex).
      """
  ```
- [ ] Escrever bloco `if __name__ == "__main__"` que imprime tamanho e 5 exemplos positivos e 5 negativos de cada léxico
- [ ] Executar `python lexicon_loader.py` e registrar os números no comentário do arquivo

### Critério de aceite

- `load_oplexicon()` e `load_sentilex()` retornam dicts não-vazios sem exceção
- Valores positivos no dict = termos positivos; valores negativos = termos negativos
- Entradas com POL:0 não aparecem no resultado
- Chaves normalizadas com NFKD (sem acentos, minúsculas) — consistente com o resto do pipeline
- Formato real de cada arquivo documentado em comentário no `lexicon_loader.py`
- Se arquivo ausente: retorna `{}` com `warnings.warn`, não levanta exceção

### Dependências

Sprint 0 concluída. Acesso à internet para download.

---

## Sprint 3 — Integração dos léxicos externos no analisador simbólico

**Objetivo:** Estender `analyze_symbolic_sentiment()` com fallback nos léxicos externos via spaCy.

**Decisões aplicadas:** D1, D4-spacy, D5-graceful.

### Tarefas

- [ ] Adicionar `spacy` ao `requirements.txt`. Adicionar comentário no `requirements.txt`:
  ```
  # após pip install spacy, rodar: python -m spacy download pt_core_news_sm
  ```
- [ ] No topo de `sentiment_analyzer.py`, após os imports existentes:
  ```python
  import warnings
  import spacy
  from lexicon_loader import load_oplexicon, load_sentilex
  from data_processing import DATA_DIR

  try:
      _nlp = spacy.load("pt_core_news_sm", disable=["parser", "ner"])
  except OSError:
      warnings.warn("pt_core_news_sm nao encontrado; fallback de lexicos externos desabilitado.")
      _nlp = None

  _OPLEXICON = load_oplexicon(DATA_DIR / "lexicons" / "<nome_real_do_arquivo>")
  _SENTILEX  = load_sentilex(DATA_DIR  / "lexicons" / "<nome_real_do_arquivo>")
  ```
  - Substituir `<nome_real_do_arquivo>` pelo nome real descoberto na Sprint 2
- [ ] Criar função `_external_lexicon_score(normalized_text, existing_excerpts)` em `sentiment_analyzer.py`:
  - `existing_excerpts`: conjunto de substrings já presentes em `rule_hits` (para evitar dupla contagem)
  - Tokeniza `normalized_text` por espaços
  - Pula tokens cujo texto aparece como substring em algum excerpt existente
  - Se `_nlp` não é None: lematiza os tokens restantes com spaCy
  - Consulta `_SENTILEX` e `_OPLEXICON` pelo lema
  - Para cada hit: aplica `symbolic_context_multiplier()` usando a posição do token no texto
  - Retorna `(positive_contrib, negative_contrib, hits_list)`
  - Se `_nlp is None` ou ambos os dicts são vazios: retorna `(0.0, 0.0, [])`
- [ ] No final de `analyze_symbolic_sentiment()`, antes do cálculo do score final, adicionar:
  ```python
  existing_excerpts = {h["excerpt"] for h in rule_hits}
  ext_pos, ext_neg, ext_hits = _external_lexicon_score(normalized, existing_excerpts)
  positive_score += ext_pos
  negative_score += ext_neg
  rule_hits.extend(ext_hits)
  ```
- [ ] Rodar `python test_symbolic_analyzer.py` — deve passar 15/15 com os mesmos rótulos
- [ ] Rodar `python main.py` — registrar novos resultados

### Critério de aceite

- `test_symbolic_analyzer.py` passa 15/15 sem regressão de rótulo
- Com léxicos externos ativos, `rule_hits` de textos com vocabulário fora do léxico interno mostram entradas com `rule` contendo `"oplexicon"` ou `"sentilex"`
- Com `_nlp = None` (modelo ausente), o comportamento é idêntico ao pré-Sprint 3
- `main.py` conclui sem erros

### Dependências

- Sprint 0 e Sprint 2 concluídas

---

## Sprint 4 — Corpus gold SentiBR

**Objetivo:** Baixar o SentiBR, implementar `evaluate_on_gold()` e expor o modelo treinado para avaliação externa.

**Decisão aplicada:** D2-gold (modelo B2W+MeLi, sem retreinamento; gold só para avaliação).

### Aviso sobre formato do SentiBR

O formato real (colunas, encoding de rótulos) só pode ser confirmado após download. SentiBR pode ser binário (positivo/negativo) ou ternário (+ neutro). Verificar antes de implementar o mapeamento de rótulos.

### Tarefas

**Parte A — obter o corpus e inspecionar**
- [ ] Baixar SentiBR: https://github.com/edilsonacjr/sentibr
  - Salvar CSV em `data/gold_corpus/sentibr.csv`
  - Inspecionar: número de colunas, nome da coluna de texto, nome da coluna de rótulo, valores únicos dos rótulos
  - Se binário (sem neutro): documentar em comentário e mapear apenas positivo/negativo; a avaliação do neutro não será calculada
- [ ] Adicionar em `data_processing.py`:
  ```python
  SENTIBR_PATH = DATA_DIR / "gold_corpus" / "sentibr.csv"

  def load_sentibr(path=SENTIBR_PATH) -> pd.DataFrame:
      """
      Retorna DataFrame com colunas: raw_text, clean_text, label.
      Rótulos mapeados para 'positivo'/'negativo'/'neutro'.
      Aplica clean_text() para uso com modelos supervisionados.
      raw_text preservado para uso com o analisador simbólico.
      """
  ```

**Parte B — expor o modelo treinado**

*Problema identificado na revisão crítica:* `run_experiment()` descarta o objeto Pipeline após treinar. Para avaliar no gold corpus, precisa do pipeline treinado.

- [ ] Modificar `run_experiment()` em `sentiment_analyzer.py` para também retornar os pipelines:
  ```python
  def run_experiment(name: str, df: pd.DataFrame) -> tuple[list[dict], dict[str, Pipeline]]:
      ...
      trained_pipelines = {}
      for model_name in ["logistic_regression", "linear_svc"]:
          pipeline = build_pipeline(model_name)
          pipeline.fit(X_train, y_train)
          trained_pipelines[model_name] = pipeline
          ...
      return rows, trained_pipelines
  ```
- [ ] Em `main.py`, capturar os pipelines do experimento B2W+MeLi:
  ```python
  metric_rows_exp, pipelines_b2w_meli = run_experiment("b2w_mais_meli_simples", df_b2w_meli)
  ```
  - Atenção: a linha `metric_rows.extend(run_experiment(name, df))` precisa ser atualizada para desempacotar a tupla em todos os 4 experimentos

**Parte C — avaliação no gold**
- [ ] Em `report_generator.py`, criar:
  ```python
  def evaluate_on_gold(
      pipelines: dict[str, Pipeline],
      df_gold: pd.DataFrame,
      output_dir: Path,
  ) -> None:
      """
      Avalia modelos supervisionados (via pipelines) e simbólico sobre df_gold.
      df_gold deve ter colunas: raw_text (para simbólico), clean_text (para supervisionados), label.
      Salva tabela em output_dir/gold_evaluation.md.
      """
  ```
- [ ] Em `main.py`, após os experimentos principais, chamar `evaluate_on_gold()` com os pipelines B2W+MeLi e o `df_sentibr`
- [ ] Rodar `python main.py` — verificar geração de `artifacts/gold_evaluation.md`

### Critério de aceite

- `data/gold_corpus/sentibr.csv` presente e carregável
- `artifacts/gold_evaluation.md` gerado com resultados de LR, LinearSVC e simbólico
- Formato do SentiBR documentado em comentário em `load_sentibr()`
- `main.py` conclui sem erros; todos os 4 experimentos originais ainda geram métricas

### Dependências

- Sprint 0 e Sprint 1 concluídas (preprocessing consistente)
- Sprint 3 recomendada (analisador com léxicos ativos), mas pode ser executada em paralelo

---

## Sprint 5 — Relatório LaTeX com resultados reais + guia de execução

**Objetivo:** Preencher os resultados reais na seção "Revisão e Extensões", atualizar `como_executar.md` e compilar o PDF final.

### Tarefas

**Parte A — LaTeX**
- [ ] Na seção "Revisão e Extensões" de `relatorio.tex`, preencher a tabela comparativa com os resultados reais:
  - Coluna "Etapa 1": F1-macro original (LR=0,7452, LinearSVC=0,7385, Simbólico=0,6036)
  - Coluna "pós-RSLP" (Sprint 1): novos valores
  - Coluna "pós-léxicos externos" (Sprint 3): novos valores
- [ ] Adicionar tabela de avaliação no corpus gold SentiBR (Sprint 4) na seção "Revisão e Extensões"
- [ ] Rodar `pdflatex relatorio.tex` duas vezes e confirmar sem erros `!`

**Parte B — `docs/como_executar.md`**
- [ ] Adicionar seção "Etapa 2 — dependências adicionais":
  ```
  pip install nltk spacy
  python -m spacy download pt_core_news_sm
  ```
- [ ] Adicionar instrução para baixar os léxicos externos (URLs) e onde salvar
- [ ] Adicionar instrução para baixar o SentiBR e onde salvar
- [ ] Documentar que, sem os léxicos e sem o SentiBR, o pipeline ainda roda (D5-graceful e D6-graceful-lex), mas sem a avaliação gold e sem fallback externo no simbólico

**Parte C — status**
- [ ] Atualizar `docs/status.md` marcando todas as sprints como concluídas

### Critério de aceite

- `relatorio.pdf` compila sem erros
- Tabela comparativa antes/depois preenchida com números reais
- Tabela de avaliação gold presente
- `docs/como_executar.md` descreve o setup completo da Etapa 2
- `docs/status.md` atualizado

### Dependências

- Todas as sprints (0–4) concluídas

---

## Resumo

| Sprint | Objetivo | Depende de | Bloqueante para |
|---|---|---|---|
| 0 | Ambiente e .gitignore | — | todas |
| 1 | RSLP em data_processing.py | 0 | 4 |
| 2 | Download e parsing dos léxicos | 0 | 3 |
| 3 | Integração léxicos + spaCy no simbólico | 0, 2 | 4 (recomendado) |
| 4 | Corpus gold + expor modelo treinado | 0, 1 | 5 |
| 5 | LaTeX com resultados + como_executar.md | 0, 1, 2, 3, 4 | — |

## Arquivos gerados ao final

| Arquivo | Sprint |
|---|---|
| `.gitignore` atualizado | 0 |
| `requirements.txt` atualizado | 1, 3 |
| `data_processing.py` com RSLP e load_sentibr() | 1, 4 |
| `data/lexicons/<oplexicon>` e `<sentilex>` | 2 |
| `lexicon_loader.py` | 2 |
| `sentiment_analyzer.py` com fallback externo e run_experiment() retornando pipeline | 3, 4 |
| `report_generator.py` com evaluate_on_gold() | 4 |
| `data/gold_corpus/sentibr.csv` | 4 |
| `artifacts/gold_evaluation.md` | 4 |
| `docs/como_executar.md` atualizado | 5 |
| `latex/gen-latex/relatorio.pdf` final | 5 |

## Problemas corrigidos nesta revisão

| ID | Problema | Correção |
|---|---|---|
| C1 | run_experiment() descartava o pipeline treinado | Sprint 4: modificar assinatura para retornar tuple(rows, pipelines) |
| C2 | dict de léxico perdia a polaridade | D1 atualizado: sinal do float codifica polaridade (+1.0 positivo, -1.0 negativo) |
| C3 | SentiLex não é CSV | Sprint 2 e D4-spacy documentam o parser real do formato |
| C4 | "tokens sem match" não tinha suporte na arquitetura | D4-spacy descreve abordagem via existing_excerpts como proxy |
| M1 | Import spaCy falhava se modelo ausente | D5-graceful: try/except no carregamento, _nlp = None como fallback |
| M2 | Arquivos de léxico ausentes quebravam import | D6-graceful-lex: retornar dict vazio com warnings.warn |
| M3 | data/lexicons/ e data/gold_corpus/ fora do .gitignore | Sprint 0 |
| M4 | como_executar.md não atualizado | Sprint 5 Parte B |
| M5 | SentiBR pode ser binário | Sprint 4 Parte A: verificar e documentar antes de implementar |
