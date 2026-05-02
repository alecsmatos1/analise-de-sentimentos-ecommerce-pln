# Relatorio da Solucao Simbolica para Analise de Sentimentos

## 1. Introducao

Este projeto trata da tarefa de analise de sentimentos em avaliacoes de produtos de e-commerce. O problema consiste em receber como entrada o texto de uma avaliacao escrita por um usuario e produzir como saida uma classe de polaridade: `positivo`, `negativo` ou `neutro`.

Exemplos de entrada e saida:

| Entrada | Saida esperada |
| --- | --- |
| `Produto excelente, chegou rapido.` | `positivo` |
| `Nao recebi o produto.` | `negativo` |
| `Produto entregue ontem.` | `neutro` |

Essa tarefa e relevante porque avaliacoes de e-commerce registram a percepcao real dos consumidores sobre produtos, entrega, atendimento, embalagem e custo-beneficio. Um sistema capaz de detectar sentimentos pode resumir grandes volumes de reviews, apoiar a comparacao entre produtos e alimentar recomendacoes de itens semelhantes aos que o usuario avaliou positivamente.

Sistemas analogos aparecem em lojas virtuais, plataformas de marketplace e paineis de reputacao, onde avaliacoes textuais sao usadas para resumir satisfacao, identificar problemas recorrentes e destacar produtos bem avaliados. No contexto deste projeto, o sentimento tambem e usado como base para uma recomendacao simples de itens.

Os principais desafios sao:

- variacao linguistica em portugues brasileiro, incluindo girias, abreviacoes e erros ortograficos;
- frases com polaridade mista, como `Produto bonito, mas nao funciona`;
- negacoes e inversores de polaridade, como `nao`, `sem`, `parou de` e `deixou de`;
- diferenca entre fato neutro e opiniao, como `produto entregue ontem` versus `entrega rapida`;
- uso de ironia, intensificadores, caixa-alta e pontuacao expressiva;
- limitacao de usar a nota numerica da avaliacao como aproximacao do sentimento textual.

Embora o projeto tambem treine dois modelos supervisionados com TF-IDF, Regressao Logistica e Linear SVC, o foco deste relatorio e a solucao simbolica. A estrategia simbolica e comparada aos modelos supervisionados como baseline interpretavel, baseado em regras explicitas.

## 2. Proposta detalhada da solucao simbolica

### 2.1 Arquitetura da solucao

A solucao simbolica esta implementada principalmente na funcao `analyze_symbolic_sentiment`, em `sentiment_analyzer.py`. Ela recebe o texto bruto de uma avaliacao e devolve um dicionario com a classe final, pontuacoes e regras acionadas.

Fluxo geral:

```text
Review bruto
  -> normalizacao simbolica
  -> deteccao de padroes fortes
  -> aplicacao de lexicos positivos e negativos
  -> ajuste por contexto linguistico
  -> tratamento de contraste, concessao e enfase
  -> agregacao de scores
  -> rotulo final e explicacao das regras acionadas
```

Saida da funcao simbolica:

```text
{
  "label": "positivo|negativo|neutro",
  "score": diferenca entre score positivo e negativo,
  "positive_score": soma das evidencias positivas,
  "negative_score": soma das evidencias negativas,
  "confidence": confianca relativa da decisao,
  "neutral_evidence": quantidade de evidencias neutras,
  "rule_hits": lista de regras acionadas
}
```

### 2.2 Processos da arquitetura

#### Normalizacao simbolica

O texto bruto e convertido para minusculas, tem acentos removidos e espacos normalizados. Essa normalizacao permite que frases como `NAO recomendo`, `Nao recomendo` e `nao recomendo` sejam tratadas de forma equivalente.

Esse processamento e separado da limpeza usada pelo TF-IDF. O analisador simbolico usa o texto bruto normalizado para preservar pistas como pontuacao expressiva, caixa-alta e emojis.

#### Padroes fortes de dominio

Algumas expressoes sao tratadas como evidencias fortes de sentimento, porque em reviews de e-commerce costumam ter polaridade clara.

Exemplos:

| Padrao | Polaridade | Justificativa |
| --- | --- | --- |
| `nao recebi`, `produto nao chegou` | negativa | indica falha grave de entrega |
| `nao funciona`, `veio com defeito` | negativa | indica defeito ou falha de funcionamento |
| `quero devolucao`, `pedi reembolso` | negativa | indica acao pos-compra negativa |
| `chegou rapido`, `chegou antes do prazo` | positiva | indica entrega satisfatoria |
| `super recomendo`, `podem comprar` | positiva | indica recomendacao direta |

Esses padroes recebem pesos maiores que palavras isoladas, pois representam eventos ou atos de fala mais informativos.

#### Lexicos positivos e negativos

A solucao usa dois lexicos internos de dominio, definidos no proprio codigo. Eles contem termos comuns em avaliacoes de e-commerce.

Exemplos de termos positivos:

- `excelente`
- `otimo`
- `recomendo`
- `perfeito`
- `top`
- `vale a pena`

Exemplos de termos negativos:

- `defeito`
- `quebrado`
- `pessimo`
- `ruim`
- `travou`
- `demorou`

Cada termo possui um peso. Termos mais fortes, como `excelente`, `pessimo` ou `quebrado`, impactam mais o score final.

#### Polarity shifters

O analisador verifica se ha inversores de polaridade antes de um termo avaliatorio. Isso permite capturar casos em que uma palavra positiva aparece dentro de uma expressao negativa.

Exemplos:

| Frase | Efeito |
| --- | --- |
| `nao funciona` | `funciona` deixa de ser positivo e vira evidencia negativa |
| `sem qualidade` | `qualidade` deixa de indicar avaliacao positiva |
| `parou de funcionar` | indica mudanca para estado negativo |

Essa regra e importante porque a simples contagem de palavras positivas e negativas tende a errar frases com negacao.

#### Intensificadores, atenuadores e hedges

O score das evidencias pode ser ajustado por marcadores de intensidade ou incerteza.

Exemplos:

| Marcador | Exemplo | Efeito |
| --- | --- | --- |
| intensificador | `muito bom`, `super recomendo` | aumenta o peso |
| atenuador | `meio ruim`, `um pouco lento` | reduz o peso |
| hedge | `talvez seja bom`, `acho que vale` | reduz a confianca |
| condicional | `seria otimo se...` | reduz a forca da avaliacao |

O objetivo nao e fazer uma analise sintatica completa, mas aproximar efeitos linguisticos comuns com regras simples.

#### Contraste e concessao

O analisador trata conectores discursivos que mudam a interpretacao global da frase.

Em frases com `mas` ou `porem`, o trecho final recebe mais importancia, pois costuma trazer a conclusao principal do avaliador.

Exemplo:

```text
O produto e bonito, mas nao funciona.
```

Apesar de `bonito` ser positivo, o trecho `nao funciona` depois de `mas` e mais relevante. A classificacao esperada e `negativo`.

Em frases concessivas, como `embora` ou `apesar de`, a conclusao tambem tende a aparecer no trecho principal.

Exemplo:

```text
Embora tenha demorado, o produto e otimo.
```

Mesmo havendo uma evidencia negativa em `demorado`, a conclusao `produto e otimo` conduz a classificacao para `positivo`.

#### Enfase, emojis e ironia conservadora

A solucao considera alguns sinais expressivos:

- palavras em caixa-alta;
- alongamento de letras;
- repeticao de exclamacoes;
- emojis positivos ou negativos.

Tambem ha uma regra conservadora de ironia. Ela so dispara quando uma palavra positiva aparece proxima de uma evidencia negativa clara.

Exemplo:

```text
Produto excelente! Chegou quebrado e nao funciona.
```

Nesse caso, `excelente` isoladamente seria positivo, mas a presenca de `quebrado` e `nao funciona` indica uso ironico ou contraditorio. O sistema classifica como `negativo`.

#### Agregacao e decisao final

Cada regra acionada adiciona peso positivo, negativo ou evidencia neutra. Ao final:

```text
score = positive_score - negative_score
```

Se o score for maior que a margem configurada, a classe final e `positivo`. Se for menor que a margem negativa, a classe e `negativo`. Valores proximos de zero sao classificados como `neutro`.

A lista `rule_hits` permite explicar a decisao, registrando quais regras foram acionadas, qual trecho foi encontrado, sua polaridade e seu peso.

### 2.3 Recursos e ferramentas utilizados

Recursos linguisticos e simbolicos:

- lexicos internos de polaridade positiva e negativa;
- padroes regulares para eventos fortes de e-commerce;
- regras para negacao e inversao de polaridade;
- regras para contraste, concessao, intensificacao, atenuacao e incerteza;
- conjunto pequeno de emojis com polaridade.

Ferramentas e bibliotecas:

- Python;
- `re`, para expressoes regulares;
- `unicodedata`, para normalizacao de acentos;
- `pandas`, para manipulacao de dados;
- `scikit-learn`, para os modelos supervisionados comparativos;
- `matplotlib`, para matrizes de confusao.

Nao foram usadas dependencias externas de parsing sintatico, WordNet, spaCy ou LIWC. Isso mantem a solucao simples, reproduzivel e adequada ao escopo academico do projeto.

### 2.4 Ilustracoes de execucao

#### Exemplo 1: avaliacao positiva simples

Entrada:

```text
Produto excelente, chegou rapido.
```

Processamento simbolico:

| Etapa | Resultado |
| --- | --- |
| normalizacao | `produto excelente, chegou rapido.` |
| lexico positivo | encontra `excelente` |
| padrao de entrega positiva | encontra `chegou rapido` |
| agregacao | score positivo maior que negativo |

Saida esperada:

```text
label = positivo
```

#### Exemplo 2: contraste com defeito

Entrada:

```text
O produto e bonito, mas nao funciona.
```

Processamento simbolico:

| Etapa | Resultado |
| --- | --- |
| lexico positivo | encontra `bonito` |
| padrao forte negativo | encontra `nao funciona` |
| contraste | trecho apos `mas` recebe peso negativo adicional |
| agregacao | score negativo domina |

Saida esperada:

```text
label = negativo
```

#### Exemplo 3: frase factual

Entrada:

```text
Produto entregue ontem.
```

Processamento simbolico:

| Etapa | Resultado |
| --- | --- |
| marcador factual | encontra `produto` |
| polaridade | nao encontra evidencia positiva ou negativa forte |
| agregacao | score proximo de zero |

Saida esperada:

```text
label = neutro
```

#### Exemplo 4: recomendacao negativa direta

Entrada:

```text
Nao recomendo, dinheiro jogado fora.
```

Processamento simbolico:

| Etapa | Resultado |
| --- | --- |
| padrao forte negativo | encontra `nao recomendo` |
| valor negativo | encontra `dinheiro jogado fora` |
| agregacao | score negativo alto |

Saida esperada:

```text
label = negativo
```

## 3. Instrucao para execucao do codigo-fonte

O guia completo de execucao esta em [`docs/como_executar.md`](como_executar.md). O resumo operacional e apresentado abaixo.

### 3.1 Preparar ambiente

```powershell
git clone <URL_DO_REPOSITORIO>
cd "SCC0633-SCC5908 - PLN"

python -m venv .venv
.\.venv\Scripts\Activate.ps1

pip install -r requirements.txt
```

### 3.2 Preparar dados

Se a pasta `data/` foi enviada por drive, copie-a para a raiz do projeto.

Estrutura esperada:

```text
data/
  B2W-Reviews01.csv
  olist_order_reviews_dataset.csv
  mercadolivre_reviews_simple.csv
```

`B2W-Reviews01.csv` e `olist_order_reviews_dataset.csv` podem ser baixados automaticamente pelo pipeline caso nao estejam presentes. A base `mercadolivre_reviews_simple.csv` depende do arquivo local.

### 3.3 Validar analisador simbolico

```powershell
python test_symbolic_analyzer.py
```

Saida esperada:

```text
OK - 15 testes do analisador simbolico passaram
```

### 3.4 Executar pipeline completo

```powershell
python main.py
```

A execucao gera:

- `metricas.csv`, com resultados dos modelos supervisionados e simbolico;
- `recomendacoes.csv`, com recomendacoes baseline;
- matrizes de confusao `cm_*.png`;
- atualizacao de `docs/relatorio_final.md`.

## 4. Consideracoes finais

A solucao simbolica proposta e interpretavel e adequada para demonstrar fenomenos linguisticos em reviews de e-commerce, como negacao, contraste, intensificacao e expressoes de dominio. Sua principal vantagem e a explicabilidade: cada classificacao pode ser acompanhada pelas regras acionadas.

Por outro lado, a abordagem simbolica depende da cobertura dos lexicos e padroes definidos manualmente. Por isso, ela foi mantida como baseline paralelo aos modelos supervisionados, permitindo comparar uma estrategia baseada em regras explicitas com estrategias estatisticas treinadas a partir dos dados.
