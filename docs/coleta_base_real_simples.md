# Coleta Complementar Simples do Mercado Livre

Este arquivo descreve uma versao simplificada da coleta de reviews do Mercado Livre, seguindo a mesma ideia geral das outras bases do projeto: obter uma base complementar real, com codigo simples e poucos criterios adicionais.

No projeto atual, esta e a base real complementar ativa integrada ao pipeline principal junto com `B2W` e `Olist`.

## Arquivo de codigo

- `collect_meli_reviews_simple.py`

## Saida

- `data/mercadolivre_reviews_simple.csv`

## Ideia da coleta

- Usar a API oficial do Mercado Livre.
- Buscar produtos por termos gerais de consumo.
- Obter um item por produto encontrado.
- Coletar reviews do item.
- Reaproveitar coletas anteriores para reduzir custo.
- Manter apenas reviews com texto e nota valida.
- Salvar a base em CSV.

## Campos do CSV

- `source`
- `site`
- `item_id`
- `item_title`
- `category_id`
- `rating`
- `review_title`
- `review_text`
- `collection_date`

## Regras desta versao simples

- Nao aplicar os mesmos filtros mais rigidos da versao principal de coleta complementar.
- Nao exigir distribuicao mista de notas por item.
- Nao exigir minimo de reviews negativas ou positivas por produto.
- Manter apenas:
  - texto nao vazio
  - nota entre 1 e 5
  - remocao de duplicatas por id de review
- Reaproveitar:
  - `data/mercadolivre_reviews.csv`
  - `data/mercadolivre_reviews_simple.csv`
- Continuar coleta ate tentar atingir `5.000` reviews consolidadas.

## Como refazer

1. Definir `MELI_ACCESS_TOKEN`.
2. Executar:

```powershell
python collect_meli_reviews_simple.py
```

3. Ler o CSV gerado em `data/mercadolivre_reviews_simple.csv`.
4. Executar `python main.py` para incluir essa base nos experimentos do trabalho.

## Observacao

Esta copia foi criada para servir como alternativa mais simples e mais proxima da logica usada nas demais bases do projeto, sem os criterios extras de controle de vies adotados na versao mais elaborada da coleta.
