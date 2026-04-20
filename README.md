# Projeto de PLN

Projeto academico simples de analise de sentimentos em avaliacoes de e-commerce brasileiro.

## Bases usadas

- `B2W-Reviews01` como base principal
- `Olist Brazilian E-Commerce Public Dataset` como base complementar
- `Mercado Livre simples` como base complementar real integrada ao pipeline

## Dependencias

Instale as bibliotecas do arquivo `requirements.txt`.

## Como executar

```powershell
pip install -r requirements.txt
python collect_meli_reviews_simple.py
python main.py
```

## Ordem metodologica

- `B2W` como base principal
- `Olist` como base complementar
- `Mercado Livre simples` como base complementar adicional

## Fluxo

- coletar ou atualizar a base real simples do Mercado Livre
- executar o pipeline principal em `main.py`
- analisar `metricas.csv`, as matrizes de confusao e `docs/relatorio_final.md`

## Saidas

- `metricas.csv`
- imagens `.png` das matrizes de confusao
- `docs/relatorio_final.md`
