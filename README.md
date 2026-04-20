# Analise de Sentimentos em E-commerce com PLN

Trabalho universitario de Processamento de Linguagem Natural voltado para classificacao de sentimentos em avaliacoes de e-commerce brasileiro.

## Objetivo

Comparar modelos de classificacao treinados sobre bases de avaliacoes de produtos, usando textos em portugues e combinando diferentes fontes de dados para observar o impacto no desempenho.

## Bases utilizadas

- `B2W-Reviews01` como base principal
- `Olist Brazilian E-Commerce Public Dataset` como base complementar
- base complementar do Mercado Livre utilizada localmente no experimento

## Estrutura do repositorio

- `main.py`: pipeline principal de preparo, treinamento e avaliacao
- `docs/relatorio_final.md`: relatorio consolidado do trabalho
- `docs/coleta_base_real.md`: documentacao da coleta complementar
- `docs/coleta_base_real_simples.md`: versao simplificada da coleta
- `docs/codigo_comentado.md`: explicacao comentada do codigo principal

## Como executar

```powershell
pip install -r requirements.txt
python main.py
```

## Observacao sobre dados e arquivos locais

Os datasets brutos, arquivos gerados durante a execucao e scripts auxiliares de coleta local nao fazem parte da arvore principal do repositorio publicado no GitHub.

## Resultado esperado

A execucao do pipeline gera metricas e artefatos de avaliacao usados na analise descrita em `docs/relatorio_final.md`.
