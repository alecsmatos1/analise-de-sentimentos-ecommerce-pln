# Como Executar o Projeto

Este guia descreve como preparar o ambiente e executar o projeto do zero, considerando que a pasta `data/` pode ter sido enviada separadamente por drive.

## Requisitos

- Python 3.10 ou superior.
- Arquivos CSV da pasta `data/`, quando forem usados os dados enviados por drive.

## Preparar o ambiente

```powershell
git clone <URL_DO_REPOSITORIO>
cd "SCC0633-SCC5908 - PLN"

python -m venv .venv
.\.venv\Scripts\Activate.ps1

pip install -r requirements.txt
```

## Preparar os dados

Se voce recebeu a pasta `data/` por drive, copie essa pasta para a raiz do projeto antes de executar.

A estrutura esperada e:

```text
data/
  B2W-Reviews01.csv
  olist_order_reviews_dataset.csv
  mercadolivre_reviews_simple.csv
```

Observacoes:

- `B2W-Reviews01.csv` e `olist_order_reviews_dataset.csv` podem ser baixados automaticamente se nao existirem localmente.
- `mercadolivre_reviews_simple.csv` nao e baixado automaticamente; ele precisa estar dentro de `data/` para ativar os experimentos com Mercado Livre e a recomendacao global dessa base.

## Validar o analisador simbolico

```powershell
python test_symbolic_analyzer.py
```

A saida esperada e:

```text
OK - 15 testes do analisador simbolico passaram
```

## Executar o pipeline completo

```powershell
python main.py
```

A execucao:

- treina os modelos supervisionados;
- avalia o analisador simbolico em paralelo;
- salva `metricas.csv`;
- gera `recomendacoes.csv`;
- gera matrizes de confusao;
- atualiza `docs/relatorio_final.md`.

## Arquivos gerados

Os arquivos `metricas.csv` e `recomendacoes.csv` sao artefatos locais regenerados por `python main.py` e nao fazem parte do Git.

As matrizes de confusao PNG sao mantidas versionaveis porque o relatorio final referencia esses arquivos pelo nome.
