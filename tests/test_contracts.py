"""Testes de contrato para a v2.

Estes testes nao executam modelos ou loaders. Eles apenas verificam que a
estrutura de dados acordada em ``v2/arquitetura.md`` e ``v2/avaliacao.md`` esta
sendo respeitada pela fixture demonstrativa e que ondas futuras nao podem
introduzir rotulos ou colunas divergentes sem atualizar tambem o contrato.
"""

from __future__ import annotations

from collections import Counter
from typing import List, Mapping


def test_fixture_nao_esta_vazia(sample_reviews: List[Mapping[str, object]]) -> None:
    assert sample_reviews, "fixture demonstrativa nao pode ser vazia"


def test_todas_as_linhas_tem_colunas_obrigatorias(
    sample_reviews: List[Mapping[str, object]],
    required_columns: frozenset,
) -> None:
    for index, row in enumerate(sample_reviews):
        missing = required_columns - set(row)
        assert not missing, f"linha {index}: colunas ausentes {sorted(missing)}"


def test_labels_estao_no_conjunto_permitido(
    sample_reviews: List[Mapping[str, object]],
    allowed_labels: frozenset,
) -> None:
    for index, row in enumerate(sample_reviews):
        assert row["label"] in allowed_labels, (
            f"linha {index}: label invalida {row['label']!r}; "
            f"esperado um de {sorted(allowed_labels)}"
        )


def test_fixture_cobre_todas_as_classes(
    sample_reviews: List[Mapping[str, object]],
    allowed_labels: frozenset,
) -> None:
    contagem = Counter(row["label"] for row in sample_reviews)
    faltantes = allowed_labels - set(contagem)
    assert not faltantes, (
        f"fixture precisa cobrir todas as classes; faltam {sorted(faltantes)}"
    )
    for classe, total in contagem.items():
        assert total >= 1, f"classe {classe} sem exemplos"


def test_textos_nao_vazios(sample_reviews: List[Mapping[str, object]]) -> None:
    for index, row in enumerate(sample_reviews):
        raw = row["raw_text"]
        clean = row["clean_text"]
        assert isinstance(raw, str) and raw.strip(), (
            f"linha {index}: raw_text precisa ser string nao vazia"
        )
        assert isinstance(clean, str) and clean.strip(), (
            f"linha {index}: clean_text precisa ser string nao vazia"
        )


def test_clean_text_e_lowercase(sample_reviews: List[Mapping[str, object]]) -> None:
    for index, row in enumerate(sample_reviews):
        clean = row["clean_text"]
        assert clean == clean.lower(), (
            f"linha {index}: clean_text {clean!r} deveria estar em caixa baixa"
        )


def test_score_quando_presente_esta_no_intervalo(
    sample_reviews: List[Mapping[str, object]],
) -> None:
    for index, row in enumerate(sample_reviews):
        if "score" not in row:
            continue
        score = row["score"]
        assert isinstance(score, int), f"linha {index}: score precisa ser inteiro"
        assert 1 <= score <= 5, f"linha {index}: score {score} fora do intervalo [1,5]"


def test_score_consistente_com_label(
    sample_reviews: List[Mapping[str, object]],
    score_to_label: Mapping[int, str],
) -> None:
    for index, row in enumerate(sample_reviews):
        if "score" not in row:
            continue
        esperado = score_to_label[row["score"]]
        assert row["label"] == esperado, (
            f"linha {index}: score {row['score']} mapeia para {esperado!r}, "
            f"mas a fixture diz {row['label']!r}"
        )


def test_source_nao_revela_dados_comerciais_reais(
    sample_reviews: List[Mapping[str, object]],
) -> None:
    fontes_proibidas = {"b2w", "olist", "mercadolivre", "mercado_livre"}
    for index, row in enumerate(sample_reviews):
        source = str(row["source"]).strip().lower()
        assert source, f"linha {index}: source nao pode ser vazio"
        assert source not in fontes_proibidas, (
            f"linha {index}: fixture demonstrativa nao deve usar fonte real {source!r}"
        )


def test_smoke_contrato_completo(
    sample_reviews: List[Mapping[str, object]],
    required_columns: frozenset,
    allowed_labels: frozenset,
) -> None:
    """Smoke test agregador.

    Roda sem dados externos e garante que qualquer alteracao futura na fixture
    continue compativel com o contrato minimo da v2.
    """

    assert len(sample_reviews) >= 3
    for row in sample_reviews:
        assert required_columns.issubset(row)
        assert row["label"] in allowed_labels
