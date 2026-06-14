"""Testes do experimento TF-IDF + modelo linear da v2."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from v2.src.experiments.run_tfidf_linear import (  # noqa: E402
    DEFAULT_LABEL_ORDER,
    TfidfLinearResult,
    build_tfidf_linear_pipeline,
    run_tfidf_linear,
)
from v2.src.models.linear import LinearModelConfig, build_linear_model  # noqa: E402
from v2.src.representations.tfidf import TfidfConfig, build_tfidf  # noqa: E402


@pytest.fixture
def synthetic_split() -> tuple[list[str], list[str], list[str], list[str]]:
    positive = [
        "produto otimo recomendo demais",
        "adorei a entrega rapida",
        "excelente qualidade gostei muito",
        "perfeito vale a pena comprar",
        "amei o produto chegou cedo",
        "muito bom durou bastante tempo",
    ]
    negative = [
        "produto pessimo nao recomendo",
        "horrivel quebrou na primeira semana",
        "atrasou demais e nao chegou",
        "decepcionado com a qualidade ruim",
        "veio com defeito horrivel",
        "nao funciona detesto totalmente",
    ]
    neutral = [
        "produto chegou conforme descrito",
        "atende ao que promete apenas",
        "embalagem padrao sem novidade",
        "entrega no prazo combinado",
        "produto comum sem destaque",
        "comprei e usei normalmente",
    ]

    # 4 exemplos por classe no treino, 2 no teste -> total 18
    X_train = positive[:4] + negative[:4] + neutral[:4]
    y_train = (
        ["positivo"] * 4 + ["negativo"] * 4 + ["neutro"] * 4
    )
    X_test = positive[4:] + negative[4:] + neutral[4:]
    y_test = (
        ["positivo"] * 2 + ["negativo"] * 2 + ["neutro"] * 2
    )
    return X_train, y_train, X_test, y_test


def test_build_tfidf_uses_default_config() -> None:
    vec = build_tfidf()
    assert vec.ngram_range == (1, 2)
    assert vec.sublinear_tf is True
    assert vec.lowercase is True


def test_build_tfidf_honors_custom_config() -> None:
    vec = build_tfidf(TfidfConfig(ngram_range=(1, 1), min_df=1, sublinear_tf=False))
    assert vec.ngram_range == (1, 1)
    assert vec.min_df == 1
    assert vec.sublinear_tf is False


def test_build_linear_model_defaults_to_logistic_regression() -> None:
    model = build_linear_model()
    assert isinstance(model, LogisticRegression)
    assert model.random_state == 42
    assert model.class_weight == "balanced"


def test_build_linear_model_supports_linear_svc() -> None:
    model = build_linear_model(LinearModelConfig(name="linear_svc"))
    assert isinstance(model, LinearSVC)
    assert model.random_state == 42


def test_build_linear_model_rejects_unknown_name() -> None:
    with pytest.raises(ValueError):
        build_linear_model(LinearModelConfig(name="random_forest"))  # type: ignore[arg-type]


def test_build_tfidf_linear_pipeline_has_two_steps() -> None:
    pipeline = build_tfidf_linear_pipeline(
        tfidf_config=TfidfConfig(min_df=1, max_df=1.0),
    )
    assert isinstance(pipeline, Pipeline)
    assert [name for name, _ in pipeline.steps] == ["tfidf", "model"]


def test_run_tfidf_linear_returns_result_contract(
    synthetic_split: tuple[list[str], list[str], list[str], list[str]],
) -> None:
    X_train, y_train, X_test, y_test = synthetic_split
    result = run_tfidf_linear(
        X_train,
        y_train,
        X_test,
        y_test,
        tfidf_config=TfidfConfig(min_df=1, max_df=1.0, ngram_range=(1, 1)),
    )

    assert isinstance(result, TfidfLinearResult)
    assert result.label_order == DEFAULT_LABEL_ORDER
    assert result.model_name == "logistic_regression"
    assert 0.0 <= result.accuracy <= 1.0
    assert 0.0 <= result.f1_macro <= 1.0
    assert len(result.confusion_matrix) == len(DEFAULT_LABEL_ORDER)
    assert all(len(row) == len(DEFAULT_LABEL_ORDER) for row in result.confusion_matrix)
    total = sum(sum(row) for row in result.confusion_matrix)
    assert total == len(y_test)
    assert isinstance(result.pipeline, Pipeline)


def test_run_tfidf_linear_separates_classes_on_synthetic_data(
    synthetic_split: tuple[list[str], list[str], list[str], list[str]],
) -> None:
    X_train, y_train, X_test, y_test = synthetic_split
    result = run_tfidf_linear(
        X_train,
        y_train,
        X_test,
        y_test,
        tfidf_config=TfidfConfig(min_df=1, max_df=1.0, ngram_range=(1, 1)),
    )
    # com o corpus sintetico as classes positivo/negativo sao claramente
    # separaveis; aceitamos margem maior para a classe neutro
    assert result.accuracy >= 0.5


def test_run_tfidf_linear_supports_linear_svc(
    synthetic_split: tuple[list[str], list[str], list[str], list[str]],
) -> None:
    X_train, y_train, X_test, y_test = synthetic_split
    result = run_tfidf_linear(
        X_train,
        y_train,
        X_test,
        y_test,
        tfidf_config=TfidfConfig(min_df=1, max_df=1.0, ngram_range=(1, 1)),
        model_config=LinearModelConfig(name="linear_svc"),
    )
    assert result.model_name == "linear_svc"
    assert 0.0 <= result.accuracy <= 1.0


def test_run_tfidf_linear_respects_custom_label_order(
    synthetic_split: tuple[list[str], list[str], list[str], list[str]],
) -> None:
    X_train, y_train, X_test, y_test = synthetic_split
    custom_order = ("positivo", "neutro", "negativo")
    result = run_tfidf_linear(
        X_train,
        y_train,
        X_test,
        y_test,
        tfidf_config=TfidfConfig(min_df=1, max_df=1.0, ngram_range=(1, 1)),
        label_order=custom_order,
    )
    assert result.label_order == custom_order


def test_result_as_dict_tem_campos_esperados(
    synthetic_split: tuple[list[str], list[str], list[str], list[str]],
) -> None:
    X_train, y_train, X_test, y_test = synthetic_split
    result = run_tfidf_linear(
        X_train,
        y_train,
        X_test,
        y_test,
        tfidf_config=TfidfConfig(min_df=1, max_df=1.0, ngram_range=(1, 1)),
    )
    expected = {
        "accuracy",
        "precision_macro",
        "recall_macro",
        "f1_macro",
        "confusion_matrix",
        "labels",
        "per_class",
    }
    assert set(result.as_dict().keys()) == expected


def test_result_as_dict_tipos_corretos(
    synthetic_split: tuple[list[str], list[str], list[str], list[str]],
) -> None:
    X_train, y_train, X_test, y_test = synthetic_split
    result = run_tfidf_linear(
        X_train,
        y_train,
        X_test,
        y_test,
        tfidf_config=TfidfConfig(min_df=1, max_df=1.0, ngram_range=(1, 1)),
    )
    payload = result.as_dict()
    assert isinstance(payload["accuracy"], float)
    assert isinstance(payload["precision_macro"], float)
    assert isinstance(payload["recall_macro"], float)
    assert isinstance(payload["f1_macro"], float)
    assert isinstance(payload["confusion_matrix"], list)
    assert all(isinstance(row, list) for row in payload["confusion_matrix"])
    assert all(
        isinstance(value, int)
        for row in payload["confusion_matrix"]
        for value in row
    )


def test_result_as_dict_nao_contem_pipeline(
    synthetic_split: tuple[list[str], list[str], list[str], list[str]],
) -> None:
    X_train, y_train, X_test, y_test = synthetic_split
    result = run_tfidf_linear(
        X_train,
        y_train,
        X_test,
        y_test,
        tfidf_config=TfidfConfig(min_df=1, max_df=1.0, ngram_range=(1, 1)),
    )
    assert "pipeline" not in result.as_dict()


def test_run_corpus_inexistente_levanta_filenotfounderror(tmp_path) -> None:
    """run() com corpus_path invalido deve levantar FileNotFoundError clara."""
    from v2.src.experiments.run_tfidf_linear import run

    with pytest.raises(FileNotFoundError, match="nao encontrado"):
        run(corpus_path=tmp_path / "inexistente.csv")


def test_as_dict_labels_eh_list(
    synthetic_split: tuple[list[str], list[str], list[str], list[str]],
) -> None:
    """as_dict() deve retornar labels como list, nao tuple — consistente com reporting.py."""
    X_train, y_train, X_test, y_test = synthetic_split
    result = run_tfidf_linear(
        X_train,
        y_train,
        X_test,
        y_test,
        tfidf_config=TfidfConfig(min_df=1, max_df=1.0, ngram_range=(1, 1)),
    )
    assert isinstance(result.as_dict()["labels"], list)


def test_as_dict_metricas_com_4_casas(
    synthetic_split: tuple[list[str], list[str], list[str], list[str]],
) -> None:
    """Metricas em as_dict() devem ter no maximo 4 casas decimais."""
    X_train, y_train, X_test, y_test = synthetic_split
    result = run_tfidf_linear(
        X_train,
        y_train,
        X_test,
        y_test,
        tfidf_config=TfidfConfig(min_df=1, max_df=1.0, ngram_range=(1, 1)),
    )
    payload = result.as_dict()
    for field in ("accuracy", "precision_macro", "recall_macro", "f1_macro"):
        val = payload[field]
        assert val == round(val, 4), (
            f"{field}={val} tem mais de 4 casas decimais"
        )


def test_run_com_csv_sintetico_retorna_dict_valido(tmp_path):
    """run() com CSV real (sintetico) deve retornar dict com campos esperados."""
    import pandas as pd
    from v2.src.experiments.run_tfidf_linear import run

    # corpus sintetico com as 4 colunas canonicas
    n = 10  # suficiente para split estratificado (min 2 por classe)
    rows = []
    for label in ["positivo", "neutro", "negativo"]:
        for i in range(n):
            rows.append({
                "raw_text": f"review {label} numero {i}",
                "label": label,
                "source": "synthetic",
            })
    csv_path = tmp_path / "corpus_test.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)

    result = run(corpus_path=csv_path)

    assert isinstance(result, dict)
    for campo in ("accuracy", "precision_macro", "recall_macro", "f1_macro",
                  "confusion_matrix", "labels"):
        assert campo in result, f"Campo ausente: {campo}"
    assert 0.0 <= result["accuracy"] <= 1.0
    assert isinstance(result["labels"], list)


def test_run_metricas_arredondadas_4_casas(tmp_path):
    """run() via CSV deve retornar metricas com no maximo 4 casas decimais."""
    import pandas as pd
    from v2.src.experiments.run_tfidf_linear import run

    rows = []
    for label in ["positivo", "neutro", "negativo"]:
        for i in range(8):
            rows.append({"raw_text": f"texto {label} {i}", "label": label, "source": "s"})
    csv_path = tmp_path / "c.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)

    result = run(corpus_path=csv_path)
    for campo in ("accuracy", "precision_macro", "recall_macro", "f1_macro"):
        val = result[campo]
        assert val == round(val, 4), f"{campo}={val} tem mais de 4 casas decimais"


def test_read_corpus_file_extensao_invalida(tmp_path):
    """_read_corpus_file deve levantar ValueError para extensao desconhecida."""
    from v2.src.experiments.run_tfidf_linear import _read_corpus_file

    invalid_file = tmp_path / "corpus.xlsx"
    invalid_file.write_text("coluna")
    with pytest.raises(ValueError, match=r"(?i)(extensao|extension|formato|format|xlsx)"):
        _read_corpus_file(invalid_file)


def test_read_corpus_file_csv(tmp_path):
    """_read_corpus_file deve ler CSV com as colunas esperadas."""
    import pandas as pd
    from v2.src.experiments.run_tfidf_linear import _read_corpus_file

    corpus = tmp_path / "corpus.csv"
    df = pd.DataFrame(
        {
            "raw_text": ["bom", "ruim", "ok"],
            "clean_text": ["bom", "ruim", "ok"],
            "label": ["positivo", "negativo", "neutro"],
            "source": ["b2w", "b2w", "b2w"],
        }
    )
    df.to_csv(corpus, index=False)
    result = _read_corpus_file(corpus)
    assert set(result.columns) >= {"raw_text", "clean_text", "label", "source"}
    assert len(result) == 3


def test_run_retorna_per_class(tmp_path):
    """run() deve retornar per_class com metricas por classe."""
    import pandas as pd
    from v2.src.experiments.run_tfidf_linear import run

    rows = []
    for lbl in ["positivo", "negativo", "neutro"]:
        for i in range(20):
            rows.append(
                {
                    "raw_text": f"{lbl} {i}",
                    "clean_text": f"{lbl} {i}",
                    "label": lbl,
                    "source": "test",
                }
            )
    df = pd.DataFrame(rows)
    corpus_path = tmp_path / "corpus.csv"
    df.to_csv(corpus_path, index=False)

    result = run(corpus_path=str(corpus_path))
    d = result.as_dict() if hasattr(result, "as_dict") else result
    assert "per_class" in d, "per_class deve estar no resultado"
    assert isinstance(d["per_class"], dict), "per_class deve ser dict"
    assert len(d["per_class"]) >= 2, "per_class deve ter pelo menos 2 classes"
    for lbl, metrics in d["per_class"].items():
        assert "precision" in metrics and "recall" in metrics and "f1" in metrics


def test_run_tfidf_linear_per_class_preenchido(
    synthetic_split: tuple[list[str], list[str], list[str], list[str]],
) -> None:
    """run_tfidf_linear() deve popular per_class com metricas por classe."""
    X_train, y_train, X_test, y_test = synthetic_split
    result = run_tfidf_linear(
        X_train,
        y_train,
        X_test,
        y_test,
        tfidf_config=TfidfConfig(min_df=1, max_df=1.0, ngram_range=(1, 1)),
    )
    assert isinstance(result.per_class, dict)
    assert set(result.per_class.keys()) == set(DEFAULT_LABEL_ORDER)
    for lbl, metrics in result.per_class.items():
        for campo in ("precision", "recall", "f1", "support"):
            assert campo in metrics, f"per_class[{lbl}] faltando {campo}"
        assert isinstance(metrics["support"], int)
