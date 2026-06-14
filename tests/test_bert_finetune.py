"""Testes do experimento BERTimbau fine-tuned (mock — sem GPU/treino real).

A logica de fine-tuning vive em ``v2.src.models.bertimbau_finetune``. Os
testes monkeypatcham ``train_and_evaluate`` para devolver predicoes
sinteticas, garantindo que o contrato de ``run_bert_finetune`` e o ciclo
de metricas/relatorios funcionem sem precisar de GPU nem baixar o BERTimbau.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from v2.src.experiments import run_bert_finetune as run_module
from v2.src.experiments.run_bert_finetune import (
    BertFinetuneResult,
    run,
    run_bert_finetune,
)
from v2.src.models.bertimbau_finetune import FinetuneConfig, FinetuneOutput


def _synthetic_corpus():
    train_texts = [
        "produto otimo entrega rapida amei recomendo",
        "muito bom recomendo loja confiavel adorei",
        "razoavel produto medio nada surpreendente",
        "normal aceitavel produto padrao regular",
        "pessimo produto chegou quebrado nao recomendo",
        "horrivel atendimento demorou pessimo defeito",
    ]
    train_labels = ["positivo", "positivo", "neutro", "neutro", "negativo", "negativo"]
    test_texts = [
        "produto otimo amei",
        "razoavel padrao normal",
        "pessimo horrivel quebrado",
    ]
    test_labels = ["positivo", "neutro", "negativo"]
    return train_texts, train_labels, test_texts, test_labels


@pytest.fixture
def perfect_predictions_patch(monkeypatch):
    """Monkeypatcha train_and_evaluate para devolver predicoes perfeitas."""

    def _fake_train(
        train_texts, train_labels, test_texts, test_labels, config=None, **kwargs
    ):
        assert len(test_texts) == len(test_labels)
        return FinetuneOutput(
            predictions=list(test_labels),
            training_time_seconds=0.123,
            device_used="cpu",
        )

    monkeypatch.setattr(run_module, "train_and_evaluate", _fake_train)
    return _fake_train


def test_run_com_mock_retorna_result_correto(perfect_predictions_patch):
    """run_bert_finetune com mock devolve BertFinetuneResult com campos esperados."""

    X_train, y_train, X_test, y_test = _synthetic_corpus()
    result = run_bert_finetune(X_train, y_train, X_test, y_test)

    assert isinstance(result, BertFinetuneResult)
    assert result.accuracy == 1.0
    assert result.f1_macro == 1.0
    assert set(result.labels) == {"negativo", "neutro", "positivo"}
    assert len(result.confusion_matrix) == 3
    assert set(result.per_class.keys()) == {"negativo", "neutro", "positivo"}
    for cls, metrics in result.per_class.items():
        assert set(metrics.keys()) >= {"precision", "recall", "f1", "support"}


def test_meta_contem_campos_obrigatorios_da_sprint(perfect_predictions_patch):
    """meta.epochs, meta.model, meta.hardware, meta.training_time_seconds."""

    X_train, y_train, X_test, y_test = _synthetic_corpus()
    cfg = FinetuneConfig(num_epochs=3)
    result = run_bert_finetune(X_train, y_train, X_test, y_test, config=cfg)
    payload = result.as_dict()
    meta = payload["meta"]
    assert meta["epochs"] == 3
    assert meta["model"] == "neuralmind/bert-base-portuguese-cased"
    assert meta["hardware"] in {"cpu", "cuda"}
    assert isinstance(meta["training_time_seconds"], float)
    assert meta["training_time_seconds"] >= 0.0


def test_as_dict_e_json_serializavel(perfect_predictions_patch):
    """as_dict() produz somente tipos JSON-serializaveis."""

    X_train, y_train, X_test, y_test = _synthetic_corpus()
    result = run_bert_finetune(X_train, y_train, X_test, y_test)
    json.dumps(result.as_dict())


def test_imperfect_predictions_propagam_para_metricas(monkeypatch):
    """Quando o mock erra uma classe, accuracy/f1 caem coerentemente."""

    def _fake_train(
        train_texts, train_labels, test_texts, test_labels, config=None, **kwargs
    ):
        # devolve sempre "positivo" -> accuracy = 1/3
        return FinetuneOutput(
            predictions=["positivo"] * len(test_texts),
            training_time_seconds=0.1,
            device_used="cpu",
        )

    monkeypatch.setattr(run_module, "train_and_evaluate", _fake_train)

    X_train, y_train, X_test, y_test = _synthetic_corpus()
    result = run_bert_finetune(X_train, y_train, X_test, y_test)
    assert result.accuracy == round(1 / 3, 4)
    assert result.f1_macro < 1.0


def test_run_lanca_quando_corpus_inexistente(tmp_path):
    """run(corpus_path=missing) levanta FileNotFoundError descritivo."""

    missing = tmp_path / "no_such_corpus.csv"
    with pytest.raises(FileNotFoundError):
        run(corpus_path=missing)


def test_finetune_config_defaults_compativeis_com_sprint():
    """FinetuneConfig tem os defaults declarados na sprint S07."""

    cfg = FinetuneConfig()
    assert cfg.model_name == "neuralmind/bert-base-portuguese-cased"
    assert cfg.num_labels == 3
    assert cfg.num_epochs == 3
    assert cfg.batch_size == 32
    assert cfg.eval_batch_size == 64
    assert cfg.learning_rate == 2e-5
    assert cfg.max_length == 128
    assert cfg.warmup_ratio == 0.1
    assert cfg.weight_decay == 0.01
    assert cfg.random_state == 42
