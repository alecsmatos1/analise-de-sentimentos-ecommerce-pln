"""Testes de configuracao da v2."""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from v2.src import config


def test_processed_corpus_path_e_path():
    """PROCESSED_CORPUS_PATH deve ser um objeto Path."""
    assert isinstance(config.PROCESSED_CORPUS_PATH, Path)


def test_processed_corpus_path_aponta_para_v2_data():
    """PROCESSED_CORPUS_PATH deve estar dentro de v2/data/."""
    assert "v2" in str(config.PROCESSED_CORPUS_PATH)
    assert "data" in str(config.PROCESSED_CORPUS_PATH)
    assert config.PROCESSED_CORPUS_PATH.suffix == ".csv"


def test_validate_config_retorna_lista():
    """validate_config() deve retornar uma lista (mesmo sem corpus)."""
    result = config.validate_config()
    assert isinstance(result, list)


def test_validate_config_reporta_corpus_ausente(tmp_path, monkeypatch):
    """validate_config() deve reportar corpus ausente quando o arquivo nao existe."""
    monkeypatch.setattr(config, "PROCESSED_CORPUS_PATH", tmp_path / "nao_existe.csv")
    issues = config.validate_config()
    assert any("nao encontrado" in issue or "nao_existe" in issue for issue in issues)


def test_validate_config_sem_issues_quando_corpus_existe(tmp_path, monkeypatch):
    """validate_config() deve retornar lista vazia quando corpus existe."""
    corpus = tmp_path / "corpus.csv"
    corpus.write_text("raw_text,clean_text,label,source\n")
    monkeypatch.setattr(config, "PROCESSED_CORPUS_PATH", corpus)
    issues = config.validate_config()
    assert issues == []
