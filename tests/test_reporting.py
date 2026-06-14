"""Tests for v2/src/reporting.py."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import pytest

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import reporting  # noqa: E402


def _sample_result() -> dict:
    return {
        "accuracy": 0.75,
        "precision_macro": 0.7,
        "recall_macro": 0.8,
        "f1_macro": 0.74,
        "labels": ["negativo", "neutro", "positivo"],
        "confusion_matrix": [
            [2, 0, 0],
            [0, 1, 1],
            [0, 1, 3],
        ],
    }


def test_save_metrics_csv_creates_header_and_appends(tmp_path: Path):
    result = _sample_result()
    target = reporting.save_metrics_csv("tfidf-linear", result, output_dir=tmp_path)

    assert target.exists()
    assert target.parent == tmp_path

    with target.open(encoding="utf-8") as fh:
        rows = list(csv.reader(fh))
    assert rows[0] == [
        "timestamp",
        "experiment",
        "accuracy",
        "precision_macro",
        "recall_macro",
        "f1_macro",
    ]
    assert rows[1][1] == "tfidf-linear"
    assert rows[1][2] == "0.7500"

    reporting.save_metrics_csv("tfidf-linear", result, output_dir=tmp_path)
    with target.open(encoding="utf-8") as fh:
        rows = list(csv.reader(fh))
    assert len(rows) == 3, "second call should append, not rewrite the header"


def test_save_result_json_round_trips_payload(tmp_path: Path):
    result = _sample_result()

    target = reporting.save_result_json(
        "tfidf-linear",
        result,
        extra={"mode": "fixture", "n_test": 6},
        output_dir=tmp_path,
    )

    payload = json.loads(target.read_text(encoding="utf-8"))
    assert payload["experiment"] == "tfidf-linear"
    assert payload["accuracy"] == 0.75
    assert payload["confusion_matrix"] == result["confusion_matrix"]
    assert payload["meta"] == {"mode": "fixture", "n_test": 6}
    assert "timestamp" in payload


def test_save_confusion_csv_writes_labelled_matrix(tmp_path: Path):
    result = _sample_result()

    target = reporting.save_confusion_csv(
        "tfidf-linear", result, output_dir=tmp_path
    )

    with target.open(encoding="utf-8") as fh:
        rows = list(csv.reader(fh))
    assert rows[0] == ["real\\predito", "negativo", "neutro", "positivo"]
    assert rows[1] == ["negativo", "2", "0", "0"]
    assert rows[3] == ["positivo", "0", "1", "3"]


def test_save_confusion_csv_rejects_inconsistent_matrix(tmp_path: Path):
    broken = _sample_result()
    broken["confusion_matrix"] = [[1, 0], [0, 1]]

    with pytest.raises(ValueError):
        reporting.save_confusion_csv("tfidf-linear", broken, output_dir=tmp_path)


def test_save_all_writes_three_artifacts(tmp_path: Path):
    result = _sample_result()

    written = reporting.save_all(
        "tfidf-linear", result, extra={"mode": "fixture"}, output_dir=tmp_path
    )

    assert set(written.keys()) == {"metrics_csv", "result_json", "confusion_csv"}
    for path in written.values():
        assert path.exists()


def test_reporting_accepts_object_with_as_dict(tmp_path: Path):
    class Stub:
        def as_dict(self):
            return _sample_result()

    target = reporting.save_result_json("tfidf-linear", Stub(), output_dir=tmp_path)
    payload = json.loads(target.read_text(encoding="utf-8"))
    assert payload["accuracy"] == 0.75


def test_reporting_rejects_missing_metric_fields(tmp_path: Path):
    incomplete = {"accuracy": 0.5}

    with pytest.raises(ValueError):
        reporting.save_metrics_csv("tfidf-linear", incomplete, output_dir=tmp_path)


def test_default_output_dir_points_to_v2_outputs():
    assert reporting.DEFAULT_OUTPUT_DIR.name == "outputs"
    assert reporting.DEFAULT_OUTPUT_DIR.parent.name == "v2"
