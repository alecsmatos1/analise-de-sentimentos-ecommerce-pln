"""Serialize v2 evaluation results to CSV/JSON artifacts.

Outputs land under ``v2/outputs/`` by default, which is git-ignored. Callers
that need a different directory can pass ``output_dir``.

The functions accept either an :class:`evaluation.EvaluationResult` instance
or a plain dict with the same shape, so this module does not import from
``evaluation`` directly. That keeps the file standalone and avoids cross
imports during script execution.
"""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parents[1] / "outputs"

_METRIC_FIELDS = ("accuracy", "precision_macro", "recall_macro", "f1_macro")


def _normalize_result(result: Any) -> dict:
    if isinstance(result, Mapping):
        payload = dict(result)
    elif hasattr(result, "as_dict"):
        payload = result.as_dict()
    else:
        raise TypeError(
            "result must be a Mapping or expose an as_dict() method"
        )
    missing = [field for field in _METRIC_FIELDS if field not in payload]
    if missing:
        raise ValueError(f"result is missing required fields: {missing}")
    payload.setdefault("labels", [])
    payload.setdefault("confusion_matrix", [])
    return payload


def ensure_output_dir(output_dir: Path | str | None = None) -> Path:
    target = Path(output_dir) if output_dir is not None else DEFAULT_OUTPUT_DIR
    target.mkdir(parents=True, exist_ok=True)
    return target


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def save_metrics_csv(
    experiment: str,
    result: Any,
    output_dir: Path | str | None = None,
    filename: str = "metrics.csv",
) -> Path:
    """Append a one-line record with the headline metrics.

    The file is created with a header on first write and re-used afterwards so
    several experiments can share the same table.
    """
    payload = _normalize_result(result)
    target_dir = ensure_output_dir(output_dir)
    target = target_dir / filename
    write_header = not target.exists()
    with target.open("a", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        if write_header:
            writer.writerow(["timestamp", "experiment", *_METRIC_FIELDS])
        writer.writerow(
            [
                _timestamp(),
                experiment,
                f"{float(payload['accuracy']):.4f}",
                f"{float(payload['precision_macro']):.4f}",
                f"{float(payload['recall_macro']):.4f}",
                f"{float(payload['f1_macro']):.4f}",
            ]
        )
    return target


def save_result_json(
    experiment: str,
    result: Any,
    extra: Mapping[str, Any] | None = None,
    output_dir: Path | str | None = None,
    filename: str | None = None,
) -> Path:
    """Persist the full evaluation payload (including confusion matrix) as JSON."""
    payload = _normalize_result(result)
    target_dir = ensure_output_dir(output_dir)
    target = target_dir / (filename or f"{experiment}.json")
    body = {
        "experiment": experiment,
        "timestamp": _timestamp(),
        **payload,
    }
    if extra:
        body["meta"] = dict(extra)
    target.write_text(
        json.dumps(body, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return target


def save_confusion_csv(
    experiment: str,
    result: Any,
    output_dir: Path | str | None = None,
    filename: str | None = None,
) -> Path:
    """Write the confusion matrix as ``real x predito`` with the fixed labels."""
    payload = _normalize_result(result)
    labels = list(payload.get("labels") or [])
    matrix = payload.get("confusion_matrix") or []
    if not labels or not matrix:
        raise ValueError("confusion matrix requires labels and matrix data")
    if any(len(row) != len(labels) for row in matrix):
        raise ValueError("confusion matrix rows must match label count")

    target_dir = ensure_output_dir(output_dir)
    target = target_dir / (filename or f"{experiment}_confusion.csv")
    with target.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["real\\predito", *labels])
        for label, row in zip(labels, matrix):
            writer.writerow([label, *[int(value) for value in row]])
    return target


def save_all(
    experiment: str,
    result: Any,
    extra: Mapping[str, Any] | None = None,
    output_dir: Path | str | None = None,
) -> dict[str, Path]:
    """Convenience wrapper writing metrics CSV, JSON and confusion CSV."""
    return {
        "metrics_csv": save_metrics_csv(experiment, result, output_dir),
        "result_json": save_result_json(experiment, result, extra, output_dir),
        "confusion_csv": save_confusion_csv(experiment, result, output_dir),
    }
