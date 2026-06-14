"""CLI entry point for v2 experiments.

The first supported experiment is ``tfidf-linear``. Without ``--fixture`` the
CLI delegates to the experiment module that other Wave 01 sprints provide
(``v2/src/experiments/run_tfidf_linear.py``); if that module is not available
it fails fast with an explanatory message. ``--fixture`` runs a tiny in-memory
pipeline so the evaluation/reporting path can be validated without external
datasets.
"""

from __future__ import annotations

import argparse
import importlib
import platform
import sys
import time
from pathlib import Path
from typing import Sequence


_THIS_DIR = Path(__file__).resolve().parent
_SRC_DIR = _THIS_DIR / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

import evaluation  # noqa: E402  (path-injection-dependent import)
import reporting  # noqa: E402


SUPPORTED_EXPERIMENTS = ("tfidf-linear",)


_FIXTURE_TRAIN: list[tuple[str, str]] = [
    ("produto excelente entrega rapida amei recomendo loja", "positivo"),
    ("muito bom recomendo demais loja confiavel adorei", "positivo"),
    ("adorei produto chegou no prazo qualidade otima", "positivo"),
    ("super recomendo produto otimo entrega rapida amei", "positivo"),
    ("perfeito atendimento nota dez excelente loja", "positivo"),
    ("comum nada demais entrega normal regular padrao", "neutro"),
    ("razoavel produto medio nada surpreendente esperado", "neutro"),
    ("ok funcionou esperava mais nada demais padrao", "neutro"),
    ("regular dentro do esperado nada de especial comum", "neutro"),
    ("normal aceitavel produto entregue padrao mediano", "neutro"),
    ("pessimo produto chegou quebrado nao recomendo horrivel", "negativo"),
    ("horrivel atendimento demorou nao gostei pessimo", "negativo"),
    ("ruim produto com defeito devolvi nao recomendo", "negativo"),
    ("nao recomendo pessimo qualidade ruim horrivel defeito", "negativo"),
    ("decepcionante produto quebrado nao funciona pessimo", "negativo"),
]

_FIXTURE_TEST: list[tuple[str, str]] = [
    ("produto otimo recomendo entrega rapida amei", "positivo"),
    ("excelente loja confiavel adorei recomendo", "positivo"),
    ("razoavel nada demais comum padrao", "neutro"),
    ("ok normal dentro do esperado regular", "neutro"),
    ("pessimo nao recomendo produto ruim horrivel", "negativo"),
    ("horrivel quebrado nao funciona decepcionante", "negativo"),
]


def _run_tfidf_fixture(seed: int) -> tuple[evaluation.EvaluationResult, dict]:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline

    pipeline = Pipeline(
        [
            ("tfidf", TfidfVectorizer(min_df=1, ngram_range=(1, 2))),
            (
                "clf",
                LogisticRegression(max_iter=1000, random_state=seed),
            ),
        ]
    )
    train_texts = [text for text, _ in _FIXTURE_TRAIN]
    train_labels = [label for _, label in _FIXTURE_TRAIN]
    test_texts = [text for text, _ in _FIXTURE_TEST]
    test_labels = [label for _, label in _FIXTURE_TEST]

    started = time.perf_counter()
    pipeline.fit(train_texts, train_labels)
    predictions = pipeline.predict(test_texts).tolist()
    elapsed = time.perf_counter() - started

    result = evaluation.evaluate(test_labels, predictions)
    meta = {
        "mode": "fixture",
        "experiment": "tfidf-linear",
        "n_train": len(train_texts),
        "n_test": len(test_texts),
        "seed": seed,
        "elapsed_seconds": round(elapsed, 4),
        "python_version": platform.python_version(),
    }
    return result, meta


def _run_tfidf_full() -> tuple[evaluation.EvaluationResult, dict]:
    try:
        module = importlib.import_module("v2.src.experiments.run_tfidf_linear")
    except ImportError as exc:  # pragma: no cover - depends on parallel sprints
        raise SystemExit(
            "Experimento tfidf-linear sem --fixture exige "
            "v2/src/experiments/run_tfidf_linear.py (sprint TF-IDF linear). "
            f"Falha ao importar: {exc}"
        )

    runner = getattr(module, "run", None)
    if runner is None:
        raise SystemExit(
            "v2.src.experiments.run_tfidf_linear precisa expor uma funcao run()"
        )
    payload = runner()
    if isinstance(payload, tuple) and len(payload) == 2:
        result, meta = payload
    else:
        result, meta = payload, {}
    if not isinstance(meta, dict):
        meta = {}
    meta.setdefault("mode", "full")
    return result, meta


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="v2/run_experiment.py",
        description="Executa experimentos da v2 e gera artefatos de avaliacao.",
    )
    parser.add_argument(
        "--experiment",
        required=True,
        choices=SUPPORTED_EXPERIMENTS,
        help="Identificador do experimento a executar.",
    )
    parser.add_argument(
        "--fixture",
        action="store_true",
        help="Roda um pipeline minimo em memoria, sem dependencia de datasets externos.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Diretorio para salvar artefatos (padrao: v2/outputs/).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Semente usada pelo modo fixture.",
    )
    parser.add_argument(
        "--no-write",
        action="store_true",
        help="Calcula metricas sem persistir arquivos (uso de inspecao rapida).",
    )
    return parser


def _format_summary(experiment: str, result: evaluation.EvaluationResult) -> str:
    return (
        f"Experimento: {experiment}\n"
        f"  accuracy        = {result.accuracy:.4f}\n"
        f"  precision_macro = {result.precision_macro:.4f}\n"
        f"  recall_macro    = {result.recall_macro:.4f}\n"
        f"  f1_macro        = {result.f1_macro:.4f}\n"
        f"  labels          = {list(result.labels)}\n"
        f"  confusion_matrix= {result.confusion_matrix}"
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    if args.experiment == "tfidf-linear":
        if args.fixture:
            result, meta = _run_tfidf_fixture(args.seed)
        else:
            result, meta = _run_tfidf_full()
    else:  # pragma: no cover - guarded by argparse choices
        raise SystemExit(f"Experimento nao suportado: {args.experiment}")

    print(_format_summary(args.experiment, result))

    if args.no_write:
        print("Persistencia desativada (--no-write).")
        return 0

    written = reporting.save_all(
        experiment=args.experiment,
        result=result,
        extra=meta,
        output_dir=args.output_dir,
    )
    print("Artefatos gerados:")
    for key, path in written.items():
        print(f"  {key}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
