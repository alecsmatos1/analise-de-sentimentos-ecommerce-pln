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
_REPO_ROOT = _THIS_DIR.parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

try:
    from dotenv import load_dotenv
    load_dotenv(_REPO_ROOT / ".env")
except ImportError:
    pass

import evaluation  # noqa: E402  (path-injection-dependent import)
import reporting  # noqa: E402
import reporting_compare  # noqa: E402


SUPPORTED_EXPERIMENTS = (
    "tfidf-linear",
    "baseline",
    "symbolic",
    "word2vec-linear",
    "bert-embeddings-linear",
    "bert-finetune",
    "llm",
)
SUPPORTED_REPORTS = ("compare",)


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
                # random_state e passado por consistencia de interface, mas o
                # solver "lbfgs" (padrao) nao usa aleatoriedade: o resultado e
                # deterministico independente do seed. Para deterministismo
                # real dependente do seed seria preciso trocar para
                # solver="saga".
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


def _load_tfidf_module():
    """Carrega run_tfidf_linear via import absoluto.

    ``v2/src/experiments/run_tfidf_linear.py`` usa imports relativos
    (``from ..models.linear import ...``), entao precisa ser carregado como
    parte do pacote ``v2.src.experiments``. O proprio modulo CLI ja insere
    ``_REPO_ROOT`` em ``sys.path``, o que torna ``v2`` resolvivel como
    namespace package (PEP 420).
    """

    mod_path = _SRC_DIR / "experiments" / "run_tfidf_linear.py"
    if not mod_path.exists():
        raise SystemExit(
            "Experimento tfidf-linear sem --fixture exige "
            f"{mod_path} (sprint TF-IDF linear). Arquivo nao encontrado."
        )
    return importlib.import_module("v2.src.experiments.run_tfidf_linear")


def _coerce_runner_result(payload: object) -> evaluation.EvaluationResult:
    """Converte o retorno de ``run()`` em ``EvaluationResult``.

    Aceita um ``EvaluationResult`` (compatibilidade) ou um dict com o contrato
    de ``TfidfLinearResult.as_dict()``.
    """

    if isinstance(payload, evaluation.EvaluationResult):
        return payload
    if not isinstance(payload, dict):
        raise SystemExit(
            "run() de tfidf-linear deve retornar dict ou EvaluationResult; "
            f"recebido {type(payload).__name__}."
        )
    missing = [
        key
        for key in (
            "accuracy",
            "precision_macro",
            "recall_macro",
            "f1_macro",
            "confusion_matrix",
        )
        if key not in payload
    ]
    if missing:
        raise SystemExit(
            f"run() de tfidf-linear retornou dict incompleto; faltam: {missing}"
        )
    labels = tuple(payload.get("labels") or evaluation.LABELS)
    return evaluation.EvaluationResult(
        accuracy=float(payload["accuracy"]),
        precision_macro=float(payload["precision_macro"]),
        recall_macro=float(payload["recall_macro"]),
        f1_macro=float(payload["f1_macro"]),
        confusion_matrix=[[int(v) for v in row] for row in payload["confusion_matrix"]],
        labels=labels,
        per_class=dict(payload.get("per_class") or {}),
    )


def _run_tfidf_full(corpus_path: Path | None = None) -> tuple[evaluation.EvaluationResult, dict]:
    module = _load_tfidf_module()

    runner = getattr(module, "run", None)
    if runner is None:
        raise SystemExit(
            "run_tfidf_linear precisa expor uma funcao run()"
        )
    payload = runner(corpus_path=corpus_path)
    if isinstance(payload, tuple) and len(payload) == 2:
        raw_result, meta = payload
    else:
        raw_result, meta = payload, {}
    if not isinstance(meta, dict):
        meta = {}
    meta.setdefault("mode", "full")
    return _coerce_runner_result(raw_result), meta


def _run_baseline_full(corpus_path: Path | None = None) -> tuple[evaluation.EvaluationResult, dict]:
    """Executa o baseline majoritario via ``v2/src/experiments/run_baseline.py``."""

    module = importlib.import_module("v2.src.experiments.run_baseline")
    runner = getattr(module, "run", None)
    if runner is None:
        raise SystemExit("run_baseline precisa expor uma funcao run()")
    raw_result = runner(corpus_path=corpus_path)
    payload = raw_result.as_dict() if hasattr(raw_result, "as_dict") else raw_result
    return _coerce_runner_result(payload), {"mode": "full"}


def _run_symbolic_full(corpus_path: Path | None = None) -> tuple[evaluation.EvaluationResult, dict]:
    """Executa o analisador simbolico via ``v2/src/experiments/run_symbolic.py``."""

    module = importlib.import_module("v2.src.experiments.run_symbolic")
    runner = getattr(module, "run", None)
    if runner is None:
        raise SystemExit("run_symbolic precisa expor uma funcao run()")
    raw_result = runner(corpus_path=corpus_path)
    payload = raw_result.as_dict() if hasattr(raw_result, "as_dict") else raw_result
    return _coerce_runner_result(payload), {"mode": "full"}


def _run_word2vec_linear_full(
    corpus_path: Path | None = None,
) -> tuple[evaluation.EvaluationResult, dict]:
    """Executa Word2Vec NILC + linear via ``v2/src/experiments/run_word2vec_linear.py``."""

    module = importlib.import_module("v2.src.experiments.run_word2vec_linear")
    runner = getattr(module, "run", None)
    if runner is None:
        raise SystemExit("run_word2vec_linear precisa expor uma funcao run()")
    raw_result = runner(corpus_path=corpus_path)
    payload = raw_result.as_dict() if hasattr(raw_result, "as_dict") else raw_result
    return _coerce_runner_result(payload), {"mode": "full"}


def _run_bert_embeddings_linear_full(
    corpus_path: Path | None = None,
) -> tuple[evaluation.EvaluationResult, dict]:
    """Executa BERTimbau (frozen) + linear via ``run_bert_embeddings_linear.py``."""

    module = importlib.import_module(
        "v2.src.experiments.run_bert_embeddings_linear"
    )
    runner = getattr(module, "run", None)
    if runner is None:
        raise SystemExit(
            "run_bert_embeddings_linear precisa expor uma funcao run()"
        )
    raw_result = runner(corpus_path=corpus_path)
    payload = raw_result.as_dict() if hasattr(raw_result, "as_dict") else raw_result
    return _coerce_runner_result(payload), {"mode": "full"}


def _run_bert_finetune_full(
    corpus_path: Path | None = None,
) -> tuple[evaluation.EvaluationResult, dict]:
    """Executa BERTimbau fine-tuned via ``run_bert_finetune.py``."""

    module = importlib.import_module("v2.src.experiments.run_bert_finetune")
    runner = getattr(module, "run", None)
    if runner is None:
        raise SystemExit("run_bert_finetune precisa expor uma funcao run()")
    payload = runner(corpus_path=corpus_path)
    if not isinstance(payload, dict):
        raise SystemExit("run_bert_finetune.run() deve devolver dict serializavel.")
    meta = dict(payload.get("meta") or {})
    meta.setdefault("mode", "full")
    return _coerce_runner_result(payload), meta


def _run_llm_full(
    corpus_path: Path | None = None,
    sample_size: int | None = None,
    write_predictions: bool = True,
) -> tuple[evaluation.EvaluationResult, dict]:
    """Executa LLM zero-shot via ``v2/src/experiments/run_llm.py``."""

    module = importlib.import_module("v2.src.experiments.run_llm")
    runner = getattr(module, "run", None)
    if runner is None:
        raise SystemExit("run_llm precisa expor uma funcao run()")
    kwargs: dict = {"corpus_path": corpus_path, "write_predictions": write_predictions}
    if sample_size is not None:
        kwargs["sample_size"] = sample_size
    raw_result = runner(**kwargs)
    payload = raw_result.as_dict() if hasattr(raw_result, "as_dict") else raw_result
    meta = dict(payload.get("meta") or {})
    meta.setdefault("mode", "full")
    return _coerce_runner_result(payload), meta


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="v2/run_experiment.py",
        description="Executa experimentos da v2 e gera artefatos de avaliacao.",
    )
    parser.add_argument(
        "--experiment",
        required=False,
        choices=SUPPORTED_EXPERIMENTS,
        help="Identificador do experimento a executar.",
    )
    parser.add_argument(
        "--report",
        required=False,
        choices=SUPPORTED_REPORTS,
        help=(
            "Gera um relatorio agregado a partir dos JSONs ja persistidos. "
            "Use 'compare' para consolidar todos os experimentos em "
            "comparison_table.csv + comparison_report.md."
        ),
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
    parser.add_argument(
        "--corpus-path",
        default=None,
        metavar="PATH",
        help=(
            "Caminho para o corpus processado (CSV ou parquet). "
            "Requerido no modo nao-fixture."
        ),
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=None,
        metavar="N",
        help="Tamanho da amostra (apenas para experimentos que suportam amostragem, ex: llm).",
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


def _run_compare_report(output_dir: str | None) -> int:
    """Consolida JSONs de v2/outputs em tabela CSV + relatorio Markdown."""
    target_dir = Path(output_dir) if output_dir else reporting.DEFAULT_OUTPUT_DIR
    df = reporting_compare.build_comparison_table(outputs_dir=target_dir)
    print(df.to_string(index=False))
    csv_path = reporting_compare.save_comparison_csv(df, outputs_dir=target_dir)
    md_path = reporting_compare.save_comparison_report(df, outputs_dir=target_dir)
    print("Artefatos gerados:")
    print(f"  comparison_csv: {csv_path}")
    print(f"  comparison_md:  {md_path}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.report is not None and args.experiment is not None:
        parser.error("--report e --experiment sao mutuamente exclusivos.")
    if args.report is None and args.experiment is None:
        parser.error("informe --experiment <id> ou --report <nome>.")

    if args.report == "compare":
        return _run_compare_report(args.output_dir)

    if args.experiment == "tfidf-linear":
        if args.fixture:
            result, meta = _run_tfidf_fixture(args.seed)
        else:
            corpus_path = Path(args.corpus_path) if args.corpus_path else None
            result, meta = _run_tfidf_full(corpus_path=corpus_path)
    elif args.experiment == "baseline":
        corpus_path = Path(args.corpus_path) if args.corpus_path else None
        result, meta = _run_baseline_full(corpus_path=corpus_path)
    elif args.experiment == "symbolic":
        corpus_path = Path(args.corpus_path) if args.corpus_path else None
        result, meta = _run_symbolic_full(corpus_path=corpus_path)
    elif args.experiment == "word2vec-linear":
        corpus_path = Path(args.corpus_path) if args.corpus_path else None
        result, meta = _run_word2vec_linear_full(corpus_path=corpus_path)
    elif args.experiment == "bert-embeddings-linear":
        corpus_path = Path(args.corpus_path) if args.corpus_path else None
        result, meta = _run_bert_embeddings_linear_full(corpus_path=corpus_path)
    elif args.experiment == "bert-finetune":
        corpus_path = Path(args.corpus_path) if args.corpus_path else None
        result, meta = _run_bert_finetune_full(corpus_path=corpus_path)
    elif args.experiment == "llm":
        corpus_path = Path(args.corpus_path) if args.corpus_path else None
        result, meta = _run_llm_full(
            corpus_path=corpus_path,
            sample_size=args.sample_size,
            write_predictions=not args.no_write,
        )
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
