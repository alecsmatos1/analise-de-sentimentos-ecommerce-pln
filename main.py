from __future__ import annotations

import pandas as pd

from data_processing import (
    B2W_PATH,
    B2W_URL,
    MELI_SIMPLE_PATH,
    OLIST_PATH,
    OLIST_URL,
    ROOT,
    download_if_needed,
    ensure_directories,
    load_b2w,
    load_meli_simple,
    load_olist,
)
from report_generator import summarize_dataset, write_report
from sentiment_analyzer import generate_recommendations, run_experiment, run_symbolic_experiment


def main() -> None:
    ensure_directories()

    availability: dict[str, str] = {}
    ok_b2w, msg_b2w = download_if_needed(B2W_URL, B2W_PATH)
    availability["b2w"] = msg_b2w
    if not ok_b2w:
        raise SystemExit(f"B2W indisponivel: {msg_b2w}")

    ok_olist, msg_olist = download_if_needed(OLIST_URL, OLIST_PATH)
    availability["olist"] = msg_olist
    availability["meli_simples"] = "arquivo local" if MELI_SIMPLE_PATH.exists() else "nao disponivel"

    b2w_df = load_b2w()
    experiments = {"b2w_principal": b2w_df}
    olist_df = None
    meli_df = None

    if ok_olist and OLIST_PATH.exists():
        olist_df = load_olist()
        if not olist_df.empty:
            experiments["b2w_mais_olist"] = pd.concat([b2w_df, olist_df], ignore_index=True)

    if MELI_SIMPLE_PATH.exists():
        meli_df = load_meli_simple()
        if not meli_df.empty:
            experiments["b2w_mais_meli_simples"] = pd.concat([b2w_df, meli_df], ignore_index=True)

    if olist_df is not None and meli_df is not None and not olist_df.empty and not meli_df.empty:
        experiments["b2w_mais_olist_mais_meli_simples"] = pd.concat(
            [b2w_df, olist_df, meli_df],
            ignore_index=True,
        )

    dataset_summaries = [summarize_dataset(name, df) for name, df in experiments.items()]
    metric_rows: list[dict[str, object]] = []
    for name, df in experiments.items():
        metric_rows.extend(run_experiment(name, df))
        metric_rows.extend(run_symbolic_experiment(name, df))

    metrics_df = pd.DataFrame(metric_rows)
    metrics_df.to_csv(ROOT / "metricas.csv", index=False)

    recommendation_frames = [generate_recommendations(b2w_df)]
    if meli_df is not None and not meli_df.empty:
        recommendation_frames.append(generate_recommendations(meli_df))
    recommendations_df = pd.concat(
        [frame for frame in recommendation_frames if not frame.empty],
        ignore_index=True,
    ) if any(not frame.empty for frame in recommendation_frames) else pd.DataFrame()
    recommendations_df.to_csv(ROOT / "recomendacoes.csv", index=False)

    write_report(availability, dataset_summaries, metrics_df, recommendations_df)

    print("Execucao concluida.")
    print(metrics_df.to_string(index=False))
    if not recommendations_df.empty:
        print("\nRecomendacoes geradas:")
        print(recommendations_df.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
