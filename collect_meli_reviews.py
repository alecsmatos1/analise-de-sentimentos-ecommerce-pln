from __future__ import annotations

import csv
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
DOCS_DIR = ROOT / "docs"
OUTPUT_CSV = DATA_DIR / "mercadolivre_reviews.csv"
OUTPUT_DOC = DOCS_DIR / "coleta_base_real.md"

SITE = "mercadolivre"
SOURCE = "mercadolivre_api_reviews"
SEARCH_TERMS = [
    "air fryer",
    "cafeteira",
    "microondas",
    "smart tv",
    "notebook",
    "fone bluetooth",
    "mouse gamer",
    "smartphone",
    "smartwatch",
    "perfume feminino",
    "protetor solar",
    "shampoo",
    "liquidificador",
    "aspirador robo",
    "cadeira escritorio",
    "tenis corrida",
    "impressora",
    "mochila",
    "caixa de som bluetooth",
    "panela eletrica",
]

TARGET_REVIEWS = 5000
MAX_REVIEWS_PER_ITEM = 100
MIN_COMMENTS_PER_ITEM = 30
MIN_NEGATIVE_PER_ITEM = 1
MIN_POSITIVE_PER_ITEM = 5
MAX_ITEMS_TO_COLLECT = 100
REQUEST_PAUSE_SECONDS = 0.25
PRODUCTS_PAGE_LIMIT = 50
MAX_PRODUCT_PAGES_PER_TERM = 1
MAX_RETRIES = 5


def ensure_directories() -> None:
    DATA_DIR.mkdir(exist_ok=True)
    DOCS_DIR.mkdir(exist_ok=True)


def get_token() -> str:
    token = os.environ.get("MELI_ACCESS_TOKEN", "").strip()
    if not token:
        raise SystemExit("Defina a variável de ambiente MELI_ACCESS_TOKEN antes de executar.")
    return token


def fetch_json(url: str, headers: dict[str, str] | None = None) -> dict:
    request_headers = {"Accept": "application/json"}
    if headers:
        request_headers.update(headers)
    last_error = None
    for attempt in range(MAX_RETRIES):
        request = Request(url, headers=request_headers)
        try:
            with urlopen(request, timeout=30) as response:
                return json.load(response)
        except HTTPError as error:
            last_error = error
            if error.code == 429 and attempt < MAX_RETRIES - 1:
                retry_after = error.headers.get("Retry-After")
                sleep_seconds = int(retry_after) if retry_after and retry_after.isdigit() else 5 * (attempt + 1)
                time.sleep(sleep_seconds)
                continue
            raise
        except URLError as error:
            last_error = error
            if attempt < MAX_RETRIES - 1:
                time.sleep(2 * (attempt + 1))
                continue
            raise
    if last_error:
        raise last_error
    raise RuntimeError("Falha inesperada ao consultar a API.")


def fetch_products(term: str, token: str, offset: int) -> list[dict]:
    url = (
        "https://api.mercadolibre.com/products/search"
        f"?site_id=MLB&status=active&q={quote(term)}&limit={PRODUCTS_PAGE_LIMIT}&offset={offset}"
    )
    try:
        data = fetch_json(url, headers={"Authorization": f"Bearer {token}"})
        time.sleep(REQUEST_PAUSE_SECONDS)
        return data.get("results", []) or []
    except (HTTPError, URLError):
        return []


def fetch_items_for_product(product_id: str, token: str) -> list[dict]:
    url = f"https://api.mercadolibre.com/products/{product_id}/items"
    try:
        data = fetch_json(url, headers={"Authorization": f"Bearer {token}"})
        time.sleep(REQUEST_PAUSE_SECONDS)
        return data.get("results", []) or []
    except (HTTPError, URLError):
        return []


def fetch_review_metadata(item_id: str, token: str) -> dict | None:
    url = f"https://api.mercadolibre.com/reviews/item/{item_id}?limit=1"
    try:
        data = fetch_json(url, headers={"Authorization": f"Bearer {token}"})
        time.sleep(REQUEST_PAUSE_SECONDS)
        return data
    except (HTTPError, URLError):
        return None


def item_passes_criteria(data: dict) -> bool:
    paging = data.get("paging", {}) or {}
    levels = data.get("rating_levels", {}) or {}
    comments = int(paging.get("reviews_with_comment", 0) or 0)
    negative = int(levels.get("one_star", 0) or 0) + int(levels.get("two_star", 0) or 0)
    positive = int(levels.get("four_star", 0) or 0) + int(levels.get("five_star", 0) or 0)
    return (
        comments >= MIN_COMMENTS_PER_ITEM
        and negative >= MIN_NEGATIVE_PER_ITEM
        and positive >= MIN_POSITIVE_PER_ITEM
    )


def fetch_reviews_for_item(item_id: str, token: str, max_reviews: int) -> list[dict]:
    rows: list[dict] = []
    offset = 0
    while len(rows) < max_reviews:
        limit = min(100, max_reviews - len(rows))
        url = f"https://api.mercadolibre.com/reviews/item/{item_id}?limit={limit}&offset={offset}"
        data = fetch_json(url, headers={"Authorization": f"Bearer {token}"})
        time.sleep(REQUEST_PAUSE_SECONDS)
        reviews = data.get("reviews", []) or []
        if not reviews:
            break
        rows.extend(reviews)
        offset += len(reviews)
    return rows


def clean_review_text(text: str) -> str:
    return " ".join(str(text or "").strip().split())


def collect() -> tuple[list[dict], list[dict], dict]:
    token = get_token()
    candidate_items: list[dict] = []
    seen_item_ids: set[str] = set()
    seen_product_ids: set[str] = set()

    # Descobre itens candidatos exclusivamente via API oficial de produtos.
    for term in SEARCH_TERMS:
        for page in range(MAX_PRODUCT_PAGES_PER_TERM):
            offset = page * PRODUCTS_PAGE_LIMIT
            products = fetch_products(term, token, offset)
            if not products:
                break
            for product in products:
                product_id = product.get("id")
                product_name = product.get("name", "")
                if not product_id or product_id in seen_product_ids:
                    continue
                for item in fetch_items_for_product(product_id, token):
                    item_id = item.get("item_id")
                    if item_id and item_id not in seen_item_ids:
                        seen_item_ids.add(item_id)
                        seen_product_ids.add(product_id)
                        candidate_items.append(
                            {
                                "item_id": item_id,
                                "item_title": product_name or item_id,
                                "category_id": item.get("category_id", ""),
                                "product_id": product_id,
                            }
                        )
                        break
            if len(candidate_items) >= MAX_ITEMS_TO_COLLECT * 4:
                break
        if len(candidate_items) >= MAX_ITEMS_TO_COLLECT * 4:
            break

    selected_items: list[dict] = []
    inspected = 0

    for candidate in candidate_items:
        if len(selected_items) >= MAX_ITEMS_TO_COLLECT:
            break
        item_id = candidate["item_id"]
        metadata = fetch_review_metadata(item_id, token)
        inspected += 1
        if not metadata or not item_passes_criteria(metadata):
            continue
        paging = metadata.get("paging", {}) or {}
        levels = metadata.get("rating_levels", {}) or {}
        selected_items.append(
            {
                "item_id": item_id,
                "item_title": candidate["item_title"],
                "category_id": candidate["category_id"],
                "product_id": candidate["product_id"],
                "rating_average": metadata.get("rating_average", 0),
                "reviews_with_comment": int(paging.get("reviews_with_comment", 0) or 0),
                "rating_total": int(paging.get("total", 0) or 0),
                "negative_total": int(levels.get("one_star", 0) or 0) + int(levels.get("two_star", 0) or 0),
                "neutral_total": int(levels.get("three_star", 0) or 0),
                "positive_total": int(levels.get("four_star", 0) or 0) + int(levels.get("five_star", 0) or 0),
            }
        )

    all_rows: list[dict] = []
    seen_review_ids: set[int] = set()

    for item in selected_items:
        if len(all_rows) >= TARGET_REVIEWS:
            break
        per_item_target = min(
            MAX_REVIEWS_PER_ITEM,
            item["reviews_with_comment"],
            TARGET_REVIEWS - len(all_rows),
        )
        reviews = fetch_reviews_for_item(item["item_id"], token, per_item_target)
        collection_date = datetime.now(timezone.utc).astimezone().date().isoformat()
        for review in reviews:
            review_id = review.get("id")
            content = clean_review_text(review.get("content", ""))
            rate = review.get("rate")
            if not content or review_id in seen_review_ids:
                continue
            if rate not in (1, 2, 3, 4, 5):
                continue
            seen_review_ids.add(review_id)
            all_rows.append(
                {
                    "source": SOURCE,
                    "site": SITE,
                    "item_id": item["item_id"],
                    "item_title": item["item_title"],
                    "category_id": item["category_id"],
                    "rating": rate,
                    "review_title": clean_review_text(review.get("title", "")),
                    "review_text": content,
                    "collection_date": collection_date,
                }
            )
            if len(all_rows) >= TARGET_REVIEWS:
                break

    summary = {
        "candidate_items": len(candidate_items),
        "inspected_items": inspected,
        "selected_items": len(selected_items),
        "collected_reviews": len(all_rows),
    }
    return all_rows, selected_items, summary


def write_csv(rows: list[dict]) -> None:
    fieldnames = [
        "source",
        "site",
        "item_id",
        "item_title",
        "category_id",
        "rating",
        "review_title",
        "review_text",
        "collection_date",
    ]
    with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_doc(rows: list[dict], selected_items: list[dict], summary: dict) -> None:
    rating_counts = {score: 0 for score in [1, 2, 3, 4, 5]}
    for row in rows:
        rating_counts[row["rating"]] += 1

    lines = [
        "# Coleta de Base Real Complementar",
        "",
        "## Fonte",
        "",
        "- Mercado Livre Brasil.",
        "- Reviews coletadas pela API oficial de opiniões de produto.",
        "- Descoberta de itens feita exclusivamente por endpoints oficiais de catálogo e produtos.",
        "",
        "## Critérios escolhidos para reduzir viés",
        "",
        "- Usar apenas a API oficial como fonte dos textos das reviews.",
        "- Tratar a base como complementar, nunca como base principal.",
        "- Selecionar itens com distribuição mista de notas.",
        f"- Exigir pelo menos `{MIN_COMMENTS_PER_ITEM}` reviews com comentário por item.",
        f"- Exigir pelo menos `{MIN_NEGATIVE_PER_ITEM}` avaliações negativas agregadas (1-2 estrelas) por item.",
        f"- Exigir pelo menos `{MIN_POSITIVE_PER_ITEM}` avaliações positivas agregadas (4-5 estrelas) por item.",
        f"- Limitar a coleta a no máximo `{MAX_REVIEWS_PER_ITEM}` reviews por item para evitar concentração excessiva.",
        "- Usar múltiplos termos de busca para espalhar a amostragem por diferentes domínios de consumo.",
        "",
        "## Procedimento para refazer",
        "",
        "1. Definir a variável de ambiente `MELI_ACCESS_TOKEN` com um token válido do app.",
        "2. Executar `python collect_meli_reviews.py` no diretório do projeto.",
        "3. Ler o arquivo `data/mercadolivre_reviews.csv` gerado ao final.",
        "",
        "## Observações de conformidade",
        "",
        "- A coleta usou a API oficial tanto para descobrir os itens quanto para obter os textos das reviews.",
        "- Os dados foram coletados para uso acadêmico complementar.",
        "- O arquivo gerado não deve ser republicado como espelho bruto sem revisão dos termos da plataforma.",
        "",
        "## Resumo da execução",
        "",
        f"- Data da coleta: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"- Itens candidatos encontrados: {summary['candidate_items']}",
        f"- Itens inspecionados na API: {summary['inspected_items']}",
        f"- Itens selecionados: {summary['selected_items']}",
        f"- Reviews coletadas: {summary['collected_reviews']}",
        "",
        "## Distribuição das notas na base coletada",
        "",
        f"- 1 estrela: {rating_counts[1]}",
        f"- 2 estrelas: {rating_counts[2]}",
        f"- 3 estrelas: {rating_counts[3]}",
        f"- 4 estrelas: {rating_counts[4]}",
        f"- 5 estrelas: {rating_counts[5]}",
        "",
        "## Limitações",
        "",
        "- A seleção dos textos depende do que a API oficial expõe para cada item.",
        "- Mesmo com filtros de distribuição, a base complementar ainda pode refletir o viés do ecossistema de avaliações da plataforma.",
    ]

    if selected_items:
        lines.extend([
            "",
            "## Itens selecionados",
            "",
            "item_id | product_id | item_title | category_id | reviews_with_comment | rating_average",
            "--- | --- | --- | --- | --- | ---",
        ])
        for item in selected_items:
            safe_title = item["item_title"].replace("|", "/")
            lines.append(
                f"{item['item_id']} | {item['product_id']} | {safe_title} | {item['category_id']} | "
                f"{item['reviews_with_comment']} | {item['rating_average']}"
            )

    OUTPUT_DOC.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ensure_directories()
    rows, selected_items, summary = collect()
    if not rows:
        raise SystemExit("Nenhuma review foi coletada. Saídas existentes foram preservadas.")
    write_csv(rows)
    write_doc(rows, selected_items, summary)
    print(json.dumps(summary, ensure_ascii=False))
    print(str(OUTPUT_CSV))
    print(str(OUTPUT_DOC))


if __name__ == "__main__":
    main()
