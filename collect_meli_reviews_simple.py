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
OUTPUT_CSV = DATA_DIR / "mercadolivre_reviews_simple.csv"
SEED_FILES = [
    DATA_DIR / "mercadolivre_reviews.csv",
    DATA_DIR / "mercadolivre_reviews_simple.csv",
]

SEARCH_TERMS = [
    "air fryer",
    "cafeteira",
    "smart tv",
    "notebook",
    "fone bluetooth",
    "smartphone",
    "perfume feminino",
    "liquidificador",
    "cadeira escritorio",
    "tenis corrida",
]

TARGET_REVIEWS = 5000
MAX_REVIEWS_PER_ITEM = 50
MAX_ITEMS = 50
PRODUCTS_PAGE_LIMIT = 50
MAX_PRODUCT_PAGES_PER_TERM = 1
REQUEST_PAUSE_SECONDS = 0.2
MAX_RETRIES = 5


def ensure_directories() -> None:
    DATA_DIR.mkdir(exist_ok=True)
    DOCS_DIR.mkdir(exist_ok=True)


def get_token() -> str:
    token = os.environ.get("MELI_ACCESS_TOKEN", "").strip()
    if not token:
        raise SystemExit("Defina MELI_ACCESS_TOKEN antes de executar.")
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
    data = fetch_json(url, headers={"Authorization": f"Bearer {token}"})
    time.sleep(REQUEST_PAUSE_SECONDS)
    return data.get("results", []) or []


def fetch_items_for_product(product_id: str, token: str) -> list[dict]:
    url = f"https://api.mercadolibre.com/products/{product_id}/items"
    try:
        data = fetch_json(url, headers={"Authorization": f"Bearer {token}"})
        time.sleep(REQUEST_PAUSE_SECONDS)
        return data.get("results", []) or []
    except (HTTPError, URLError):
        return []


def fetch_reviews_for_item(item_id: str, token: str, max_reviews: int) -> list[dict]:
    rows: list[dict] = []
    offset = 0
    while len(rows) < max_reviews:
        limit = min(100, max_reviews - len(rows))
        url = f"https://api.mercadolibre.com/reviews/item/{item_id}?limit={limit}&offset={offset}"
        try:
            data = fetch_json(url, headers={"Authorization": f"Bearer {token}"})
        except (HTTPError, URLError):
            break
        time.sleep(REQUEST_PAUSE_SECONDS)
        reviews = data.get("reviews", []) or []
        if not reviews:
            break
        rows.extend(reviews)
        offset += len(reviews)
    return rows


def clean_text(text: str) -> str:
    return " ".join(str(text or "").strip().split())


def collect() -> tuple[list[dict], dict]:
    token = get_token()
    existing_rows, seen_review_ids = load_existing_rows()
    candidate_items: list[dict] = []
    seen_products: set[str] = set()
    seen_items: set[str] = set()

    for term in SEARCH_TERMS:
        for page in range(MAX_PRODUCT_PAGES_PER_TERM):
            offset = page * PRODUCTS_PAGE_LIMIT
            products = fetch_products(term, token, offset)
            if not products:
                break
            for product in products:
                product_id = product.get("id")
                if not product_id or product_id in seen_products:
                    continue
                seen_products.add(product_id)
                items = fetch_items_for_product(product_id, token)
                for item in items:
                    item_id = item.get("item_id")
                    if item_id and item_id not in seen_items:
                        seen_items.add(item_id)
                        candidate_items.append(
                            {
                                "item_id": item_id,
                                "item_title": product.get("name", item_id),
                                "category_id": item.get("category_id", ""),
                            }
                        )
                        break
            if len(candidate_items) >= MAX_ITEMS * 3:
                break
        if len(candidate_items) >= MAX_ITEMS * 3:
            break

    rows: list[dict] = list(existing_rows)
    selected_items = 0

    for candidate in candidate_items:
        if selected_items >= MAX_ITEMS or len(rows) >= TARGET_REVIEWS:
            break
        reviews = fetch_reviews_for_item(candidate["item_id"], token, MAX_REVIEWS_PER_ITEM)
        valid_reviews = 0
        collection_date = datetime.now(timezone.utc).astimezone().date().isoformat()
        for review in reviews:
            review_id = review.get("id")
            content = clean_text(review.get("content", ""))
            rate = review.get("rate")
            if not content or review_id in seen_review_ids:
                continue
            if rate not in (1, 2, 3, 4, 5):
                continue
            seen_review_ids.add(review_id)
            valid_reviews += 1
            rows.append(
                {
                    "source": "mercadolivre_api_reviews_simple",
                    "site": "mercadolivre",
                    "item_id": candidate["item_id"],
                    "item_title": candidate["item_title"],
                    "category_id": candidate["category_id"],
                    "rating": rate,
                    "review_title": clean_text(review.get("title", "")),
                    "review_text": content,
                    "collection_date": collection_date,
                }
            )
            if len(rows) >= TARGET_REVIEWS:
                break
        if valid_reviews:
            selected_items += 1

    summary = {
        "candidate_items": len(candidate_items),
        "seed_reviews": len(existing_rows),
        "selected_items": selected_items,
        "collected_reviews": len(rows),
    }
    return rows, summary


def load_existing_rows() -> tuple[list[dict], set[int]]:
    rows: list[dict] = []
    seen_review_ids: set[int] = set()
    fieldnames = {
        "source",
        "site",
        "item_id",
        "item_title",
        "category_id",
        "rating",
        "review_title",
        "review_text",
        "collection_date",
    }

    for path in SEED_FILES:
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8", newline="") as file:
            reader = csv.DictReader(file)
            for row in reader:
                if set(row.keys()) >= fieldnames:
                    key = (
                        str(row.get("item_id", "")).strip(),
                        str(row.get("rating", "")).strip(),
                        str(row.get("review_title", "")).strip(),
                        str(row.get("review_text", "")).strip(),
                    )
                    review_id = hash(key)
                    if review_id in seen_review_ids:
                        continue
                    seen_review_ids.add(review_id)
                    rows.append({name: row.get(name, "") for name in fieldnames})
    return rows, seen_review_ids


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


def main() -> None:
    ensure_directories()
    rows, summary = collect()
    if not rows:
        raise SystemExit("Nenhuma review foi coletada.")
    write_csv(rows)
    print(json.dumps(summary, ensure_ascii=False))
    print(str(OUTPUT_CSV))


if __name__ == "__main__":
    main()
