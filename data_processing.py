from __future__ import annotations

import re
import unicodedata
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlretrieve

import pandas as pd


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
DOCS_DIR = ROOT / "docs"

B2W_URL = "https://raw.githubusercontent.com/b2wdigital/b2w-reviews01/master/B2W-Reviews01.csv"
OLIST_URL = (
    "https://raw.githubusercontent.com/Athospd/work-at-olist-data/master/"
    "datasets/olist_order_reviews_dataset.csv"
)

B2W_PATH = DATA_DIR / "B2W-Reviews01.csv"
OLIST_PATH = DATA_DIR / "olist_order_reviews_dataset.csv"
MELI_SIMPLE_PATH = DATA_DIR / "mercadolivre_reviews_simple.csv"

OUTPUT_COLUMNS = [
    "source",
    "rating",
    "label",
    "raw_text",
    "clean_text",
    "sentiment_label",
    "user_id",
    "order_id",
    "product_id",
    "product_name",
    "category",
]

PORTUGUESE_STOPWORDS = {
    "a",
    "ao",
    "aos",
    "aquela",
    "aquele",
    "aqueles",
    "as",
    "ate",
    "com",
    "como",
    "da",
    "das",
    "de",
    "dela",
    "dele",
    "deles",
    "depois",
    "do",
    "dos",
    "e",
    "ela",
    "ele",
    "eles",
    "em",
    "entre",
    "era",
    "essa",
    "esse",
    "esta",
    "estao",
    "estar",
    "este",
    "eu",
    "foi",
    "foram",
    "ha",
    "isso",
    "ja",
    "lhe",
    "mais",
    "mas",
    "me",
    "mesmo",
    "meu",
    "minha",
    "muito",
    "na",
    "nas",
    "no",
    "nos",
    "o",
    "os",
    "ou",
    "para",
    "pela",
    "pelo",
    "por",
    "porque",
    "que",
    "se",
    "seu",
    "sua",
    "tambem",
    "tem",
    "ter",
    "um",
    "uma",
    "voce",
}

STEM_SUFFIXES = (
    "amentos",
    "imentos",
    "amento",
    "imento",
    "adoras",
    "adores",
    "acao",
    "acoes",
    "mente",
    "idades",
    "idade",
    "ivos",
    "ivas",
    "oso",
    "osa",
    "osos",
    "osas",
    "ado",
    "ada",
    "ados",
    "adas",
    "ido",
    "ida",
    "idos",
    "idas",
    "ar",
    "er",
    "ir",
    "s",
)


def ensure_directories() -> None:
    DATA_DIR.mkdir(exist_ok=True)
    DOCS_DIR.mkdir(exist_ok=True)


def download_if_needed(url: str, path: Path) -> tuple[bool, str]:
    if path.exists():
        return True, "arquivo local"
    try:
        urlretrieve(url, path)
        return True, "download concluido"
    except URLError as exc:
        return False, f"falha no download: {exc}"


def simple_stem(token: str) -> str:
    if len(token) <= 4:
        return token
    for suffix in STEM_SUFFIXES:
        if token.endswith(suffix) and len(token) - len(suffix) >= 4:
            return token[: -len(suffix)]
    return token


def clean_text(text: str) -> str:
    text = "" if text is None else str(text).strip().lower()
    text = unicodedata.normalize("NFKD", text)
    text = "".join(char for char in text if not unicodedata.combining(char))
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"[\r\n\t]+", " ", text)
    text = re.sub(r"[^0-9a-zA-Z\s]", " ", text)
    text = re.sub(r"\d+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    tokens = [
        simple_stem(token)
        for token in text.split()
        if token not in PORTUGUESE_STOPWORDS and len(token) > 1
    ]
    return " ".join(tokens)


def rating_to_label(value: int) -> str:
    if value in (1, 2):
        return "negativo"
    if value == 3:
        return "neutro"
    if value in (4, 5):
        return "positivo"
    return "indefinido"


def load_b2w() -> pd.DataFrame:
    df = pd.read_csv(B2W_PATH, low_memory=False)
    df["raw_text"] = (
        df["review_title"].fillna("").astype(str).str.strip()
        + " "
        + df["review_text"].fillna("").astype(str).str.strip()
    ).str.strip()
    df["rating"] = pd.to_numeric(df["overall_rating"], errors="coerce")
    df["label"] = df["rating"].apply(lambda x: rating_to_label(int(x)) if pd.notna(x) else "indefinido")
    df["clean_text"] = df["raw_text"].apply(clean_text)
    df = df[(df["label"] != "indefinido") & (df["clean_text"] != "")].copy()
    df["source"] = "b2w"
    df["sentiment_label"] = df["label"]
    df["user_id"] = df["reviewer_id"].fillna("").astype(str)
    df["order_id"] = ""
    df["product_id"] = df["product_id"].fillna("").astype(str)
    df["product_name"] = df["product_name"].fillna("").astype(str)
    df["category"] = df["site_category_lv1"].fillna(df["site_category_lv2"]).fillna("").astype(str)
    return df[OUTPUT_COLUMNS]


def load_olist() -> pd.DataFrame:
    df = pd.read_csv(OLIST_PATH, low_memory=False)
    df["raw_text"] = (
        df["review_comment_title"].fillna("").astype(str).str.strip()
        + " "
        + df["review_comment_message"].fillna("").astype(str).str.strip()
    ).str.strip()
    df["rating"] = pd.to_numeric(df["review_score"], errors="coerce")
    df["label"] = df["rating"].apply(lambda x: rating_to_label(int(x)) if pd.notna(x) else "indefinido")
    df["clean_text"] = df["raw_text"].apply(clean_text)
    df = df[(df["label"] != "indefinido") & (df["clean_text"] != "")].copy()
    df["source"] = "olist"
    df["sentiment_label"] = df["label"]
    df["user_id"] = ""
    df["order_id"] = df["order_id"].fillna("").astype(str)
    df["product_id"] = ""
    df["product_name"] = ""
    df["category"] = ""
    return df[OUTPUT_COLUMNS]


def load_meli_simple() -> pd.DataFrame:
    df = pd.read_csv(MELI_SIMPLE_PATH, low_memory=False)
    df["raw_text"] = (
        df["review_title"].fillna("").astype(str).str.strip()
        + " "
        + df["review_text"].fillna("").astype(str).str.strip()
    ).str.strip()
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
    df["label"] = df["rating"].apply(lambda x: rating_to_label(int(x)) if pd.notna(x) else "indefinido")
    df["clean_text"] = df["raw_text"].apply(clean_text)
    df = df[(df["label"] != "indefinido") & (df["clean_text"] != "")].copy()
    df["source"] = "mercadolivre_simples"
    df["sentiment_label"] = df["label"]
    df["user_id"] = ""
    df["order_id"] = ""
    df["product_id"] = df["item_id"].fillna("").astype(str)
    df["product_name"] = df["item_title"].fillna("").astype(str)
    df["category"] = df["category_id"].fillna("").astype(str)
    return df[OUTPUT_COLUMNS]
