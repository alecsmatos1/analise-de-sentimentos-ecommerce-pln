from __future__ import annotations

import re
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


def clean_text(text: str) -> str:
    text = "" if text is None else str(text).strip().lower()
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"[\r\n\t]+", " ", text)
    text = re.sub(r"[^0-9a-zA-ZÃ€-Ã¿\s]", " ", text)
    text = re.sub(r"\d+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


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
        df["review_title"].fillna("").astype(str).str.strip() + " " +
        df["review_text"].fillna("").astype(str).str.strip()
    ).str.strip()
    df["rating"] = pd.to_numeric(df["overall_rating"], errors="coerce")
    df["label"] = df["rating"].apply(lambda x: rating_to_label(int(x)) if pd.notna(x) else "indefinido")
    df["clean_text"] = df["raw_text"].apply(clean_text)
    df = df[(df["label"] != "indefinido") & (df["clean_text"] != "")].copy()
    df["source"] = "b2w"
    return df[["source", "rating", "label", "raw_text", "clean_text"]]


def load_olist() -> pd.DataFrame:
    df = pd.read_csv(OLIST_PATH, low_memory=False)
    df["raw_text"] = (
        df["review_comment_title"].fillna("").astype(str).str.strip() + " " +
        df["review_comment_message"].fillna("").astype(str).str.strip()
    ).str.strip()
    df["rating"] = pd.to_numeric(df["review_score"], errors="coerce")
    df["label"] = df["rating"].apply(lambda x: rating_to_label(int(x)) if pd.notna(x) else "indefinido")
    df["clean_text"] = df["raw_text"].apply(clean_text)
    df = df[(df["label"] != "indefinido") & (df["clean_text"] != "")].copy()
    df["source"] = "olist"
    return df[["source", "rating", "label", "raw_text", "clean_text"]]


def load_meli_simple() -> pd.DataFrame:
    df = pd.read_csv(MELI_SIMPLE_PATH, low_memory=False)
    df["raw_text"] = (
        df["review_title"].fillna("").astype(str).str.strip() + " " +
        df["review_text"].fillna("").astype(str).str.strip()
    ).str.strip()
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
    df["label"] = df["rating"].apply(lambda x: rating_to_label(int(x)) if pd.notna(x) else "indefinido")
    df["clean_text"] = df["raw_text"].apply(clean_text)
    df = df[(df["label"] != "indefinido") & (df["clean_text"] != "")].copy()
    df["source"] = "mercadolivre_simples"
    return df[["source", "rating", "label", "raw_text", "clean_text"]]
