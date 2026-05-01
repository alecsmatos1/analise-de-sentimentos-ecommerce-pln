from __future__ import annotations

import re
import unicodedata
from math import log1p

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from data_processing import ROOT
from report_generator import save_confusion_matrix


LABEL_ORDER = ["negativo", "neutro", "positivo"]
LABEL_SCORE = {"negativo": -1.0, "neutro": 0.0, "positivo": 1.0}
SYMBOLIC_SCORE_MARGIN = 0.75

SYMBOLIC_POSITIVE_LEXICON = {
    "adorei": 1.4,
    "amei": 1.5,
    "aprovado": 1.2,
    "barato": 0.8,
    "bom": 1.0,
    "bonito": 0.8,
    "confortavel": 1.0,
    "cumpre": 0.9,
    "excelente": 1.6,
    "funciona": 1.0,
    "gostei": 1.1,
    "maravilhoso": 1.6,
    "melhor": 1.0,
    "original": 1.0,
    "otimo": 1.4,
    "perfeito": 1.5,
    "qualidade": 0.8,
    "rapido": 1.0,
    "recomendo": 1.3,
    "resistente": 1.0,
    "satisfeito": 1.2,
    "superou": 1.4,
    "top": 1.4,
    "vale a pena": 1.4,
}

SYMBOLIC_NEGATIVE_LEXICON = {
    "arrependi": 1.4,
    "atrasado": 1.2,
    "atrasou": 1.4,
    "bugou": 1.4,
    "caro": 0.9,
    "decepcionado": 1.5,
    "defeito": 1.7,
    "demorou": 1.1,
    "desliga": 1.4,
    "enganosa": 1.6,
    "falso": 1.7,
    "fraco": 1.1,
    "horrivel": 1.6,
    "lento": 1.0,
    "pessimo": 1.7,
    "quebrado": 1.8,
    "ruim": 1.3,
    "travou": 1.5,
}

SYMBOLIC_STRONG_PATTERNS = [
    ("nao_recebimento", r"\b(nao recebi|nunca recebi|nem recebi|nao chegou|produto nao chegou)\b", "negativo", 2.6, "nao recebimento do produto"),
    ("defeito_funcionamento", r"\b(nao funciona|nao funcionou|parou de funcionar|parou de carregar|nao liga|veio com defeito|chegou quebrado)\b", "negativo", 2.4, "defeito ou falha de funcionamento"),
    ("devolucao_garantia", r"\b(quero devolucao|pedi devolucao|pedi reembolso|estorno|garantia|troca|procon)\b", "negativo", 1.8, "acao pos-compra negativa"),
    ("expectativa_negativa", r"\b(esperava mais|nao era o que esperava|decepcionou|nao condiz|diferente da foto|propaganda enganosa|veio errado)\b", "negativo", 1.8, "expectativa violada"),
    ("entrega_negativa", r"\b(entrega atrasou|atrasou a entrega|demorou para chegar|prazo nao cumprido|transportadora ruim)\b", "negativo", 1.5, "problema de entrega"),
    ("entrega_positiva", r"\b(chegou rapido|chegou antes do prazo|entrega rapida|entrega perfeita|chegou no prazo)\b", "positivo", 1.4, "entrega positiva"),
    ("recomendacao_negativa", r"\b(nao recomendo|nao comprem|evitem|nunca mais compro)\b", "negativo", 2.0, "recomendacao negativa"),
    ("recomendacao_positiva", r"\b(super recomendo|recomendo muito|recomendo a todos|podem comprar)\b", "positivo", 1.7, "recomendacao positiva"),
    ("valor_positivo", r"\b(bom custo beneficio|custo beneficio excelente|preco justo|vale muito a pena)\b", "positivo", 1.5, "avaliacao positiva de valor"),
    ("valor_negativo", r"\b(nao vale a pena|nao vale o preco|dinheiro jogado fora|caro pelo que entrega)\b", "negativo", 1.7, "avaliacao negativa de valor"),
    ("condicional_contrafactual", r"\b(seria|teria|poderia)\b.{0,35}\b(se|caso)\b", "negativo", 0.9, "avaliacao condicional ou contrafactual"),
    ("mudanca_temporal_negativa", r"\b(no comeco|inicialmente|primeiro|antes)\b.{0,80}\b(mas|porem|depois)\b.{0,80}\b(parou|quebrou|defeito|nao funciona|travou)\b", "negativo", 1.8, "mudanca temporal negativa"),
    ("mudanca_temporal_positiva", r"\b(no comeco|inicialmente|primeiro)\b.{0,80}\b(mas|porem|depois)\b.{0,80}\b(funcionou|melhorou|resolveu)\b", "positivo", 1.0, "mudanca temporal positiva"),
    ("atribuicao_positiva", r"\b(minha filha|meu filho|minha esposa|meu marido|familia)\b.{0,30}\b(adorou|amou|gostou)\b", "positivo", 1.1, "opiniao positiva atribuida"),
    ("atribuicao_negativa", r"\b(minha filha|meu filho|minha esposa|meu marido|familia)\b.{0,30}\b(odiou|reclamou|nao gostou)\b", "negativo", 1.1, "opiniao negativa atribuida"),
    ("ironia_conservadora", r"\b(excelente|parabens|maravilha|otimo)\b.{0,80}\b(quebrado|nao funciona|defeito|atrasou|nao recebi)\b", "negativo", 1.6, "possivel ironia com evidencia negativa"),
]

SYMBOLIC_EMOJIS = {
    "\U0001f60d": ("positivo", 1.2, "emoji positivo"),
    "\U0001f525": ("positivo", 1.0, "emoji positivo"),
    "\U0001f44d": ("positivo", 1.0, "emoji positivo"),
    "\U0001f600": ("positivo", 0.8, "emoji positivo"),
    "\U0001f603": ("positivo", 0.8, "emoji positivo"),
    "\U0001f604": ("positivo", 0.8, "emoji positivo"),
    "\U0001f622": ("negativo", 1.0, "emoji negativo"),
    "\U0001f620": ("negativo", 1.0, "emoji negativo"),
    "\U0001f44e": ("negativo", 1.0, "emoji negativo"),
}

SYMBOLIC_INTENSIFIER_PATTERN = r"\b(muito|mto|super|mega|extremamente|demais|bastante|totalmente)\b"
SYMBOLIC_ATTENUATOR_PATTERN = r"\b(pouco|meio|quase|um pouco|mais ou menos)\b"
SYMBOLIC_HEDGE_PATTERN = r"\b(talvez|acho que|parece|pode ser|acredito que|ainda preciso testar)\b"
SYMBOLIC_CONDITIONAL_PATTERN = r"\b(seria|teria|poderia|se|caso)\b"
SYMBOLIC_SUBJECTIVE_PATTERN = r"\b(achei|senti|gostei|odiei|adorei|amei|recomendo|decepcionou|esperava)\b"
SYMBOLIC_OBJECTIVE_PATTERN = r"\b(comprei|recebi|chegou|produto|pedido|entrega)\b"


def normalize_symbolic_text(text: str) -> str:
    text = "" if text is None else str(text).strip().lower()
    text = unicodedata.normalize("NFKD", text)
    text = "".join(char for char in text if not unicodedata.combining(char))
    text = re.sub(r"[\r\n\t]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def phrase_pattern(phrase: str) -> str:
    escaped = re.escape(phrase).replace(r"\ ", r"\s+")
    return rf"(?<!\w){escaped}(?!\w)"


def symbolic_context_multiplier(text: str, start: int, end: int) -> float:
    prefix = text[max(0, start - 70):start]
    context = text[max(0, start - 70):min(len(text), end + 50)]
    multiplier = 1.0
    if re.search(SYMBOLIC_INTENSIFIER_PATTERN, prefix):
        multiplier *= 1.25
    if re.search(SYMBOLIC_ATTENUATOR_PATTERN, prefix):
        multiplier *= 0.65
    if re.search(SYMBOLIC_HEDGE_PATTERN, prefix):
        multiplier *= 0.55
    if re.search(SYMBOLIC_CONDITIONAL_PATTERN, context):
        multiplier *= 0.65
    return multiplier


def has_shifter_before(text: str, start: int) -> bool:
    prefix = text[max(0, start - 45):start]
    prefix = re.sub(r"[^\w\s]", " ", prefix)
    return bool(
        re.search(
            r"(?:\b(?:nao|nunca|sem|perdeu)\s+(?:\w+\s+){0,1}|\b(?:deixou de|parou de|falta de)\s+(?:\w+\s+){0,1})$",
            prefix,
        )
    )


def quick_segment_signal(text: str) -> float:
    signal = 0.0
    for phrase, weight in SYMBOLIC_POSITIVE_LEXICON.items():
        if re.search(phrase_pattern(phrase), text):
            signal += weight
    for phrase, weight in SYMBOLIC_NEGATIVE_LEXICON.items():
        if re.search(phrase_pattern(phrase), text):
            signal -= weight
    for _, pattern, polarity, weight, _ in SYMBOLIC_STRONG_PATTERNS:
        if re.search(pattern, text):
            signal += weight if polarity == "positivo" else -weight
    return signal


def analyze_symbolic_sentiment(raw_text: str) -> dict[str, object]:
    text = "" if raw_text is None else str(raw_text)
    normalized = normalize_symbolic_text(text)
    positive_score = 0.0
    negative_score = 0.0
    neutral_evidence = 0
    rule_hits: list[dict[str, object]] = []

    def add_hit(rule: str, excerpt: str, polarity: str, weight: float, reason: str) -> None:
        nonlocal positive_score, negative_score, neutral_evidence
        if weight <= 0:
            return
        clean_excerpt = re.sub(r"\s+", " ", excerpt).strip()
        if polarity == "positivo":
            positive_score += weight
        elif polarity == "negativo":
            negative_score += weight
        else:
            neutral_evidence += 1
        rule_hits.append(
            {
                "rule": rule,
                "excerpt": clean_excerpt[:90],
                "polarity": polarity,
                "weight": round(float(weight), 3),
                "reason": reason,
            }
        )

    if not normalized:
        return {
            "label": "neutro",
            "score": 0.0,
            "confidence": 0.0,
            "positive_score": 0.0,
            "negative_score": 0.0,
            "neutral_evidence": 0,
            "rule_hits": [],
        }

    for emoji, (polarity, weight, reason) in SYMBOLIC_EMOJIS.items():
        occurrences = text.count(emoji)
        if occurrences:
            add_hit("emoji", emoji, polarity, min(weight * occurrences, weight * 2.0), reason)

    section_markers = [
        ("pros", r"\b(pontos positivos?|pros?|vantagens?)\s*:"),
        ("pros", r"(?<!nao )\bgostei\s*:"),
        ("cons", r"\b(pontos negativos?|contras?|desvantagens?)\s*:"),
        ("cons", r"\bnao gostei\s*:"),
    ]
    for section, pattern in section_markers:
        for match in re.finditer(pattern, normalized):
            polarity = "positivo" if section == "pros" else "negativo"
            add_hit("pros_contras", match.group(0), polarity, 0.8, "cabecalho de pros/contras")

    for rule, pattern, polarity, weight, reason in SYMBOLIC_STRONG_PATTERNS:
        for match in re.finditer(pattern, normalized):
            multiplier = symbolic_context_multiplier(normalized, match.start(), match.end())
            add_hit(rule, match.group(0), polarity, weight * multiplier, reason)

    for phrase, weight in SYMBOLIC_POSITIVE_LEXICON.items():
        for match in re.finditer(phrase_pattern(phrase), normalized):
            multiplier = symbolic_context_multiplier(normalized, match.start(), match.end())
            if has_shifter_before(normalized, match.start()):
                add_hit("polarity_shifter", match.group(0), "negativo", weight * multiplier * 1.15, "shifter antes de termo positivo")
            else:
                add_hit("lexico_positivo", match.group(0), "positivo", weight * multiplier, "termo positivo de dominio")

    for phrase, weight in SYMBOLIC_NEGATIVE_LEXICON.items():
        for match in re.finditer(phrase_pattern(phrase), normalized):
            multiplier = symbolic_context_multiplier(normalized, match.start(), match.end())
            if has_shifter_before(normalized, match.start()):
                add_hit("polarity_shifter", match.group(0), "positivo", weight * multiplier * 0.8, "shifter antes de termo negativo")
            else:
                add_hit("lexico_negativo", match.group(0), "negativo", weight * multiplier, "termo negativo de dominio")

    if re.search(SYMBOLIC_HEDGE_PATTERN, normalized):
        add_hit("hedge", re.search(SYMBOLIC_HEDGE_PATTERN, normalized).group(0), "neutro", 0.1, "marcador de incerteza")

    if re.search(SYMBOLIC_SUBJECTIVE_PATTERN, normalized):
        add_hit("subjetividade", re.search(SYMBOLIC_SUBJECTIVE_PATTERN, normalized).group(0), "neutro", 0.1, "marcador subjetivo")
    elif re.search(SYMBOLIC_OBJECTIVE_PATTERN, normalized):
        add_hit("objetividade", re.search(SYMBOLIC_OBJECTIVE_PATTERN, normalized).group(0), "neutro", 0.1, "marcador factual sem polaridade clara")

    if " mas " in normalized or " porem " in normalized:
        connector = " mas " if " mas " in normalized else " porem "
        after_connector = normalized.rsplit(connector, 1)[-1]
        signal = quick_segment_signal(after_connector)
        if signal > 0.4:
            add_hit("contraste", after_connector[:90], "positivo", min(abs(signal) * 0.6, 1.4), "trecho final apos contraste")
        elif signal < -0.4:
            add_hit("contraste", after_connector[:90], "negativo", min(abs(signal) * 0.6, 1.4), "trecho final apos contraste")

    if re.search(r"\b(embora|apesar de|mesmo que)\b", normalized):
        tail = normalized.split(",", 1)[-1] if "," in normalized else normalized
        signal = quick_segment_signal(tail)
        if signal > 0.4:
            add_hit("concessao", tail[:90], "positivo", min(abs(signal) * 0.5, 1.1), "conclusao positiva apos concessao")
        elif signal < -0.4:
            add_hit("concessao", tail[:90], "negativo", min(abs(signal) * 0.5, 1.1), "conclusao negativa apos concessao")

    uppercase_words = re.findall(r"\b[A-Z]{4,}\b", text)
    if uppercase_words:
        dominant = "positivo" if positive_score >= negative_score else "negativo"
        add_hit("caixa_alta", " ".join(uppercase_words[:3]), dominant, min(0.2 * len(uppercase_words), 0.6), "enfase por caixa-alta")

    elongated_words = re.findall(r"\b\w*(\w)\1{2,}\w*\b", normalized)
    if elongated_words:
        dominant = "positivo" if positive_score >= negative_score else "negativo"
        add_hit("alongamento", "alongamento", dominant, min(0.15 * len(elongated_words), 0.45), "enfase por alongamento")

    if re.search(r"!{2,}", text):
        dominant = "positivo" if positive_score >= negative_score else "negativo"
        add_hit("pontuacao_expressiva", "!!", dominant, 0.25, "enfase por pontuacao")

    score = positive_score - negative_score
    if score > SYMBOLIC_SCORE_MARGIN:
        label = "positivo"
    elif score < -SYMBOLIC_SCORE_MARGIN:
        label = "negativo"
    else:
        label = "neutro"

    total_evidence = positive_score + negative_score
    confidence = 0.0 if total_evidence == 0 else min(1.0, abs(score) / total_evidence)
    return {
        "label": label,
        "score": round(float(score), 4),
        "confidence": round(float(confidence), 4),
        "positive_score": round(float(positive_score), 4),
        "negative_score": round(float(negative_score), 4),
        "neutral_evidence": int(neutral_evidence),
        "rule_hits": rule_hits,
    }


def build_pipeline(model_name: str) -> Pipeline:
    if model_name == "logistic_regression":
        model = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
    else:
        model = LinearSVC(class_weight="balanced", random_state=42)
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=False,
                    ngram_range=(1, 2),
                    min_df=3,
                    max_df=0.95,
                    sublinear_tf=True,
                ),
            ),
            ("model", model),
        ]
    )


def run_experiment(name: str, df: pd.DataFrame) -> list[dict[str, object]]:
    X_train, X_test, y_train, y_test = train_test_split(
        df["clean_text"],
        df["label"],
        test_size=0.2,
        stratify=df["label"],
        random_state=42,
    )

    rows: list[dict[str, object]] = []
    for model_name in ["logistic_regression", "linear_svc"]:
        pipeline = build_pipeline(model_name)
        pipeline.fit(X_train, y_train)
        predictions = pipeline.predict(X_test)

        precision, recall, f1, _ = precision_recall_fscore_support(
            y_test,
            predictions,
            average="macro",
            zero_division=0,
        )
        accuracy = accuracy_score(y_test, predictions)
        matrix = confusion_matrix(y_test, predictions, labels=LABEL_ORDER)

        image_name = f"cm_{name}_{model_name}.png"
        save_confusion_matrix(matrix, LABEL_ORDER, ROOT / image_name, f"{name} - {model_name}")

        rows.append(
            {
                "experimento": name,
                "modelo": model_name,
                "linhas": int(len(df)),
                "accuracy": round(float(accuracy), 4),
                "precision_macro": round(float(precision), 4),
                "recall_macro": round(float(recall), 4),
                "f1_macro": round(float(f1), 4),
                "matriz_confusao": image_name,
            }
        )
    return rows


def run_symbolic_experiment(name: str, df: pd.DataFrame) -> list[dict[str, object]]:
    _, X_test, _, y_test = train_test_split(
        df["raw_text"],
        df["label"],
        test_size=0.2,
        stratify=df["label"],
        random_state=42,
    )

    predictions = [analyze_symbolic_sentiment(text)["label"] for text in X_test]
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test,
        predictions,
        average="macro",
        zero_division=0,
    )
    accuracy = accuracy_score(y_test, predictions)
    matrix = confusion_matrix(y_test, predictions, labels=LABEL_ORDER)

    image_name = f"cm_{name}_symbolic_rules.png"
    save_confusion_matrix(matrix, LABEL_ORDER, ROOT / image_name, f"{name} - symbolic_rules")

    return [
        {
            "experimento": name,
            "modelo": "symbolic_rules",
            "linhas": int(len(df)),
            "accuracy": round(float(accuracy), 4),
            "precision_macro": round(float(precision), 4),
            "recall_macro": round(float(recall), 4),
            "f1_macro": round(float(f1), 4),
            "matriz_confusao": image_name,
        }
    ]


def build_product_recommendation_profile(df: pd.DataFrame, min_reviews: int = 2) -> pd.DataFrame:
    required = {"product_id", "product_name", "category", "rating", "sentiment_label"}
    if df.empty or not required.issubset(df.columns):
        return pd.DataFrame()

    candidates = df.copy()
    candidates = candidates[candidates["product_id"].fillna("").astype(str).str.strip() != ""].copy()
    if candidates.empty:
        return pd.DataFrame()

    candidates["sentiment_score"] = candidates["sentiment_label"].map(LABEL_SCORE).fillna(0.0)
    grouped = (
        candidates.groupby(["source", "product_id", "product_name", "category"], dropna=False)
        .agg(
            reviews=("sentiment_label", "size"),
            rating_mean=("rating", "mean"),
            sentiment_mean=("sentiment_score", "mean"),
            positive_reviews=("sentiment_label", lambda values: int((values == "positivo").sum())),
            negative_reviews=("sentiment_label", lambda values: int((values == "negativo").sum())),
        )
        .reset_index()
    )
    grouped = grouped[grouped["reviews"] >= min_reviews].copy()
    if grouped.empty:
        return grouped

    grouped["positive_rate"] = grouped["positive_reviews"] / grouped["reviews"]
    grouped["negative_rate"] = grouped["negative_reviews"] / grouped["reviews"]
    max_volume = max(float(grouped["reviews"].max()), 1.0)
    grouped["volume_score"] = grouped["reviews"].apply(lambda value: log1p(float(value)) / log1p(max_volume))
    grouped["recommendation_score"] = (
        grouped["positive_rate"] * 0.45
        + (grouped["rating_mean"].fillna(0) / 5.0) * 0.35
        + ((grouped["sentiment_mean"] + 1.0) / 2.0) * 0.15
        + grouped["volume_score"] * 0.05
        - grouped["negative_rate"] * 0.25
    )
    return grouped.sort_values(
        ["recommendation_score", "positive_rate", "rating_mean", "reviews"],
        ascending=[False, False, False, False],
    )


def generate_recommendations(
    df: pd.DataFrame,
    users_limit: int = 20,
    recommendations_per_user: int = 5,
    min_reviews: int = 2,
) -> pd.DataFrame:
    profile = build_product_recommendation_profile(df, min_reviews=min_reviews)
    if profile.empty:
        return pd.DataFrame()

    rows: list[dict[str, object]] = []
    user_ready = {"user_id", "product_id", "category", "sentiment_label"}.issubset(df.columns)
    if user_ready:
        user_df = df.copy()
        user_df = user_df[user_df["user_id"].fillna("").astype(str).str.strip() != ""].copy()
        positive_history = user_df[user_df["sentiment_label"] == "positivo"].copy()
        user_ids = positive_history["user_id"].drop_duplicates().head(users_limit).tolist()

        for user_id in user_ids:
            user_reviews = user_df[user_df["user_id"] == user_id]
            seen_products = set(user_reviews["product_id"].fillna("").astype(str))
            liked_categories = (
                positive_history[positive_history["user_id"] == user_id]["category"]
                .fillna("")
                .astype(str)
                .str.strip()
            )
            liked_categories = [category for category in liked_categories.drop_duplicates().tolist() if category]
            if not liked_categories:
                continue

            candidates = profile[
                profile["category"].fillna("").astype(str).isin(liked_categories)
                & ~profile["product_id"].fillna("").astype(str).isin(seen_products)
            ].head(recommendations_per_user)

            for rank, (_, candidate) in enumerate(candidates.iterrows(), start=1):
                rows.append(
                    {
                        "tipo": "personalizada",
                        "user_id": user_id,
                        "categoria_base": candidate["category"],
                        "rank": rank,
                        "source": candidate["source"],
                        "product_id": candidate["product_id"],
                        "product_name": candidate["product_name"],
                        "category": candidate["category"],
                        "recommendation_score": round(float(candidate["recommendation_score"]), 4),
                        "rating_mean": round(float(candidate["rating_mean"]), 4),
                        "positive_rate": round(float(candidate["positive_rate"]), 4),
                        "reviews": int(candidate["reviews"]),
                        "motivo": "categoria com historico positivo do usuario",
                    }
                )

    if rows:
        return pd.DataFrame(rows)

    fallback = profile.head(recommendations_per_user)
    for rank, (_, candidate) in enumerate(fallback.iterrows(), start=1):
        rows.append(
            {
                "tipo": "global",
                "user_id": "",
                "categoria_base": candidate["category"],
                "rank": rank,
                "source": candidate["source"],
                "product_id": candidate["product_id"],
                "product_name": candidate["product_name"],
                "category": candidate["category"],
                "recommendation_score": round(float(candidate["recommendation_score"]), 4),
                "rating_mean": round(float(candidate["rating_mean"]), 4),
                "positive_rate": round(float(candidate["positive_rate"]), 4),
                "reviews": int(candidate["reviews"]),
                "motivo": "ranking global por sentimento positivo",
            }
        )
    return pd.DataFrame(rows)
