"""
Voice-of-customer analysis for the (Brazilian Portuguese) review comments.

There is no labelled sentiment in the raw data, and running a full NLP model
is overkill for short e-commerce comments, so this uses two simple, explainable
rules instead of a black box:

    sentiment -- from review_score (the star rating the customer already gave):
                 1-2 stars = negative, 3 = neutral, 4-5 = positive
    themes    -- keyword matching in Portuguese for the topics that come up
                 again and again in Olist reviews: delivery, product quality,
                 customer service, price and the seller/store.

This is a "bag of known phrases" approach, not a trained model -- it is
intentionally simple so every tag can be explained by the exact words that
triggered it.
"""

import re

import pandas as pd

_THEME_KEYWORDS = {
    "delivery": [
        "entrega", "entregue", "prazo", "chegou", "chegar", "atras", "demora",
        "rapid", "no prazo", "antes do prazo", "transportadora", "correios",
    ],
    "product_quality": [
        "qualidade", "quebrado", "defeito", "estragado", "veio errado",
        "produto ruim", "excelente produto", "ótimo produto", "bom produto",
    ],
    "customer_service": [
        "atendimento", "suporte", "resposta", "contato", "sac", "vendedor",
    ],
    "price_value": [
        "preço", "caro", "barato", "valor", "custo",
    ],
    "store_experience": [
        "loja", "site", "recomendo", "compraria", "voltarei", "nunca mais",
    ],
}


def _sentiment_from_score(score) -> str:
    if pd.isna(score):
        return "unknown"
    if score <= 2:
        return "negative"
    if score == 3:
        return "neutral"
    return "positive"


def _themes_in(text: str) -> list[str]:
    if not isinstance(text, str) or not text.strip():
        return []
    lowered = text.lower()
    return [theme for theme, keywords in _THEME_KEYWORDS.items() if any(kw in lowered for kw in keywords)]


def analyze_reviews(reviews: pd.DataFrame) -> pd.DataFrame:
    """Add `sentiment` and `themes` columns to a reviews DataFrame.

    Expects the columns `review_score` and `review_comment_message` (as in
    dim_reviews / olist_order_reviews_dataset.csv).
    """
    result = reviews.copy()
    result["sentiment"] = result["review_score"].apply(_sentiment_from_score)
    result["themes"] = result["review_comment_message"].apply(_themes_in)
    return result


if __name__ == "__main__":
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from database import query_dataframe

    reviews = query_dataframe(
        "SELECT review_score, review_comment_message FROM dim_reviews WHERE review_comment_message IS NOT NULL"
    )
    analyzed = analyze_reviews(reviews)
    print(analyzed["sentiment"].value_counts())
    all_themes = [t for row in analyzed["themes"] for t in row]
    print(pd.Series(all_themes).value_counts())
