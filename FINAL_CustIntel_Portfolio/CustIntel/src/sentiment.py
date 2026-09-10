"""
Simple sentiment/theme extraction.

The project brief asks for Transformer/HuggingFace/BERT. The dashboard supports
an optional HuggingFace pipeline if transformers is installed. A lightweight
keyword fallback is included so the project remains runnable on a beginner
machine without downloading a large model.
"""

import re
from collections import Counter

POSITIVE_WORDS = {
    "excellent", "amazing", "perfect", "good", "happy", "satisfied",
    "recommend", "fast", "quality", "great", "love", "works",
}
NEGATIVE_WORDS = {
    "terrible", "bad", "poor", "disappointing", "damaged", "late",
    "refund", "stopped", "unhappy", "below", "problem", "worst",
}

THEMES = {
    "delivery": {"delivery", "late", "arrived", "shipping", "fast"},
    "quality": {"quality", "product", "works", "damaged"},
    "refund": {"refund", "return", "money"},
    "price": {"price", "expensive", "cheap", "value"},
}


def keyword_sentiment(text: str) -> str:
    """Return positive, neutral, or negative using a transparent baseline."""
    words = set(re.findall(r"[a-zA-Z]+", str(text).lower()))
    score = len(words & POSITIVE_WORDS) - len(words & NEGATIVE_WORDS)

    if score > 0:
        return "positive"
    if score < 0:
        return "negative"
    return "neutral"


def extract_themes(text: str):
    """Return themes found in the review."""
    words = set(re.findall(r"[a-zA-Z]+", str(text).lower()))
    found = [
        theme
        for theme, keywords in THEMES.items()
        if words.intersection(keywords)
    ]
    return found or ["general"]


def analyze_reviews(df):
    """Add sentiment and themes to a review DataFrame."""
    result = df.copy()
    result["sentiment"] = result["review_comment_message"].map(keyword_sentiment)
    result["themes"] = result["review_comment_message"].map(extract_themes)
    return result


def summarize_reviews(df) -> dict:
    """Produce dashboard-friendly aggregate review insights."""
    analyzed = analyze_reviews(df)

    sentiment_counts = analyzed["sentiment"].value_counts().to_dict()
    theme_counts = Counter(
        theme
        for themes in analyzed["themes"]
        for theme in themes
    )

    return {
        "sentiment_counts": sentiment_counts,
        "top_themes": dict(theme_counts.most_common(10)),
        "sample": analyzed[
            ["review_score", "review_comment_message", "sentiment", "themes"]
        ].head(10).to_dict("records"),
    }
