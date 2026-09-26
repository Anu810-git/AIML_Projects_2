"""
The "ask anything" business chatbot.

This is an AI *agent*, not a fixed list of canned answers: it is given the
database schema and told to write its own read-only SQL to answer whatever
the person asks, using Gemini's function-calling. If no Google API key is
configured it falls back to a small set of hard-coded, keyword-matched
answers so the rest of the app still works offline.

Setup for the full agent:
    pip install google-genai
    export GOOGLE_API_KEY="your-key"      # or GEMINI_API_KEY
    (or put GOOGLE_API_KEY=... in a .env file in the project root)
"""

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

_SRC = str(Path(__file__).resolve().parent)
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from config import DB_PATH, REPORT_DIR
from database import query_dataframe

load_dotenv()

# ---------------------------------------------------------------------------
# Configuration

# Google retires Gemini model names over time, and its flash-tier models have
# also had well-documented, Google-side capacity crunches (503 "high demand")
# that can last a while. So instead of one fixed model, try a short list in
# order and fall back to the next one if a model is genuinely unavailable.
# If you hit a 404 "no longer available" error, check
# https://ai.google.dev/gemini-api/docs/models for current names and update
# this list.
_MODEL_CANDIDATES = ["gemini-3.8-flash", "gemini-2.5-flash", "gemini-3.1-flash-lite"]

_SCHEMA_DESCRIPTION = """
SQLite database with these tables (read-only):

fact_orders(order_id, customer_id, order_status, order_purchase_timestamp,
            order_delivered_customer_date, order_estimated_delivery_date,
            item_count, price, freight_value, payment_value, review_score)
    -- one row per ORDER. order_status is usually 'delivered'; filter on it
    -- for revenue questions unless asked otherwise.

fact_order_items(order_id, order_item_id, product_id, seller_id, customer_id,
                  order_status, order_purchase_timestamp, price, freight_value)
    -- one row per ITEM bought (an order can have several items/products).

dim_customers(customer_id, customer_unique_id, customer_city, customer_state)
    -- customer_id is per-order; customer_unique_id identifies the actual person.

dim_products(product_id, product_category, product_weight_g, product_length_cm,
             product_height_cm, product_width_cm)
    -- product_category is already translated to English.

dim_sellers(seller_id, seller_city, seller_state)

dim_reviews(review_id, order_id, review_score, review_comment_title,
            review_comment_message, review_creation_date)
    -- review_score is 1-5. review_comment_message is in Portuguese and is
    -- often blank.

dim_state_geo(state, lat, lng)
    -- one row per Brazilian state, median coordinates (for maps only).

Model reports (already computed, not in the database):
    reports/model_metrics.json -- LTV (regression) and churn (classification) scores
    reports/dl_history.json    -- next-purchase-category model scores
    reports/customer_features.csv -- one row per customer: order_count, total_spend,
        avg_order_value, avg_review_score, days_since_last_purchase,
        customer_tenure_days, future_spend_180d, will_churn
"""

_SYSTEM_INSTRUCTION = f"""
You are CustIntel's business analyst chatbot for a Brazilian e-commerce
marketplace (the Olist dataset). Prices are in Brazilian Real (R$).

{_SCHEMA_DESCRIPTION}

Rules:
- You do NOT know any specific numbers, rankings, totals or category names in
  this database from memory. For every question that involves a number, a
  ranking ("highest", "lowest", "top", "most", "least"), a comparison, or any
  other specific fact, you MUST call `run_sql_query` (or `read_model_report`
  for the model reports) and base your answer only on what it returns.
  Guessing an answer instead of calling the tool is a serious error.
- Only ever write SELECT statements. Never INSERT/UPDATE/DELETE/DROP.
- Prefer aggregated queries (COUNT, SUM, AVG, GROUP BY) over dumping raw rows.
- If a query fails, read the error and try a corrected query once.
- Keep the final answer short and specific; do not restate the SQL to the user.
"""


def _run_sql_query(sql: str) -> str:
    """Run a read-only SQL SELECT against the CustIntel database and return the result as text."""
    cleaned = sql.strip().rstrip(";")
    if not cleaned.lower().startswith(("select", "with")):
        return "Error: only SELECT queries are allowed."
    try:
        result = query_dataframe(cleaned)
    except Exception as exc:  # surfaced back to the model so it can retry
        return f"Error running query: {exc}"
    if result.empty:
        return "Query ran successfully but returned no rows."
    return result.head(30).to_string(index=False)


def _read_model_report(report_name: str) -> str:
    """Read one of the precomputed report files: 'model_metrics', 'dl_history', or 'customer_features'."""
    files = {
        "model_metrics": REPORT_DIR / "model_metrics.json",
        "dl_history": REPORT_DIR / "dl_history.json",
        "customer_features": REPORT_DIR / "customer_features.csv",
    }
    path = files.get(report_name)
    if path is None:
        return f"Unknown report '{report_name}'. Valid options: {list(files)}"
    if not path.exists():
        return f"Report file {path.name} does not exist yet. Run the training scripts first."
    if path.suffix == ".csv":
        import pandas as pd
        return pd.read_csv(path).describe().to_string()
    return path.read_text()


def _has_api_key() -> bool:
    return bool(os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY"))


def _offline_answer(question: str) -> str:
    """A small, keyword-matched fallback used only when no API key is configured."""
    q = question.lower()
    try:
        if "churn" in q:
            metrics = _read_model_report("model_metrics")
            return f"(Offline mode) Churn model report:\n{metrics}"
        if "review" in q or "sentiment" in q:
            avg = query_dataframe("SELECT AVG(review_score) AS avg_score FROM dim_reviews")
            return f"(Offline mode) Average review score: {avg.iloc[0]['avg_score']:.2f} / 5."
        if "revenue" in q or "sales" in q or "category" in q:
            top = query_dataframe("""
                SELECT p.product_category, ROUND(SUM(o.price), 2) AS revenue
                FROM fact_order_items o JOIN dim_products p ON o.product_id = p.product_id
                WHERE o.order_status = 'delivered'
                GROUP BY p.product_category ORDER BY revenue DESC LIMIT 5
            """)
            return "(Offline mode) Top 5 categories by revenue:\n" + top.to_string(index=False)
        if "ltv" in q or "lifetime" in q:
            metrics = _read_model_report("model_metrics")
            return f"(Offline mode) LTV model report:\n{metrics}"
    except Exception as exc:
        return f"(Offline mode) Could not answer: {exc}"
    return (
        "(Offline mode) I can only answer basic churn / reviews / sales / LTV "
        "questions without a GOOGLE_API_KEY. Set that environment variable and "
        "`pip install google-genai` to enable the full ask-anything AI agent."
    )


def answer_question(question: str, history: list[dict] | None = None) -> str:
    """Answer a business question. Uses the full AI agent if a Google API key is set,
    otherwise falls back to a small offline mode."""
    if not _has_api_key():
        return _offline_answer(question)

    import time

    from google import genai
    from google.genai import types

    client = genai.Client()

    contents = []
    for turn in (history or []):
        role = "model" if turn["role"] == "assistant" else "user"
        contents.append(types.Content(role=role, parts=[types.Part.from_text(text=turn["content"])]))
    contents.append(types.Content(role="user", parts=[types.Part.from_text(text=question)]))

    config = types.GenerateContentConfig(
        system_instruction=_SYSTEM_INSTRUCTION,
        tools=[_run_sql_query, _read_model_report],
        temperature=0.0,
    )

    # Google's flash-tier models occasionally hit a 503 ("high demand") or 429
    # (rate limit) that can persist for a while on Google's side. Rather than
    # just retrying the same model, work through _MODEL_CANDIDATES in order --
    # a short retry on each, then move on to the next model.
    last_error: Exception | None = None
    for model_name in _MODEL_CANDIDATES:
        for attempt in range(2):
            try:
                response = client.models.generate_content(model=model_name, contents=contents, config=config)
                return response.text or "(The model returned an empty response.)"
            except Exception as exc:
                last_error = exc
                message = str(exc)
                is_transient = (
                    "503" in message or "UNAVAILABLE" in message
                    or "429" in message or "RESOURCE_EXHAUSTED" in message
                )
                if is_transient and attempt == 0:
                    time.sleep(1.5)
                    continue
                break  # not transient, or already retried once: try the next model

    if last_error is not None and ("503" in str(last_error) or "UNAVAILABLE" in str(last_error)):
        return (
            "Google's Gemini models are currently overloaded on their end (a 503 'high demand' "
            f"error), and this happened across all of {', '.join(_MODEL_CANDIDATES)}. This is a "
            "known, ongoing capacity issue on Google's side, not something wrong with this app -- "
            "please wait a bit and try again."
        )
    return f"Sorry, the AI agent hit an error: {last_error}"


if __name__ == "__main__":
    print(answer_question("What are the top 3 product categories by revenue?"))
