"""
CustIntel business chatbot.

This is an *agentic* chatbot: instead of pattern-matching a fixed list of
questions, it hands the question to an LLM (Google Gemini) that can call a
`run_sql` tool to query the live SQLite database and read the precomputed
model reports (LTV/churn metrics, customer features). This lets it answer
open-ended business questions correctly and specifically, the way a
ChatGPT-style analyst agent would -- not just the four canned questions the
old rule-based version understood.

Setup
-----
    pip install google-genai python-dotenv
    Put this in your .env file:
        GOOGLE_API_KEY=your-key-here

If no API key is configured (or the `google-genai` package isn't installed),
`answer_question` transparently falls back to a smaller set of
statistics-based answers computed directly from the database, so the
Streamlit app never crashes -- it just loses the "ask anything" ability.

Public API
----------
    answer_question(question: str, history: list[dict] | None = None) -> str
"""

from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List, Optional

import pandas as pd
from dotenv import load_dotenv

from config import DB_PATH, REPORT_DIR
from database import query_dataframe

load_dotenv()

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODEL = "gemini-3.6-flash"         # per Google's API: gemini-2.5-flash was retired, use this instead
MAX_TOOL_TURNS = 6                 # how many SQL round-trips the agent may take per question
MAX_ROWS_RETURNED = 200            # cap on rows sent back to the model per query
MAX_TOKENS = 1024

_SELECT_ONLY = re.compile(r"^\s*select\b", re.IGNORECASE)
_FORBIDDEN = re.compile(
    r"\b(insert|update|delete|drop|alter|attach|detach|pragma|vacuum|create)\b",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Grounding context: live schema + precomputed reports
# ---------------------------------------------------------------------------

def _get_live_schema() -> str:
    """Introspect the actual SQLite schema so the agent's knowledge of the
    database never drifts out of sync with reality."""
    try:
        tables = query_dataframe(
            "SELECT name, sql FROM sqlite_master "
            "WHERE type='table' AND name NOT LIKE 'sqlite_%'"
        )
    except Exception as exc:  # pragma: no cover - defensive
        return f"(schema unavailable: {exc})"

    parts = []
    for _, row in tables.iterrows():
        parts.append(row["sql"])
    return "\n\n".join(parts) if parts else "(no tables found)"


def _load_report_context() -> str:
    """Summarize the precomputed model reports (metrics + feature table) so
    the agent can answer model-performance questions without re-querying."""
    chunks = []

    metrics_path = REPORT_DIR / "model_metrics.json"
    if metrics_path.exists():
        try:
            metrics = json.loads(metrics_path.read_text())
            chunks.append("model_metrics.json:\n" + json.dumps(metrics, indent=2))
        except Exception as exc:  # pragma: no cover - defensive
            chunks.append(f"model_metrics.json unavailable: {exc}")

    dl_history_path = REPORT_DIR / "dl_history.json"
    if dl_history_path.exists():
        try:
            dl_history = json.loads(dl_history_path.read_text())
            chunks.append("dl_history.json:\n" + json.dumps(dl_history, indent=2))
        except Exception as exc:  # pragma: no cover - defensive
            chunks.append(f"dl_history.json unavailable: {exc}")

    features_path = REPORT_DIR / "customer_features.csv"
    if features_path.exists():
        try:
            features = pd.read_csv(features_path)
            desc = features.describe(include="all").round(2).to_string()
            chunks.append(
                f"customer_features.csv has {len(features)} rows, "
                f"columns: {list(features.columns)}\nSummary stats:\n{desc}"
            )
        except Exception as exc:  # pragma: no cover - defensive
            chunks.append(f"customer_features.csv unavailable: {exc}")

    return "\n\n".join(chunks) if chunks else "(no precomputed reports found)"


def _build_system_prompt() -> str:
    return f"""You are the CustIntel Business Analyst, an assistant embedded in an
e-commerce analytics dashboard. You answer any question a business user asks
about the company's customers, orders, revenue, reviews, churn, LTV, or the
trained models -- not just a fixed set of pre-written questions.

You have a `run_sql` tool that runs a single read-only SELECT statement
against the live SQLite database and returns the result as JSON rows. Use it
whenever you need a number, a ranking, a trend, or any fact you are not
already certain of. Never invent numbers -- run a query instead. You may
call the tool more than once (e.g. to explore a table before writing the
real query).

Database schema (live, from sqlite_master):
{_get_live_schema()}

Precomputed reports available (already-trained model metrics and customer
feature summary -- you can cite these directly without a query):
{_load_report_context()}

Guidelines:
- This is a demo/synthetic dataset, not production data -- you can mention
  that if relevant, but don't be repetitive about it.
- Ground every specific number or ranking in a `run_sql` result or in the
  precomputed reports above.
- If a question is ambiguous, make a reasonable assumption, state it in one
  short clause, and answer -- don't just ask for clarification.
- If a question is entirely unrelated to this business/dataset (e.g. general
  knowledge), answer helpfully from general knowledge, but say briefly that
  it's outside the dashboard's data.
- Keep answers concise and business-friendly: lead with the answer, then a
  short supporting detail or two. Use a small markdown table only when a
  ranked list of rows genuinely helps.
- Never write or suggest INSERT/UPDATE/DELETE/DROP/ALTER/PRAGMA statements.
"""


_RUN_SQL_SCHEMA = {
    "type": "object",
    "properties": {
        "sql": {
            "type": "string",
            "description": "A single SELECT statement.",
        }
    },
    "required": ["sql"],
}


# ---------------------------------------------------------------------------
# Tool execution
# ---------------------------------------------------------------------------

def _run_readonly_sql(sql: str) -> str:
    sql = sql.strip().rstrip(";")

    if not _SELECT_ONLY.match(sql):
        return json.dumps({"error": "Only SELECT statements are allowed."})
    if _FORBIDDEN.search(sql):
        return json.dumps({"error": "Query contains a forbidden keyword."})
    if ";" in sql:
        return json.dumps({"error": "Only a single statement is allowed."})

    try:
        df = query_dataframe(sql)
    except Exception as exc:
        return json.dumps({"error": f"Query failed: {exc}"})

    truncated = len(df) > MAX_ROWS_RETURNED
    if truncated:
        df = df.head(MAX_ROWS_RETURNED)

    result = {
        "row_count": len(df),
        "truncated": truncated,
        "rows": json.loads(df.to_json(orient="records")),
    }
    return json.dumps(result, default=str)


def _dispatch_tool(name: str, tool_input: Dict[str, Any]) -> str:
    if name == "run_sql":
        return _run_readonly_sql(tool_input.get("sql", ""))
    return json.dumps({"error": f"Unknown tool '{name}'"})


# ---------------------------------------------------------------------------
# Offline fallback (no API key / package configured)
# ---------------------------------------------------------------------------

def _fallback_answer(question: str) -> str:
    q = question.lower()

    try:
        if "churn" in q:
            churn = query_dataframe(
                "SELECT AVG(CASE WHEN days_since_last_purchase > 180 THEN 1.0 ELSE 0.0 END) "
                "AS approx_churn_rate FROM (SELECT customer_id, "
                "MAX(julianday('now') - julianday(order_date)) AS days_since_last_purchase "
                "FROM fact_orders GROUP BY customer_id)"
            )
            rate = churn.iloc[0]["approx_churn_rate"]
            return (
                f"Roughly {rate:.1%} of customers haven't ordered in 180+ days "
                "(offline estimate -- connect a GOOGLE_API_KEY for a full "
                "data-grounded churn analysis, including model-based churn "
                "probabilities per customer)."
            )

        if "review" in q or "sentiment" in q:
            reviews = query_dataframe("SELECT AVG(review_score) AS avg_score FROM dim_reviews")
            return f"The average review score is {reviews.iloc[0]['avg_score']:.2f}/5."

        if "categor" in q or "sales" in q or "revenue" in q:
            cats = query_dataframe(
                "SELECT p.product_category, "
                "ROUND(SUM(o.price * o.quantity + o.freight_value), 2) AS revenue "
                "FROM fact_orders o JOIN dim_products p ON o.product_id = p.product_id "
                "WHERE o.order_status = 'delivered' "
                "GROUP BY p.product_category ORDER BY revenue DESC LIMIT 5"
            )
            lines = "\n".join(
                f"{i+1}. {r.product_category}: {r.revenue:,.0f}"
                for i, r in cats.iterrows()
            )
            return f"Top categories by revenue:\n{lines}"

        if "ltv" in q or "lifetime value" in q:
            features_path = REPORT_DIR / "customer_features.csv"
            if features_path.exists():
                features = pd.read_csv(features_path)
                return (
                    f"Historical average spend per customer is "
                    f"{features['total_spend'].mean():,.2f}. Open the "
                    "'Customer Intelligence' tab for model-predicted 12-month LTV "
                    "per customer."
                )

        return (
            "I can only answer a few basic questions right now (churn, reviews, "
            "sales by category, LTV) because no GOOGLE_API_KEY is configured. "
            "Set that environment variable and install the `google-genai` "
            "package to unlock the full ask-anything business chatbot."
        )
    except Exception as exc:  # pragma: no cover - defensive
        return f"I couldn't compute that: {exc}"


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def answer_question(question: str, history: Optional[List[Dict[str, str]]] = None) -> str:
    """Answer any business question about the CustIntel dataset.

    Parameters
    ----------
    question : str
        The user's latest message.
    history : list of {"role": "user"|"assistant", "content": str}, optional
        Prior turns in the conversation, oldest first, NOT including the
        current `question`. Passing this lets the agent handle follow-ups
        like "and last quarter?" correctly.
    """
    try:
        from google import genai
        from google.genai import types
    except ImportError:
        return _fallback_answer(question)

    api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
    if not api_key:
        return _fallback_answer(question)

    client = genai.Client(api_key=api_key)

    tool = types.Tool(
        function_declarations=[
            types.FunctionDeclaration(
                name="run_sql",
                description=(
                    "Run a single read-only SQL SELECT statement against the "
                    "CustIntel SQLite database and return the resulting rows as "
                    "JSON. Only SELECT statements are permitted."
                ),
                parameters=_RUN_SQL_SCHEMA,
            )
        ]
    )

    config = types.GenerateContentConfig(
        system_instruction=_build_system_prompt(),
        tools=[tool],
        max_output_tokens=MAX_TOKENS,
    )

    # Build the conversation: prior history (role names translated to
    # Gemini's "user" / "model") plus the new question.
    contents: List[Any] = []
    for turn in history or []:
        role = "model" if turn.get("role") == "assistant" else "user"
        contents.append(
            types.Content(role=role, parts=[types.Part(text=turn.get("content", ""))])
        )
    contents.append(types.Content(role="user", parts=[types.Part(text=question)]))

    try:
        for _ in range(MAX_TOOL_TURNS):
            response = client.models.generate_content(
                model=MODEL,
                contents=contents,
                config=config,
            )

            candidate = response.candidates[0]
            parts = candidate.content.parts or []

            function_calls = [p.function_call for p in parts if getattr(p, "function_call", None)]

            if not function_calls:
                text_parts = [p.text for p in parts if getattr(p, "text", None)]
                return "\n".join(text_parts).strip() or "I don't have an answer for that."

            # Record the model's turn (including its function call) in the
            # conversation, then run each requested tool and feed the
            # results back as the next turn.
            contents.append(candidate.content)

            response_parts = []
            for fc in function_calls:
                result_text = _dispatch_tool(fc.name, dict(fc.args or {}))
                response_parts.append(
                    types.Part.from_function_response(
                        name=fc.name,
                        response={"result": result_text},
                    )
                )
            contents.append(types.Content(role="user", parts=response_parts))

        return (
            "That question needed more exploration than I could finish -- "
            "try asking something more specific."
        )
    except Exception as exc:  # pragma: no cover - defensive
        return f"The AI chatbot hit an error ({exc}). Falling back to basic stats.\n\n" + _fallback_answer(
            question
        )
