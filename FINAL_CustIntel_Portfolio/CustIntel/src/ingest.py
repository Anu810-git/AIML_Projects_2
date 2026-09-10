"""
Clean CSV/JSON data and load it into the normalized SQLite star schema.

This is intentionally written with small functions so a beginner can follow
the complete data pipeline.
"""

import json
from pathlib import Path
import pandas as pd

try:
    from config import RAW_DIR, DB_PATH
    from database import get_connection, initialize_database
except ImportError:
    from src.config import RAW_DIR, DB_PATH
    from src.database import get_connection, initialize_database


REQUIRED_FILES = [
    "customers.csv",
    "products.csv",
    "orders.csv",
    "reviews.csv",
    "browsing_events.csv",
]


def clean_customers(df: pd.DataFrame) -> pd.DataFrame:
    """Fill missing text and standardize customer columns."""
    df = df.copy()
    for column in ["customer_city", "customer_state"]:
        df[column] = df[column].fillna("Unknown").astype(str).str.strip()
    df["signup_date"] = pd.to_datetime(df["signup_date"], errors="coerce")
    df = df.dropna(subset=["customer_id", "signup_date"])
    df["signup_date"] = df["signup_date"].dt.strftime("%Y-%m-%d")
    return df


def clean_products(df: pd.DataFrame) -> pd.DataFrame:
    """Clean product dimensions and category names."""
    df = df.copy()
    df["product_category"] = (
        df["product_category"].fillna("unknown").astype(str).str.strip()
    )
    numeric_cols = [
        "product_weight_g", "product_length_cm",
        "product_height_cm", "product_width_cm"
    ]
    for column in numeric_cols:
        df[column] = pd.to_numeric(df[column], errors="coerce").fillna(
            df[column].median()
        )
    return df.dropna(subset=["product_id"])


def clean_orders(df: pd.DataFrame) -> pd.DataFrame:
    """Validate order dates and numeric transaction values."""
    df = df.copy()

    df["order_status"] = df["order_status"].fillna("canceled").str.lower().str.strip()
    valid_statuses = {"delivered", "shipped", "canceled"}
    df.loc[~df["order_status"].isin(valid_statuses), "order_status"] = "canceled"

    df["order_purchase_date"] = pd.to_datetime(
        df["order_purchase_date"], errors="coerce"
    )

    for column in ["price", "freight_value", "quantity"]:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df["price"] = df["price"].fillna(0).clip(lower=0)
    df["freight_value"] = df["freight_value"].fillna(0).clip(lower=0)
    df["quantity"] = df["quantity"].fillna(1).round().clip(lower=1).astype(int)

    df = df.dropna(subset=["order_id", "customer_id", "product_id", "order_purchase_date"])
    df["order_purchase_date"] = df["order_purchase_date"].dt.strftime("%Y-%m-%d")
    return df


def clean_reviews(df: pd.DataFrame) -> pd.DataFrame:
    """Clean review scores and text."""
    df = df.copy()
    df["review_score"] = pd.to_numeric(df["review_score"], errors="coerce")
    df["review_score"] = df["review_score"].fillna(3).round().clip(1, 5).astype(int)
    df["review_comment_message"] = (
        df["review_comment_message"].fillna("No written comment.").astype(str).str.strip()
    )
    df["review_date"] = pd.to_datetime(df["review_date"], errors="coerce")
    df = df.dropna(subset=["review_id", "order_id", "review_date"])
    df["review_date"] = df["review_date"].dt.strftime("%Y-%m-%d")
    return df


def clean_browsing(df: pd.DataFrame) -> pd.DataFrame:
    """Clean browsing event data."""
    df = df.copy()
    df["product_category"] = (
        df["product_category"].fillna("unknown").astype(str).str.strip()
    )
    df["event_time"] = pd.to_datetime(df["event_time"], errors="coerce")
    df = df.dropna(subset=["event_id", "customer_id", "event_time"])
    df["event_time"] = df["event_time"].astype(str)
    return df


def parse_nested_json(path: Path) -> pd.DataFrame:
    """Parse the mock API JSON and flatten one item per order."""
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not parse JSON file: {exc}") from exc

    rows = []
    for record in raw:
        try:
            item = record["items"][0]
            rows.append({
                "order_id": record["order_id"],
                "customer_id": record["customer"]["customer_id"],
                "product_id": item["product_id"],
                "order_status": record["status"],
                "order_purchase_date": record["purchase_date"],
                "price": item["price"],
                "freight_value": item["freight_value"],
                "quantity": item["quantity"],
            })
        except (KeyError, IndexError, TypeError) as exc:
            # Bad API records are caught here instead of reaching SQL.
            print(f"Skipping malformed JSON record: {exc}")

    return pd.DataFrame(rows)


def load_all() -> None:
    """Run the full ingestion pipeline inside SQL transactions."""
    missing = [name for name in REQUIRED_FILES if not (RAW_DIR / name).exists()]
    if missing:
        raise FileNotFoundError(
            "Missing raw files. Run generate_demo_data.py first. "
            f"Missing: {missing}"
        )

    customers = clean_customers(pd.read_csv(RAW_DIR / "customers.csv"))
    products = clean_products(pd.read_csv(RAW_DIR / "products.csv"))
    orders = clean_orders(pd.read_csv(RAW_DIR / "orders.csv"))
    reviews = clean_reviews(pd.read_csv(RAW_DIR / "reviews.csv"))
    browsing = clean_browsing(pd.read_csv(RAW_DIR / "browsing_events.csv"))

    # Demonstrate JSON ingestion as requested by the project brief.
    api_orders = parse_nested_json(RAW_DIR / "mock_ecommerce_api.json")
    print(f"Parsed {len(api_orders)} records from mock JSON API.")

    initialize_database()

    with get_connection() as conn:
        # Explicit transaction: either all inserts succeed or the transaction
        # can be rolled back by SQLite if an error occurs.
        try:
            customers.to_sql("dim_customers", conn, if_exists="append", index=False)
            products.to_sql("dim_products", conn, if_exists="append", index=False)
            orders.to_sql("fact_orders", conn, if_exists="append", index=False)
            reviews.to_sql("dim_reviews", conn, if_exists="append", index=False)
            browsing.to_sql("fact_browsing", conn, if_exists="append", index=False)
            conn.commit()
        except Exception:
            conn.rollback()
            raise

    print(f"SQLite database created at: {DB_PATH}")
    print("Rows loaded:")
    print(f"  Customers: {len(customers):,}")
    print(f"  Products:  {len(products):,}")
    print(f"  Orders:    {len(orders):,}")
    print(f"  Reviews:   {len(reviews):,}")
    print(f"  Browsing:  {len(browsing):,}")


if __name__ == "__main__":
    load_all()
