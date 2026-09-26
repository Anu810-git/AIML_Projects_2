"""
Turn the 9 raw Olist CSV files into a clean SQLite database.

Star schema:
    fact_order_items  -- one row per item bought (grain of the raw data)
    fact_orders       -- one row per order (amounts rolled up from the items)
    dim_customers
    dim_products      -- category name translated to English
    dim_sellers
    dim_reviews       -- one row per order (duplicates/blanks removed)

Run directly:
    python src/ingest.py
"""

import sys
from pathlib import Path

import pandas as pd

_SRC = str(Path(__file__).resolve().parent)
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from config import OLIST_FILES, RAW_DIR, REPORT_DIR
from database import get_connection

# Category names in the raw file that are missing from the official
# translation table (checked by hand against the Kaggle dataset).
_MISSING_CATEGORY_TRANSLATION = {
    "pc_gamer": "pc_gamer",
    "portateis_cozinha_e_preparadores_de_alimentos": "kitchen_portables_and_preparers",
}


def _read(name: str) -> pd.DataFrame:
    path = RAW_DIR / OLIST_FILES[name]
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path.name}. Download the 9 CSVs from "
            "https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce "
            f"and place them in {RAW_DIR}"
        )
    # category_translation.csv is saved with a UTF-8 BOM; utf-8-sig handles that
    # and is harmless for the other files.
    return pd.read_csv(path, encoding="utf-8-sig")


def _check_raw_files() -> list[str]:
    """Return the list of missing raw file names (empty list = all present)."""
    return [name for name in OLIST_FILES.values() if not (RAW_DIR / name).exists()]


def load_all() -> dict:
    """Clean the 9 CSVs and write the star schema to SQLite. Returns a data-quality report."""
    quality: dict = {}

    customers = _read("customers")
    geolocation = _read("geolocation")
    order_items = _read("order_items")
    payments = _read("payments")
    reviews = _read("reviews")
    orders = _read("orders")
    products = _read("products")
    sellers = _read("sellers")
    translation = _read("category_translation")

    date_cols = [
        "order_purchase_timestamp", "order_approved_at",
        "order_delivered_carrier_date", "order_delivered_customer_date",
        "order_estimated_delivery_date",
    ]
    for col in date_cols:
        orders[col] = pd.to_datetime(orders[col], errors="coerce")

    # ---- dim_products: fill missing categories, translate to English --------
    quality["products_missing_category"] = int(products["product_category_name"].isna().sum())
    products["product_category_name"] = products["product_category_name"].fillna("unknown")

    cat_map = dict(zip(translation["product_category_name"], translation["product_category_name_english"]))
    cat_map.update(_MISSING_CATEGORY_TRANSLATION)
    cat_map["unknown"] = "unknown"
    products["product_category"] = products["product_category_name"].map(cat_map).fillna(
        products["product_category_name"]
    )
    dim_products = products[[
        "product_id", "product_category", "product_weight_g",
        "product_length_cm", "product_height_cm", "product_width_cm",
    ]].copy()

    # ---- dim_customers -------------------------------------------------------
    dim_customers = customers[[
        "customer_id", "customer_unique_id", "customer_city", "customer_state",
    ]].copy()

    # ---- dim_sellers -----------------------------------------------------------
    dim_sellers = sellers[["seller_id", "seller_city", "seller_state"]].copy()

    # ---- dim_reviews: one row per order, keep the most recent if duplicated --
    quality["review_id_duplicates"] = int(reviews["review_id"].duplicated().sum())
    quality["orders_with_multiple_reviews"] = int(reviews["order_id"].duplicated().sum())
    reviews["review_answer_timestamp"] = pd.to_datetime(reviews["review_answer_timestamp"], errors="coerce")
    reviews = reviews.sort_values("review_answer_timestamp").drop_duplicates("order_id", keep="last")
    dim_reviews = reviews[[
        "review_id", "order_id", "review_score",
        "review_comment_title", "review_comment_message", "review_creation_date",
    ]].copy()

    # ---- fact_order_items: item grain, one row per product bought -----------
    quality["order_items_orphaned"] = int((~order_items["order_id"].isin(orders["order_id"])).sum())
    fact_order_items = order_items.merge(
        orders[["order_id", "customer_id", "order_status", "order_purchase_timestamp"]],
        on="order_id", how="left",
    )
    fact_order_items = fact_order_items[[
        "order_id", "order_item_id", "product_id", "seller_id", "customer_id",
        "order_status", "order_purchase_timestamp", "price", "freight_value",
    ]]

    # ---- fact_orders: order grain, amounts rolled up from items/payments ----
    item_rollup = order_items.groupby("order_id").agg(
        item_count=("order_item_id", "count"),
        price=("price", "sum"),
        freight_value=("freight_value", "sum"),
    ).reset_index()
    payment_rollup = payments.groupby("order_id").agg(payment_value=("payment_value", "sum")).reset_index()

    fact_orders = orders.merge(item_rollup, on="order_id", how="left")
    fact_orders = fact_orders.merge(payment_rollup, on="order_id", how="left")
    fact_orders = fact_orders.merge(dim_reviews[["order_id", "review_score"]], on="order_id", how="left")
    quality["orders_without_items"] = int(fact_orders["item_count"].isna().sum())
    fact_orders["item_count"] = fact_orders["item_count"].fillna(0)
    fact_orders["price"] = fact_orders["price"].fillna(0.0)
    fact_orders["freight_value"] = fact_orders["freight_value"].fillna(0.0)
    fact_orders["payment_value"] = fact_orders["payment_value"].fillna(fact_orders["price"] + fact_orders["freight_value"])
    fact_orders = fact_orders[[
        "order_id", "customer_id", "order_status", "order_purchase_timestamp",
        "order_delivered_customer_date", "order_estimated_delivery_date",
        "item_count", "price", "freight_value", "payment_value", "review_score",
    ]]

    quality["orders_total"] = int(len(orders))
    quality["orders_delivered"] = int((orders["order_status"] == "delivered").sum())
    quality["customers_total"] = int(customers["customer_id"].nunique())
    quality["unique_people"] = int(customers["customer_unique_id"].nunique())
    quality["date_range"] = [
        str(orders["order_purchase_timestamp"].min()),
        str(orders["order_purchase_timestamp"].max()),
    ]

    tables = {
        "dim_customers": dim_customers,
        "dim_products": dim_products,
        "dim_sellers": dim_sellers,
        "dim_reviews": dim_reviews,
        "fact_order_items": fact_order_items,
        "fact_orders": fact_orders,
    }

    connection = get_connection()
    try:
        for name, table in tables.items():
            table.to_sql(name, connection, if_exists="replace", index=False)
        connection.execute("CREATE INDEX IF NOT EXISTS idx_fo_customer ON fact_orders(customer_id)")
        connection.execute("CREATE INDEX IF NOT EXISTS idx_foi_order ON fact_order_items(order_id)")
        connection.execute("CREATE INDEX IF NOT EXISTS idx_foi_product ON fact_order_items(product_id)")
        connection.commit()
    finally:
        connection.close()

    # geolocation is large (1.1M rows) and only used for an optional map, so it
    # is kept as a small per-state summary table instead of loading it whole.
    geo_summary = geolocation.groupby("geolocation_state").agg(
        lat=("geolocation_lat", "median"), lng=("geolocation_lng", "median"),
    ).reset_index().rename(columns={"geolocation_state": "state"})
    connection = get_connection()
    try:
        geo_summary.to_sql("dim_state_geo", connection, if_exists="replace", index=False)
        connection.commit()
    finally:
        connection.close()

    import json
    (REPORT_DIR / "data_quality_report.json").write_text(json.dumps(quality, indent=2))
    return quality


if __name__ == "__main__":
    missing = _check_raw_files()
    if missing:
        raise SystemExit(
            "Missing raw files: " + ", ".join(missing) +
            f"\nDownload them from https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce and put them in {RAW_DIR}"
        )
    report = load_all()
    print("Database built. Data-quality report:")
    for key, value in report.items():
        print(f"  {key}: {value}")
