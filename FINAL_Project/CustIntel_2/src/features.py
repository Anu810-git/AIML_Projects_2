"""
Turn the order history into one row per customer: recency/frequency/monetary
features, plus the two things we want to predict:

    future_spend_180d  -- money spent in the 180 days AFTER the cutoff date (the LTV target)
    will_churn          -- 1 if the customer did NOT buy again in those 180 days

To avoid leakage, every feature is computed using only orders up to a single
CUTOFF_DATE, and the targets are computed only from orders after it. The
cutoff is chosen so both "before" and "after" windows have real data.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

_SRC = str(Path(__file__).resolve().parent)
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from config import HORIZON_DAYS, REPORT_DIR
from database import query_dataframe


def build_customer_features(cutoff_date: str | None = None) -> pd.DataFrame:
    """Build the customer_features table and save it to reports/customer_features.csv."""
    orders = query_dataframe("""
        SELECT customer_id, order_status, order_purchase_timestamp, payment_value, review_score
        FROM fact_orders
        WHERE order_status = 'delivered'
    """)
    orders["order_purchase_timestamp"] = pd.to_datetime(orders["order_purchase_timestamp"])

    customers = query_dataframe("SELECT customer_id, customer_unique_id FROM dim_customers")
    orders = orders.merge(customers, on="customer_id", how="left")

    max_date = orders["order_purchase_timestamp"].max()
    if cutoff_date is None:
        # Leave one full HORIZON_DAYS window after the cutoff to compute the targets.
        cutoff = max_date - pd.Timedelta(days=HORIZON_DAYS)
    else:
        cutoff = pd.Timestamp(cutoff_date)

    before = orders[orders["order_purchase_timestamp"] <= cutoff]
    after = orders[
        (orders["order_purchase_timestamp"] > cutoff)
        & (orders["order_purchase_timestamp"] <= cutoff + pd.Timedelta(days=HORIZON_DAYS))
    ]

    grouped = before.groupby("customer_unique_id")
    features = grouped.agg(
        order_count=("order_purchase_timestamp", "count"),
        total_spend=("payment_value", "sum"),
        avg_order_value=("payment_value", "mean"),
        avg_review_score=("review_score", "mean"),
        first_purchase=("order_purchase_timestamp", "min"),
        last_purchase=("order_purchase_timestamp", "max"),
    ).reset_index()

    features["days_since_last_purchase"] = (cutoff - features["last_purchase"]).dt.days
    features["customer_tenure_days"] = (cutoff - features["first_purchase"]).dt.days
    features["avg_review_score"] = features["avg_review_score"].fillna(features["avg_review_score"].mean())

    future_spend = after.groupby("customer_unique_id")["payment_value"].sum().rename("future_spend_180d")
    features = features.merge(future_spend, on="customer_unique_id", how="left")
    features["future_spend_180d"] = features["future_spend_180d"].fillna(0.0)
    features["will_churn"] = (features["future_spend_180d"] <= 0).astype(int)

    features = features.drop(columns=["first_purchase", "last_purchase"])
    features = features.rename(columns={"customer_unique_id": "customer_id"})

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    features.to_csv(REPORT_DIR / "customer_features.csv", index=False)
    return features


if __name__ == "__main__":
    result = build_customer_features()
    print(f"Built features for {len(result)} customers.")
    print("Churn rate:", round(result["will_churn"].mean(), 3))
    print(result.head())
