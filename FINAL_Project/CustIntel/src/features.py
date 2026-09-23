"""Feature engineering functions shared by ML and Streamlit."""

import numpy as np
import pandas as pd

try:
    from database import query_dataframe
except ImportError:
    from src.database import query_dataframe


LTV_FEATURES = [
    "order_count",
    "total_spend",
    "avg_order_value",
    "avg_freight",
    "avg_review_score",
    "days_since_last_purchase",
]

CHURN_FEATURES = [
    "order_count",
    "total_spend",
    "avg_order_value",
    "avg_review_score",
    "days_since_last_purchase",
]


def build_customer_features() -> pd.DataFrame:
    """
    Build customer-level historical features and future targets.

    Observation period:
        orders purchased on or before 2024-06-30

    Future period:
        orders purchased after 2024-06-30

    This avoids using future information as input features.
    """
    sql = """
    WITH history AS (
        SELECT
            o.customer_id,
            COUNT(*) AS order_count,
            SUM(o.price * o.quantity + o.freight_value) AS total_spend,
            AVG(o.price * o.quantity + o.freight_value) AS avg_order_value,
            AVG(o.freight_value) AS avg_freight,
            MAX(o.order_purchase_date) AS last_purchase_date
        FROM fact_orders o
        WHERE o.order_purchase_date <= '2024-06-30'
          AND o.order_status = 'delivered'
        GROUP BY o.customer_id
    ),
    review_features AS (
        SELECT
            o.customer_id,
            AVG(r.review_score) AS avg_review_score
        FROM fact_orders o
        JOIN dim_reviews r ON o.order_id = r.order_id
        WHERE o.order_purchase_date <= '2024-06-30'
        GROUP BY o.customer_id
    ),
    future AS (
        SELECT
            customer_id,
            SUM(price * quantity + freight_value) AS future_12m_spend,
            COUNT(*) AS future_orders
        FROM fact_orders
        WHERE order_purchase_date > '2024-06-30'
          AND order_purchase_date <= '2025-06-30'
          AND order_status = 'delivered'
        GROUP BY customer_id
    )
    SELECT
        c.customer_id,
        h.order_count,
        h.total_spend,
        h.avg_order_value,
        h.avg_freight,
        COALESCE(r.avg_review_score, 3.0) AS avg_review_score,
        CAST(julianday('2024-06-30') - julianday(h.last_purchase_date) AS INTEGER)
            AS days_since_last_purchase,
        COALESCE(f.future_12m_spend, 0.0) AS future_12m_spend,
        COALESCE(f.future_orders, 0) AS future_orders
    FROM dim_customers c
    JOIN history h ON c.customer_id = h.customer_id
    LEFT JOIN review_features r ON c.customer_id = r.customer_id
    LEFT JOIN future f ON c.customer_id = f.customer_id
    """

    df = query_dataframe(sql)

    # Churn is defined as no future delivered order during the next 12 months.
    df["will_churn"] = (df["future_orders"] == 0).astype(int)

    # A log transform reduces the effect of very large spenders.
    df["log_total_spend"] = np.log1p(df["total_spend"])

    return df
