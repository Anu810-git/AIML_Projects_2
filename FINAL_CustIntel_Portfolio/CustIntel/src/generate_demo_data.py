"""
Generate a small, reproducible e-commerce dataset.

Why do we generate demo data?
The project brief mentions Olist or a similar open-source dataset, but the
brief itself does not contain the actual CSV files. This generator lets a
beginner run the entire project immediately.

The generated data intentionally contains relationships that a model can learn:
- customers with high historical spend tend to have higher future spend;
- long inactivity and poor reviews increase churn probability;
- browsing categories follow simple transition patterns.
"""

from pathlib import Path
import json
import numpy as np
import pandas as pd

try:
    from config import RAW_DIR, CATEGORIES, RANDOM_STATE
except ImportError:
    from src.config import RAW_DIR, CATEGORIES, RANDOM_STATE


def generate_data(
    n_customers: int = 1500,
    n_orders: int = 15000,
    n_browsing_events: int = 15000,
) -> None:
    """Create CSV and JSON files used by the rest of the project."""
    rng = np.random.default_rng(RANDOM_STATE)
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    # -------------------------
    # 1. Customers
    # -------------------------
    customer_ids = [f"CUST_{i:05d}" for i in range(1, n_customers + 1)]
    states = ["SP", "RJ", "MG", "PR", "RS", "BA", "SC", "PE"]
    cities = ["Sao Paulo", "Rio de Janeiro", "Belo Horizonte",
              "Curitiba", "Porto Alegre", "Salvador", "Florianopolis"]

    signup_start = pd.Timestamp("2022-01-01")
    signup_days = rng.integers(0, 500, size=n_customers)

    customers = pd.DataFrame({
        "customer_id": customer_ids,
        "customer_city": rng.choice(cities, n_customers),
        "customer_state": rng.choice(states, n_customers),
        "signup_date": (
            signup_start + pd.to_timedelta(signup_days, unit="D")
        ).strftime("%Y-%m-%d"),
    })

    # -------------------------
    # 2. Products
    # -------------------------
    product_ids = [f"PROD_{i:04d}" for i in range(1, 501)]
    products = pd.DataFrame({
        "product_id": product_ids,
        "product_category": rng.choice(
            CATEGORIES, size=len(product_ids),
            p=[.14, .12, .12, .10, .10, .10, .08, .07, .09, .08]
        ),
        "product_weight_g": rng.normal(1200, 500, len(product_ids)).clip(100, 5000),
        "product_length_cm": rng.normal(25, 8, len(product_ids)).clip(5, 70),
        "product_height_cm": rng.normal(15, 5, len(product_ids)).clip(3, 50),
        "product_width_cm": rng.normal(20, 7, len(product_ids)).clip(3, 60),
    })

    # -------------------------
    # 3. Orders
    # -------------------------
    # Each customer gets a persistent "buying propensity".
    # This makes historical behavior genuinely predictive of future behavior.
    customer_affinity = rng.lognormal(mean=0.0, sigma=0.45, size=n_customers)

    # 20% of customers are deliberately modeled as churners.
    # They buy in the historical period but make no future purchase.
    churn_segment = rng.random(n_customers) < 0.20

    order_rows = []
    order_number = 1

    for customer_index, customer_id in enumerate(customer_ids):
        affinity = customer_affinity[customer_index]

        # Historical orders are observed before the prediction cutoff.
        historical_lambda = 8.0 + 4.0 * affinity
        historical_count = max(
            1,
            int(rng.poisson(historical_lambda))
        )

        if churn_segment[customer_index]:
            # Churners become inactive earlier.
            hist_end = pd.Timestamp("2024-01-15")
        else:
            # Active customers can buy close to the observation cutoff.
            hist_end = pd.Timestamp("2024-06-30")

        hist_start = pd.Timestamp("2023-01-01")
        hist_span = max(1, (hist_end - hist_start).days)

        for _ in range(historical_count):
            purchase_date = hist_start + pd.to_timedelta(
                int(rng.integers(0, hist_span + 1)), unit="D"
            )

            product_index = int(rng.integers(0, len(product_ids)))
            base_price = 55 + 95 * affinity
            price = float(
                np.clip(
                    rng.normal(base_price, 5 + 3 * affinity),
                    10,
                    1500,
                )
            )

            order_rows.append({
                "order_id": f"ORD_{order_number:06d}",
                "customer_id": customer_id,
                "product_id": product_ids[product_index],
                "order_status": rng.choice(
                    ["delivered", "shipped", "canceled"],
                    p=[0.93, 0.05, 0.02],
                ),
                "order_purchase_date": purchase_date.strftime("%Y-%m-%d"),
                "price": round(price, 2),
                "freight_value": round(
                    float(np.clip(price * rng.uniform(0.04, 0.10), 3, 100)),
                    2,
                ),
                "quantity": int(rng.choice([1, 1, 1, 2])),
            })
            order_number += 1

        # Future orders are used ONLY as prediction targets.
        if not churn_segment[customer_index]:
            future_count = max(1, int(round(historical_count * (0.70 + 0.02 * affinity))))

            future_start = pd.Timestamp("2024-07-01")
            future_end = pd.Timestamp("2025-06-30")
            future_span = (future_end - future_start).days

            for _ in range(future_count):
                purchase_date = future_start + pd.to_timedelta(
                    int(rng.integers(0, future_span + 1)), unit="D"
                )

                product_index = int(rng.integers(0, len(product_ids)))
                base_price = 55 + 95 * affinity
                price = float(
                    np.clip(
                        rng.normal(base_price, 5 + 3 * affinity),
                        10,
                        1500,
                    )
                )

                order_rows.append({
                    "order_id": f"ORD_{order_number:06d}",
                    "customer_id": customer_id,
                    "product_id": product_ids[product_index],
                    "order_status": rng.choice(
                        ["delivered", "shipped", "canceled"],
                        p=[0.95, 0.04, 0.01],
                    ),
                    "order_purchase_date": purchase_date.strftime("%Y-%m-%d"),
                    "price": round(price, 2),
                    "freight_value": round(
                        float(np.clip(price * rng.uniform(0.04, 0.10), 3, 100)),
                        2,
                    ),
                    "quantity": int(rng.choice([1, 1, 1, 2])),
                })
                order_number += 1

    orders = pd.DataFrame(order_rows)

    # Keep the default parameter useful as a minimum-size target, while
    # preserving all generated future records needed for evaluation.
    if len(orders) < n_orders:
        print(
            f"Warning: generated {len(orders)} orders, below requested "
            f"minimum {n_orders}."
        )

    # -------------------------
    # 4. Reviews
    # -------------------------
    delivered_orders = orders.loc[orders["order_status"] == "delivered"].copy()
    review_orders = delivered_orders.sample(
        n=min(5500, len(delivered_orders)), random_state=RANDOM_STATE
    )

    review_templates = {
        1: [
            "Very disappointing product and poor quality.",
            "Item arrived damaged and I want a refund.",
            "Terrible experience. The product stopped working.",
        ],
        2: [
            "The product is okay but delivery was late.",
            "Quality is below expectations.",
            "Not very happy with the purchase.",
        ],
        3: [
            "Average product. It works as expected.",
            "It is okay for the price.",
            "Nothing special, but acceptable.",
        ],
        4: [
            "Good product and fast delivery.",
            "Happy with the purchase and quality.",
            "Works well and arrived on time.",
        ],
        5: [
            "Excellent product. I highly recommend it.",
            "Amazing quality and very fast delivery.",
            "Perfect purchase. Very satisfied.",
        ],
    }

    # High-value orders get slightly better reviews in this demo.
    scores = np.clip(
        np.round(
            3.3
            + rng.normal(0, 0.9, len(review_orders))
            + (review_orders["price"].to_numpy() > 250) * 0.15
        ),
        1,
        5,
    ).astype(int)

    reviews = pd.DataFrame({
        "review_id": [f"REV_{i:06d}" for i in range(1, len(review_orders) + 1)],
        "order_id": review_orders["order_id"].to_numpy(),
        "review_score": scores,
        "review_comment_message": [
            rng.choice(review_templates[int(score)])
            for score in scores
        ],
        "review_date": (
            pd.to_datetime(review_orders["order_purchase_date"].to_numpy())
            + pd.to_timedelta(rng.integers(1, 15, len(review_orders)), unit="D")
        ).strftime("%Y-%m-%d"),
    })

    # -------------------------
    # 5. Browsing events
    # -------------------------
    # The transition map is intentionally learnable:
    # category A usually leads to category B.
    # Event timestamps are strictly increasing within each customer so the
    # sequence order is preserved after SQL/CSV sorting.
    category_next = {
        "health_beauty": "watches_gifts",
        "bed_bath": "furniture",
        "sports_leisure": "health_beauty",
        "computers": "electronics",
        "watches_gifts": "health_beauty",
        "housewares": "bed_bath",
        "furniture": "housewares",
        "auto": "electronics",
        "toys": "electronics",
        "electronics": "computers",
    }

    browsing_rows = []
    event_counter = 1

    for customer_id in customer_ids:
        seq_len = int(rng.integers(8, 18))
        current = rng.choice(CATEGORIES)
        customer_start = pd.Timestamp("2024-01-01")

        for step in range(seq_len):
            if rng.random() < 0.92:
                current = category_next[current]
            else:
                current = rng.choice(CATEGORIES)

            browsing_rows.append({
                "event_id": f"EVT_{event_counter:07d}",
                "customer_id": customer_id,
                "event_time": (
                    customer_start
                    + pd.to_timedelta(step * 2, unit="h")
                ).isoformat(),
                "product_category": current,
            })
            event_counter += 1

    browsing = pd.DataFrame(browsing_rows).head(n_browsing_events)

    # -------------------------
    # 6. Save flat files
    # -------------------------
    customers.to_csv(RAW_DIR / "customers.csv", index=False)
    products.to_csv(RAW_DIR / "products.csv", index=False)
    orders.to_csv(RAW_DIR / "orders.csv", index=False)
    reviews.to_csv(RAW_DIR / "reviews.csv", index=False)
    browsing.to_csv(RAW_DIR / "browsing_events.csv", index=False)

    # Also create a nested JSON file to demonstrate JSON parsing.
    nested_orders = []
    for _, row in orders.head(100).iterrows():
        nested_orders.append({
            "order_id": row["order_id"],
            "customer": {"customer_id": row["customer_id"]},
            "items": [{
                "product_id": row["product_id"],
                "price": float(row["price"]),
                "freight_value": float(row["freight_value"]),
                "quantity": int(row["quantity"]),
            }],
            "status": row["order_status"],
            "purchase_date": row["order_purchase_date"],
        })

    (RAW_DIR / "mock_ecommerce_api.json").write_text(
        json.dumps(nested_orders, indent=2),
        encoding="utf-8",
    )

    print(f"Created demo data in: {RAW_DIR}")


if __name__ == "__main__":
    generate_data()
