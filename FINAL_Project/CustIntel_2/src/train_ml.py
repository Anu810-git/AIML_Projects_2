"""
Train two classic ML models on the customer_features table:

    LTV model    -- RandomForestRegressor predicting future_spend_180d
    Churn model  -- RandomForestClassifier predicting will_churn

Saves the fitted models (joblib) and a metrics report (JSON).

Note on this dataset: on Olist, the huge majority of customers never buy a
second time (see README), so "will_churn" is 1 for about 98-99% of rows. The
churn model is trained with class_weight="balanced" and judged on F1 for the
minority ("still active") class rather than accuracy, which would look good
by just always predicting churn.
"""

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import f1_score, mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

_SRC = str(Path(__file__).resolve().parent)
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from config import MODEL_DIR, RANDOM_STATE, REPORT_DIR

LTV_FEATURES = [
    "order_count", "total_spend", "avg_order_value",
    "avg_review_score", "days_since_last_purchase", "customer_tenure_days",
]
CHURN_FEATURES = [
    "order_count", "total_spend", "avg_order_value",
    "avg_review_score", "days_since_last_purchase",
]


def train_models() -> dict:
    features_path = REPORT_DIR / "customer_features.csv"
    if not features_path.exists():
        raise FileNotFoundError("Run src/features.py first to build reports/customer_features.csv")
    data = pd.read_csv(features_path)

    metrics: dict = {}

    # ---------------- LTV regressor -----------------------------------------
    X = data[LTV_FEATURES]
    y = data["future_spend_180d"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)

    ltv_model = RandomForestRegressor(n_estimators=300, max_depth=10, random_state=RANDOM_STATE, n_jobs=-1)
    ltv_model.fit(X_train, y_train)
    ltv_pred = np.maximum(0, ltv_model.predict(X_test))

    metrics["ltv"] = {
        "random_forest_r2": round(float(r2_score(y_test, ltv_pred)), 4),
        "random_forest_rmse": round(float(np.sqrt(mean_squared_error(y_test, ltv_pred))), 2),
        "random_forest_mae": round(float(mean_absolute_error(y_test, ltv_pred)), 2),
        "n_train": len(X_train), "n_test": len(X_test),
    }
    joblib.dump(ltv_model, MODEL_DIR / "ltv_random_forest.joblib")

    # ---------------- Churn classifier ---------------------------------------
    Xc = data[CHURN_FEATURES]
    yc = data["will_churn"]
    Xc_train, Xc_test, yc_train, yc_test = train_test_split(
        Xc, yc, test_size=0.2, random_state=RANDOM_STATE, stratify=yc
    )

    churn_model = RandomForestClassifier(
        n_estimators=300, max_depth=8, random_state=RANDOM_STATE, n_jobs=-1, class_weight="balanced",
    )
    churn_model.fit(Xc_train, yc_train)
    churn_pred = churn_model.predict(Xc_test)

    metrics["churn"] = {
        "random_forest_f1": round(float(f1_score(yc_test, churn_pred)), 4),
        "random_forest_f1_active_customers": round(
            float(f1_score(yc_test, churn_pred, pos_label=0)), 4
        ),
        "churn_rate": round(float(yc.mean()), 4),
        "n_train": len(Xc_train), "n_test": len(Xc_test),
        "note": (
            "This dataset has a very low repeat-purchase rate (Olist is mostly "
            "one-time buyers), so churn is heavily imbalanced. The model is "
            "trained with class_weight='balanced'; judge it on the 'active "
            "customers' F1, not on accuracy."
        ),
    }
    joblib.dump(churn_model, MODEL_DIR / "churn_random_forest.joblib")

    (REPORT_DIR / "model_metrics.json").write_text(json.dumps(metrics, indent=2))
    return metrics


if __name__ == "__main__":
    result = train_models()
    print(json.dumps(result, indent=2))
