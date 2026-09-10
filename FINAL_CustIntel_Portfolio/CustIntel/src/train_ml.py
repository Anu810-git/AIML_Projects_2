"""Train classical ML models for LTV and churn."""

import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm

try:
    from config import MODEL_DIR, REPORT_DIR, RANDOM_STATE
    from features import build_customer_features, LTV_FEATURES, CHURN_FEATURES
except ImportError:
    from src.config import MODEL_DIR, REPORT_DIR, RANDOM_STATE
    from src.features import build_customer_features, LTV_FEATURES, CHURN_FEATURES


def train_models():
    MODEL_DIR.mkdir(exist_ok=True)
    REPORT_DIR.mkdir(exist_ok=True)

    df = build_customer_features()

    # Remove rows that cannot be modeled.
    df = df.replace([np.inf, -np.inf], np.nan).dropna(
        subset=LTV_FEATURES + ["future_12m_spend", "will_churn"]
    )

    # -------------------------
    # LTV Regression
    # -------------------------
    X_ltv = df[LTV_FEATURES]
    y_ltv = df["future_12m_spend"]

    X_train, X_test, y_train, y_test = train_test_split(
        X_ltv, y_ltv, test_size=0.20, random_state=RANDOM_STATE
    )

    ltv_linear = Pipeline([
        ("scaler", StandardScaler()),
        ("model", LinearRegression()),
    ])

    ltv_rf = RandomForestRegressor(
        n_estimators=250,
        max_depth=10,
        min_samples_leaf=3,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    ltv_linear.fit(X_train, y_train)
    ltv_rf.fit(X_train, y_train)

    linear_pred = np.maximum(0, ltv_linear.predict(X_test))
    rf_pred = np.maximum(0, ltv_rf.predict(X_test))

    ltv_metrics = {
        "linear_regression_rmse": float(np.sqrt(mean_squared_error(y_test, linear_pred))),
        "linear_regression_r2": float(r2_score(y_test, linear_pred)),
        "random_forest_rmse": float(np.sqrt(mean_squared_error(y_test, rf_pred))),
        "random_forest_r2": float(r2_score(y_test, rf_pred)),
    }

    joblib.dump(ltv_linear, MODEL_DIR / "ltv_linear_regression.joblib")
    joblib.dump(ltv_rf, MODEL_DIR / "ltv_random_forest.joblib")

    # OLS diagnostics: useful for the project requirement.
    X_ols = sm.add_constant(X_train)
    ols_model = sm.OLS(y_train, X_ols).fit()
    (REPORT_DIR / "ols_summary.txt").write_text(
        ols_model.summary().as_text(), encoding="utf-8"
    )

    # -------------------------
    # Churn Classification
    # -------------------------
    X_churn = df[CHURN_FEATURES]
    y_churn = df["will_churn"]

    X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
        X_churn,
        y_churn,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y_churn,
    )

    churn_knn = Pipeline([
        ("scaler", StandardScaler()),
        ("model", KNeighborsClassifier(n_neighbors=11)),
    ])

    churn_rf = RandomForestClassifier(
        n_estimators=250,
        max_depth=10,
        min_samples_leaf=3,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    churn_knn.fit(X_train_c, y_train_c)
    churn_rf.fit(X_train_c, y_train_c)

    knn_pred = churn_knn.predict(X_test_c)
    rf_pred_c = churn_rf.predict(X_test_c)

    churn_metrics = {
        "knn_f1": float(f1_score(y_test_c, knn_pred)),
        "random_forest_f1": float(f1_score(y_test_c, rf_pred_c)),
        "random_forest_confusion_matrix": confusion_matrix(
            y_test_c, rf_pred_c
        ).tolist(),
        "random_forest_report": classification_report(
            y_test_c, rf_pred_c, output_dict=True
        ),
    }

    joblib.dump(churn_knn, MODEL_DIR / "churn_knn.joblib")
    joblib.dump(churn_rf, MODEL_DIR / "churn_random_forest.joblib")

    # -------------------------
    # Feature importance chart
    # -------------------------
    importances = pd.Series(
        churn_rf.feature_importances_, index=CHURN_FEATURES
    ).sort_values(ascending=True)

    plt.figure(figsize=(8, 4))
    importances.plot(kind="barh")
    plt.title("Random Forest Churn Feature Importance")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig(REPORT_DIR / "churn_feature_importance.png", dpi=150)
    plt.close()

    metrics = {
        "dataset_rows": int(len(df)),
        "churn_rate": float(df["will_churn"].mean()),
        "ltv": ltv_metrics,
        "churn": churn_metrics,
    }

    (REPORT_DIR / "model_metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )

    # Save the engineered data so the dashboard can load it quickly.
    df.to_csv(
        REPORT_DIR / "customer_features.csv",
        index=False,
    )

    print(json.dumps(ltv_metrics, indent=2))
    print(json.dumps({
        "knn_f1": churn_metrics["knn_f1"],
        "random_forest_f1": churn_metrics["random_forest_f1"],
    }, indent=2))


if __name__ == "__main__":
    train_models()
