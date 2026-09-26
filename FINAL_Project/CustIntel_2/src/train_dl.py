"""
"Deep learning" section of the project: predicts the category of a
customer's NEXT purchase from the categories of their last few purchases.

Uses scikit-learn's MLPClassifier -- a genuine multi-layer perceptron
(the same kind of feed-forward neural network PyTorch would give you here),
without needing a multi-gigabyte GPU-toolkit download for a small, tabular
model like this one.

Sequence example: a customer who bought [bed_bath_table, toys, toys] gets
padded to length HISTORY_LEN with the "<none>" token and the model tries to
guess the category of the purchase that came after.
"""

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

_SRC = str(Path(__file__).resolve().parent)
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from config import HISTORY_LEN, MODEL_DIR, NONE_TOKEN, RANDOM_STATE, REPORT_DIR
from database import query_dataframe


def _build_sequences() -> tuple[np.ndarray, np.ndarray]:
    """For every customer with 2+ purchases, build (last-N-categories -> next-category) rows."""
    purchases = query_dataframe("""
        SELECT c.customer_unique_id AS customer_id,
               p.product_category AS category,
               f.order_purchase_timestamp AS ts
        FROM fact_order_items f
        JOIN dim_products p ON f.product_id = p.product_id
        JOIN dim_customers c ON f.customer_id = c.customer_id
        WHERE f.order_status = 'delivered'
        ORDER BY c.customer_unique_id, f.order_purchase_timestamp
    """)

    X_rows, y_rows = [], []
    for _, group in purchases.groupby("customer_id"):
        cats = group["category"].tolist()
        if len(cats) < 2:
            continue
        for i in range(1, len(cats)):
            history = cats[max(0, i - HISTORY_LEN):i]
            history = [NONE_TOKEN] * (HISTORY_LEN - len(history)) + history
            X_rows.append(history)
            y_rows.append(cats[i])

    return np.array(X_rows), np.array(y_rows)


def train() -> dict:
    X_raw, y_raw = _build_sequences()
    if len(X_raw) < 50:
        raise RuntimeError("Not enough repeat-purchase sequences to train the next-category model.")

    category_encoder = LabelEncoder()
    all_categories = np.unique(np.concatenate([X_raw.ravel(), y_raw]))
    category_encoder.fit(all_categories)

    X_encoded = np.vectorize(lambda c: category_encoder.transform([c])[0])(X_raw)
    y_encoded = category_encoder.transform(y_raw)

    one_hot = OneHotEncoder(categories=[range(len(category_encoder.classes_))] * HISTORY_LEN, sparse_output=False)
    X_features = one_hot.fit_transform(X_encoded)

    X_train, X_test, y_train, y_test = train_test_split(
        X_features, y_encoded, test_size=0.2, random_state=RANDOM_STATE
    )

    model = MLPClassifier(
        hidden_layer_sizes=(128, 64),
        activation="relu",
        max_iter=300,
        early_stopping=True,
        random_state=RANDOM_STATE,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    top1_accuracy = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)

    # Also report top-3 accuracy: is the true next category in the model's top 3 guesses?
    # model.classes_ holds the *encoded label value* for each column of predict_proba,
    # so the column index has to be mapped back through model.classes_ before comparing.
    probabilities = model.predict_proba(X_test)
    top3_cols = np.argsort(probabilities, axis=1)[:, -3:]
    top3_labels = model.classes_[top3_cols]
    top3_accuracy = np.mean([y_test[i] in top3_labels[i] for i in range(len(y_test))])

    history = {
        "model_type": "MLPClassifier (scikit-learn), hidden layers (128, 64)",
        "epochs_ran": int(model.n_iter_),
        "n_categories": int(len(category_encoder.classes_)),
        "n_sequences": int(len(X_raw)),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "test_accuracy": round(float(top1_accuracy), 4),
        "test_top3_accuracy": round(float(top3_accuracy), 4),
        "test_macro_f1": round(float(macro_f1), 4),
        "note": (
            "Repeat purchases are rare on Olist, so this model is trained on a "
            "small number of sequences; treat the numbers as illustrative, not "
            "production-grade."
        ),
    }

    joblib.dump(model, MODEL_DIR / "next_category_mlp.joblib")
    joblib.dump(category_encoder, MODEL_DIR / "category_encoder.joblib")
    joblib.dump(one_hot, MODEL_DIR / "category_one_hot.joblib")
    (REPORT_DIR / "dl_history.json").write_text(json.dumps(history, indent=2))
    return history


def load_next_category_model():
    """Load the trained model + encoders for inference (used by the Streamlit app)."""
    model = joblib.load(MODEL_DIR / "next_category_mlp.joblib")
    category_encoder = joblib.load(MODEL_DIR / "category_encoder.joblib")
    one_hot = joblib.load(MODEL_DIR / "category_one_hot.joblib")
    return model, category_encoder, one_hot


if __name__ == "__main__":
    result = train()
    print(json.dumps(result, indent=2))
