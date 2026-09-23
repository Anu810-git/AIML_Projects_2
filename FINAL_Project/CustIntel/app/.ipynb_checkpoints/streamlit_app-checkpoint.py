"""
CustIntel Streamlit dashboard.

Run:
    streamlit run app/streamlit_app.py
"""

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
import torch

# Add src/ to Python's import path.
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from config import DB_PATH, MODEL_DIR, REPORT_DIR
from database import query_dataframe
from chatbot import answer_question
from sentiment import analyze_reviews
from train_dl import NextCategoryMLP


st.set_page_config(
    page_title="CustIntel",
    page_icon="📊",
    layout="wide",
)

st.title("📊 CustIntel")
st.caption("AI-Powered Smart Customer Intelligence Platform")

if not DB_PATH.exists():
    st.error("Database not found. Run the pipeline first:")
    st.code(
        "python src/generate_demo_data.py\n"
        "python src/ingest.py\n"
        "python src/train_ml.py\n"
        "python src/train_dl.py"
    )
    st.stop()


@st.cache_data
def load_features():
    return pd.read_csv(REPORT_DIR / "customer_features.csv")


@st.cache_data
def load_metrics():
    return json.loads((REPORT_DIR / "model_metrics.json").read_text())


features = load_features()
metrics = load_metrics()

# -------------------------
# Sidebar navigation
# -------------------------
page = st.sidebar.radio(
    "Navigate",
    [
        "Executive Dashboard",
        "Customer Intelligence",
        "Next Category AI",
        "Voice of Customer",
        "Business Chatbot",
    ],
)

# -------------------------
# Executive dashboard
# -------------------------
if page == "Executive Dashboard":
    st.header("Executive Dashboard")

    orders_df = query_dataframe("""
        SELECT
            COUNT(*) AS orders,
            SUM(price * quantity + freight_value) AS revenue,
            COUNT(DISTINCT customer_id) AS customers
        FROM fact_orders
        WHERE order_status = 'delivered'
    """)
    reviews_df = query_dataframe("""
        SELECT AVG(review_score) AS avg_score
        FROM dim_reviews
    """)

    row = orders_df.iloc[0]

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Delivered Orders", f"{int(row['orders']):,}")
    col2.metric("Revenue", f"{row['revenue']:,.0f}")
    col3.metric("Customers", f"{int(row['customers']):,}")
    col4.metric("Avg Review", f"{reviews_df.iloc[0]['avg_score']:.2f}/5")

    st.subheader("Model performance")

    ltv = metrics["ltv"]
    churn = metrics["churn"]

    m1, m2, m3 = st.columns(3)
    m1.metric("LTV RF R²", f"{ltv['random_forest_r2']:.3f}")
    m2.metric("LTV RF RMSE", f"{ltv['random_forest_rmse']:.2f}")
    m3.metric("Churn RF F1", f"{churn['random_forest_f1']:.3f}")

    st.info(
        "These numbers are measured on the generated demo dataset. "
        "Do not claim they represent production performance."
    )

    category_sales = query_dataframe("""
        SELECT
            p.product_category,
            SUM(o.price * o.quantity + o.freight_value) AS revenue
        FROM fact_orders o
        JOIN dim_products p ON o.product_id = p.product_id
        WHERE o.order_status = 'delivered'
        GROUP BY p.product_category
        ORDER BY revenue DESC
    """)

    fig, ax = plt.subplots(figsize=(10, 4))
    category_sales.plot(
        x="product_category",
        y="revenue",
        kind="bar",
        ax=ax,
        legend=False,
    )
    ax.set_title("Revenue by Product Category")
    ax.set_xlabel("Category")
    ax.set_ylabel("Revenue")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    st.pyplot(fig)

# -------------------------
# Customer intelligence
# -------------------------
elif page == "Customer Intelligence":
    st.header("Customer Intelligence")

    customer_ids = features["customer_id"].tolist()
    selected = st.selectbox("Select customer", customer_ids)

    customer = features.loc[features["customer_id"] == selected].iloc[0]

    ltv_model = joblib.load(MODEL_DIR / "ltv_random_forest.joblib")
    churn_model = joblib.load(MODEL_DIR / "churn_random_forest.joblib")

    ltv_features = [
        "order_count", "total_spend", "avg_order_value",
        "avg_freight", "avg_review_score", "days_since_last_purchase"
    ]
    churn_features = [
        "order_count", "total_spend", "avg_order_value",
        "avg_review_score", "days_since_last_purchase"
    ]

    predicted_ltv = max(
        0,
        float(ltv_model.predict(customer[ltv_features].to_frame().T)[0])
    )
    churn_probability = float(
        churn_model.predict_proba(
            customer[churn_features].to_frame().T
        )[0][1]
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Historical Spend", f"{customer['total_spend']:,.2f}")
    c2.metric("Predicted 12M LTV", f"{predicted_ltv:,.2f}")
    c3.metric("Churn Probability", f"{churn_probability:.1%}")
    c4.metric("Last Purchase Gap", f"{int(customer['days_since_last_purchase'])} days")

    if churn_probability >= 0.70:
        st.error("High churn risk — consider a targeted retention action.")
    elif churn_probability >= 0.40:
        st.warning("Medium churn risk — monitor customer engagement.")
    else:
        st.success("Lower churn risk.")

    st.subheader("Top churn-risk customers")

    display = features.copy()
    display["churn_probability"] = churn_model.predict_proba(
        display[churn_features]
    )[:, 1]
    display["predicted_ltv"] = np.maximum(
        0,
        ltv_model.predict(display[ltv_features])
    )

    st.dataframe(
        display.sort_values("churn_probability", ascending=False)[
            [
                "customer_id",
                "total_spend",
                "predicted_ltv",
                "churn_probability",
                "days_since_last_purchase",
            ]
        ].head(20),
        use_container_width=True,
    )

# -------------------------
# Next-category AI
# -------------------------
elif page == "Next Category AI":
    st.header("Next-Category Browsing Prediction")

    encoder = joblib.load(MODEL_DIR / "category_encoder.joblib")
    checkpoint = torch.load(
        MODEL_DIR / "next_category_mlp.pt",
        map_location="cpu",
        weights_only=False,
    )

    model = NextCategoryMLP(checkpoint["n_categories"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    categories = list(encoder.classes_)

    st.write(
        "Enter the last five categories the customer viewed. "
        "The neural network predicts the next category."
    )

    cols = st.columns(5)
    selected_categories = []

    for i, col in enumerate(cols):
        selected_categories.append(
            col.selectbox(
                f"Step {i + 1}",
                categories,
                key=f"category_{i}",
            )
        )

    encoded = encoder.transform(selected_categories).astype("float32")
    x = torch.tensor(encoded).reshape(1, 5)

    with torch.no_grad():
        probabilities = torch.softmax(model(x), dim=1)[0].numpy()

    results = pd.DataFrame({
        "category": categories,
        "probability": probabilities,
    }).sort_values("probability", ascending=False)

    st.subheader("Prediction")
    st.dataframe(results, use_container_width=True)

    fig, ax = plt.subplots(figsize=(9, 4))
    results.head(5).plot(
        x="category",
        y="probability",
        kind="bar",
        ax=ax,
        legend=False,
    )
    ax.set_ylabel("Probability")
    ax.set_title("Top 5 Next-Category Predictions")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    st.pyplot(fig)

# -------------------------
# Voice of customer
# -------------------------
elif page == "Voice of Customer":
    st.header("Voice of the Customer")

    reviews = query_dataframe("""
        SELECT review_score, review_comment_message
        FROM dim_reviews
        WHERE review_comment_message IS NOT NULL
    """)

    analyzed = analyze_reviews(reviews)

    sentiment_counts = analyzed["sentiment"].value_counts()
    st.subheader("Sentiment distribution")
    st.bar_chart(sentiment_counts)

    theme_rows = []
    for themes in analyzed["themes"]:
        for theme in themes:
            theme_rows.append(theme)

    st.subheader("Top themes")
    st.bar_chart(pd.Series(theme_rows).value_counts().head(10))

    st.subheader("Sample analyzed reviews")
    st.dataframe(
        analyzed.head(30),
        use_container_width=True,
    )

# -------------------------
# Chatbot
# -------------------------
elif page == "Business Chatbot":
    st.header("💬 Business Chatbot")
    st.write(
        "Ask questions such as: "
        "`Why is churn high?`, `What categories have the highest sales?`, "
        "`What is the average review score?`, or `What is customer LTV?`"
    )

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    question = st.chat_input("Ask a business question...")

    if question:
        st.session_state.messages.append({
            "role": "user",
            "content": question,
        })

        with st.chat_message("user"):
            st.markdown(question)

        answer = answer_question(question)

        with st.chat_message("assistant"):
            st.markdown(answer)

        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
        })
