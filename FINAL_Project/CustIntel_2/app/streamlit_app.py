"""
CustIntel Streamlit dashboard.

Run:
    streamlit run app/streamlit_app.py
"""

import os
import sys
from pathlib import Path

import joblib
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = str(ROOT / "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from config import DB_PATH, HISTORY_LEN, MODEL_DIR, NONE_TOKEN, REPORT_DIR
from database import query_dataframe
from chatbot import answer_question
from sentiment import analyze_reviews


st.set_page_config(page_title="CustIntel", page_icon="\U0001F4CA", layout="wide")
st.title("\U0001F4CA CustIntel")
st.caption("AI-Powered Smart Customer Intelligence Platform \u2014 Olist Brazilian E-commerce")

if not DB_PATH.exists():
    st.error("Database not found. Run the pipeline first:")
    st.code(
        "python src/ingest.py\n"
        "python src/features.py\n"
        "python src/train_ml.py\n"
        "python src/train_dl.py"
    )
    st.stop()


@st.cache_data
def load_features():
    return pd.read_csv(REPORT_DIR / "customer_features.csv")


@st.cache_data
def load_metrics():
    import json
    return json.loads((REPORT_DIR / "model_metrics.json").read_text())


@st.cache_resource
def load_next_category_model():
    model = joblib.load(MODEL_DIR / "next_category_mlp.joblib")
    category_encoder = joblib.load(MODEL_DIR / "category_encoder.joblib")
    one_hot = joblib.load(MODEL_DIR / "category_one_hot.joblib")
    return model, category_encoder, one_hot


features = load_features()
metrics = load_metrics()

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
            SUM(payment_value) AS revenue,
            COUNT(DISTINCT customer_id) AS customers
        FROM fact_orders
        WHERE order_status = 'delivered'
    """)
    reviews_df = query_dataframe("SELECT AVG(review_score) AS avg_score FROM dim_reviews")

    row = orders_df.iloc[0]

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Delivered Orders", f"{int(row['orders']):,}")
    col2.metric("Revenue (R$)", f"{row['revenue']:,.0f}")
    col3.metric("Customers", f"{int(row['customers']):,}")
    col4.metric("Avg Review", f"{reviews_df.iloc[0]['avg_score']:.2f}/5")

    st.subheader("Model performance")

    ltv = metrics["ltv"]
    churn = metrics["churn"]

    m1, m2, m3 = st.columns(3)
    m1.metric("LTV RF R\u00b2", f"{ltv['random_forest_r2']:.3f}")
    m2.metric("LTV RF RMSE (R$)", f"{ltv['random_forest_rmse']:.2f}")
    m3.metric("Churn RF F1 (active)", f"{churn['random_forest_f1_active_customers']:.3f}")

    st.info(
        "These numbers are measured on real Olist data. Olist has a very low "
        "repeat-purchase rate (see churn_rate below), which limits how well "
        "LTV/churn can be predicted from a single purchase's features \u2014 "
        "this is a property of the dataset, not a modelling bug."
    )
    st.caption(f"Churn rate in the data: {churn['churn_rate']:.1%} of customers did not buy again within 180 days.")

    all_category_sales = query_dataframe("""
        SELECT p.product_category, SUM(o.price) AS revenue
        FROM fact_order_items o
        JOIN dim_products p ON o.product_id = p.product_id
        WHERE o.order_status = 'delivered'
        GROUP BY p.product_category
        ORDER BY revenue DESC
    """)
    n_categories = len(all_category_sales)
    category_sales = all_category_sales.head(15)

    fig, ax = plt.subplots(figsize=(10, 4))
    category_sales.plot(x="product_category", y="revenue", kind="bar", ax=ax, legend=False)
    ax.set_title("Top 15 Categories by Revenue")
    ax.set_xlabel("Category")
    ax.set_ylabel("Revenue (R$)")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    st.pyplot(fig)
    st.caption(
        f"This chart only shows the 15 HIGHEST-revenue categories out of {n_categories} total. "
        "The smallest bar shown here is the 15th-highest category, not the lowest-revenue "
        "category overall \u2014 see the lowest 10 below for that."
    )

    with st.expander(f"Show the 10 LOWEST-revenue categories (out of {n_categories} total)"):
        lowest = all_category_sales.tail(10).sort_values("revenue")
        fig2, ax2 = plt.subplots(figsize=(9, 3.5))
        lowest.plot(x="product_category", y="revenue", kind="bar", ax=ax2, legend=False, color="indianred")
        ax2.set_title("10 Lowest-Revenue Categories")
        ax2.set_xlabel("Category")
        ax2.set_ylabel("Revenue (R$)")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        st.pyplot(fig2)
        st.dataframe(lowest.reset_index(drop=True), use_container_width=True)

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
        "avg_review_score", "days_since_last_purchase", "customer_tenure_days",
    ]
    churn_features = [
        "order_count", "total_spend", "avg_order_value",
        "avg_review_score", "days_since_last_purchase",
    ]

    predicted_ltv = max(0, float(ltv_model.predict(customer[ltv_features].to_frame().T)[0]))
    churn_probability = float(churn_model.predict_proba(customer[churn_features].to_frame().T)[0][1])

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Historical Spend (R$)", f"{customer['total_spend']:,.2f}")
    c2.metric("Predicted 180d Spend (R$)", f"{predicted_ltv:,.2f}")
    c3.metric("Churn Probability", f"{churn_probability:.1%}")
    c4.metric("Last Purchase Gap", f"{int(customer['days_since_last_purchase'])} days")

    if churn_probability >= 0.70:
        st.error("High churn risk \u2014 consider a targeted retention action.")
    elif churn_probability >= 0.40:
        st.warning("Medium churn risk \u2014 monitor customer engagement.")
    else:
        st.success("Lower churn risk.")

    st.subheader("Top churn-risk customers")

    display = features.copy()
    display["churn_probability"] = churn_model.predict_proba(display[churn_features])[:, 1]
    display["predicted_ltv"] = np.maximum(0, ltv_model.predict(display[ltv_features]))

    st.dataframe(
        display.sort_values("churn_probability", ascending=False)[
            ["customer_id", "total_spend", "predicted_ltv", "churn_probability", "days_since_last_purchase"]
        ].head(20),
        use_container_width=True,
    )

# -------------------------
# Next-category AI
# -------------------------
elif page == "Next Category AI":
    st.header("Next-Category Purchase Prediction")

    model, category_encoder, one_hot = load_next_category_model()
    categories = list(category_encoder.classes_)
    default_categories = [c for c in categories if c != NONE_TOKEN][:HISTORY_LEN]

    st.write(
        f"Enter the last {HISTORY_LEN} product categories the customer bought "
        f"(use '{NONE_TOKEN}' for unknown/none). A neural network (MLP) predicts the next category."
    )

    cols = st.columns(HISTORY_LEN)
    selected_categories = []
    for i, col in enumerate(cols):
        default_idx = categories.index(default_categories[i]) if i < len(default_categories) else 0
        selected_categories.append(col.selectbox(f"Purchase {i + 1}", categories, index=default_idx, key=f"category_{i}"))

    encoded = np.array([[category_encoder.transform([c])[0] for c in selected_categories]])
    X = one_hot.transform(encoded)
    probabilities = model.predict_proba(X)[0]

    # probabilities columns follow model.classes_, which are encoded label values
    # for the categories that appear as *targets* (NONE_TOKEN is never a target).
    class_labels = category_encoder.inverse_transform(model.classes_)
    results = pd.DataFrame({"category": class_labels, "probability": probabilities}).sort_values(
        "probability", ascending=False
    )

    st.subheader("Prediction")
    st.dataframe(results, use_container_width=True)

    fig, ax = plt.subplots(figsize=(9, 4))
    results.head(5).plot(x="category", y="probability", kind="bar", ax=ax, legend=False)
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

    theme_rows = [theme for themes in analyzed["themes"] for theme in themes]
    st.subheader("Top themes")
    st.bar_chart(pd.Series(theme_rows).value_counts())

    st.caption("Reviews are in Brazilian Portuguese; sentiment comes from the star rating, themes from Portuguese keyword matching.")

    st.subheader("Sample analyzed reviews")
    st.dataframe(analyzed.head(30), use_container_width=True)

# -------------------------
# Chatbot
# -------------------------
elif page == "Business Chatbot":
    st.header("\U0001F4AC Business Chatbot")
    st.write(
        "Ask anything about the business \u2014 revenue, categories, churn, LTV, "
        "reviews, specific customers, model performance, trends, comparisons \u2014 "
        "in plain English. The chatbot is an AI agent that queries the live "
        "database itself, so it isn't limited to a fixed list of questions."
    )

    if not os.environ.get("GOOGLE_API_KEY") and not os.environ.get("GEMINI_API_KEY"):
        st.warning(
            "No `GOOGLE_API_KEY` is set, so the chatbot is running in a "
            "limited offline mode (churn / reviews / sales / LTV basics only). "
            "Set that environment variable and `pip install google-genai` to "
            "enable the full ask-anything AI agent."
        )

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    question = st.chat_input("Ask a business question...")

    if question:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Analyzing..."):
                history = st.session_state.messages[:-1]
                answer = answer_question(question, history=history)
            st.markdown(answer)

        st.session_state.messages.append({"role": "assistant", "content": answer})

    if st.session_state.messages:
        if st.button("Clear conversation"):
            st.session_state.messages = []
            st.rerun()
