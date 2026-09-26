# CustIntel \u2014 AI-Powered Smart Customer Intelligence Platform

An end-to-end customer-intelligence project built on the real
[Olist Brazilian E-commerce dataset](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce)
(Kaggle): ~100k real orders from a Brazilian multi-seller marketplace, 2016\u20132018.

Raw CSVs \u2192 SQLite database \u2192 EDA \u2192 customer features (RFM) \u2192 ML (LTV + churn) \u2192
deep learning (next-category MLP) \u2192 Portuguese review sentiment \u2192 AI business chatbot \u2192
Streamlit dashboard.

## 1. Get the data

This repo does **not** include the raw CSVs (they're ~140 MB and licensed by Olist/Kaggle, not by
this project). Download them yourself:

1. Go to https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce and download the dataset
   (you'll need a free Kaggle account).
2. Unzip it and copy these **9 files** into `data/raw/`:

   | File | What it is |
   |---|---|
   | `olist_customers_dataset.csv` | customer id + city/state |
   | `olist_orders_dataset.csv` | one row per order, with status and timestamps |
   | `olist_order_items_dataset.csv` | one row per product bought (price, freight) |
   | `olist_order_payments_dataset.csv` | payment method/value per order |
   | `olist_order_reviews_dataset.csv` | star rating + (often blank) Portuguese comment |
   | `olist_products_dataset.csv` | product category + dimensions |
   | `olist_sellers_dataset.csv` | seller id + city/state |
   | `olist_geolocation_dataset.csv` | zip-code \u2192 lat/lng lookup (large, ~1.1M rows) |
   | `product_category_name_translation.csv` | Portuguese \u2192 English category names |

## 2. Install

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## 3. (Optional) enable the full AI chatbot

Without this step the chatbot still works, in a small offline mode (churn / reviews / sales / LTV
basics only). To get the full "ask anything" agent:

```bash
cp .env.example .env
# then edit .env and set GOOGLE_API_KEY=your-key
# (get a free key at https://ai.google.dev/gemini-api/docs/api-key)
```

## 4. Run the pipeline

```bash
python src/ingest.py       # builds database/custintel.db from the 9 CSVs
python src/features.py     # builds reports/customer_features.csv
python src/train_ml.py     # trains LTV + churn models -> models/, reports/model_metrics.json
python src/train_dl.py     # trains the next-category MLP -> models/, reports/dl_history.json
```

Or just open `notebooks/CustIntel_End_to_End.ipynb` and run all cells \u2014 it does the same steps
with explanations and charts along the way.

## 5. Run the dashboard

```bash
streamlit run app/streamlit_app.py
```

Five pages: **Executive Dashboard**, **Customer Intelligence**, **Next Category AI**,
**Voice of Customer**, **Business Chatbot**.

## Project layout

```
src/            ingest, features, ML/DL training, sentiment, chatbot, config
app/            streamlit_app.py (the dashboard)
notebooks/      CustIntel_End_to_End.ipynb
data/raw/       put the 9 Olist CSVs here (not included)
database/       custintel.db is built here by src/ingest.py
models/         trained models (.joblib) land here
reports/        metrics, feature tables, data-quality report
```

See `ARCHITECTURE.md` for the data flow diagram and the honest limitations of this dataset
(most notably: repeat purchases are rare on Olist, which caps churn/LTV accuracy \u2014 that's a
property of the data, not a bug).
