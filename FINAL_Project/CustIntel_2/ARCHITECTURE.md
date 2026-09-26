# CustIntel \u2014 Architecture

## Data flow

```
data/raw/*.csv (9 Olist files)
        |
        v
  src/ingest.py  --------> database/custintel.db   (SQLite star schema)
        |                          |
        v                          v
src/features.py            src/database.py (query_dataframe helper,
        |                   used by everything below)
        v
reports/customer_features.csv
        |
        +---------------------------+
        v                           v
src/train_ml.py              src/train_dl.py
(LTV + churn models)         (next-category MLP)
        |                           |
        v                           v
models/*.joblib             reports/model_metrics.json, dl_history.json
        |                           |
        +-------------+-------------+
                      v
              app/streamlit_app.py  (5-page dashboard)
                      ^
                      |
        src/sentiment.py, src/chatbot.py
        (reviews & AI agent, called live by the app)
```

## Database star schema (`database/custintel.db`)

- `fact_orders` \u2014 one row per **order** (amounts rolled up from items + payments)
- `fact_order_items` \u2014 one row per **item bought** (product/seller-level detail)
- `dim_customers`, `dim_products` (category translated to English), `dim_sellers`, `dim_reviews`
- `dim_state_geo` \u2014 small per-state summary of `olist_geolocation_dataset.csv` (that file is
  1.1M rows and only used for an optional map, so it isn't loaded row-by-row)

## Why scikit-learn instead of PyTorch for the "deep learning" model

The next-category model is a genuine multi-layer perceptron (`MLPClassifier`), trained on one-hot
encoded purchase-category sequences. PyTorch would work the same way conceptually, but installing it
pulls in several gigabytes of CUDA/cuDNN packages even for CPU-only use in some environments \u2014
overkill for a small tabular model like this one. Swap in a `torch.nn.Module` version of
`src/train_dl.py` if you want to run this on GPU-scale data later.

## Known data characteristics (not bugs)

- **Repeat purchases are rare.** ~96% of unique Olist customers buy exactly once, so `will_churn`
  is close to 1.0 for almost everyone and the LTV target (`future_spend_180d`) is 0 for most rows.
  This caps how well churn/LTV can be predicted from a single order's features \u2014 it's a property
  of this specific marketplace snapshot, not a modelling mistake.
- **~59% of reviews have no written comment**, only a star rating. Sentiment therefore comes from
  `review_score`, and the Portuguese keyword theme-tagger only fires on the ~41% with text.
- **A few products (610) have no category** and are labelled `"unknown"`.
