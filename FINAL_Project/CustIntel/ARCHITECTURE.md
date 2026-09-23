# CustIntel Architecture

```text
CSV / JSON
   |
   v
src/generate_demo_data.py
   |
   v
src/ingest.py  ---> cleaning + validation + transaction
   |
   v
SQLite Star Schema
   |-- dim_customers
   |-- dim_products
   |-- dim_reviews
   |-- fact_orders
   `-- fact_browsing
   |
   +--------------------+
   |                    |
   v                    v
features.py         train_dl.py
   |                    |
   v                    v
train_ml.py        next_category_mlp.pt
   |
   +--> LTV models
   +--> Churn models
   |
   v
Streamlit
   |
   +--> KPI dashboard
   +--> Customer intelligence
   +--> Next-category AI
   +--> Voice of customer
   `--> Business chatbot
```

## Why the chatbot does not execute arbitrary SQL

Letting an LLM generate unrestricted SQL against a production database is a bad beginner architecture because it creates security, correctness, and governance problems.

CustIntel instead uses:

1. User question
2. Intent detection
3. Predefined safe SQL
4. Retrieved facts
5. Optional Gemini summarization

This keeps database access deterministic and limits hallucination risk.
