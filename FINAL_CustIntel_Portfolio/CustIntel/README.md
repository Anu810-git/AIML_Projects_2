# CustIntel — AI-Powered Smart Customer Intelligence Platform

A beginner-friendly end-to-end e-commerce customer intelligence project based on the supplied GUVI/HCL project brief.

## What this project demonstrates

1. **Data Engineering**
   - CSV/JSON ingestion
   - Cleaning and validation
   - SQLite star-schema style warehouse
   - Primary/foreign-key constraints
   - Transactional inserts

2. **EDA + Classical Machine Learning**
   - Customer/LTV feature engineering
   - Linear Regression + Random Forest Regression
   - KNN + Random Forest Churn Classification
   - RMSE, R², F1, confusion matrix
   - OLS diagnostics

3. **Deep Learning**
   - PyTorch MLP
   - Last 5 browsing categories -> next category
   - Train/validation/test split
   - Dropout
   - Cross-entropy loss

4. **GenAI / NLP**
   - Review sentiment and theme extraction
   - Optional Gemini API integration
   - Local fallback when no API key is configured
   - Natural-language business questions

5. **Streamlit Dashboard**
   - KPI cards
   - Churn-risk customers
   - LTV predictions
   - Sales/review charts
   - Next-category prediction
   - Sentiment analysis
   - Business chatbot

## Important note about the dataset

The supplied project brief names the **Olist Brazilian E-commerce Dataset or similar open-source data** and asks for CSV/JSON ingestion with SQLite storage. The original Olist dataset is not included in the uploaded brief, so this portfolio package includes a **reproducible synthetic e-commerce dataset generator** with the same major concepts.

If you later download Olist, put its CSV files under `data/raw/` and adapt `src/ingest.py` to map the real column names.

## Project structure

```text
CustIntel/
├── app/
│   └── streamlit_app.py
├── data/
│   ├── raw/
│   └── processed/
├── database/
│   ├── custintel.db
│   └── schema.sql
├── models/
│   ├── ltv_linear_regression.joblib
│   ├── ltv_random_forest.joblib
│   ├── churn_knn.joblib
│   ├── churn_random_forest.joblib
│   ├── ltv_scaler.joblib
│   ├── churn_scaler.joblib
│   ├── category_encoder.joblib
│   └── next_category_mlp.pt
├── notebooks/
│   └── CustIntel_End_to_End.ipynb
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── generate_demo_data.py
│   ├── ingest.py
│   ├── database.py
│   ├── features.py
│   ├── train_ml.py
│   ├── train_dl.py
│   ├── sentiment.py
│   └── chatbot.py
├── reports/
│   └── model_metrics.json
├── .env.example
├── requirements.txt
└── README.md
```

## Setup

### 1. Create a virtual environment

Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
```

macOS/Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install packages

```bash
pip install -r requirements.txt
```

### 3. Run the complete pipeline

```bash
python src/generate_demo_data.py
python src/ingest.py
python src/train_ml.py
python src/train_dl.py
```

The scripts create the database, model files, metrics, and charts.

### 4. Start Streamlit

```bash
streamlit run app/streamlit_app.py
```

## Run the notebook

```bash
jupyter notebook notebooks/CustIntel_End_to_End.ipynb
```

Run cells from top to bottom.

## Optional Gemini chatbot

Create a `.env` file:

```text
GEMINI_API_KEY=your_real_key
```

The dashboard still works without this key. Without it, the chatbot uses a deterministic local business-insight fallback instead of pretending that an LLM is available.

## Beginner learning order

Read these files in this order:

1. `src/generate_demo_data.py`
2. `src/ingest.py`
3. `src/database.py`
4. `src/features.py`
5. `src/train_ml.py`
6. `src/train_dl.py`
7. `src/sentiment.py`
8. `src/chatbot.py`
9. `app/streamlit_app.py`
10. `notebooks/CustIntel_End_to_End.ipynb`

## Evaluation targets from the supplied brief

The brief specifies:
- LTV: R² > 0.75
- Churn: F1 >= 0.80
- SQL joins: < 1 second
- Loss curve without validation overfitting
- Chatbot answers should reflect SQL data without hallucinating

These are **targets, not guaranteed outcomes**. The generated demo data is designed to produce learnable relationships, but model quality must be measured rather than claimed.

