"""
Project settings: folders, file names and modelling choices in one place.

Every other file imports from here, so if you move a folder or change the
prediction window you only edit this file.
"""

from pathlib import Path

# Project root = the folder that contains src/, app/, data/ ...
ROOT = Path(__file__).resolve().parents[1]

# ---- Folders -------------------------------------------------------------
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"          # put the 9 Olist CSV files here
DB_DIR = ROOT / "database"
DB_PATH = DB_DIR / "custintel.db"   # the SQLite database that ingest.py builds
MODEL_DIR = ROOT / "models"         # trained models (.joblib / .pt)
REPORT_DIR = ROOT / "reports"       # metrics, feature tables, quality report

for _folder in (RAW_DIR, DB_DIR, MODEL_DIR, REPORT_DIR):
    _folder.mkdir(parents=True, exist_ok=True)

# ---- The 9 Olist files (Kaggle: olistbr/brazilian-ecommerce) ---------------
OLIST_FILES = {
    "customers": "olist_customers_dataset.csv",
    "geolocation": "olist_geolocation_dataset.csv",
    "order_items": "olist_order_items_dataset.csv",
    "payments": "olist_order_payments_dataset.csv",
    "reviews": "olist_order_reviews_dataset.csv",
    "orders": "olist_orders_dataset.csv",
    "products": "olist_products_dataset.csv",
    "sellers": "olist_sellers_dataset.csv",
    "category_translation": "product_category_name_translation.csv",
}

# ---- Modelling choices -----------------------------------------------------
# LTV = money a customer spends in the NEXT `HORIZON_DAYS` days.
# Churn = the customer does NOT buy again in the next `HORIZON_DAYS` days.
HORIZON_DAYS = 180

# Next-category model: how many previous purchases it looks at.
HISTORY_LEN = 5
NONE_TOKEN = "<none>"   # padding used when a customer has fewer than 5 purchases

RANDOM_STATE = 42
