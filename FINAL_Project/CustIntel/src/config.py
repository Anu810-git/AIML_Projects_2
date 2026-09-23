from pathlib import Path

# Project root directory.
ROOT_DIR = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

DATABASE_DIR = ROOT_DIR / "database"
DB_PATH = DATABASE_DIR / "custintel.db"

MODEL_DIR = ROOT_DIR / "models"
REPORT_DIR = ROOT_DIR / "reports"
LOG_DIR = ROOT_DIR / "logs"

RANDOM_STATE = 42

# A small list keeps the demo easy to understand.
CATEGORIES = [
    "health_beauty",
    "bed_bath",
    "sports_leisure",
    "computers",
    "watches_gifts",
    "housewares",
    "furniture",
    "auto",
    "toys",
    "electronics",
]
