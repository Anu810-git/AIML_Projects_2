"""SQLite database helper functions."""

import sqlite3
from pathlib import Path
import pandas as pd

try:
    from config import DB_PATH, DATABASE_DIR
except ImportError:
    from src.config import DB_PATH, DATABASE_DIR


def get_connection() -> sqlite3.Connection:
    """Open a SQLite connection with foreign-key checks enabled."""
    DATABASE_DIR.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(DB_PATH)
    connection.execute("PRAGMA foreign_keys = ON;")
    return connection


def initialize_database() -> None:
    """Create all project tables from schema.sql."""
    schema_path = DATABASE_DIR / "schema.sql"

    with get_connection() as conn:
        conn.executescript(schema_path.read_text(encoding="utf-8"))
        conn.commit()


def insert_dataframe(
    df: pd.DataFrame,
    table_name: str,
    if_exists: str = "append",
) -> None:
    """Insert a DataFrame into a SQL table."""
    with get_connection() as conn:
        df.to_sql(table_name, conn, if_exists=if_exists, index=False)
        conn.commit()


def query_dataframe(sql: str, params=None) -> pd.DataFrame:
    """Run a read-only SQL query and return a DataFrame."""
    with get_connection() as conn:
        return pd.read_sql_query(sql, conn, params=params)
