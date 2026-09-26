"""Small helpers for talking to the SQLite database."""

import sqlite3
import sys
from pathlib import Path

import pandas as pd

_SRC = str(Path(__file__).resolve().parent)
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from config import DB_PATH


def get_connection(read_only: bool = False) -> sqlite3.Connection:
    """Open the database. read_only=True makes writing physically impossible."""
    if read_only:
        return sqlite3.connect(f"file:{DB_PATH.as_posix()}?mode=ro", uri=True)
    return sqlite3.connect(DB_PATH)


def query_dataframe(sql: str, params=None) -> pd.DataFrame:
    """Run a SELECT query and return the result as a pandas table."""
    connection = get_connection(read_only=True)
    try:
        return pd.read_sql_query(sql, connection, params=params)
    finally:
        connection.close()
