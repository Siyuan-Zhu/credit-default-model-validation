"""DuckDB connection, raw loader and a minimal SQL-file runner.

SQL files may contain ``{{name}}`` placeholders. Values are restricted to
integers and simple identifiers so the substitution cannot inject SQL.
"""
import re
from pathlib import Path

import duckdb
import pandas as pd

from .config import WAREHOUSE

RAW_COLUMNS = [
    "ID", "LIMIT_BAL", "SEX", "EDUCATION", "MARRIAGE", "AGE",
    "PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6",
    "BILL_AMT1", "BILL_AMT2", "BILL_AMT3", "BILL_AMT4", "BILL_AMT5", "BILL_AMT6",
    "PAY_AMT1", "PAY_AMT2", "PAY_AMT3", "PAY_AMT4", "PAY_AMT5", "PAY_AMT6",
    "default payment next month",
]

_PLACEHOLDER = re.compile(r"\{\{\s*(\w+)\s*\}\}")
_SAFE_VALUE = re.compile(r"^-?\w+$")


def connect(path=WAREHOUSE):
    if str(path) != ":memory:":
        Path(path).parent.mkdir(parents=True, exist_ok=True)
    return duckdb.connect(str(path))


def render(sql, params=None):
    params = params or {}

    def sub(match):
        name = match.group(1)
        if name not in params:
            raise KeyError(f"SQL parameter '{name}' was not provided")
        value = str(params[name])
        if not _SAFE_VALUE.match(value):
            raise ValueError(f"Unsafe value for SQL parameter '{name}': {value!r}")
        return value

    return _PLACEHOLDER.sub(sub, sql)


def run_sql_file(con, path, params=None):
    con.execute(render(Path(path).read_text(), params))


def run_sql_dir(con, directory, params=None):
    files = sorted(Path(directory).glob("*.sql"))
    for path in files:
        run_sql_file(con, path, params)
    return files


def query_file(con, path, params=None):
    return con.execute(render(Path(path).read_text(), params)).df()


def raw_frame_from_xls(xls_path):
    """Parse the source workbook (header is on the second row)."""
    df = pd.read_excel(xls_path, header=1, engine="xlrd")
    if list(df.columns) != RAW_COLUMNS:
        raise ValueError(f"Unexpected columns in source file: {list(df.columns)}")
    return df


def load_raw(con, df):
    """Write a source-shaped DataFrame to raw.credit_card_clients (lower-case names)."""
    df = df.rename(columns=lambda c: c.lower().replace(" ", "_"))
    con.execute("create schema if not exists raw")
    con.register("raw_df", df)
    casts = ", ".join(f"cast({c} as bigint) as {c}" for c in df.columns)
    con.execute(f"create or replace table raw.credit_card_clients as select {casts} from raw_df")
    con.unregister("raw_df")
