"""Paths and constants shared by every pipeline step.

Generated files (warehouse, output, reports) go under ``CDMV_WORKDIR`` when that
environment variable is set, otherwise under the repository root. The tests use
it to run the pipeline end to end in a temporary directory.
"""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

DATA_URL = (
    "https://archive.ics.uci.edu/static/public/350/"
    "default+of+credit+card+clients.zip"
)
RAW_DIR = ROOT / "data" / "raw"
CHECKSUMS = ROOT / "data" / "checksums.sha256"
ZIP_NAME = "default_of_credit_card_clients.zip"
XLS_NAME = "default_of_credit_card_clients.xls"
XLS_MEMBER = "default of credit card clients.xls"

SQL_DIR = ROOT / "sql"
WORKDIR = Path(os.environ.get("CDMV_WORKDIR", ROOT))
WAREHOUSE = WORKDIR / "warehouse" / "credit.duckdb"
OUTPUT_DIR = WORKDIR / "output"
MODEL_DIR = OUTPUT_DIR / "models"
REPORTS_DIR = WORKDIR / "reports"
README = WORKDIR / "README.md"
FIGURES_DIR = REPORTS_DIR / "figures"
TABLES_DIR = REPORTS_DIR / "tables"

EXPECTED_ROWS = 30_000
SEED = 42
