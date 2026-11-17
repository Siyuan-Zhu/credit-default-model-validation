"""Paths and constants shared by every pipeline step."""
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
WAREHOUSE = ROOT / "warehouse" / "credit.duckdb"
OUTPUT_DIR = ROOT / "output"
MODEL_DIR = OUTPUT_DIR / "models"
REPORTS_DIR = ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
TABLES_DIR = REPORTS_DIR / "tables"

EXPECTED_ROWS = 30_000
SEED = 42
