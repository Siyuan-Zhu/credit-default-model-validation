"""Load the raw workbook into DuckDB and build the staging and feature tables."""
from .config import RAW_DIR, SEED, SQL_DIR, XLS_NAME
from .warehouse import connect, load_raw, raw_frame_from_xls, run_sql_dir

PARAMS = {"seed": SEED}


def build(con, params=PARAMS):
    """Run staging then features. Assumes raw.credit_card_clients exists."""
    ran = []
    for layer in ("staging", "features"):
        ran += run_sql_dir(con, SQL_DIR / layer, params)
    return ran


def main():
    con = connect()
    load_raw(con, raw_frame_from_xls(RAW_DIR / XLS_NAME))
    for path in build(con):
        print(f"ran {path.relative_to(SQL_DIR.parent)}")
    for table in ("raw.credit_card_clients", "staging.stg_accounts",
                  "staging.stg_account_months", "features.account_features",
                  "features.model_split"):
        n = con.execute(f"select count(*) from {table}").fetchone()[0]
        print(f"{table}: {n:,} rows")
    con.close()


if __name__ == "__main__":
    main()
