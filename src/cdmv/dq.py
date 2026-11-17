"""Run the SQL data-quality checks and fail loudly on any error-severity failure."""
import sys

import pandas as pd

from .config import EXPECTED_ROWS, SQL_DIR, TABLES_DIR
from .warehouse import connect, query_file


class DataQualityError(RuntimeError):
    pass


def run_checks(con, params=None, dq_dir=SQL_DIR / "dq"):
    params = params or {"expected_rows": EXPECTED_ROWS}
    frames = [query_file(con, p, params) for p in sorted(dq_dir.glob("*.sql"))]
    results = pd.concat(frames, ignore_index=True)
    results["failing_rows"] = results["failing_rows"].astype(int)
    results["total_rows"] = results["total_rows"].astype(int)
    results["status"] = "PASS"
    failing = results["failing_rows"] > 0
    results.loc[failing & (results["severity"] == "warn"), "status"] = "WARN"
    results.loc[failing & (results["severity"] == "error"), "status"] = "FAIL"
    con.execute("create schema if not exists dq")
    con.register("dq_df", results)
    con.execute("create or replace table dq.results as select * from dq_df")
    con.unregister("dq_df")
    return results


def enforce(results):
    failed = results[results["status"] == "FAIL"]
    if len(failed):
        lines = [f"  {r.check_id} {r.description}: {r.failing_rows} failing rows"
                 for r in failed.itertuples()]
        raise DataQualityError("Data-quality checks failed:\n" + "\n".join(lines))


def main():
    con = connect()
    results = run_checks(con)
    con.close()
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    results.to_csv(TABLES_DIR / "dq_results.csv", index=False)
    print(results[["check_id", "severity", "status", "failing_rows", "description"]]
          .to_string(index=False))
    try:
        enforce(results)
    except DataQualityError as exc:
        print(f"\n{exc}", file=sys.stderr)
        sys.exit(1)
    counts = results["status"].value_counts().to_dict()
    print(f"\nDQ passed: {counts}")


if __name__ == "__main__":
    main()
