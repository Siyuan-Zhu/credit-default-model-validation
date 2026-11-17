"""Train, validate, draw figures and write the report on synthetic data in a temp directory."""
import os
import subprocess
import sys

from cdmv.config import ROOT
from cdmv.dq import run_checks
from cdmv.pipeline import build
from cdmv.warehouse import connect, load_raw

from conftest import synthetic_raw

STEPS = ["cdmv.train", "cdmv.validate", "cdmv.figures", "cdmv.report"]


def test_pipeline_end_to_end(tmp_path):
    raw = synthetic_raw(n=3000, seed=7)
    con = connect(tmp_path / "warehouse" / "credit.duckdb")
    load_raw(con, raw)
    build(con)
    tables = tmp_path / "reports" / "tables"
    tables.mkdir(parents=True)
    dq = run_checks(con, {"expected_rows": len(raw)})
    assert (dq.status != "FAIL").all()
    dq.to_csv(tables / "dq_results.csv", index=False)
    con.close()
    (tmp_path / "README.md").write_text("x\n<!-- results:start -->\nold\n<!-- results:end -->\n")

    env = {**os.environ, "CDMV_WORKDIR": str(tmp_path),
           "PYTHONPATH": str(ROOT / "src") + os.pathsep + os.environ.get("PYTHONPATH", "")}
    for step in STEPS:
        res = subprocess.run([sys.executable, "-m", step], env=env, capture_output=True,
                             text=True, cwd=tmp_path)
        assert res.returncode == 0, f"{step} failed:\n{res.stderr[-2000:]}"

    report = (tmp_path / "reports" / "model_validation.md").read_text()
    for heading in ("## Summary", "### 5.1 Score bands", "## 6. Calibration",
                    "Example reason codes", "## 11. Ongoing monitoring plan", "## 12. Conclusions"):
        assert heading in report, heading
    assert " nan" not in report and "None" not in report
    readme = (tmp_path / "README.md").read_text()
    assert "old" not in readme and "The report lists" in readme
    assert len(list((tmp_path / "reports" / "figures").glob("*.png"))) == 6
