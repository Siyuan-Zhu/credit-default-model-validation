import pytest

from cdmv.dq import DataQualityError, enforce, run_checks
from cdmv.warehouse import render

from conftest import build_warehouse


def checks(con, n):
    return run_checks(con, {"expected_rows": n}).set_index("check_id")


def test_clean_data_passes(warehouse, raw_df):
    res = checks(warehouse, len(raw_df))
    assert (res.status != "FAIL").all(), res[res.status == "FAIL"]
    enforce(res.reset_index())  # does not raise
    assert len(res) >= 25
    assert warehouse.execute("select count(*) from dq.results").fetchone()[0] == len(res)


def test_row_count_mismatch_fails(warehouse, raw_df):
    res = checks(warehouse, len(raw_df) + 1)
    assert res.loc["RAW_01", "status"] == "FAIL"
    with pytest.raises(DataQualityError, match="RAW_01"):
        enforce(res.reset_index())


def test_duplicate_id_fails_loudly(raw_df):
    raw_df.loc[1, "ID"] = raw_df.loc[0, "ID"]
    con = build_warehouse(raw_df)
    res = checks(con, len(raw_df))
    assert res.loc["RAW_02", "status"] == "FAIL"
    assert res.loc["FEA_02", "status"] == "FAIL"
    with pytest.raises(DataQualityError, match="RAW_02"):
        enforce(res.reset_index())


def test_null_value_fails(raw_df):
    raw_df["LIMIT_BAL"] = raw_df["LIMIT_BAL"].astype(float)
    raw_df.loc[3, "LIMIT_BAL"] = float("nan")
    con = build_warehouse(raw_df)
    res = checks(con, len(raw_df))
    assert res.loc["RAW_03", "failing_rows"] == 1
    assert res.loc["RAW_03", "status"] == "FAIL"


def test_out_of_range_values_fail(raw_df):
    raw_df.loc[0, "PAY_AMT3"] = -5
    raw_df.loc[1, "AGE"] = 150
    raw_df.loc[2, "default payment next month"] = 2
    con = build_warehouse(raw_df)
    res = checks(con, len(raw_df))
    for check in ("RAW_09", "RAW_06", "RAW_04"):
        assert res.loc[check, "status"] == "FAIL", check


def test_undocumented_codes_warn_and_are_mapped(raw_df):
    raw_df.loc[0:4, "EDUCATION"] = [0, 5, 6, 5, 0]
    raw_df.loc[0:1, "MARRIAGE"] = 0
    con = build_warehouse(raw_df)
    res = checks(con, len(raw_df))
    assert res.loc["RAW_10", "status"] == "WARN" and res.loc["RAW_10", "failing_rows"] == 5
    assert res.loc["RAW_11", "status"] == "WARN" and res.loc["RAW_11", "failing_rows"] == 2
    assert res.loc["STG_03", "status"] == "PASS"
    mapped = con.execute(
        "select education_code, marriage_code, education_code_undocumented "
        "from staging.stg_accounts where account_id = 1").fetchone()
    assert mapped == (4, 3, True)


def test_render_rejects_unsafe_parameters():
    assert render("select {{n}}", {"n": 5}) == "select 5"
    with pytest.raises(ValueError):
        render("select {{n}}", {"n": "1; drop table x"})
    with pytest.raises(KeyError):
        render("select {{missing}}", {})
