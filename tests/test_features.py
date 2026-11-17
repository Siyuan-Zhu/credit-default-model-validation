"""Feature SQL on hand-built accounts with known answers."""
import pandas as pd

from conftest import build_warehouse, synthetic_raw


def features_for(pay_statuses, bills, pays, limit=10_000):
    df = synthetic_raw(n=2)
    pay_cols = ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]
    df.loc[0, pay_cols] = pay_statuses
    df.loc[0, [f"BILL_AMT{k}" for k in range(1, 7)]] = bills
    df.loc[0, [f"PAY_AMT{k}" for k in range(1, 7)]] = pays
    df.loc[0, "LIMIT_BAL"] = limit
    con = build_warehouse(df)
    return con.execute("select * from features.account_features where account_id = 1").df().iloc[0]


def test_delinquency_counts_and_streaks():
    # m1..m6 = Sep..Apr: late 2, 1 then on time, then late 3, 2, 1
    f = features_for([2, 1, -1, 3, 2, 1], [1000] * 6, [100] * 6)
    assert f.months_delinquent_m1 == 2
    assert f.max_months_delinquent_6m == 3
    assert f.n_months_delinquent_6m == 5
    assert f.current_delinq_streak == 2
    assert f.longest_delinq_streak_6m == 3
    assert f.n_months_paid_duly_6m == 1


def test_no_delinquency_and_undocumented_codes():
    f = features_for([-2, -2, 0, 0, 0, -1], [0, 0, 500, 500, 500, 500], [0, 0, 0, 500, 500, 0])
    assert f.n_months_delinquent_6m == 0
    assert f.current_delinq_streak == 0
    assert f.longest_delinq_streak_6m == 0
    assert f.n_months_status_neg2_6m == 2
    assert f.n_months_status_zero_6m == 3
    assert f.n_months_zero_payment_6m == 4


def test_payment_ratio_rules():
    # pay m1 against bill m2, etc.; zero previous bill counts as fully paid; capped at 1
    bills = [1000, 2000, 0, 1000, 500, 400]
    pays = [500, 300, 0, 900, 1000, 0]
    f = features_for([0] * 6, bills, pays, limit=10_000)
    expected = [500 / 2000, 1.0, 0 / 1000, 900 / 500, 1000 / 400]
    expected = [min(x, 1.0) for x in expected]
    assert abs(f.pay_ratio_m1 - 0.25) < 1e-12
    assert abs(f.avg_pay_ratio_5m - sum(expected) / 5) < 1e-12
    assert f.min_pay_ratio_5m == 0
    assert abs(f.util_m1 - 0.1) < 1e-12
    assert abs(f.util_change_6m - (0.1 - 0.04)) < 1e-12
    assert abs(f.bill_growth_3m - 0.1) < 1e-12
    assert abs(f.pay_to_limit_6m - sum(pays) / 10_000) < 1e-12


def test_split_is_stratified_and_complete():
    con = build_warehouse(synthetic_raw(n=1000, seed=3))
    s = con.execute(
        "select split, count(*) n, avg(default_flag) dr from features.model_split "
        "join features.account_features using (account_id) group by 1").df().set_index("split")
    assert s.n.sum() == 1000
    assert abs(s.loc["train", "n"] / 1000 - 0.6) < 0.01
    assert (s.dr - s.dr.mean()).abs().max() < 0.01
