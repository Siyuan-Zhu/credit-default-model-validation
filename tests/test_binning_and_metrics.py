"""Monotone coarse classing, train-only binning, bootstrap AUC and the DQ command line."""
import numpy as np
import pandas as pd
import pytest

import cdmv.dq
import cdmv.models
from cdmv.metrics import bootstrap_auc, calibration_table, gains_table, hosmer_lemeshow
from cdmv.models import fit_champion, load_modeling_frame, reason_codes, scorecard_points
from cdmv.woe import WoEBinner

from conftest import build_warehouse, synthetic_raw


def test_monotone_binning_gives_monotone_woe():
    rng = np.random.default_rng(1)
    n = 8000
    x = rng.uniform(0, 1, n)
    # U-shaped risk plus noise: the raw quantile bins are not monotone
    p = 0.15 + 0.5 * (x - 0.4) ** 2 + 0.03 * np.sin(40 * x)
    y = (rng.random(n) < p).astype(int)
    X = pd.DataFrame({"u": x, "steps": rng.integers(0, 7, n)})
    raw = WoEBinner(monotone=False).fit(X, y)
    assert not raw.iv_table().set_index("feature").loc["u", "monotone_woe"]
    b = WoEBinner().fit(X, y)
    for col, t in b.tables_.items():
        d = np.diff(t.woe.to_numpy())
        assert (d >= 0).all() or (d <= 0).all(), col
        assert (t.share >= b.min_bin_frac).all()
        assert b.direction_[col] in (1, -1)
    assert b.iv_table().monotone_woe.all()


def test_monotone_direction_follows_risk():
    rng = np.random.default_rng(2)
    x = rng.normal(size=5000)
    y = (rng.random(5000) < 1 / (1 + np.exp(-x))).astype(int)  # risk rises with x
    b = WoEBinner().fit(pd.DataFrame({"x": x}), y)
    assert b.direction_["x"] == -1 and b.tables_["x"].woe.is_monotonic_decreasing


def test_binner_sees_training_rows_only(monkeypatch):
    con = build_warehouse(synthetic_raw(n=2000, seed=4))
    df = load_modeling_frame(con)
    train, valid = df[df.split == "train"], df[df.split == "validation"]
    seen = []

    class Recording(WoEBinner):
        def fit(self, X, y):
            seen.append(X.index.copy())
            return super().fit(X, y)

    monkeypatch.setattr(cdmv.models, "WoEBinner", Recording)
    art = fit_champion(train, valid)
    assert len(seen) == 1
    assert set(seen[0]) == set(train.index)
    assert not set(seen[0]) & set(df[df.split != "train"].index)
    assert art["binner"].n_fit_rows_ == len(train)


def test_champion_drops_flat_features_and_gives_reasons():
    con = build_warehouse(synthetic_raw(n=3000, seed=5))
    df = load_modeling_frame(con)
    train, valid, test = (df[df.split == s] for s in ("train", "validation", "test"))
    art = fit_champion(train, valid)
    pts = scorecard_points(art)
    spread = pts.groupby("feature").points.agg(lambda p: p.max() - p.min())
    assert (spread >= cdmv.models.MIN_POINTS_SPREAD).all()
    assert set(art["c_grid"].C) == set(cdmv.models.C_GRID)
    rc = reason_codes(art, test.head(4))
    assert len(rc) == 4
    if len(art["features"]) >= 2:
        assert (rc.points_lost_1 >= rc.points_lost_2).all()
    assert (rc.points_lost_1 >= 0).all()


def test_bootstrap_auc():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 3000)
    good = y + rng.normal(scale=0.8, size=3000)
    weak = y + rng.normal(scale=3.0, size=3000)
    same = bootstrap_auc(y, good, good, n_boot=200)
    assert same["diff_mean"] == 0 and same["diff_ci"] == [0.0, 0.0]
    better = bootstrap_auc(y, weak, good, n_boot=200)
    assert better["diff_ci"][0] > 0 and better["n_boot"] == 200
    lo, hi = better["auc_b_ci"]
    assert lo < hi and lo > 0.5


def test_hosmer_lemeshow_degrees_of_freedom():
    rng = np.random.default_rng(0)
    p = rng.uniform(0.05, 0.6, 5000)
    y = (rng.random(5000) < p).astype(int)
    t = calibration_table(y, p)
    assert hosmer_lemeshow(t)["dof"] == 8
    holdout = hosmer_lemeshow(t, fitted_on_sample=False)
    assert holdout["dof"] == 10 and holdout["chi2"] == hosmer_lemeshow(t)["chi2"]


def test_gains_table():
    y = np.array([1, 1, 0, 1, 0, 0, 0, 0, 0, 0])
    p = np.linspace(0.9, 0.1, 10)
    g = gains_table(y, p, n_bins=5)
    assert g.band.tolist() == [1, 2, 3, 4, 5]
    assert g.defaults.tolist() == [2, 1, 0, 0, 0]
    assert g.cum_share_defaults.iloc[-1] == 1.0
    assert abs(g.cum_share_defaults.iloc[0] - 2 / 3) < 1e-12


@pytest.mark.parametrize("break_it", [False, True])
def test_dq_command_line_exit_code(monkeypatch, tmp_path, raw_df, break_it):
    if break_it:
        raw_df.loc[1, "ID"] = raw_df.loc[0, "ID"]  # duplicate key: error-level failure
    con = build_warehouse(raw_df)
    monkeypatch.setattr(cdmv.dq, "connect", lambda: con)
    monkeypatch.setattr(cdmv.dq, "TABLES_DIR", tmp_path)
    monkeypatch.setattr(cdmv.dq, "EXPECTED_ROWS", len(raw_df))
    if break_it:
        with pytest.raises(SystemExit) as exc:
            cdmv.dq.main()
        assert exc.value.code == 1
    else:
        cdmv.dq.main()
    assert (tmp_path / "dq_results.csv").exists()
